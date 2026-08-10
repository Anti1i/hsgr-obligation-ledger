"""Phase-G1: causal MVP for hierarchy-conditioned hidden-state guidance.

Question
--------
Can a typed dependency relation produce a reusable activation direction that
improves held-out hop execution beyond (a) the original structured prompt and
(b) spelling out the same guide in text?

The experiment uses MuSiQue gold decompositions only to define a controlled
dependency-use task.  Every arm sees identical evidence, current-hop text, and
verified predecessor values.  A calibration split is used to construct guide
directions and select one feature layer / intervention strength.  The held-out
test split is touched once with four greedy-decoding arms:

  base       structured prompt, no additional guide
  text       base + an explicit dependency-use guide
  latent     base + a hidden-state direction distilled from guide contrasts
  anti       base - the same hidden-state direction

Pre-registered success gate (held-out): latent-base >= 5pp with paired p<.05,
latent > text, and latent-anti >= 5pp.  Failure stops the mechanism line; a
readable or behavior-changing direction alone is not enough.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import answers_match, evidence_from_row, extract_boxed  # noqa: E402
from mh_e0 import hop_deps, load_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You execute exactly one node in a typed multi-hop reasoning hierarchy. "
    "Use only the supplied evidence and verified predecessor mappings. "
    "Return the current node answer in \\boxed{}."
)

USER = """Evidence (identical across experimental arms):
{evidence}

Original question: {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[GOAL] {goal}
[DEPENDS_ON]
{dependencies}

Execute only the current node and put its answer in \\boxed{{}}.{guide}"""

TEXT_GUIDE = (
    "\n[GUIDE STATE] dependency-use=REQUIRED. Ground the computation in the "
    "verified predecessor mappings; do not bypass or replace them."
)
ANTI_GUIDE = (
    "\n[GUIDE STATE] dependency-use=FORBIDDEN. Ignore the predecessor mappings "
    "and reason as if they were unavailable."
)

# These are hidden-state indices as returned by output_hidden_states.  The
# corresponding intervention is placed after decoder block feature_layer - 1.
FEATURE_LAYERS = (7, 14, 21, 28)
BETAS = (0.03, 0.06, 0.12, 0.20)


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test via a Binomial(n, .5) tail."""
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p": p}


def make_units(rows: list[dict], seed: int) -> list[dict]:
    units = []
    for row in rows:
        decomp = row.get("question_decomposition") or []
        if not decomp:
            continue
        deps_all = hop_deps(decomp)
        hop = len(decomp) - 1
        deps = deps_all[hop]
        if not deps:
            continue
        dep_lines = [f"  - #{j + 1} = {decomp[j]['answer']} (verified)" for j in deps]
        base = USER.format(
            evidence=evidence_from_row(row),
            question=row["question"],
            hop=hop + 1,
            n_hops=len(decomp),
            goal=decomp[hop]["question"],
            dependencies="\n".join(dep_lines),
            guide="",
        )
        units.append({
            "id": row.get("id") or row.get("_uid"),
            "n_hops": len(decomp),
            "goal": decomp[hop]["question"],
            "gold": str(row["answer"]),
            "aliases": list(row.get("answer_aliases") or []),
            "base_user": base,
            "text_user": base + TEXT_GUIDE,
            "anti_text_user": base + ANTI_GUIDE,
        })
    random.Random(seed).shuffle(units)
    return units


class GuidedRunner:
    def __init__(self, model_id: str, max_context: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        torch.manual_seed(0)
        self.torch = torch
        self.max_context = max_context
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tok.truncation_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()
        self.vectors: dict[int, object] = {}
        self.token_counts = Counter()
        print(f"[model] loaded {model_id} in {time.time() - t0:.1f}s", flush=True)

    def template(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode(self, users: list[str]):
        return self.tok(
            [self.template(u) for u in users],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context,
        ).to("cuda")

    def last_hidden(self, users: list[str], feature_layers: tuple[int, ...], bs: int):
        feats = {layer: [] for layer in feature_layers}
        for start in range(0, len(users), bs):
            enc = self.encode(users[start:start + bs])
            with self.torch.no_grad():
                out = self.model(**enc, output_hidden_states=True, use_cache=False)
            for layer in feature_layers:
                feats[layer].append(out.hidden_states[layer][:, -1, :].float().cpu())
            del enc, out
            self.torch.cuda.empty_cache()
        return {layer: self.torch.cat(parts) for layer, parts in feats.items()}

    def build_vectors(self, units: list[dict], bs: int):
        pos = self.last_hidden([u["text_user"] for u in units], FEATURE_LAYERS, bs)
        neg = self.last_hidden([u["anti_text_user"] for u in units], FEATURE_LAYERS, bs)
        for layer in FEATURE_LAYERS:
            paired = pos[layer] - neg[layer]
            vec = paired.mean(0)
            self.vectors[layer] = (vec / (vec.norm() + 1e-8)).to(
                "cuda", dtype=self.torch.bfloat16
            )
            mean_cos = self.torch.nn.functional.cosine_similarity(
                paired, vec.unsqueeze(0), dim=-1
            ).mean().item()
            print(
                f"[vector] feature_layer={layer} block={layer - 1} "
                f"norm={vec.norm().item():.3f} mean_pair_cos={mean_cos:.3f}",
                flush=True,
            )

    def _install_hook(self, feature_layer: int, beta: float):
        vec = self.vectors[feature_layer].view(1, 1, -1)
        block = self.model.model.layers[feature_layer - 1]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Intervene only at the active (last) position: the final prompt
            # position during prefill and the newly decoded position thereafter.
            h = hidden[:, -1:, :]
            norms = h.norm(dim=-1, keepdim=True)
            mixed = h / (norms + 1e-6) + float(beta) * vec
            mixed = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-6)
            changed = hidden.clone()
            changed[:, -1:, :] = mixed * norms
            if isinstance(output, tuple):
                return (changed,) + output[1:]
            return changed

        return block.register_forward_hook(hook)

    def generate(
        self,
        users: list[str],
        arm: str,
        bs: int,
        max_new: int,
        feature_layer: int | None = None,
        beta: float = 0.0,
    ) -> list[str]:
        outputs = []
        for start in range(0, len(users), bs):
            chunk = users[start:start + bs]
            enc = self.encode(chunk)
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            self.token_counts[f"{arm}_prompt"] += sum(int(x) for x in prompt_lens)
            handle = None
            if feature_layer is not None and beta:
                handle = self._install_hook(feature_layer, beta)
            try:
                with self.torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
            finally:
                if handle is not None:
                    handle.remove()
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new = gen[j, plen:]
                n_new = int((new != self.tok.pad_token_id).sum().item())
                self.token_counts[f"{arm}_gen"] += n_new
                outputs.append(self.tok.decode(new, skip_special_tokens=True))
            del enc, gen
            self.torch.cuda.empty_cache()
        return outputs


def score(units: list[dict], texts: list[str]) -> tuple[list[bool], list[dict]]:
    hits, details = [], []
    for unit, text in zip(units, texts):
        ans = extract_boxed(text)
        hit = bool(ans and answers_match(ans, unit["gold"], unit["aliases"]))
        hits.append(hit)
        details.append({"answer": ans, "hit": hit, "text": text[:600]})
    return hits, details


def acc(hits: list[bool]) -> float:
    return sum(hits) / max(1, len(hits))


def main(args):
    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    rows = load_rows(data, args.limit, seed=0)
    units = make_units(rows, args.seed)
    if len(units) <= args.calib:
        raise SystemExit(f"need more than {args.calib} dependency units; found {len(units)}")
    calib, test = units[:args.calib], units[args.calib:]
    print(
        f"[data] dependency units={len(units)} calib={len(calib)} test={len(test)} "
        f"hop_counts={dict(Counter(u['n_hops'] for u in units))}",
        flush=True,
    )

    runner = GuidedRunner(args.model, args.max_context)
    runner.build_vectors(calib, args.bs_hidden)

    # Calibration baselines are descriptive.  Only latent accuracy selects the
    # layer/beta; the held-out split remains untouched until selection is fixed.
    cal_base_texts = runner.generate(
        [u["base_user"] for u in calib], "cal_base", args.bs_generate, args.max_new
    )
    cal_text_texts = runner.generate(
        [u["text_user"] for u in calib], "cal_text", args.bs_generate, args.max_new
    )
    cal_base, _ = score(calib, cal_base_texts)
    cal_text, _ = score(calib, cal_text_texts)
    grid = []
    best = None
    for layer in FEATURE_LAYERS:
        for beta in BETAS:
            name = f"cal_l{layer}_b{beta:.2f}"
            texts = runner.generate(
                [u["base_user"] for u in calib],
                name,
                args.bs_generate,
                args.max_new,
                feature_layer=layer,
                beta=beta,
            )
            hits, _ = score(calib, texts)
            row = {"feature_layer": layer, "beta": beta, "acc": acc(hits)}
            grid.append(row)
            key = (row["acc"], -beta, -layer)
            if best is None or key > best[0]:
                best = (key, row)
            print(f"[calib] layer={layer} beta={beta:.2f} acc={row['acc']:.3f}", flush=True)
    selected = best[1]
    print(
        f"[select] feature_layer={selected['feature_layer']} beta={selected['beta']:.2f} "
        f"cal_acc={selected['acc']:.3f} base={acc(cal_base):.3f} text={acc(cal_text):.3f}",
        flush=True,
    )

    # One held-out evaluation after the layer/strength has been fixed.
    arm_specs = {
        "base": ([u["base_user"] for u in test], None, 0.0),
        "text": ([u["text_user"] for u in test], None, 0.0),
        "latent": (
            [u["base_user"] for u in test],
            int(selected["feature_layer"]),
            float(selected["beta"]),
        ),
        "anti": (
            [u["base_user"] for u in test],
            int(selected["feature_layer"]),
            -float(selected["beta"]),
        ),
    }
    hits, details = {}, {}
    for arm, (users, layer, beta) in arm_specs.items():
        texts = runner.generate(
            users,
            arm,
            args.bs_generate,
            args.max_new,
            feature_layer=layer,
            beta=beta,
        )
        hits[arm], details[arm] = score(test, texts)
        print(f"[test] {arm} acc={acc(hits[arm]):.3f}", flush=True)

    accuracy = {arm: acc(v) for arm, v in hits.items()}
    paired = {
        "latent_vs_base": exact_mcnemar(hits["latent"], hits["base"]),
        "latent_vs_text": exact_mcnemar(hits["latent"], hits["text"]),
        "latent_vs_anti": exact_mcnemar(hits["latent"], hits["anti"]),
    }
    delta_base = accuracy["latent"] - accuracy["base"]
    delta_text = accuracy["latent"] - accuracy["text"]
    delta_anti = accuracy["latent"] - accuracy["anti"]
    gates = {
        "gain_vs_base": delta_base >= 0.05 and paired["latent_vs_base"]["p"] < 0.05,
        "beats_text_guide": delta_text > 0,
        "direction_specific": delta_anti >= 0.05,
    }
    gate_pass = all(gates.values())

    cases_path = os.path.join(args.out_dir, "hsgr_guide_cases.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(cases_path, "w", encoding="utf-8") as f:
        for i, unit in enumerate(test):
            f.write(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "goal": unit["goal"],
                "gold": unit["gold"],
                "aliases": unit["aliases"],
                "arms": {arm: details[arm][i] for arm in arm_specs},
            }, ensure_ascii=False) + "\n")

    report = {
        "experiment": "HSGR dependency-use latent guide MVP",
        "n": {"all": len(units), "calib": len(calib), "test": len(test)},
        "split_seed": args.seed,
        "selected": selected,
        "calibration": {
            "base_acc": acc(cal_base),
            "text_acc": acc(cal_text),
            "grid": grid,
        },
        "test_accuracy": accuracy,
        "delta": {
            "latent_vs_base": delta_base,
            "latent_vs_text": delta_text,
            "latent_vs_anti": delta_anti,
        },
        "paired_exact_mcnemar": paired,
        "gates": gates,
        "gate_pass": gate_pass,
        "token_counts": dict(runner.token_counts),
        "feature_layers": list(FEATURE_LAYERS),
        "betas": list(BETAS),
    }
    report_path = os.path.join(args.out_dir, "hsgr_guide_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print("\n== HSGR dependency-use guide held-out result ==")
    print(json.dumps({
        "selected": selected,
        "accuracy": accuracy,
        "delta": report["delta"],
        "paired": paired,
        "gates": gates,
        "gate_pass": gate_pass,
    }, indent=1))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--out-dir", default="hsgr_guide")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--bs-hidden", type=int, default=8)
    ap.add_argument("--bs-generate", type=int, default=8)
    main(ap.parse_args())
