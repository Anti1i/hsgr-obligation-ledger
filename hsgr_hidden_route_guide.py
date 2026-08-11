"""Held-out sample-specific hidden RouteGuide action ceiling.

This experiment is intentionally different from the failed fixed
dependency-use direction.  Every arm receives the exact same prompt, compiled
current goal, verified predecessor values, and complete gold-support evidence.
For each item, a clean prefill produces token hidden states.  A relation-query
conditioned route vector is then built from either:

* the oracle final-hop evidence block (correct_route), or
* a length-matched predecessor block (wrong_route).

The selected vector is sample-specific and is injected only at the active last
position of one decoder layer during greedy decoding.  Thus the comparison
tests a hidden-state control channel, not compliance with a textual marker and
not evidence pruning/retrieval.

The calibration split selects pooling mode, feature layer, and beta by the
minimum of correct-vs-base and correct-vs-wrong normalized-EM deltas.  The
held-out split is decoded exactly once after selection.  Pre-registered gates:

1. correct_route beats base by >=3pp EM and >=.03 official answer F1, with
   paired p<.05 for both;
2. correct_route beats wrong_route by >=3pp EM with exact McNemar p<.05;
3. correct-minus-base EM does not decrease from 2-hop to 3-hop to 4-hop.

All gates must pass before training a deployable hidden-state route observer.
Gold route indices measure oracle action headroom only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

from hsgr_focus_route_ceiling import (
    SYSTEM,
    build_base_prompt,
    exact_mcnemar,
    make_unit,
    paired_signflip_p,
    score_text,
    unit_id,
)
from mh_e0 import load_rows


FEATURE_LAYERS = (14, 21, 28)
BETAS = (0.04, 0.08, 0.16, 0.32)
POOL_MODES = ("mean", "qtopk")


def mean(values):
    return sum(values) / max(1, len(values))


def read_ids(path: str) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        return {str(json.loads(line)["id"]) for line in handle if line.strip()}


def select_units(
    data: str,
    exclude_cases: list[str],
    original_limit: int,
    calib_n: int,
    test_n: int,
    seed: int,
):
    original_ids = {unit_id(row) for row in load_rows(data, original_limit, seed=0)}
    exclude_sets = []
    for path in exclude_cases:
        ids = read_ids(path)
        exclude_sets.append((os.path.abspath(path), ids))
    excluded = set(original_ids)
    for _, ids in exclude_sets:
        excluded.update(ids)

    pool = []
    skipped = 0
    for row in load_rows(data, 0, seed=0):
        if unit_id(row) in excluded:
            continue
        unit = make_unit(row)
        if unit is None:
            skipped += 1
            continue
        pool.append(unit)
    need = calib_n + test_n
    if len(pool) < need:
        raise SystemExit(f"need {need} untouched units; found {len(pool)}")
    chosen = random.Random(seed).sample(pool, need)
    calib, test = chosen[:calib_n], chosen[calib_n:]

    meta = {
        "original_excluded": len(original_ids),
        "exclude_sources": [
            {
                "path": path,
                "n": len(ids),
                "id_sha256": hashlib.sha256(
                    "\n".join(sorted(ids)).encode("utf-8")
                ).hexdigest(),
            }
            for path, ids in exclude_sets
        ],
        "excluded_union": len(excluded),
        "untouched_pool": len(pool),
        "skipped_incomplete_support": skipped,
        "selection_seed": seed,
        "calib_n": len(calib),
        "test_n": len(test),
        "calib_hops": dict(Counter(u["n_hops"] for u in calib)),
        "test_hops": dict(Counter(u["n_hops"] for u in test)),
        "calib_id_sha256": hashlib.sha256(
            "\n".join(sorted(u["id"] for u in calib)).encode("utf-8")
        ).hexdigest(),
        "test_id_sha256": hashlib.sha256(
            "\n".join(sorted(u["id"] for u in test)).encode("utf-8")
        ).hexdigest(),
    }
    return calib, test, meta


def char_spans(chat_text: str, n_blocks: int):
    goal_marker = "[COMPILED CURRENT GOAL] "
    goal_start = chat_text.index(goal_marker) + len(goal_marker)
    goal_end = chat_text.index("\n", goal_start)
    evidence = []
    cursor = goal_end
    for i in range(n_blocks):
        marker = f"[E{i + 1}]\n"
        marker_pos = chat_text.index(marker, cursor)
        start = marker_pos + len(marker)
        if i + 1 < n_blocks:
            end = chat_text.index(f"\n\n[E{i + 2}]\n", start)
        else:
            end = chat_text.index("\n\nExecute only the compiled current goal.", start)
        evidence.append((start, end))
        cursor = end
    return (goal_start, goal_end), evidence


def token_indices(offsets, span, pad: int):
    start, end = span
    idxs = [
        i + pad
        for i, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not idxs:
        raise RuntimeError(f"empty token span for chars {span}")
    return idxs


class HiddenRouteRunner:
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
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()
        self.token_counts = defaultdict(int)
        print(
            f"[model] {model_id} layers={len(self.model.model.layers)} "
            f"hidden={self.model.config.hidden_size}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def encode_texts(self, texts):
        return self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context,
        ).to("cuda")

    def _pool(self, hidden, idxs, q, mode: str):
        torch = self.torch
        values = hidden[idxs].float()
        if mode == "mean":
            return values.mean(dim=0)
        if mode != "qtopk":
            raise ValueError(mode)
        qn = q / (q.norm() + 1e-8)
        vn = values / (values.norm(dim=-1, keepdim=True) + 1e-8)
        scores = vn @ qn
        k = min(16, values.shape[0])
        keep = torch.topk(scores, k=k, largest=True).indices
        return values[keep].mean(dim=0)

    def extract_guides(self, units, users, feature_layers, pool_modes, bs):
        torch = self.torch
        store = {
            mode: {
                layer: {"correct_route": [], "wrong_route": []}
                for layer in feature_layers
            }
            for mode in pool_modes
        }
        for start in range(0, len(units), bs):
            chunk_units = units[start : start + bs]
            chunk_users = users[start : start + bs]
            texts = [self.chat_text(user) for user in chunk_users]
            enc = self.encode_texts(texts)
            with torch.no_grad():
                out = self.model(
                    **enc, output_hidden_states=True, use_cache=False, return_dict=True
                )
            max_len = enc["input_ids"].shape[1]
            index_meta = []
            for text, unit in zip(texts, chunk_units):
                single = self.tok(
                    text,
                    truncation=True,
                    max_length=self.max_context,
                    return_offsets_mapping=True,
                )
                ids = single["input_ids"]
                offsets = single["offset_mapping"]
                pad = max_len - len(ids)
                goal_chars, evidence_chars = char_spans(text, len(unit["blocks"]))
                goal_idxs = token_indices(offsets, goal_chars, pad)
                evidence_idxs = [token_indices(offsets, span, pad) for span in evidence_chars]
                index_meta.append((goal_idxs, evidence_idxs))

            for feature_layer in feature_layers:
                hidden = out.hidden_states[feature_layer]
                for b, unit in enumerate(chunk_units):
                    goal_idxs, evidence_idxs = index_meta[b]
                    q = hidden[b, goal_idxs].float().mean(dim=0)
                    for mode in pool_modes:
                        for arm, selected in (
                            ("correct_route", unit["correct_focus"]),
                            ("wrong_route", unit["wrong_focus"]),
                        ):
                            other = [
                                idx
                                for j, block_idxs in enumerate(evidence_idxs)
                                if j != selected
                                for idx in block_idxs
                            ]
                            chosen = self._pool(
                                hidden[b], evidence_idxs[selected], q, mode
                            )
                            contrast = self._pool(hidden[b], other, q, mode)
                            guide = chosen - contrast
                            guide = guide / (guide.norm() + 1e-8)
                            store[mode][feature_layer][arm].append(guide.cpu())
            del enc, out
            torch.cuda.empty_cache()
            print(
                f"[guide-prefill] {min(start + bs, len(units))}/{len(units)}",
                flush=True,
            )
        return {
            mode: {
                layer: {
                    arm: torch.stack(parts)
                    for arm, parts in arms.items()
                }
                for layer, arms in layers.items()
            }
            for mode, layers in store.items()
        }

    def _install_hook(self, feature_layer: int, beta: float, guides):
        torch = self.torch
        vec = guides.to("cuda", dtype=torch.bfloat16).unsqueeze(1)
        block = self.model.model.layers[feature_layer - 1]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
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

    def generate(self, users, arm, bs, max_new, feature_layer=None, beta=0.0, guides=None):
        torch = self.torch
        outputs = []
        for start in range(0, len(users), bs):
            chunk = users[start : start + bs]
            texts = [self.chat_text(user) for user in chunk]
            enc = self.encode_texts(texts)
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            self.token_counts[f"{arm}_prompt"] += sum(int(x) for x in prompt_lens)
            handle = None
            if feature_layer is not None:
                if guides is None:
                    raise ValueError("guided generation requires vectors")
                handle = self._install_hook(
                    int(feature_layer), float(beta), guides[start : start + len(chunk)]
                )
            try:
                with torch.no_grad():
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
                self.token_counts[f"{arm}_generated"] += int(
                    (new != self.tok.pad_token_id).sum().item()
                )
                outputs.append(self.tok.decode(new, skip_special_tokens=True))
            del enc, gen
            torch.cuda.empty_cache()
        return outputs


def evaluate(units, texts):
    em, f1, answers = [], [], []
    for unit, text in zip(units, texts):
        hit, score, answer = score_text(unit, text)
        em.append(hit)
        f1.append(score)
        answers.append(answer)
    return em, f1, answers


def direct_delta(a, b):
    return mean(a) - mean(b)


def main(args):
    calib, test, data_meta = select_units(
        data=args.data,
        exclude_cases=args.exclude_cases,
        original_limit=args.original_exclude_limit,
        calib_n=args.calib,
        test_n=args.test,
        seed=args.seed,
    )
    print("[data] " + json.dumps(data_meta, ensure_ascii=False), flush=True)
    if args.dry_run:
        for unit in (calib[:1] + test[:1]):
            user = build_base_prompt(unit)
            goal_chars, evidence_chars = char_spans(user, len(unit["blocks"]))
            print(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "goal": unit["fields"]["compiled_goal"],
                "correct_route": unit["correct_focus"],
                "wrong_route": unit["wrong_focus"],
                "goal_chars": goal_chars,
                "evidence_chars": evidence_chars,
            }, ensure_ascii=False))
        return

    runner = HiddenRouteRunner(args.model, args.max_context)
    cal_users = [build_base_prompt(u) for u in calib]
    cal_guides = runner.extract_guides(
        calib, cal_users, FEATURE_LAYERS, POOL_MODES, args.bs_hidden
    )
    cal_base_texts = runner.generate(
        cal_users, "cal_base", args.bs_generate, args.max_new
    )
    cal_base_em, cal_base_f1, _ = evaluate(calib, cal_base_texts)
    print(
        f"[cal-base] em={mean(cal_base_em):.4f} f1={mean(cal_base_f1):.4f}",
        flush=True,
    )

    grid = []
    best = None
    for mode in POOL_MODES:
        for layer in FEATURE_LAYERS:
            for beta in BETAS:
                row = {"pool": mode, "feature_layer": layer, "beta": beta}
                arm_scores = {}
                for arm in ("correct_route", "wrong_route"):
                    name = f"cal_{mode}_l{layer}_b{beta:.2f}_{arm}"
                    texts = runner.generate(
                        cal_users,
                        name,
                        args.bs_generate,
                        args.max_new,
                        feature_layer=layer,
                        beta=beta,
                        guides=cal_guides[mode][layer][arm],
                    )
                    em, f1, _ = evaluate(calib, texts)
                    arm_scores[arm] = {"em": mean(em), "f1": mean(f1)}
                row.update(
                    correct_em=arm_scores["correct_route"]["em"],
                    correct_f1=arm_scores["correct_route"]["f1"],
                    wrong_em=arm_scores["wrong_route"]["em"],
                    wrong_f1=arm_scores["wrong_route"]["f1"],
                )
                row["delta_correct_base_em"] = row["correct_em"] - mean(cal_base_em)
                row["delta_correct_wrong_em"] = row["correct_em"] - row["wrong_em"]
                row["delta_correct_base_f1"] = row["correct_f1"] - mean(cal_base_f1)
                grid.append(row)
                key = (
                    min(row["delta_correct_base_em"], row["delta_correct_wrong_em"]),
                    row["delta_correct_base_em"] + row["delta_correct_wrong_em"],
                    row["delta_correct_base_f1"],
                    -beta,
                    -layer,
                    mode == "qtopk",
                )
                if best is None or key > best[0]:
                    best = (key, row)
                print("[cal-grid] " + json.dumps(row, sort_keys=True), flush=True)

    selected = best[1]
    print("[selected] " + json.dumps(selected, sort_keys=True), flush=True)
    del cal_guides
    runner.torch.cuda.empty_cache()

    test_users = [build_base_prompt(u) for u in test]
    test_guides = runner.extract_guides(
        test,
        test_users,
        [int(selected["feature_layer"])],
        [selected["pool"]],
        args.bs_hidden,
    )
    test_outputs = {}
    test_outputs["base"] = runner.generate(
        test_users, "test_base", args.bs_generate, args.max_new
    )
    for arm in ("correct_route", "wrong_route"):
        test_outputs[arm] = runner.generate(
            test_users,
            f"test_{arm}",
            args.bs_generate,
            args.max_new,
            feature_layer=int(selected["feature_layer"]),
            beta=float(selected["beta"]),
            guides=test_guides[selected["pool"]][int(selected["feature_layer"])][arm],
        )

    results = {}
    for arm, texts in test_outputs.items():
        em, f1, answers = evaluate(test, texts)
        results[arm] = {"em": em, "f1": f1, "answers": answers}
        print(f"[test] {arm} em={mean(em):.4f} f1={mean(f1):.4f}", flush=True)

    paired = {
        "correct_vs_base_em": exact_mcnemar(
            results["correct_route"]["em"], results["base"]["em"]
        ),
        "correct_vs_wrong_em": exact_mcnemar(
            results["correct_route"]["em"], results["wrong_route"]["em"]
        ),
        "correct_vs_base_f1": paired_signflip_p(
            results["correct_route"]["f1"],
            results["base"]["f1"],
            seed=args.seed,
            samples=args.permutation_samples,
        ),
    }
    deltas = {
        "correct_vs_base_em": direct_delta(
            results["correct_route"]["em"], results["base"]["em"]
        ),
        "correct_vs_wrong_em": direct_delta(
            results["correct_route"]["em"], results["wrong_route"]["em"]
        ),
        "correct_vs_base_f1": direct_delta(
            results["correct_route"]["f1"], results["base"]["f1"]
        ),
    }
    depth_delta = {}
    for hop in sorted({u["n_hops"] for u in test}):
        idxs = [i for i, u in enumerate(test) if u["n_hops"] == hop]
        depth_delta[str(hop)] = mean(
            [results["correct_route"]["em"][i] for i in idxs]
        ) - mean([results["base"]["em"][i] for i in idxs])
    depth_keys = [str(h) for h in (2, 3, 4) if str(h) in depth_delta]
    depth_non_decreasing = all(
        depth_delta[b] >= depth_delta[a] - 1e-12
        for a, b in zip(depth_keys, depth_keys[1:])
    )
    gates = {
        "gain_vs_base": (
            deltas["correct_vs_base_em"] >= 0.03
            and deltas["correct_vs_base_f1"] >= 0.03
            and paired["correct_vs_base_em"]["p"] < 0.05
            and paired["correct_vs_base_f1"]["p"] < 0.05
        ),
        "route_specificity": (
            deltas["correct_vs_wrong_em"] >= 0.03
            and paired["correct_vs_wrong_em"]["p"] < 0.05
        ),
        "depth_non_decreasing": depth_non_decreasing,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    cases_path = os.path.join(args.out_dir, "hsgr_hidden_route_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for i, unit in enumerate(test):
            handle.write(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "compiled_goal": unit["fields"]["compiled_goal"],
                "gold": unit["gold"],
                "correct_route": unit["correct_focus"],
                "wrong_route": unit["wrong_focus"],
                "arms": {
                    arm: {
                        "answer": results[arm]["answers"][i],
                        "em": results[arm]["em"][i],
                        "f1": results[arm]["f1"][i],
                        "text": test_outputs[arm][i][:1000],
                    }
                    for arm in test_outputs
                },
            }, ensure_ascii=False) + "\n")

    report = {
        "experiment": "HSGR sample-specific hidden RouteGuide action ceiling",
        "claim_boundary": (
            "Gold final-hop route indices measure oracle hidden-control headroom only; "
            "they do not establish a deployable hidden route observer."
        ),
        "data": data_meta,
        "calibration": {
            "base_em": mean(cal_base_em),
            "base_f1": mean(cal_base_f1),
            "grid": grid,
            "selection_rule": (
                "maximize min(correct-base EM, correct-wrong EM), then summed EM "
                "deltas, correct-base F1, smaller beta, lower layer"
            ),
        },
        "selected": selected,
        "test": {
            "official_em": {arm: mean(v["em"]) for arm, v in results.items()},
            "official_f1": {arm: mean(v["f1"]) for arm, v in results.items()},
            "deltas": deltas,
            "paired": paired,
            "depth_correct_vs_base_em": depth_delta,
            "gates": gates,
            "advance_to_hidden_route_observer": all(gates.values()),
        },
        "token_counts": dict(runner.token_counts),
        "feature_layers": list(FEATURE_LAYERS),
        "betas": list(BETAS),
        "pool_modes": list(POOL_MODES),
        "model": args.model,
    }
    report_path = os.path.join(args.out_dir, "hsgr_hidden_route_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print("\n== HSGR hidden RouteGuide held-out result ==")
    print(json.dumps({
        "selected": selected,
        "official_em": report["test"]["official_em"],
        "official_f1": report["test"]["official_f1"],
        "deltas": deltas,
        "paired": paired,
        "depth": depth_delta,
        "gates": gates,
        "advance": report["test"]["advance_to_hidden_route_observer"],
    }, indent=1, ensure_ascii=False))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--exclude-cases", action="append", default=[])
    ap.add_argument("--out-dir", default="hsgr_hidden_route")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--original-exclude-limit", type=int, default=200)
    ap.add_argument("--calib", type=int, default=80)
    ap.add_argument("--test", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--bs-hidden", type=int, default=4)
    ap.add_argument("--bs-generate", type=int, default=8)
    ap.add_argument("--permutation-samples", type=int, default=50000)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
