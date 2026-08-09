"""Phase-0.7: steering-based candidate expansion vs temperature sampling.

Question: can Fractional-Reasoning-style latent steering produce MORE DISTINCT
value classes per semantic node than pure temperature sampling, at matched
generation budget? (Training-free attack on candidate-domain collapse.)

Design:
  - Steering vector: mean last-prompt-token hidden diff between an
    "explore an alternative approach" variant and the plain subquestion prompt,
    computed over the first N_VEC subquestions, at layer L.
  - Arms per subquestion (matched budget, 3 sampled candidates each, T=0.8):
      baseline : existing subcands (1 greedy + 3 samples) from the pilot runs
      steer    : 3 samples with direction-mix coefficients beta in BETAS
                 (norm-preserving: h' = |h| * normalize(h_hat + beta * v_hat))
  - Metrics per node: distinct value classes, collapse rate (=1 class),
    new-value rate vs baseline union.

Usage:
  python phase07_steering.py --dirs outputs_gsm_test,outputs --limit 60
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import extract_boxed, normalize_answer  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

LAYER = 17
N_VEC = 96
BETAS = [0.06, 0.12, 0.20]
TEMP = 0.8

ALT_SUFFIX = ("\n\nImportant: deliberately explore a DIFFERENT solution method or "
              "interpretation than the first one that comes to mind. If several "
              "values seem plausible, pursue the less obvious one.")


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, out_dir, pattern))):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


class Steerer:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()
        self.vec = None
        self.beta = 0.0
        self._hook = None
        print("[model] loaded", flush=True)

    def _tmpl(self, user):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )

    def last_token_hidden(self, users, layer, bs=8):
        feats = []
        for i in range(0, len(users), bs):
            texts = [self._tmpl(u) for u in users[i:i + bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to("cuda")
            with self.torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            feats.append(out.hidden_states[layer][:, -1, :].float().cpu())
            del enc, out
            self.torch.cuda.empty_cache()
        return self.torch.cat(feats)

    def build_vector(self, subq_prompts, layer):
        plain = self.last_token_hidden(subq_prompts, layer)
        alt = self.last_token_hidden([p + ALT_SUFFIX for p in subq_prompts], layer)
        v = (alt - plain).mean(0)
        self.vec = (v / v.norm()).to("cuda", self.torch.bfloat16)
        print(f"[vector] layer {layer}, built from {len(subq_prompts)} prompts", flush=True)

    def enable(self, betas):
        """betas: per-row coefficients for the current batch."""
        self.beta = self.torch.tensor(betas, device="cuda",
                                      dtype=self.torch.bfloat16).view(-1, 1, 1)
        if self._hook is not None:
            return
        layer_mod = self.model.model.layers[LAYER]

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.beta is not None and h.shape[0] == self.beta.shape[0]:
                norms = h.norm(dim=-1, keepdim=True)
                hn = h / (norms + 1e-6)
                mixed = hn + self.beta * self.vec
                mixed = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-6)
                h = mixed * norms
            return (h,) + out[1:] if isinstance(out, tuple) else h

        self._hook = layer_mod.register_forward_hook(hook)

    def disable(self):
        self.beta = None

    def sample(self, user, beta_list, max_new=400):
        """One batched generate: row k steered with beta_list[k]."""
        text = self._tmpl(user)
        self.enable(beta_list)
        enc = self.tok([text] * len(beta_list), return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=TEMP, top_p=0.95, pad_token_id=self.tok.pad_token_id,
            )
        outs = [
            self.tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j in range(len(beta_list))
        ]
        self.disable()
        del enc, gen
        self.torch.cuda.empty_cache()
        return outs


def main(args):
    all_units = []  # (dir, pid, sub_idx, problem, subq, baseline_norms)
    for d in args.dirs.split(","):
        d = d.strip()
        dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
        subc = jread_glob(d, "subcands.s*.jsonl")
        seen_p = set()
        for r in subc:
            pid = r["id"]
            dd = dec.get(pid)
            if dd is None or not dd.get("subquestions"):
                continue
            if pid not in seen_p and len(seen_p) >= args.limit:
                continue
            seen_p.add(pid)
            base_norms = [c["norm"] for c in r["cands"] if c["norm"] is not None]
            all_units.append(
                {"dir": d, "id": pid, "sub_idx": r["sub_idx"], "problem": dd["problem"],
                 "subq": r["subq"], "base_norms": base_norms}
            )
    print(f"{len(all_units)} nodes from {args.dirs} (limit {args.limit}/dir)", flush=True)

    S = Steerer(args.model)
    vec_prompts = [
        P.SUBQ_USER.format(problem=u["problem"], subquestion=u["subq"])
        for u in all_units[:N_VEC]
    ]
    S.build_vector(vec_prompts, LAYER)

    out_path = os.path.join(HERE, "phase07_steer_cands.jsonl")
    done = {(r["dir"], r["id"], r["sub_idx"])
            for r in (json.loads(l) for l in open(out_path))} if os.path.exists(out_path) else set()
    f = open(out_path, "a", buffering=1)
    for i, u in enumerate(all_units):
        if (u["dir"], u["id"], u["sub_idx"]) in done:
            continue
        prompt = P.SUBQ_USER.format(problem=u["problem"], subquestion=u["subq"])
        texts = S.sample(prompt, BETAS)
        cands = []
        for beta, t in zip(BETAS, texts):
            a = extract_boxed(t)
            cands.append({"beta": beta, "ans": a, "norm": normalize_answer(a)})
        f.write(json.dumps({**{k: u[k] for k in ("dir", "id", "sub_idx", "base_norms")},
                            "steer_cands": cands}, ensure_ascii=False) + "\n")
        if i % 10 == 0:
            print(f"[steer] {i}/{len(all_units)}", flush=True)

    # ---- report ----
    rows = [json.loads(l) for l in open(out_path)]
    stats = Counter()
    for r in rows:
        base = set(x for x in r["base_norms"] if x is not None)
        steer_all = set(x["norm"] for x in r["steer_cands"] if x["norm"] is not None)
        # matched budget: baseline sampled = base minus nothing (4 cands incl greedy)
        # steer arm = greedy(first base value if any) + 3 steered
        greedy = r["base_norms"][0] if r["base_norms"] else None
        steer_arm = set([greedy] if greedy else []) | steer_all
        stats["nodes"] += 1
        stats["base_collapsed"] += (len(base) <= 1)
        stats["steer_collapsed"] += (len(steer_arm) <= 1)
        stats["base_classes"] += len(base)
        stats["steer_classes"] += len(steer_arm)
        stats["new_value_nodes"] += bool(steer_all - base)
    n = max(1, stats["nodes"])
    print(f"\nnodes={stats['nodes']}")
    print(f"collapse rate : baseline {stats['base_collapsed']/n:.3f}  "
          f"steer {stats['steer_collapsed']/n:.3f}")
    print(f"avg classes   : baseline {stats['base_classes']/n:.2f}  "
          f"steer {stats['steer_classes']/n:.2f}")
    print(f"nodes where steering found value outside baseline union: "
          f"{stats['new_value_nodes']/n:.3f}")
    with open(os.path.join(HERE, "phase07_report.json"), "w") as fo:
        json.dump({k: v for k, v in stats.items()}, fo, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dirs", default="outputs_gsm_test,outputs")
    ap.add_argument("--limit", type=int, default=60)
    main(ap.parse_args())
