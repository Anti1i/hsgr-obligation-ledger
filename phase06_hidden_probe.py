"""Phase-0.6: hidden-state probe vs text scorer (latent semantic interface potential, level-A pretest).

Hypothesis under test (user): correctness of an assignment is linearly readable
from the frozen generator's own hidden states, and this latent signal is
competitive with (or better than) a fine-tuned 0.5B text scorer / the 7B
prompted verify-vote.

Design:
  - Input prompt identical to the trained scorer (SCORER_USER), so the only
    difference is WHERE we read the signal: internal states + tiny linear probe
    vs fine-tuned text model logits vs prompted generation.
  - Extract last-token hidden states at several layers from frozen
    Qwen2.5-7B-Instruct (the generator itself). Train logistic probes per layer.
  - Train on outputs_gsm_train + outputs_math_train (problem-disjoint from both
    eval sets). Evaluate on outputs (MATH pilot 200) and outputs_gsm_test.
  - Report: overall AUROC (natural), within-problem AUROC, synthetic-negative
    AUROC (locally-plausible-but-inconsistent detection), per-layer comparison.

Stages (resumable):
  extract : forward passes, save features per dir  -> <dir>/hidden_feats.pt
  probe   : train + evaluate probes                -> phase06_report.json

Usage:
  python phase06_hidden_probe.py --stage extract --dirs outputs_gsm_train,outputs_math_train,outputs,outputs_gsm_test
  python phase06_hidden_probe.py --stage probe --train-dirs outputs_gsm_train,outputs_math_train --eval-dirs outputs,outputs_gsm_test
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SCORER_USER = """Problem:
{problem}

Proposed intermediate results:
{facts}

Proposed final answer: {final}

Is this solution sketch correct and internally consistent? Answer VALID or INVALID."""

LAYERS = [7, 14, 21, 28]  # Qwen2.5-7B has 28 layers; index into hidden_states tuple
MAX_SYN_PER_PROBLEM = 3


def facts_str(subs, answers):
    return "\n".join(f"{i+1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(zip(subs, answers)))


def jread_glob(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_units(out_dir):
    """One unit per (assignment | synthetic negative): prompt + label + meta."""
    dec = {r["id"]: r for r in jread_glob(os.path.join(out_dir, "decompose.s*.jsonl"))}
    units = []
    seen = defaultdict(set)
    for r in jread_glob(os.path.join(out_dir, "aggregate.s*.jsonl")):
        pid = r["id"]
        d = dec.get(pid)
        if d is None or not d.get("subquestions") or r.get("final_ans") is None:
            continue
        key = (tuple(r.get("sub_norms", [])), r.get("final_norm"))
        if key in seen[pid]:
            continue
        seen[pid].add(key)
        if "label" in r:
            label = int(r["label"])
        else:
            label = int(answers_equal(r["final_ans"], d["gold"]))
        units.append(
            {
                "id": pid,
                "assign_idx": r.get("assign_idx", -1),
                "kind": "nat",
                "label": label,
                "prompt": SCORER_USER.format(
                    problem=d["problem"],
                    facts=facts_str(d["subquestions"], r["sub_answers"]),
                    final=r["final_ans"],
                ),
            }
        )
    syn_by_p = defaultdict(list)
    for r in jread_glob(os.path.join(out_dir, "synthetic_negatives.jsonl")):
        syn_by_p[r["id"]].append(r)
    for pid, rows in syn_by_p.items():
        d = dec.get(pid)
        if d is None:
            continue
        rng = random.Random(pid)
        rng.shuffle(rows)
        for r in rows[:MAX_SYN_PER_PROBLEM]:
            units.append(
                {
                    "id": pid,
                    "assign_idx": -1,
                    "kind": "syn",
                    "label": 0,
                    "prompt": SCORER_USER.format(
                        problem=d["problem"],
                        facts=facts_str(d["subquestions"], r["sub_answers"]),
                        final=r["final_ans"],
                    ),
                }
            )
    return units


def stage_extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print(f"[model] {args.model} loaded", flush=True)

    for d in args.dirs.split(","):
        out_dir = os.path.join(HERE, d.strip())
        feat_path = os.path.join(out_dir, args.feat_file)
        if os.path.exists(feat_path):
            print(f"[extract] {d}: exists, skip", flush=True)
            continue
        units = build_units(out_dir)
        print(f"[extract] {d}: {len(units)} units", flush=True)
        feats = {l: [] for l in LAYERS}
        metas = []
        bs = args.bs
        for i in range(0, len(units), bs):
            batch = units[i : i + bs]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": u["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for u in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to("cuda")
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for l in LAYERS:
                # left padding -> last position is the true last token
                feats[l].append(out.hidden_states[l][:, -1, :].float().cpu())
            for u in batch:
                metas.append({k: u[k] for k in ("id", "assign_idx", "kind", "label")})
            del enc, out
            torch.cuda.empty_cache()
            if (i // bs) % 20 == 0:
                print(f"  {min(i+bs, len(units))}/{len(units)}", flush=True)
        payload = {
            "layers": LAYERS,
            "feats": {l: torch.cat(feats[l]).half() for l in LAYERS},
            "metas": metas,
        }
        torch.save(payload, feat_path)
        print(f"[extract] saved {feat_path}", flush=True)


def _auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _within_problem_auroc(scores, labels, pids):
    byp = defaultdict(list)
    for s, y, p in zip(scores, labels, pids):
        byp[p].append((s, y))
    vals = []
    for p, items in byp.items():
        a = _auroc([s for s, _ in items], [y for _, y in items])
        if a == a:  # not nan
            vals.append(a)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def stage_probe(args):
    import torch

    def load(dirs):
        packs = []
        for d in dirs.split(","):
            p = os.path.join(HERE, d.strip(), args.feat_file)
            packs.append((d.strip(), torch.load(p)))
        return packs

    train_packs = load(args.train_dirs)
    eval_packs = load(args.eval_dirs)

    report = {"layers": {}, "config": {"train_dirs": args.train_dirs, "eval_dirs": args.eval_dirs}}
    best = None

    for l in LAYERS:
        X = torch.cat([pk["feats"][l].float() for _, pk in train_packs])
        metas = [m for _, pk in train_packs for m in pk["metas"]]
        y = torch.tensor([m["label"] for m in metas], dtype=torch.float32)
        # problem-level split (dir-qualified id to avoid collisions)
        pids = [f"{d}:{m['id']}" for (d, pk) in train_packs for m in pk["metas"]]
        upids = sorted(set(pids))
        random.Random(42).shuffle(upids)
        val_p = set(upids[: max(1, len(upids) // 20)])
        vmask = torch.tensor([p in val_p for p in pids])
        tmask = ~vmask

        mu, sd = X[tmask].mean(0), X[tmask].std(0) + 1e-6
        Xn = (X - mu) / sd

        torch.manual_seed(0)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.AdamW([w, b], lr=1e-2, weight_decay=1e-3)
        Xt, yt = Xn[tmask], y[tmask]
        for _ in range(args.probe_epochs):
            opt.zero_grad()
            logit = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
            loss.backward()
            opt.step()

        with torch.no_grad():
            val_scores = (Xn[vmask] @ w + b).tolist()
        val_labels = y[vmask].tolist()
        val_kinds = [m["kind"] for m, keep in zip(metas, vmask.tolist()) if keep]
        val_nat = _auroc(
            [s for s, k in zip(val_scores, val_kinds) if k == "nat"],
            [yy for yy, k in zip(val_labels, val_kinds) if k == "nat"],
        )
        entry = {"val_auroc_all": _auroc(val_scores, val_labels), "val_auroc_natural": val_nat,
                 "train_loss": loss.item(), "evals": {}}

        for d, pk in eval_packs:
            Xe = (pk["feats"][l].float() - mu) / sd
            with torch.no_grad():
                sc = (Xe @ w + b).tolist()
            ms = pk["metas"]
            lab = [m["label"] for m in ms]
            kin = [m["kind"] for m in ms]
            pid = [m["id"] for m in ms]
            nat_idx = [i for i, k in enumerate(kin) if k == "nat"]
            nat_sc = [sc[i] for i in nat_idx]
            nat_lab = [lab[i] for i in nat_idx]
            nat_pid = [pid[i] for i in nat_idx]
            wp, nwp = _within_problem_auroc(nat_sc, nat_lab, nat_pid)
            # synthetic negatives vs natural positives: inconsistency detection
            syn_sc = [sc[i] for i, k in enumerate(kin) if k == "syn"]
            pos_sc = [s for s, yy in zip(nat_sc, nat_lab) if yy == 1]
            syn_auroc = _auroc(pos_sc + syn_sc, [1] * len(pos_sc) + [0] * len(syn_sc))
            entry["evals"][d] = {
                "n_natural": len(nat_idx),
                "auroc_natural": _auroc(nat_sc, nat_lab),
                "within_problem_auroc": wp,
                "n_problems_scored": nwp,
                "auroc_pos_vs_synthetic": syn_auroc,
                "n_synthetic": len(syn_sc),
            }
        report["layers"][l] = entry
        crit = val_nat if val_nat == val_nat else entry["val_auroc_all"]
        if best is None or crit > best[1]:
            best = (l, crit)
        print(f"[probe] layer {l}: val_nat={val_nat:.3f} " +
              " ".join(f"{d}:{e['auroc_natural']:.3f}/wp{e['within_problem_auroc']:.3f}"
                       for d, e in entry["evals"].items()), flush=True)

    report["best_layer_by_val"] = best[0]
    out = os.path.join(HERE, args.report)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nbest layer (val natural AUROC): {best[0]}")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "probe"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dirs", default="outputs_gsm_train,outputs_math_train,outputs,outputs_gsm_test")
    ap.add_argument("--train-dirs", default="outputs_gsm_train,outputs_math_train")
    ap.add_argument("--eval-dirs", default="outputs,outputs_gsm_test")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--feat-file", default="hidden_feats.pt")
    ap.add_argument("--report", default="phase06_report.json")
    args = ap.parse_args()
    if args.stage == "extract":
        stage_extract(args)
    else:
        stage_probe(args)
