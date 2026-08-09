"""Phase-0.6b: multi-layer fusion probe + end-to-end score export.

Concatenates hidden features from layers {7,14,21,28} (extracted by
phase06_hidden_probe.py), trains one logistic probe, evaluates, and writes
sigmoid scores for eval dirs to probe.s0.jsonl so analyze.py can plug them
into DCH selection.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [7, 14, 21, 28]  # overridden by --layers


def _auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _wp_auroc(scores, labels, pids):
    byp = defaultdict(list)
    for s, y, p in zip(scores, labels, pids):
        byp[p].append((s, y))
    vals = []
    for items in byp.values():
        a = _auroc([s for s, _ in items], [y for _, y in items])
        if a == a:
            vals.append(a)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def main(args):
    import torch

    global LAYERS
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",")]

    def load(dirs):
        packs = []
        for d in dirs.split(","):
            p = os.path.join(HERE, d.strip(), "hidden_feats.pt")
            packs.append((d.strip(), torch.load(p)))
        return packs

    def fuse(pk):
        return torch.cat([pk["feats"][l].float() for l in LAYERS], dim=1)

    train_packs = load(args.train_dirs)
    eval_packs = load(args.eval_dirs)

    X = torch.cat([fuse(pk) for _, pk in train_packs])
    metas = [m for _, pk in train_packs for m in pk["metas"]]
    y = torch.tensor([m["label"] for m in metas], dtype=torch.float32)
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
    opt = torch.optim.AdamW([w, b], lr=5e-3, weight_decay=3e-3)
    Xt, yt = Xn[tmask], y[tmask]
    for _ in range(args.probe_epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        vs = (Xn[vmask] @ w + b).tolist()
    vy = y[vmask].tolist()
    vk = [m["kind"] for m, keep in zip(metas, vmask.tolist()) if keep]
    val_nat = _auroc([s for s, k in zip(vs, vk) if k == "nat"],
                     [yy for yy, k in zip(vy, vk) if k == "nat"])
    print(f"[fusion] dim={X.shape[1]}  val_all={_auroc(vs, vy):.3f}  val_natural={val_nat:.3f}")

    report = {"val_auroc_natural": val_nat, "evals": {}}
    for d, pk in eval_packs:
        Xe = (fuse(pk) - mu) / sd
        with torch.no_grad():
            logits = (Xe @ w + b)
            probs = torch.sigmoid(logits).tolist()
        ms = pk["metas"]
        nat = [(p, m) for p, m in zip(probs, ms) if m["kind"] == "nat"]
        sc = [p for p, _ in nat]
        lab = [m["label"] for _, m in nat]
        pid = [m["id"] for _, m in nat]
        wp, nwp = _wp_auroc(sc, lab, pid)
        report["evals"][d] = {
            "auroc_natural": _auroc(sc, lab),
            "within_problem_auroc": wp,
            "n_problems": nwp,
            "n": len(nat),
        }
        print(f"[fusion] {d}: auroc={report['evals'][d]['auroc_natural']:.3f} "
              f"wp={wp:.3f} (n={len(nat)})")
        out_path = os.path.join(HERE, d, "probe.s0.jsonl")
        with open(out_path, "w") as f:
            for p, m in nat:
                if m["assign_idx"] >= 0:
                    f.write(json.dumps({"id": m["id"], "assign_idx": m["assign_idx"],
                                        "probe": p}) + "\n")
        print(f"  wrote {out_path}")

    with open(os.path.join(HERE, "phase06b_report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dirs", default="outputs_gsm_train,outputs_math_train")
    ap.add_argument("--eval-dirs", default="outputs,outputs_gsm_test")
    ap.add_argument("--probe-epochs", type=int, default=400)
    ap.add_argument("--layers", default="")
    main(ap.parse_args())
