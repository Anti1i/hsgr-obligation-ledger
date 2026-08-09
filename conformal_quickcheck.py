"""Conformal quick-check (zero GPU): root-level prediction sets with coverage.

Split-conformal over the DCH root value domain. Belief per value uses the
frequency-only evidence (scorer-independent, so the guarantee is about the
structure, not a particular scorer).

Calibrate on one dir (with gold), evaluate coverage / set size / selective
accuracy on another. Validates the statistical layer of the route (conformal
candidate domains) at the root level.

Usage:
  python conformal_quickcheck.py --calib outputs_gsm_train --test outputs_gsm_test --alpha 0.1
  python conformal_quickcheck.py --calib outputs_math_train --test outputs --alpha 0.1
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, normalize_answer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, out_dir, pattern))):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def problem_beliefs(out_dir):
    """Per problem: normalized freq-only belief over the root value domain + gold."""
    dec = {r["id"]: r for r in jread_glob(out_dir, "decompose.s*.jsonl")}
    roots = {r["id"]: r for r in jread_glob(out_dir, "rootcands.s*.jsonl")}
    aggs = defaultdict(list)
    for r in jread_glob(out_dir, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)

    out = []
    for pid, d in dec.items():
        if not d.get("subquestions") or pid not in aggs:
            continue
        rows = aggs[pid]
        belief = {}
        rc = roots.get(pid, {}).get("cands", [])
        rfreq = Counter(c["norm"] for c in rc if c["norm"] is not None)
        nroot = max(1, sum(rfreq.values()))
        for v, c in rfreq.items():
            belief[v] = 0.5 * c / nroot
        for r in rows:
            v = r.get("final_norm")
            if v is None:
                continue
            sub_ev = sum(r["sub_freqs"]) / max(1, len(r["sub_freqs"]))
            belief[v] = max(belief.get(v, 0.0), 0.5 * sub_ev)
        tot = sum(belief.values())
        if tot <= 0:
            continue
        belief = {v: s / tot for v, s in belief.items()}
        out.append({"id": pid, "gold": d["gold"], "belief": belief})
    return out


def gold_score(rec):
    for v, s in rec["belief"].items():
        if answers_equal(v, rec["gold"]):
            return s
    return 0.0


def main(args):
    calib = problem_beliefs(args.calib)
    test = problem_beliefs(args.test)
    print(f"calib problems={len(calib)}  test problems={len(test)}")

    # nonconformity = 1 - belief(gold); threshold = ceil((n+1)(1-alpha))/n quantile
    scores = sorted(1.0 - gold_score(r) for r in calib)
    n = len(scores)
    k = min(n - 1, max(0, int(-(-((n + 1) * (1 - args.alpha)) // 1)) - 1))  # ceil-1
    qhat = scores[k]
    print(f"alpha={args.alpha}  qhat={qhat:.4f} (rank {k+1}/{n})")

    covered, sizes, singleton_correct, singletons = 0, [], 0, 0
    for r in test:
        pset = [v for v, s in r["belief"].items() if 1.0 - s <= qhat]
        if not pset:
            pset = [max(r["belief"], key=r["belief"].get)]
        sizes.append(len(pset))
        cov = any(answers_equal(v, r["gold"]) for v in pset)
        covered += cov
        if len(pset) == 1:
            singletons += 1
            singleton_correct += cov

    m = len(test)
    print(f"coverage={covered/m:.3f} (target {1-args.alpha:.2f})")
    print(f"avg set size={sum(sizes)/m:.2f}  singleton rate={singletons/m:.3f}  "
          f"selective acc on singletons={singleton_correct/max(1,singletons):.3f}")
    rep = {"alpha": args.alpha, "qhat": qhat, "n_calib": n, "n_test": m,
           "coverage": covered / m, "avg_set_size": sum(sizes) / m,
           "singleton_rate": singletons / m,
           "selective_acc_singletons": singleton_correct / max(1, singletons)}
    out = os.path.join(HERE, f"conformal_{args.calib}_to_{args.test}.json".replace("/", "_"))
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    main(ap.parse_args())
