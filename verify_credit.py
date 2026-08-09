"""V4: credit fidelity audit (zero GPU).

Two questions the RL reward design depends on:

  (a) Is value-class LOO credit a faithful proxy for the exact counterfactual
      credit  P(root correct | do(z_v=k)) - P(root correct | z_v != k) ?
  (b) Does LOO credit carry information BEYOND candidate frequency? If the
      credit ranking equals the frequency ranking, the reward is a rebranded
      majority vote and the structural claim is empty.

Reports pooled/within-node Spearman and argmax agreement for both.

Usage: python verify_credit.py --dirs outputs,outputs_gsm_test
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402
from phase0_reward_audit import build_domains, jread_glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > EPS and dy > EPS else float("nan")


def analyze(out_dir):
    d = os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    subc = defaultdict(dict)
    for r in jread_glob(d, "subcands.s*.jsonl"):
        subc[r["id"]][r["sub_idx"]] = r
    aggs = defaultdict(list)
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)

    ids = sorted(i for i in dec if dec[i].get("subquestions") and i in subc
                 and i in aggs and len(subc[i]) == len(dec[i]["subquestions"]))

    pooled = {"loo": [], "cf": [], "freq": []}
    node_sp_cf, node_sp_freq = [], []
    agree_cf, agree_freq, n_multi = 0, 0, 0
    disagree_examples = []

    for pid in ids:
        gold = dec[pid]["gold"]
        rows = aggs[pid]
        n_assign = len(rows)
        domains = build_domains([subc[pid][si] for si in range(len(dec[pid]["subquestions"]))])
        for si in sorted(domains):
            classes = domains[si]
            loo, cf, freq = [], [], []
            for cls in classes:
                k = cls["norm"]
                using = [r for r in rows if r["sub_norms"][si] == k]
                not_using = [r for r in rows if r["sub_norms"][si] != k]

                def ok(rs):
                    return sum(1 for r in rs if r.get("final_ans")
                               and answers_equal(r["final_ans"], gold))

                loo.append(ok(using) / max(1, n_assign))
                p_k = ok(using) / max(1, len(using))
                p_not = ok(not_using) / max(1, len(not_using))
                cf.append(p_k - p_not)
                freq.append(cls["freq"])
            pooled["loo"] += loo
            pooled["cf"] += cf
            pooled["freq"] += freq
            if len(classes) >= 2:
                n_multi += 1
                node_sp_cf.append(spearman(loo, cf))
                node_sp_freq.append(spearman(loo, freq))
                bl = max(range(len(loo)), key=lambda i: loo[i])
                bc = max(range(len(cf)), key=lambda i: cf[i])
                bf = max(range(len(freq)), key=lambda i: freq[i])
                agree_cf += int(bl == bc)
                agree_freq += int(bl == bf)
                if bl != bf and len(disagree_examples) < 6:
                    disagree_examples.append(
                        {"id": pid, "sub_idx": si, "loo": loo, "freq": freq, "cf": cf,
                         "argmax_loo": classes[bl]["norm"], "argmax_freq": classes[bf]["norm"]}
                    )

    vals = [v for v in node_sp_cf if v == v]
    vals_f = [v for v in node_sp_freq if v == v]
    rep = {
        "n_problems": len(ids),
        "n_nodes_multiclass": n_multi,
        "pooled_spearman_loo_vs_cf": spearman(pooled["loo"], pooled["cf"]),
        "pooled_spearman_loo_vs_freq": spearman(pooled["loo"], pooled["freq"]),
        "mean_within_node_spearman_loo_vs_cf": sum(vals) / len(vals) if vals else float("nan"),
        "mean_within_node_spearman_loo_vs_freq": sum(vals_f) / len(vals_f) if vals_f else float("nan"),
        "argmax_agreement_loo_vs_cf": agree_cf / max(1, n_multi),
        "argmax_agreement_loo_vs_freq": agree_freq / max(1, n_multi),
        "disagreement_examples": disagree_examples,
    }
    print(f"== {out_dir}: problems={len(ids)} multiclass nodes={n_multi} ==")
    print(f"  LOO vs exact counterfactual : pooled rho {rep['pooled_spearman_loo_vs_cf']:.3f}  "
          f"within-node rho {rep['mean_within_node_spearman_loo_vs_cf']:.3f}  "
          f"argmax agree {rep['argmax_agreement_loo_vs_cf']:.3f}")
    print(f"  LOO vs frequency           : pooled rho {rep['pooled_spearman_loo_vs_freq']:.3f}  "
          f"within-node rho {rep['mean_within_node_spearman_loo_vs_freq']:.3f}  "
          f"argmax agree {rep['argmax_agreement_loo_vs_freq']:.3f}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="outputs,outputs_gsm_test")
    a = ap.parse_args()
    out = {d.strip(): analyze(d.strip()) for d in a.dirs.split(",")}
    with open(os.path.join(HERE, "verify_v4_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved verify_v4_report.json")
