"""Phase-0 reward design audit (zero GPU).

On existing pilot outputs, compute:
  - Root recoverability mass U(C) and leave-one-out credits (raw + value-class)
  - LOO degeneration rate
  - Aggregator leakage proxies
  - Exact counterfactual node credits from enumerated assignments

Usage:
  python phase0_reward_audit.py [--out-dir outputs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import product  # noqa: F401  (reserved for extended enum)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, normalize_answer  # noqa: E402

EPS = 1e-9


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_domains(subc_rows):
    """Per sub_idx: ordered list of value classes {norm, ans, freq, raw_count}."""
    domains = {}
    for r in subc_rows:
        si = r["sub_idx"]
        cands = r["cands"]
        g = next((c for c in cands if c["kind"] == "greedy" and c["ans"]), None)
        cnt = Counter(c["norm"] for c in cands if c["norm"] is not None)
        keys = sorted(cnt, key=lambda k: -cnt[k])
        gnorm = normalize_answer(g["ans"]) if g and g.get("ans") else None
        if gnorm in cnt:
            keys = [gnorm] + [k for k in keys if k != gnorm]
        classes = []
        for k in keys:
            rep = next(c["ans"] for c in cands if c["norm"] == k and c["ans"])
            raw_n = sum(1 for c in cands if c["norm"] == k)
            classes.append(
                {
                    "norm": k,
                    "ans": rep,
                    "freq": cnt[k] / max(1, sum(cnt.values())),
                    "raw_count": raw_n,
                }
            )
        if not classes:
            classes = [{"norm": None, "ans": "(none)", "freq": 0.0, "raw_count": 1}]
        domains[si] = classes
    return domains


def greedy_norms(domains):
    return [domains[si][0]["norm"] for si in sorted(domains)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--report", default="phase0_reward_audit.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, args.out_dir)

    dec = {r["id"]: r for r in jread_glob(out_dir, "decompose.s*.jsonl")}
    subc = defaultdict(dict)
    for r in jread_glob(out_dir, "subcands.s*.jsonl"):
        subc[r["id"]][r["sub_idx"]] = r
    aggs = defaultdict(list)
    for r in jread_glob(out_dir, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)

    ids = sorted(
        i
        for i in dec
        if dec[i].get("subquestions")
        and i in subc
        and i in aggs
        and len(subc[i]) == len(dec[i]["subquestions"])
    )

    # ---- accumulators ----
    node_stats = []  # one row per (pid, sub_idx)
    loo_class_credits = []
    loo_raw_degen_nodes = 0
    loo_class_degen_nodes = 0
    total_nodes = 0

    leakage = Counter()
    leakage_examples = []

    recoverability = []
    cf_credits = []  # exact counterfactual per (pid, sub_idx, norm)

    for pid in ids:
        gold = dec[pid]["gold"]
        subs = dec[pid]["subquestions"]
        domains = build_domains([subc[pid][si] for si in range(len(subs))])
        gnorms = greedy_norms(domains)
        rows = aggs[pid]
        n_assign = len(rows)
        n_correct = sum(
            1 for r in rows if r.get("final_ans") and answers_equal(r["final_ans"], gold)
        )
        u_mass = n_correct / max(1, n_assign)
        recoverability.append(
            {
                "id": pid,
                "n_assign": n_assign,
                "n_correct": n_correct,
                "u_mass": u_mass,
                "recoverable": n_correct > 0,
            }
        )

        hc = next((r for r in rows if r.get("is_hardcommit")), None)
        hc_ok = (
            hc is not None
            and hc.get("final_ans")
            and answers_equal(hc["final_ans"], gold)
        )

        # map assignment -> sub_norm tuple (for leakage examples)
        for r in rows:
            if not r.get("final_ans") or not answers_equal(r["final_ans"], gold):
                continue
            t = tuple(r["sub_norms"])
            all_non_greedy = all(
                t[si] != gnorms[si] for si in range(len(t))
            )
            if all_non_greedy:
                leakage["correct_all_non_greedy"] += 1
                if len(leakage_examples) < 8:
                    leakage_examples.append(
                        {
                            "id": pid,
                            "type": "all_non_greedy_correct",
                            "sub_norms": list(t),
                            "greedy_norms": gnorms,
                            "final": r["final_ans"],
                        }
                    )
            if not hc_ok:
                leakage["correct_while_hc_wrong"] += 1
            if all_non_greedy and not hc_ok:
                leakage["leakage_strict"] += 1

        # per-node LOO on value classes + exact counterfactual credit
        for si in sorted(domains):
            total_nodes += 1
            classes = domains[si]
            class_credits = []
            raw_credits = []

            # value-class LOO: marginal correct assignments using class k
            for cls in classes:
                k = cls["norm"]
                use = [
                    r
                    for r in rows
                    if r["sub_norms"][si] == k
                    and r.get("final_ans")
                    and answers_equal(r["final_ans"], gold)
                ]
                credit = len(use) / max(1, n_assign)
                class_credits.append(credit)
                loo_class_credits.append(credit)

                # exact counterfactual: P(correct | do(z=k)) - P(correct | z!=k)
                n_k = [r for r in rows if r["sub_norms"][si] == k]
                n_not_k = [r for r in rows if r["sub_norms"][si] != k]
                p_k = sum(
                    1
                    for r in n_k
                    if r.get("final_ans") and answers_equal(r["final_ans"], gold)
                ) / max(1, len(n_k))
                p_not = sum(
                    1
                    for r in n_not_k
                    if r.get("final_ans") and answers_equal(r["final_ans"], gold)
                ) / max(1, len(n_not_k))
                cf_credits.append(
                    {
                        "id": pid,
                        "sub_idx": si,
                        "norm": k,
                        "cf_credit": p_k - p_not,
                        "p_correct_given_k": p_k,
                        "n_k": len(n_k),
                    }
                )

            # raw-candidate LOO: redundant if same norm appears >1 time
            cands = subc[pid][si]["cands"]
            norm_to_credit = {cls["norm"]: cr for cls, cr in zip(classes, class_credits)}
            for c in cands:
                if c.get("norm") is None:
                    continue
                if sum(1 for x in cands if x.get("norm") == c["norm"]) > 1:
                    raw_credits.append(0.0)
                else:
                    raw_credits.append(norm_to_credit.get(c["norm"], 0.0))

            class_degen = all(abs(x) < EPS for x in class_credits)
            raw_degen = all(abs(x) < EPS for x in raw_credits) if raw_credits else class_degen
            if class_degen:
                loo_class_degen_nodes += 1
            if raw_degen:
                loo_raw_degen_nodes += 1

            distinct_vals = len(classes)
            collapsed = distinct_vals <= 1
            node_stats.append(
                {
                    "id": pid,
                    "sub_idx": si,
                    "n_classes": distinct_vals,
                    "collapsed": collapsed,
                    "class_credits": class_credits,
                    "class_degen": class_degen,
                    "raw_degen": raw_degen,
                    "max_class_credit": max(class_credits) if class_credits else 0.0,
                }
            )

    n_prob = len(ids)
    n_rec = sum(1 for x in recoverability if x["recoverable"])
    mean_u = sum(x["u_mass"] for x in recoverability) / max(1, n_prob)

    collapsed_nodes = sum(1 for x in node_stats if x["collapsed"])
    n_nodes = len(node_stats)

    report = {
        "n_problems": n_prob,
        "n_nodes": n_nodes,
        "root_recoverability": {
            "problems_with_correct_assignment": n_rec,
            "fraction": n_rec / max(1, n_prob),
            "mean_u_mass": mean_u,
        },
        "candidate_domain_collapse": {
            "nodes_collapsed_to_1_value": collapsed_nodes,
            "fraction": collapsed_nodes / max(1, n_nodes),
        },
        "loo_degeneration": {
            "class_level_degen_nodes": loo_class_degen_nodes,
            "class_level_degen_rate": loo_class_degen_nodes / max(1, total_nodes),
            "raw_level_degen_nodes": loo_raw_degen_nodes,
            "raw_level_degen_rate": loo_raw_degen_nodes / max(1, total_nodes),
            "mean_class_credit": sum(loo_class_credits) / max(1, len(loo_class_credits)),
            "fraction_zero_class_credit": sum(1 for x in loo_class_credits if abs(x) < EPS)
            / max(1, len(loo_class_credits)),
        },
        "aggregator_leakage": {
            "counts": dict(leakage),
            "examples": leakage_examples,
            "note": (
                "correct_all_non_greedy: final correct while every sub != greedy norm; "
                "correct_while_hc_wrong: any correct assignment when hard-commit wrong; "
                "leakage_strict: both"
            ),
        },
        "counterfactual_credit_summary": {
            "n_entries": len(cf_credits),
            "mean_cf": sum(x["cf_credit"] for x in cf_credits) / max(1, len(cf_credits)),
            "fraction_positive": sum(1 for x in cf_credits if x["cf_credit"] > EPS)
            / max(1, len(cf_credits)),
        },
        "per_problem_recoverability": recoverability,
        "per_node_stats": node_stats,
        "counterfactual_credits": cf_credits,
    }

    out_path = os.path.join(out_dir, args.report)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)

    print(f"Phase-0 reward audit  (problems={n_prob}, nodes={n_nodes})")
    print(f"  Root recoverability (≥1 correct assignment): {n_rec}/{n_prob} ({n_rec/max(1,n_prob):.3f})")
    print(f"  Mean U(C) mass (uniform):                  {mean_u:.3f}")
    print(f"  Domain collapse (1 distinct value/node):   {collapsed_nodes}/{n_nodes} ({collapsed_nodes/max(1,n_nodes):.3f})")
    print(f"  LOO degen (value-class level):             {loo_class_degen_nodes}/{total_nodes} ({loo_class_degen_nodes/max(1,total_nodes):.3f})")
    print(f"  LOO degen (raw-candidate level):           {loo_raw_degen_nodes}/{total_nodes} ({loo_raw_degen_nodes/max(1,total_nodes):.3f})")
    print(f"  Zero class-level LOO credits:              {report['loo_degeneration']['fraction_zero_class_credit']:.3f}")
    print(f"  Aggregator leakage (all non-greedy, correct): {leakage['correct_all_non_greedy']}")
    print(f"  Aggregator leakage (strict=non-greedy & hc wrong): {leakage['leakage_strict']}")
    print(f"  Mean exact counterfactual credit:          {report['counterfactual_credit_summary']['mean_cf']:.4f}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
