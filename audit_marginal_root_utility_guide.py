"""V0 oracle action-headroom audit for marginal root-utility Guide credit.

This is deliberately a zero-GPU *ceiling* test.  It asks whether a graph-level
credit target is worth learning before extracting any new hidden states.  The
oracle policies use root gold labels to construct credit and are not deployable.

Usage:
  python audit_marginal_root_utility_guide.py \
    --dirs outputs,outputs_gsm_test,outputs_gsm_train,outputs_math_train
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

from answer_check import answers_equal
from phase0_reward_audit import build_domains, jread_glob


HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = {"outputs", "outputs_gsm_test"}
EPS = 1e-12


def exact_mcnemar(candidate, baseline):
    """Two-sided exact McNemar test and discordant counts."""
    wins = sum(bool(a) and not bool(b) for a, b in zip(candidate, baseline))
    losses = sum(bool(b) and not bool(a) for a, b in zip(candidate, baseline))
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return {"wins": wins, "losses": losses, "p": min(1.0, 2.0 * tail)}


def choose(classes, scores):
    """Score, then frequency, then frozen domain order (lower index wins)."""
    return max(
        range(len(classes)),
        key=lambda i: (scores[i], classes[i]["freq"], -i),
    )


def load(out_dir):
    path = os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(path, "decompose.s*.jsonl")}
    subc = defaultdict(dict)
    for r in jread_glob(path, "subcands.s*.jsonl"):
        subc[r["id"]][r["sub_idx"]] = r
    aggs = defaultdict(list)
    for r in jread_glob(path, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)
    ids = sorted(
        pid for pid, d in dec.items()
        if d.get("subquestions")
        and pid in subc
        and pid in aggs
        and len(subc[pid]) == len(d["subquestions"])
    )
    return dec, subc, aggs, ids


def summarize_pair(candidate, baseline, mask=None):
    if mask is None:
        mask = [True] * len(candidate)
    a = [x for x, keep in zip(candidate, mask) if keep]
    b = [x for x, keep in zip(baseline, mask) if keep]
    if not a:
        return {"n": 0, "candidate_acc": None, "baseline_acc": None,
                "delta": None, "mcnemar": {"wins": 0, "losses": 0, "p": 1.0}}
    return {
        "n": len(a),
        "candidate_acc": sum(a) / len(a),
        "baseline_acc": sum(b) / len(b),
        "delta": (sum(a) - sum(b)) / len(a),
        "mcnemar": exact_mcnemar(a, b),
    }


def analyze(out_dir):
    dec, subc, aggs, ids = load(out_dir)
    outcomes = defaultdict(list)
    selected_tuples = defaultdict(list)
    multi_masks = []
    actionable_masks = []
    argmax_equal = []
    missing_selected = defaultdict(int)
    duplicate_tuples = 0
    node_total = 0
    node_multi = 0
    node_collapsed = 0

    for pid in ids:
        rows = [r for r in aggs[pid] if r.get("final_ans") is not None]
        gold = dec[pid]["gold"]
        row_ok = [bool(answers_equal(r["final_ans"], gold)) for r in rows]
        by_tuple = defaultdict(list)
        for r, ok in zip(rows, row_ok):
            by_tuple[tuple(r["sub_norms"])].append((r, ok))
        duplicate_tuples += sum(len(v) - 1 for v in by_tuple.values() if len(v) > 1)

        domains = build_domains([
            subc[pid][si] for si in range(len(dec[pid]["subquestions"]))
        ])
        policy_values = {k: [] for k in ("frequency", "loo", "exact_cf")}
        problem_multi = False

        for si in sorted(domains):
            classes = domains[si]
            node_total += 1
            if len(classes) == 1:
                node_collapsed += 1
            else:
                node_multi += 1
                problem_multi = True
            freq, loo, cf = [], [], []
            for cls in classes:
                k = cls["norm"]
                use = [ok for r, ok in zip(rows, row_ok) if r["sub_norms"][si] == k]
                not_use = [ok for r, ok in zip(rows, row_ok) if r["sub_norms"][si] != k]
                freq.append(cls["freq"])
                loo.append(sum(use) / max(1, len(rows)))
                cf.append(sum(use) / max(1, len(use)) -
                          sum(not_use) / max(1, len(not_use)))
            picks = {
                "frequency": choose(classes, freq),
                "loo": choose(classes, loo),
                "exact_cf": choose(classes, cf),
            }
            if len(classes) > 1:
                argmax_equal.append(picks["loo"] == picks["exact_cf"])
            for name, idx in picks.items():
                policy_values[name].append(classes[idx]["norm"])

        for name, vals in policy_values.items():
            key = tuple(vals)
            selected_tuples[name].append(key)
            matches = by_tuple.get(key, [])
            if not matches:
                missing_selected[name] += 1
                outcomes[name].append(False)
            else:
                # Aggregate enumeration is expected to contain one row per tuple.
                # If malformed duplicates occur, use the lowest assignment index.
                _, ok = min(matches, key=lambda x: x[0].get("assign_idx", 10**9))
                outcomes[name].append(ok)

        hard = [x for x in zip(rows, row_ok) if x[0].get("is_hardcommit")]
        outcomes["hardcommit"].append(hard[0][1] if hard else False)
        outcomes["recoverable"].append(any(row_ok))
        multi_masks.append(problem_multi)
        actionable_masks.append(
            selected_tuples["loo"][-1] != selected_tuples["frequency"][-1]
        )

    n = len(ids)
    acc = {k: sum(v) / max(1, n) for k, v in outcomes.items()}
    baseline_name = max(("hardcommit", "frequency"), key=lambda k: acc[k])
    all_pair = summarize_pair(outcomes["loo"], outcomes[baseline_name])
    multi_pair = summarize_pair(outcomes["loo"], outcomes[baseline_name], multi_masks)
    actionable_pair = summarize_pair(
        outcomes["loo"], outcomes[baseline_name], actionable_masks
    )
    exact_pair = summarize_pair(outcomes["loo"], outcomes["exact_cf"])
    arg_agree = sum(argmax_equal) / max(1, len(argmax_equal))

    checks = {
        "n_at_least_100": n >= 100,
        "actionable_at_least_20": sum(actionable_masks) >= 20,
        "loo_cf_argmax_agreement_at_least_095": arg_agree >= 0.95,
        "loo_delta_at_least_003": all_pair["delta"] is not None and all_pair["delta"] >= 0.03,
        "loo_mcnemar_p_below_005": all_pair["mcnemar"]["p"] < 0.05,
        "positive_actionable_delta": actionable_pair["delta"] is not None and actionable_pair["delta"] > 0,
    }
    rep = {
        "dir": out_dir,
        "primary": out_dir in PRIMARY,
        "n_problems": n,
        "nodes": {
            "n_total": node_total,
            "n_multiclass": node_multi,
            "collapse_rate": node_collapsed / max(1, node_total),
            "loo_exact_cf_argmax_agreement": arg_agree,
        },
        "problems": {
            "n_multiclass": sum(multi_masks),
            "n_actionable_loo_vs_frequency": sum(actionable_masks),
            "duplicate_assignment_tuples": duplicate_tuples,
            "missing_selected_tuple": dict(missing_selected),
        },
        "accuracy": acc,
        "best_nonoracle_baseline": baseline_name,
        "loo_vs_best_baseline_all": all_pair,
        "loo_vs_best_baseline_multiclass": multi_pair,
        "loo_vs_best_baseline_actionable": actionable_pair,
        "loo_vs_exact_cf": exact_pair,
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
    }
    return rep


def print_report(rep):
    n = rep["n_problems"]
    nd = rep["nodes"]
    pr = rep["problems"]
    print(f"== {rep['dir']}  n={n} primary={rep['primary']} ==")
    print(f"  nodes multiclass={nd['n_multiclass']}/{nd['n_total']} "
          f"collapse={nd['collapse_rate']:.3f} "
          f"LOO/CF argmax={nd['loo_exact_cf_argmax_agreement']:.3f}")
    print(f"  problems multiclass={pr['n_multiclass']} actionable={pr['n_actionable_loo_vs_frequency']} "
          f"duplicates={pr['duplicate_assignment_tuples']} missing={pr['missing_selected_tuple']}")
    print("  accuracy " + " ".join(f"{k}={v:.3f}" for k, v in rep["accuracy"].items()))
    p = rep["loo_vs_best_baseline_all"]
    print(f"  LOO vs {rep['best_nonoracle_baseline']} all: delta={p['delta']:+.3f} "
          f"W/L={p['mcnemar']['wins']}/{p['mcnemar']['losses']} p={p['mcnemar']['p']:.6g}")
    a = rep["loo_vs_best_baseline_actionable"]
    delta = "NA" if a["delta"] is None else f"{a['delta']:+.3f}"
    print(f"  actionable n={a['n']} delta={delta} "
          f"W/L={a['mcnemar']['wins']}/{a['mcnemar']['losses']} p={a['mcnemar']['p']:.6g}")
    print(f"  gate={'PASS' if rep['gate_pass'] else 'FAIL'} {rep['gate_checks']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs",
        default="outputs,outputs_gsm_test,outputs_gsm_train,outputs_math_train",
    )
    ap.add_argument("--report", default="marginal_root_utility_v0_report.json")
    args = ap.parse_args()
    report = {}
    for out_dir in (x.strip() for x in args.dirs.split(",") if x.strip()):
        report[out_dir] = analyze(out_dir)
        print_report(report[out_dir])
    primary = [r for r in report.values() if r["primary"]]
    report["overall_primary_gate_pass"] = bool(primary) and all(r["gate_pass"] for r in primary)
    out = os.path.join(HERE, args.report)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"overall primary gate={'PASS' if report['overall_primary_gate_pass'] else 'FAIL'}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

