"""V8: headroom decomposition (zero GPU).

Splits the remaining accuracy gap into two budgets:

  potential-side headroom = oracle over the EXISTING candidate pool minus the
                            best achieved selection accuracy. This is the most
                            a perfect edge potential / selector could still buy.
  generation-side headroom = 1 - oracle over the existing pool. Only new or
                            better candidates can recover this part.

If the potential-side headroom is small, further scorer work is capped and the
route must prioritize candidate-domain generation (RL), not the potential.

Usage: python verify_headroom.py --dirs outputs,outputs_gsm_test
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402
from phase0_reward_audit import jread_glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def analyze(out_dir, best_acc=None):
    d = os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    aggs = defaultdict(list)
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)
    roots = {r["id"]: r for r in jread_glob(d, "rootcands.s*.jsonl")}

    ids = sorted(i for i in dec if dec[i].get("subquestions") and i in aggs)
    n = len(ids)
    in_assign = in_root = in_union = 0
    only_assign = only_root = 0
    for pid in ids:
        gold = dec[pid]["gold"]
        a_ok = any(r.get("final_ans") and answers_equal(r["final_ans"], gold)
                   for r in aggs[pid])
        r_ok = any(c.get("ans") and answers_equal(c["ans"], gold)
                   for c in roots.get(pid, {}).get("cands", []))
        in_assign += a_ok
        in_root += r_ok
        in_union += (a_ok or r_ok)
        only_assign += (a_ok and not r_ok)
        only_root += (r_ok and not a_ok)

    rep = {
        "n_problems": n,
        "oracle_assignments": in_assign / max(1, n),
        "oracle_root_cot": in_root / max(1, n),
        "oracle_union": in_union / max(1, n),
        "only_in_assignments": only_assign / max(1, n),
        "only_in_root_cot": only_root / max(1, n),
    }
    if best_acc is not None:
        rep["best_achieved"] = best_acc
        rep["potential_side_headroom"] = rep["oracle_union"] - best_acc
        rep["generation_side_headroom"] = 1.0 - rep["oracle_union"]
    print(f"== {out_dir}: {n} problems ==")
    print(f"  oracle over assignments only : {rep['oracle_assignments']:.3f}")
    print(f"  oracle over root CoT only    : {rep['oracle_root_cot']:.3f}")
    print(f"  oracle over union pool       : {rep['oracle_union']:.3f}")
    print(f"  gold only reachable via decomposition: {rep['only_in_assignments']:.3f}"
          f"   only via direct CoT: {rep['only_in_root_cot']:.3f}")
    if best_acc is not None:
        print(f"  best achieved selection      : {best_acc:.3f}")
        print(f"  -> potential-side headroom   : {rep['potential_side_headroom']:.3f}")
        print(f"  -> generation-side headroom  : {rep['generation_side_headroom']:.3f}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="outputs,outputs_gsm_test")
    # best achieved DCH accuracy per dir (from analyze.py, dch_trained)
    ap.add_argument("--best", default="outputs=0.810,outputs_gsm_test=0.931")
    a = ap.parse_args()
    best = {}
    for kv in a.best.split(","):
        k, v = kv.split("=")
        best[k.strip()] = float(v)
    out = {d.strip(): analyze(d.strip(), best.get(d.strip()))
           for d in a.dirs.split(",")}
    with open(os.path.join(HERE, "verify_v8_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved verify_v8_report.json")
