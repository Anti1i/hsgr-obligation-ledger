"""S1 acceptance gate: does this dataset give the hierarchy anything to do?

V8 finding: on MATH-500 / GSM8K the gold answer is reachable ONLY through the
decomposition path for 3.6% / 0.0% of problems, while direct CoT sampling
uniquely covers 12.4% / 6.9%. With that little unique coverage the hierarchical
story has no empirical basis, no matter how good the potential or the RL is.

Gate (must hold before spending H200 time on the main experiments):
    fraction(gold reachable ONLY via the decomposition path) > GATE   (default 10%)

Also prints hard-commit vs delayed-commitment accuracy on the same set, so a
failing gate can be distinguished from a set where the structure helps but
direct CoT happens to be strong too.

Usage: python s1_gate.py --dirs outputs_chain,outputs_mathl5 --gate 0.10
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def analyze(out_dir, gate):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    aggs = defaultdict(list)
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)
    roots = {r["id"]: r for r in jread_glob(d, "rootcands.s*.jsonl")}

    ids = sorted(i for i in dec if dec[i].get("subquestions") and i in aggs)
    n = max(1, len(ids))
    c = Counter()
    for pid in ids:
        gold = dec[pid]["gold"]
        rows = aggs[pid]
        rc = roots.get(pid, {}).get("cands", [])
        a_ok = any(r.get("final_ans") and answers_equal(r["final_ans"], gold) for r in rows)
        r_ok = any(x.get("ans") and answers_equal(x["ans"], gold) for x in rc)
        c["oracle_assign"] += a_ok
        c["oracle_root"] += r_ok
        c["oracle_union"] += (a_ok or r_ok)
        c["only_decomp"] += (a_ok and not r_ok)
        c["only_root"] += (r_ok and not a_ok)

        hc = next((r for r in rows if r.get("is_hardcommit")), None)
        c["hard_commit"] += bool(hc and hc.get("final_ans")
                                 and answers_equal(hc["final_ans"], gold))
        # delayed commitment, frequency evidence only (scorer-free)
        belief = {}
        cnt = Counter(x["norm"] for x in rc if x.get("norm"))
        tot = max(1, sum(cnt.values()))
        for v, k in cnt.items():
            belief[v] = 0.5 * k / tot
        for r in rows:
            v = r.get("final_norm")
            if v is None:
                continue
            ev = sum(r["sub_freqs"]) / max(1, len(r["sub_freqs"]))
            belief[v] = max(belief.get(v, 0.0), 0.5 * ev)
        if belief:
            top = max(belief, key=belief.get)
            rep = next((r["final_ans"] for r in rows if r.get("final_norm") == top),
                       next((x["ans"] for x in rc if x.get("norm") == top), None))
            c["dch_freq"] += bool(rep and answers_equal(rep, gold))
        greedy = next((x["ans"] for x in rc if x.get("kind") == "greedy"), None)
        c["cot_greedy"] += bool(greedy and answers_equal(greedy, gold))
        if cnt:
            top = cnt.most_common(1)[0][0]
            rep = next(x["ans"] for x in rc if x.get("norm") == top)
            c["sc5"] += bool(rep and answers_equal(rep, gold))

    rep = {k: v / n for k, v in c.items()}
    rep["n_problems"] = len(ids)
    rep["gate_value"] = rep.get("only_decomp", 0.0)
    rep["gate_pass"] = rep["gate_value"] > gate
    print(f"== S1 gate: {out_dir} ({len(ids)} usable problems) ==")
    print(f"  CoT greedy {rep.get('cot_greedy',0):.3f}   SC@5 {rep.get('sc5',0):.3f}   "
          f"hard-commit {rep.get('hard_commit',0):.3f}   DCH-freq {rep.get('dch_freq',0):.3f}")
    print(f"  oracle: assignments {rep['oracle_assign']:.3f}  root CoT {rep['oracle_root']:.3f}"
          f"  union {rep['oracle_union']:.3f}")
    print(f"  gold ONLY via decomposition: {rep['gate_value']:.3f}"
          f"   ONLY via direct CoT: {rep['only_root']:.3f}")
    print(f"  potential-side headroom: {rep['oracle_union'] - rep.get('dch_freq', 0):.3f}"
          f"   generation-side headroom: {1 - rep['oracle_union']:.3f}")
    print(f"  GATE (>{gate:.2f}): {'PASS' if rep['gate_pass'] else 'FAIL'}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--gate", type=float, default=0.10)
    a = ap.parse_args()
    out = {d.strip(): analyze(d.strip(), a.gate) for d in a.dirs.split(",")}
    with open(os.path.join(HERE, "s1_gate_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved s1_gate_report.json")
    passed = [d for d, r in out.items() if r["gate_pass"]]
    print(f"datasets passing the gate: {passed if passed else 'NONE -> see route v4 stop-loss'}")
