"""Within-problem AUROC: how well each scorer separates correct from incorrect
assignments *of the same problem* (what actually matters for selection)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import jread_glob  # noqa: E402
from answer_check import answers_equal  # noqa: E402

OUT = "outputs"


def load(pattern, key):
    return {(r["id"], r["assign_idx"]): r[key] for r in jread_glob(pattern)}


def within_auroc(score_map, dec, aggs):
    per, used = [], 0
    for pid, rows in aggs.items():
        gold = dec[pid]["gold"]
        pts = []
        for r in rows:
            s = score_map.get((pid, r["assign_idx"]))
            if s is not None and r["final_ans"]:
                pts.append((s, 1 if answers_equal(r["final_ans"], gold) else 0))
        pos = [s for s, y in pts if y == 1]
        neg = [s for s, y in pts if y == 0]
        if pos and neg:
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            per.append(wins / (len(pos) * len(neg)))
            used += 1
    return (sum(per) / len(per) if per else float("nan")), used


def main():
    dec = {r["id"]: r for r in jread_glob("decompose.s*.jsonl")}
    aggs = {}
    for r in jread_glob("aggregate.s*.jsonl"):
        aggs.setdefault(r["id"], []).append(r)
    maps = {
        "compat-0-10 ": load("compat.s*.jsonl", "compat"),
        "verify-vote ": load("verify.s*.jsonl", "verify"),
        "trained-v1  ": load("trained_v1.s*.jsonl.bak", "trained"),
        "trained-v2  ": load("trained_v2.s*.jsonl.bak", "trained"),
        "trained-v3  ": load("trained.s*.jsonl", "trained"),
    }
    print("within-problem AUROC (problems with >=1 pos and >=1 neg assignment):")
    for name, m in maps.items():
        a, n = within_auroc(m, dec, aggs)
        print(f"  {name}: {a:.3f}  (over {n} problems)")


if __name__ == "__main__":
    main()
