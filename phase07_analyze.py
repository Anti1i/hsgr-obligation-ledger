"""Fine-grained Phase-0.7 analysis: by dataset and beta."""
import json
from collections import Counter, defaultdict

rows = [json.loads(l) for l in open("phase07_steer_cands.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["dir"]].append(r)


def stats(name, rs):
    s = Counter()
    by_beta = Counter()
    new_by_beta = Counter()
    count_by_beta = Counter()
    n = 0
    for r in rs:
        base = set(x for x in r["base_norms"] if x is not None)
        steer_all = set(x["norm"] for x in r["steer_cands"] if x["norm"] is not None)
        greedy = r["base_norms"][0] if r["base_norms"] else None
        steer_arm = set([greedy] if greedy else []) | steer_all
        n += 1
        s["base_collapsed"] += int(len(base) <= 1)
        s["steer_collapsed"] += int(len(steer_arm) <= 1)
        s["base_classes"] += len(base)
        s["steer_classes"] += len(steer_arm)
        s["new_value_nodes"] += int(bool(steer_all - base))
        s["steer3_classes"] += len(steer_all)
        s["steer3_collapsed"] += int(len(steer_all) <= 1)
        for c in r["steer_cands"]:
            count_by_beta[c["beta"]] += 1
            if c["norm"] is not None:
                by_beta[c["beta"]] += 1
                if c["norm"] not in base:
                    new_by_beta[c["beta"]] += 1
    print(f"== {name} n={n} ==")
    print(
        f"collapse: base {s['base_collapsed']/n:.3f}  "
        f"steer(+greedy) {s['steer_collapsed']/n:.3f}  "
        f"steer3 {s['steer3_collapsed']/n:.3f}"
    )
    print(
        f"avg class: base {s['base_classes']/n:.2f}  "
        f"steer(+greedy) {s['steer_classes']/n:.2f}  "
        f"steer3 {s['steer3_classes']/n:.2f}"
    )
    print(f"new-value nodes vs base union: {s['new_value_nodes']/n:.3f}")
    for b in sorted(count_by_beta):
        print(
            f"  beta={b}: valid_norm {by_beta[b]/count_by_beta[b]:.3f}  "
            f"new_vs_base {new_by_beta[b]/count_by_beta[b]:.3f}"
        )
    return {
        "n": n,
        "collapse_base": s["base_collapsed"] / n,
        "collapse_steer": s["steer_collapsed"] / n,
        "avg_base": s["base_classes"] / n,
        "avg_steer": s["steer_classes"] / n,
        "new_rate": s["new_value_nodes"] / n,
    }


rep = {}
for d, rs in by.items():
    rep[d] = stats(d, rs)
rep["ALL"] = stats("ALL", rows)
with open("phase07_fine.json", "w") as f:
    json.dump(rep, f, indent=1)
print("saved phase07_fine.json")
