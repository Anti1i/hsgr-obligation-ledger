"""Four-metric benchmark screen for multi-hop QA (CPU, no model).

Metrics (cheap structural / label-based; SC ceiling is a separate GPU job):
  1. Ceiling proxy   — hop-count mix; harder hops should dominate (report only)
  2. Node ambiguity  — gold hop answers: string diversity / alias count
  3. Structure       — depth, mean in-degree, reconvergence (#refs to prior hops)
  4. Error locality  — not measurable without model; leave as N/A here

MuSiQue gold decomposition uses `#k` placeholders referencing prior hop answers;
those references define the dependency edges.

Usage:
  python multihop_screen.py --data data/musique_ans_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REF_RE = re.compile(r"#(\d+)")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def hop_deps(decomp):
    """Return list[list[int]] of 0-based predecessor hop indices."""
    deps = []
    for i, step in enumerate(decomp):
        q = step.get("question") or ""
        preds = sorted({int(x) - 1 for x in REF_RE.findall(q)
                        if 0 < int(x) <= i})
        deps.append(preds)
    return deps


def has_reconvergence(deps):
    """True if some node has in-degree >= 2 (true join, not a pure chain)."""
    return any(len(d) >= 2 for d in deps)


def screen(rows, name="musique"):
    n = len(rows)
    if not n:
        print(f"== {name}: empty ==")
        return None

    depths, n_edges, n_nodes = [], 0, 0
    indeg_sum = 0
    n_recon = 0
    n_chain = 0
    hop_hist = Counter()
    ans_lens = []
    alias_ns = []
    distinct_hop_ans = []

    for r in rows:
        decomp = r.get("question_decomposition") or []
        h = len(decomp)
        hop_hist[h] += 1
        depths.append(h)
        deps = hop_deps(decomp)
        n_nodes += h
        e = sum(len(d) for d in deps)
        n_edges += e
        indeg_sum += e
        if has_reconvergence(deps):
            n_recon += 1
        elif h >= 2 and all(deps[i] == ([i - 1] if i else []) for i in range(h)):
            n_chain += 1
        answers = [str(s.get("answer") or "") for s in decomp]
        distinct_hop_ans.append(len(set(a.lower() for a in answers if a)))
        ans_lens.append(sum(len(a) for a in answers) / max(1, h))
        alias_ns.append(len(r.get("answer_aliases") or []))

    mean_depth = sum(depths) / n
    mean_indeg = indeg_sum / max(1, n_nodes)
    print(f"== multihop screen: {name} ({n} problems) ==")
    print(f"  [structure] mean depth={mean_depth:.2f}  "
          f"edges/node={mean_indeg:.2f}  "
          f"reconvergence={n_recon/n:.1%}  pure_chain={n_chain/n:.1%}")
    print(f"  [hop mix]   " + " ".join(f"{k}hop={v}({v/n:.0%})"
                                       for k, v in sorted(hop_hist.items())))
    print(f"  [ambiguity proxies] mean distinct hop-answers/problem="
          f"{sum(distinct_hop_ans)/n:.2f}  "
          f"mean hop-answer chars={sum(ans_lens)/n:.1f}  "
          f"mean final aliases={sum(alias_ns)/n:.2f}")
    print("  [ceiling]   SC@1 / SC@8 require a GPU pass (mh_ceiling.py)")
    print("  [locality]  error-locality requires model failures (later)")

    # Pass/fail heuristics relative to GSM-deep (chain, numeric, SC@1~0.92)
    flags = []
    if mean_depth < 2.2:
        flags.append("FAIL:depth_too_shallow")
    if n_recon / n < 0.05 and mean_depth >= 3:
        flags.append("WARN:almost_no_reconvergence")
    if n_chain / n > 0.85:
        flags.append("WARN:mostly_pure_chains")
    if not flags:
        flags.append("PASS:structure_ok_for_candidate_domain")
    print(f"  [verdict]   {', '.join(flags)}")

    rep = {
        "name": name, "n": n,
        "mean_depth": mean_depth,
        "edges_per_node": mean_indeg,
        "reconvergence_frac": n_recon / n,
        "pure_chain_frac": n_chain / n,
        "hop_hist": dict(hop_hist),
        "mean_distinct_hop_answers": sum(distinct_hop_ans) / n,
        "flags": flags,
    }
    out = os.path.join(HERE, f"mh_screen_{name}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"  saved {out}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    path = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    name = a.name or os.path.splitext(os.path.basename(path))[0]
    screen(load_jsonl(path), name=name)
