"""Paired significance + confound audit for the MuSiQue E0 result (CPU).

The headline (predicted-hops 0.535 vs SC@1 0.440) is a paired comparison on the
same 200 problems, so use McNemar rather than two independent proportions.
Also reports the comparison against SC@8, which is the budget-fairer baseline.

Usage:
  python mh_e0_stats.py --dir <work>/mh_e0 --data <work>/data/musique_ans_val.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import answers_match, evidence_from_row  # noqa: E402
from mh_e0 import hop_deps, load_rows  # noqa: E402


def jread(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    return rows


def mcnemar(a_only, b_only):
    """Exact-ish two-sided McNemar via normal approx with continuity correction."""
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    # two-sided p from chi2 with 1 dof
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return chi2, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--suffix", default="",
                    help="'' for evidence=hop arm, '_evall' for the matched-evidence arm")
    a = ap.parse_args()

    rows = load_rows(a.data, 200)
    by_uid = {r["_uid"]: r for r in rows}

    hops = {m: {} for m in ("predicted", "oracle")}
    for m in hops:
        f = os.path.join(a.dir, f"hops_{m}{a.suffix}.s0.jsonl")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        for r in jread(f):
            hops[m].setdefault(r["id"], {})[r["hop"]] = r
    print(f"[stats] arm = evidence={'hop' if not a.suffix else 'all'}")
    sc = {r["id"]: r for r in jread(os.path.join(a.dir, "sc.s0.jsonl"))}

    ids = [u for u in by_uid if u in sc and u in hops["predicted"]]
    n = len(ids)

    def hop_final_ok(mode, uid):
        row = by_uid[uid]
        nd = hops[mode].get(uid, {})
        last = nd.get(len(row["question_decomposition"]) - 1)
        if not last or not last.get("ans"):
            return False
        return answers_match(last["ans"], row["answer"], row.get("answer_aliases") or [])

    res = {
        "pred": {u: hop_final_ok("predicted", u) for u in ids},
        "ora": {u: hop_final_ok("oracle", u) for u in ids},
        "sc1": {u: bool(sc[u]["hit1"]) for u in ids},
        "sc8": {u: bool(sc[u]["sc"]) for u in ids},
        "ora8": {u: bool(sc[u]["oracle"]) for u in ids},
    }
    print(f"== paired stats on n={n} ==")
    for k, v in res.items():
        print(f"  {k:5s} acc={sum(v.values())/n:.3f}")

    print("\n== McNemar (two-sided) ==")
    for x, y in (("pred", "sc1"), ("pred", "sc8"), ("ora", "sc1"), ("ora", "sc8"),
                 ("ora", "ora8")):
        xo = sum(1 for u in ids if res[x][u] and not res[y][u])
        yo = sum(1 for u in ids if res[y][u] and not res[x][u])
        chi2, p = mcnemar(xo, yo)
        d = sum(res[x].values()) / n - sum(res[y].values()) / n
        sig = "SIG" if p < 0.05 else "ns"
        print(f"  {x:5s} vs {y:5s}: delta={d:+.3f}  "
              f"{x}-only={xo} {y}-only={yo}  chi2={chi2:.2f} p={p:.4f}  [{sig}]")

    # SC@k curve, subsampled from the 8 stored candidates. The hop pipeline
    # re-sends evidence once per hop (~2.6 hops), so its true cost sits near
    # SC@3, not SC@1 -- this is the budget-matched comparison point.
    print("\n== SC@k curve (subsampled from stored 8 candidates, 200 reps) ==")
    rng = random.Random(0)
    curve = {}
    for k in range(1, 9):
        accs = []
        for _ in range(200):
            hit = 0
            for u in ids:
                rec = sc[u]
                cands = rec["cands"]
                pick = rng.sample(cands, k) if k < len(cands) else cands
                vote = Counter(c["norm"] for c in pick if c["norm"])
                if not vote:
                    continue
                top = vote.most_common(1)[0][0]
                if answers_match(top, rec["gold"], rec.get("aliases") or []):
                    hit += 1
            accs.append(hit / n)
        curve[k] = sum(accs) / len(accs)
        print(f"  SC@{k} = {curve[k]:.3f}")
    hops_per_problem = sum(len(by_uid[u]["question_decomposition"]) for u in ids) / n
    kbud = max(1, round(hops_per_problem))
    print(f"\n  mean hops/problem = {hops_per_problem:.2f} "
          f"-> budget-matched baseline ~ SC@{kbud} = {curve[kbud]:.3f}")
    for name in ("pred", "ora"):
        d = sum(res[name].values()) / n - curve[kbud]
        print(f"  {name} vs SC@{kbud} (budget-matched): {d:+.3f}")

    # Confound probe: how much evidence targeting does the gold decomposition give?
    print("\n== evidence-targeting confound ==")
    n_para = []
    n_support_mapped = 0
    for u in ids:
        r = by_uid[u]
        decomp = r["question_decomposition"]
        n_para.append(len(r.get("paragraphs") or []))
        if all(s.get("paragraph_support_idx") is not None
               or isinstance(s.get("support_paragraph"), dict) for s in decomp):
            n_support_mapped += 1
    print(f"  support paragraphs per problem: mean={sum(n_para)/max(1,len(n_para)):.2f}")
    print(f"  problems where every hop has a mapped gold paragraph: "
          f"{n_support_mapped}/{n} ({n_support_mapped/max(1,n):.1%})")
    print("  -> hop executor sees ONE pre-selected gold paragraph per hop;")
    print("     SC baseline must find the right facts inside all of them.")
    print("     This is retrieval supervision on top of structural supervision.")

    # Accuracy by hop count: does the gain concentrate where structure matters?
    print("\n== by hop count ==")
    buckets = {}
    for u in ids:
        h = len(by_uid[u]["question_decomposition"])
        buckets.setdefault(h, []).append(u)
    for h in sorted(buckets):
        us = buckets[h]
        m = len(us)
        print(f"  {h}hop n={m:3d}  pred={sum(res['pred'][u] for u in us)/m:.3f}  "
              f"ora={sum(res['ora'][u] for u in us)/m:.3f}  "
              f"sc1={sum(res['sc1'][u] for u in us)/m:.3f}  "
              f"sc8={sum(res['sc8'][u] for u in us)/m:.3f}")

    rep = {"n": n,
           "acc": {k: sum(v.values()) / n for k, v in res.items()},
           "sc_curve": curve, "hops_per_problem": hops_per_problem,
           "budget_k": kbud,
           "support_paras_mean": sum(n_para) / max(1, len(n_para))}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"mh_e0_stats{a.suffix}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
