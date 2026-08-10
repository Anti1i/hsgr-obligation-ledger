"""Analysis for the decomposition-level candidate domain (CPU only).

Answers the three questions that decide whether the route survives:

  1. Do sampled decompositions actually differ? The pilot only ever ran
     `decompose` greedily, so this is unmeasured. Reported as distinct
     subquestion sets and distinct node-value sets per problem, next to the 1.16
     distinct node answers that node_oracle.py measured for the old design.
  2. Does cross-decomposition node evidence beat plain voting over the derived
     answers? If not, the method degenerates to "self-consistency over
     decompositions" and the node-level structure earns nothing.
  3. Does any of it beat SC@k over direct CoT at MATCHED token cost? Token costs
     come from the measured per-call averages in tokens.s*.json, not from
     max_new ceilings (the V7 weakness).

Usage:
  python decomp_analyze.py --dir dd_deep --data data/gsm_deep_test.jsonl
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import _to_number, answers_equal  # noqa: E402
from decomp_domain import qkey  # noqa: E402
from node_oracle import load_gold_steps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIXES = (1, 2, 3, 5, 9)


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def vkey(norm):
    """Canonical key for a value, numeric when possible."""
    if norm is None:
        return None
    v = _to_number(norm)
    return ("n", v) if v is not None else ("s", norm)


def pick_vote(ms, ymap):
    """Majority over the derived answers; ties break toward the earlier m
    (m=0 is the greedy decomposition)."""
    cnt = Counter(vkey(ymap[m]) for m in ms if ymap.get(m) is not None)
    if not cnt:
        return None
    best = max(cnt.items(), key=lambda kv: kv[1])[1]
    for m in ms:
        if ymap.get(m) is not None and cnt[vkey(ymap[m])] == best:
            return m
    return None


def corrob_scores(ms, vals):
    """For each decomposition, the mean fraction of OTHER decompositions in the
    prefix that independently produced each of its node values."""
    out = {}
    for m in ms:
        vm = [v for v in vals.get(m, []) if v is not None]
        others = [m2 for m2 in ms if m2 != m]
        if not vm or not others:
            out[m] = 0.0
            continue
        s = 0.0
        for v in vm:
            s += sum(1 for m2 in others if v in set(vals.get(m2, []))) / len(others)
        out[m] = s / len(vm)
    return out


def analyze(out_dir, data_path):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    # MATH carries no intermediate annotations; keep those rows for their gold.
    gold_steps = load_gold_steps(data_path, require_steps=False)
    decs = defaultdict(dict)
    for r in jread_glob(d, "decomps.s*.jsonl"):
        decs[r["id"]][r["m"]] = r["subquestions"]
    nodes = defaultdict(dict)
    for r in jread_glob(d, "nodeans.s*.jsonl"):
        nodes[r["id"]][r["qkey"]] = r
    agg = defaultdict(dict)
    for r in jread_glob(d, "aggm.s*.jsonl"):
        agg[r["id"]][r["m"]] = r
    roots = {r["id"]: r["cands"] for r in jread_glob(d, "rootext.s*.jsonl")}
    gold = {pid: g["gold"] for pid, g in gold_steps.items()}

    toks = Counter()
    for p in sorted(glob.glob(os.path.join(d, "tokens.s*.json"))):
        with open(p) as f:
            toks.update(json.load(f))

    ids = sorted(i for i in decs if i in agg and i in roots and i in gold)
    if not ids:
        print(f"== {out_dir}: no usable problems ==")
        return None
    n = len(ids)

    # ---- per-call token costs, measured ----
    n_dec_calls = sum(len(decs[i]) for i in ids)
    n_node_calls = sum(len(nodes.get(i, {})) for i in ids)
    n_agg_calls = sum(len(agg[i]) for i in ids)
    n_root_calls = sum(len(roots[i]) for i in ids)
    c_dec = toks.get("decomps", 0) / max(1, n_dec_calls)
    c_node = toks.get("nodeans", 0) / max(1, n_node_calls)
    c_agg = toks.get("aggm", 0) / max(1, n_agg_calls)
    c_root = toks.get("rootext", 0) / max(1, n_root_calls)

    st = Counter()
    struct = {"sub_sets": [], "val_sets": [], "y_sets": [], "n_usable": []}
    cov = defaultdict(list)
    res = {k: Counter() for k in PREFIXES}
    cost = {k: [] for k in PREFIXES}
    sc_res, sc_cost = Counter(), {}
    examples = []

    for pid in ids:
        g = gold[pid]
        usable = sorted(m for m in agg[pid] if agg[pid][m].get("final_ans"))
        struct["n_usable"].append(len(usable))
        if not usable:
            continue
        st["problems"] += 1
        ymap = {m: agg[pid][m]["final_norm"] for m in usable}
        vals = {m: [vkey(x) for x in agg[pid][m]["sub_norms"]] for m in usable}
        subsets = {m: frozenset(qkey(q) for q in agg[pid][m]["subs"]) for m in usable}

        struct["sub_sets"].append(len({subsets[m] for m in usable}))
        struct["val_sets"].append(len({frozenset(v for v in vals[m] if v)
                                       for m in usable}))
        struct["y_sets"].append(len({vkey(ymap[m]) for m in usable
                                     if ymap[m] is not None}))

        inter = {vkey(v) for v in gold_steps[pid]["inter"]}
        inter.discard(None)

        for k in PREFIXES:
            ms = [m for m in usable if m < k]
            if not ms:
                continue
            res[k]["n"] += 1
            # oracle over the decomposition-derived answers
            res[k]["oracle"] += any(ymap[m] and answers_equal(ymap[m], g) for m in ms)
            mv = pick_vote(ms, ymap)
            res[k]["vote"] += bool(mv is not None and answers_equal(ymap[mv], g))
            sc = corrob_scores(ms, vals)
            mb = max(ms, key=lambda m: (sc[m], -m))
            res[k]["corrob_best"] += bool(ymap[mb] and answers_equal(ymap[mb], g))
            agg_sc = defaultdict(float)
            for m in ms:
                if ymap[m] is not None:
                    agg_sc[vkey(ymap[m])] += sc[m]
            if agg_sc:
                topv = max(agg_sc, key=lambda v: agg_sc[v])
                rep = next(ymap[m] for m in ms if vkey(ymap[m]) == topv)
                res[k]["corrob_vote"] += bool(answers_equal(rep, g))
            # intermediate coverage from the union of node values in the prefix
            if inter:
                union = set()
                for m in ms:
                    union |= {v for v in vals[m] if v}
                cov[k].append(len(inter & union) / len(inter))
                res[k]["n_inter"] += 1
                res[k]["all_inter"] += (inter & union) == inter
            # measured token cost of this prefix
            nq = len({q for m in ms for q in subsets[m]})
            cost[k].append(len(ms) * c_dec + nq * c_node + len(ms) * c_agg)
            if (k == 9 and len(examples) < 3 and struct["y_sets"][-1] >= 3
                    and any(ymap[m] and answers_equal(ymap[m], g) for m in ms)):
                examples.append({
                    "id": pid, "gold": g,
                    "decomps": [{"m": m, "subs": agg[pid][m]["subs"],
                                 "vals": agg[pid][m]["sub_norms"],
                                 "y": ymap[m], "corrob": round(sc[m], 2)}
                                for m in ms],
                })

        rc = roots[pid]
        for k in (1, 2, 3, 5, 9, 11, 21):
            sub = rc[:k]
            cnt = Counter(x["norm"] for x in sub if x.get("norm"))
            sc_res[f"oracle@{k}"] += any(x.get("ans") and answers_equal(x["ans"], g)
                                         for x in sub)
            if cnt:
                top = cnt.most_common(1)[0][0]
                rep = next(x["ans"] for x in sub if x["norm"] == top)
                sc_res[f"sc@{k}"] += bool(rep and answers_equal(rep, g))
            sc_res[f"n@{k}"] += 1
            sc_cost[k] = k * c_root

    npb = st["problems"]
    print(f"== decomposition-level candidate domain: {out_dir} "
          f"({npb} problems) ==")
    print(f"  measured tokens/call: decompose {c_dec:.0f}  node {c_node:.0f}  "
          f"aggregate {c_agg:.0f}  rootCoT {c_root:.0f}")
    print(f"  usable decompositions/problem: "
          f"{sum(struct['n_usable'])/max(1,len(struct['n_usable'])):.2f} of 9")

    print("\n  -- Q1: do sampled decompositions differ? --")
    for name, key in (("distinct subquestion sets", "sub_sets"),
                      ("distinct node-value sets ", "val_sets"),
                      ("distinct derived answers ", "y_sets")):
        v = struct[key]
        print(f"  {name}: {sum(v)/max(1,len(v)):.2f} per problem")
    print("  (old design, for reference: 1.16 distinct answers per NODE)")

    print("\n  -- Q2/Q3: selection rules vs SC, with measured token cost --")
    hdr = (f"  {'k':>2} {'oracle':>7} {'vote':>7} {'corrobB':>8} {'corrobV':>8} "
           f"{'allInter':>9} {'tokens':>7} {'~SC@':>5}")
    print(hdr)
    rep = {"n_problems": npb, "per_call_tokens":
           {"decompose": c_dec, "node": c_node, "aggregate": c_agg, "root": c_root},
           "structure": {k: sum(v) / max(1, len(v)) for k, v in struct.items()},
           "prefix": {}, "sc": {}}
    for k in PREFIXES:
        r = res[k]
        if not r["n"]:
            continue
        t = sum(cost[k]) / max(1, len(cost[k]))
        row = {"n": r["n"],
               "oracle": r["oracle"] / r["n"], "vote": r["vote"] / r["n"],
               "corrob_best": r["corrob_best"] / r["n"],
               "corrob_vote": r["corrob_vote"] / r["n"],
               "all_inter": r["all_inter"] / max(1, r["n_inter"]),
               "inter_cov": sum(cov[k]) / max(1, len(cov[k])),
               "tokens": t, "sc_equiv": t / max(1e-9, c_root)}
        rep["prefix"][k] = row
        print(f"  {k:>2} {row['oracle']:>7.3f} {row['vote']:>7.3f} "
              f"{row['corrob_best']:>8.3f} {row['corrob_vote']:>8.3f} "
              f"{row['all_inter']:>9.3f} {t:>7.0f} {row['sc_equiv']:>5.1f}")

    print(f"\n  {'k':>2} {'SC@k':>7} {'oracle':>7} {'tokens':>7}")
    for k in (1, 2, 3, 5, 9, 11, 21):
        nk = sc_res[f"n@{k}"]
        if not nk:
            continue
        row = {"sc": sc_res[f"sc@{k}"] / nk, "oracle": sc_res[f"oracle@{k}"] / nk,
               "tokens": sc_cost[k]}
        rep["sc"][k] = row
        print(f"  {k:>2} {row['sc']:>7.3f} {row['oracle']:>7.3f} "
              f"{row['tokens']:>7.0f}")

    if examples:
        print("\n  -- example problems with >=3 distinct derived answers --")
        for e in examples:
            print(f"   [{e['id']}] gold={e['gold']}")
            for dd in e["decomps"]:
                qs = " | ".join(q[:45] for q in dd["subs"])
                print(f"     m{dd['m']} corrob={dd['corrob']:.2f} y={dd['y']} "
                      f"vals={dd['vals']}")
                print(f"        {qs}")
    with open(os.path.join(HERE, f"decomp_report_{os.path.basename(d)}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    for dd, dp in zip([x.strip() for x in a.dir.split(",")],
                      [x.strip() for x in a.data.split(",")]):
        analyze(dd, dp)
        print()
