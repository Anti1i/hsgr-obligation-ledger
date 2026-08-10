"""Node-level oracle: is the low assignment-oracle caused by the decomposition
or by the aggregation?

S1 showed the decomposition path has a LOWER ceiling than plain root CoT sampling
(gsm_deep 0.877 vs 0.947, math_l5 0.494 vs 0.620). Two very different causes:

  (a) the decomposer asks for quantities that are not on the reference solution
      path, or the nodes compute them wrong  -> the premise, as operationalized,
      is dead;
  (b) the nodes do hold the gold intermediates, but the aggregation step fails to
      turn correct facts into the correct final answer -> fixable, premise alive.

GSM8K's <<expr=value>> annotations give gold intermediate values, so this is
directly measurable. Requires a dataset carrying them: gsm_deep_*.jsonl (field
`steps`) or raw gsm8k_*.jsonl (annotations parsed on the fly).

Also splits node-level hits into "greedy candidate already had it" vs "only a
sampled candidate had it": the latter is the node-level size of the
delayed-commitment effect, which the accuracy numbers can only measure
indirectly.

Usage:
  python node_oracle.py --dir outputs_gsm_test --data data/gsm8k_test.jsonl
  python node_oracle.py --dir outputs_deep2   --data data/gsm_deep_test.jsonl
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import _to_number, normalize_answer  # noqa: E402
from data_prep import num_str, parse_steps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SUB_DOMAIN = 3  # must match pilot.py
NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def load_gold_steps(data_path, require_steps=True):
    """{row_index: {"gold": str, "inter": [str], "problem": str}}

    `inter` holds the genuine intermediates: annotated step values with the final
    answer removed (the last step usually equals it, and counting it would credit
    the decomposition for reproducing the answer, not for decomposing).

    With require_steps=False, rows carrying no intermediate annotations (e.g.
    MATH) are kept with an empty `inter` so callers that only need gold answers
    can share this loader.
    """
    p = data_path if os.path.isabs(data_path) else os.path.join(HERE, data_path)
    out = {}
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            r = json.loads(line)
            problem = r.get("problem") or r.get("question")
            if "steps" in r:
                gold = str(r["answer"])
                vals = [s["value"] for s in r["steps"]]
            else:
                steps, g = parse_steps(r["answer"])
                if not steps or g is None:
                    if require_steps:
                        continue
                    out[i] = {"gold": str(r["answer"]), "inter": [],
                              "problem": problem or ""}
                    continue
                gold = num_str(g)
                vals = [num_str(v) for _, v in steps]
            gn = _to_number(gold)
            inter, seen = [], set()
            for v in vals:
                vn = _to_number(v)
                if vn is None or (gn is not None and vn == gn) or vn in seen:
                    continue
                seen.add(vn)
                inter.append(v)
            out[i] = {"gold": gold, "inter": inter, "problem": problem or ""}
    return out


def literal_in_problem(value, problem):
    """True if the value already appears verbatim in the problem statement.

    A node answering with a number copied from the question is not evidence that
    it computed a reference quantity, so these are reported separately.
    """
    vn = _to_number(value)
    if vn is None:
        return False
    for m in NUM_RE.finditer(problem):
        if _to_number(m.group(0).replace(",", "")) == vn:
            return True
    return False


def domains_for(subc_row_by_idx, n_subs):
    """Rebuild the candidate domains exactly as pilot.py's aggregate stage does:
    greedy first, then remaining norms by descending frequency, capped."""
    doms = []
    for si in range(n_subs):
        cands = subc_row_by_idx[si]["cands"]
        g = next((c["ans"] for c in cands if c["kind"] == "greedy" and c["ans"]), None)
        cnt = Counter(c["norm"] for c in cands if c["norm"] is not None)
        keys = sorted(cnt, key=lambda k: -cnt[k])
        gnorm = normalize_answer(g) if g else None
        if gnorm in cnt:
            keys = [gnorm] + [k for k in keys if k != gnorm]
        doms.append({"norms": keys[:MAX_SUB_DOMAIN], "greedy": gnorm})
    return doms


def num_set(norms):
    out = set()
    for s in norms:
        v = _to_number(s) if s is not None else None
        if v is not None:
            out.add(v)
    return out


def analyze(out_dir, data_path):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    gold = load_gold_steps(data_path)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    subc = {}
    for r in jread_glob(d, "subcands.s*.jsonl"):
        subc.setdefault(r["id"], {})[r["sub_idx"]] = r
    aggs = {}
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        aggs.setdefault(r["id"], []).append(r)

    # The join is positional (pilot's `id` is the row index of --data), so verify
    # it before trusting anything downstream.
    shared = [i for i in sorted(dec) if i in gold]
    mism = sum(1 for i in shared
               if (dec[i].get("problem") or "")[:60].strip()
               != gold[i]["problem"][:60].strip())
    if shared and mism / len(shared) > 0.02:
        raise SystemExit(
            f"FATAL: id misalignment between {out_dir} and {data_path} "
            f"({mism}/{len(shared)} problem texts differ) -- wrong --data file?")

    ids = [i for i in sorted(dec)
           if dec[i].get("subquestions") and i in subc and i in gold and i in aggs
           and len(subc[i]) == len(dec[i]["subquestions"])]

    st = Counter()
    n_nodes = 0
    cov_union, cov_greedy, dom_sizes = [], [], []
    fail_cov, fail_full = [], 0
    examples = []

    for pid in ids:
        g = gold[pid]
        inter = g["inter"]
        nontrivial = [v for v in inter if not literal_in_problem(v, g["problem"])]
        if not inter:
            st["skipped_no_intermediates"] += 1
            continue
        st["problems"] += 1
        subs = dec[pid]["subquestions"]
        doms = domains_for(subc[pid], len(subs))
        n_nodes += len(doms)
        dom_sizes += [len(x["norms"]) for x in doms]

        inter_nums = num_set(inter)
        nontriv_nums = num_set(nontrivial)
        union_all, union_greedy = set(), set()
        for x in doms:
            dn = num_set(x["norms"])
            gn = num_set([x["greedy"]])
            union_all |= dn
            union_greedy |= gn
            hit_any = bool(dn & inter_nums)
            hit_greedy = bool(gn & inter_nums)
            st["node_hit_any"] += hit_any
            st["node_hit_greedy"] += hit_greedy
            st["node_hit_domain_only"] += (hit_any and not hit_greedy)
            st["node_hit_any_nontrivial"] += bool(dn & nontriv_nums)

        hit_u = inter_nums & union_all
        cov_union.append(len(hit_u) / len(inter_nums))
        cov_greedy.append(len(inter_nums & union_greedy) / len(inter_nums))
        full = hit_u == inter_nums
        st["all_inter_covered"] += full
        if nontriv_nums:
            st["problems_with_nontrivial"] += 1
            st["all_nontrivial_covered"] += (nontriv_nums & union_all) == nontriv_nums

        a_ok = any(r.get("final_ans") and _to_number(normalize_answer(r["final_ans"]))
                   is not None
                   and _to_number(normalize_answer(r["final_ans"]))
                   == _to_number(normalize_answer(g["gold"]))
                   for r in aggs[pid])
        st["a_ok"] += a_ok
        if full:
            st["full_and_a_ok"] += a_ok
        else:
            st["notfull"] += 1
        if not a_ok:
            fail_cov.append(len(hit_u) / len(inter_nums))
            fail_full += full
            if len(examples) < 3 and full:
                examples.append({
                    "id": pid, "gold": g["gold"], "inter": inter,
                    "subquestions": subs,
                    "domains": [x["norms"] for x in doms],
                    "agg_finals": sorted({r.get("final_ans") for r in aggs[pid]}),
                })

    n = st["problems"]
    if not n:
        print(f"== node oracle: {out_dir}: no usable problems ==")
        return None
    nf = max(1, n - st["a_ok"])
    rep = {
        "dir": out_dir, "n_problems": n, "n_nodes": n_nodes,
        "mean_nodes_per_problem": n_nodes / n,
        "mean_domain_size": sum(dom_sizes) / max(1, len(dom_sizes)),
        "mean_gold_intermediates": sum(len(gold[p]["inter"]) for p in ids) / max(1, len(ids)),
        "node_hit_any": st["node_hit_any"] / max(1, n_nodes),
        "node_hit_greedy": st["node_hit_greedy"] / max(1, n_nodes),
        "node_hit_domain_only": st["node_hit_domain_only"] / max(1, n_nodes),
        "node_hit_any_nontrivial": st["node_hit_any_nontrivial"] / max(1, n_nodes),
        "inter_cov_union": sum(cov_union) / n,
        "inter_cov_greedy": sum(cov_greedy) / n,
        "all_inter_covered": st["all_inter_covered"] / n,
        "all_nontrivial_covered": (st["all_nontrivial_covered"]
                                   / max(1, st["problems_with_nontrivial"])),
        "oracle_assign": st["a_ok"] / n,
        "agg_success_given_full_facts": st["full_and_a_ok"] / max(1, st["all_inter_covered"]),
        "fail_inter_cov": sum(fail_cov) / nf,
        "fail_all_inter_covered": fail_full / nf,
        "skipped_no_intermediates": st["skipped_no_intermediates"],
    }

    print(f"== node oracle: {out_dir}  ({n} problems, {n_nodes} nodes) ==")
    print(f"  {rep['mean_nodes_per_problem']:.2f} nodes/problem, "
          f"domain size {rep['mean_domain_size']:.2f}, "
          f"{rep['mean_gold_intermediates']:.2f} gold intermediates/problem")
    print("  -- does a node target a reference quantity? --")
    print(f"  node hit (any candidate)     {rep['node_hit_any']:.3f}")
    print(f"  node hit (greedy only)       {rep['node_hit_greedy']:.3f}")
    print(f"  node hit (ONLY via sampling) {rep['node_hit_domain_only']:.3f}"
          "   <- node-level delayed-commitment gain")
    print(f"  node hit, non-trivial values {rep['node_hit_any_nontrivial']:.3f}"
          "   <- excludes values copied from the question")
    print("  -- are the reference intermediates recovered at all? --")
    print(f"  intermediate coverage, union of domains {rep['inter_cov_union']:.3f}"
          f"   greedy-only {rep['inter_cov_greedy']:.3f}")
    print(f"  problems with ALL intermediates covered {rep['all_inter_covered']:.3f}"
          f"   (non-trivial only: {rep['all_nontrivial_covered']:.3f})")
    print("  -- decomposition bug or aggregation bug? --")
    print(f"  oracle_assign (gold reachable via decomposition) {rep['oracle_assign']:.3f}")
    print(f"  P(oracle_assign | all intermediates present)     "
          f"{rep['agg_success_given_full_facts']:.3f}"
          "   <- aggregation quality")
    print(f"  among FAILED problems: intermediate coverage {rep['fail_inter_cov']:.3f}, "
          f"all covered {rep['fail_all_inter_covered']:.3f}"
          "   <- high => aggregation is the bug")
    if examples:
        print("  -- failures WITH all gold intermediates present (aggregation lost it) --")
        for e in examples:
            print(f"   [{e['id']}] gold={e['gold']} inter={e['inter']}")
            for q, dm in zip(e["subquestions"], e["domains"]):
                print(f"       Q: {q[:90]}  -> {dm}")
            print(f"       aggregate finals: {e['agg_finals']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="comma-separated output dirs")
    ap.add_argument("--data", required=True, help="comma-separated source jsonl, aligned with --dir")
    a = ap.parse_args()
    dirs = [x.strip() for x in a.dir.split(",")]
    datas = [x.strip() for x in a.data.split(",")]
    if len(dirs) != len(datas):
        raise SystemExit("--dir and --data must have the same number of entries")
    out = {}
    for d, dp in zip(dirs, datas):
        r = analyze(d, dp)
        if r:
            out[d] = r
        print()
    with open(os.path.join(HERE, "node_oracle_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved node_oracle_report.json")
