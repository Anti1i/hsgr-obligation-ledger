"""Candidate domain over DECOMPOSITIONS instead of over node answers.

Why: node_oracle.py measured that per-node answer domains collapse to ~1.16
distinct values (cap 3, 4 samples), so the product space the original pipeline
enumerates is effectively a single point, and keeping K answers per node buys
only 1.8-2.6% at node level. Meanwhile 31-42% of nodes ask for quantities that
are not on the reference solution path, and aggregation succeeds 95-99% of the
time once the nodes do hold the gold intermediates. The variance lives in WHICH
subquestions get asked, and `decompose` is the one stage the pilot ran greedily.

So: sample M decompositions per problem, answer each node greedily (justified by
the 1.16), aggregate once per decomposition. The candidate domain becomes
{(H_m, node values, y_m)}, and delayed commitment means not committing to a
decomposition.

Stages (resumable, one process per GPU):
  decomps  : 1 greedy + M sampled decompositions
  nodeans  : greedy answer per UNIQUE subquestion within a problem (decompositions
             share subquestions, so dedup by text saves a large fraction)
  aggm     : one aggregation per decomposition, from its greedy node answers
  rootext  : N direct CoT samples, for the budget-matched SC@k baseline curve

Real generated-token counts per stage are written to tokens.s*.json so the
accuracy/token Pareto does not have to be estimated from max_new ceilings.

Usage:
  python decomp_domain.py --data data/gsm_deep_test.jsonl --out-dir dd_deep \
      --shard 0 --num-shards 2 --limit 300
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import extract_boxed, normalize_answer  # noqa: E402
from pilot import JWriter, Runner, facts_str, jread, load_problems, parse_decompose  # noqa: E402
import prompts as P  # noqa: E402

M_DECOMP = 8       # sampled decompositions per problem (greedy is extra, m=0)
N_ROOT_EXT = 20    # direct CoT samples for the SC@k baseline
TEMP_DECOMP = 1.0  # structural diversity is the whole point of this run
TEMP_ROOT = 0.8    # matches the pilot, so SC@k stays comparable

HERE = os.path.dirname(os.path.abspath(__file__))


def qkey(text):
    """Dedup key for a subquestion: different decompositions often ask the same
    thing with trivial wording differences."""
    s = re.sub(r"\s+", " ", text.strip().lower())
    return s.rstrip(" ?.!")


def run(args):
    tag = f"s{args.shard}"
    OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    probs = load_problems(args.data, args.limit, args.shard, args.num_shards)
    print(f"[shard {args.shard}] {len(probs)} problems from {args.data}", flush=True)
    R = Runner(args.model)
    toks = {}
    tok_path = os.path.join(OUT, f"tokens.{tag}.json")
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            toks = json.load(f)

    def mark(stage):
        toks[stage] = toks.get(stage, 0) + R.n_new_tokens
        R.n_new_tokens = 0
        with open(tok_path, "w") as f:
            json.dump(toks, f, indent=1)

    # ---------------- stage 1: M decompositions ----------------
    path = os.path.join(OUT, f"decomps.{tag}.jsonl")
    have = {(r["id"], r["m"]) for r in jread(path)}
    todo = [p for p in probs if (p["id"], 0) not in have]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 32):
            batch = todo[i : i + 32]
            ps = [P.DECOMPOSE_USER.format(problem=p["problem"]) for p in batch]
            greedy = R.chat_batch(ps, system=P.DECOMPOSE_SYSTEM, max_new=300,
                                  bs=args.bs_decompose)
            sampled = R.chat_batch(ps, system=P.DECOMPOSE_SYSTEM, max_new=300,
                                   temperature=TEMP_DECOMP, n=M_DECOMP,
                                   bs=args.bs_decompose)
            for p, g, ss in zip(batch, greedy, sampled):
                for m, (kind, text) in enumerate([("greedy", g[0])]
                                                 + [("sample", t) for t in ss]):
                    w.write({"id": p["id"], "m": m, "kind": kind,
                             "subquestions": parse_decompose(text)})
            print(f"[decomps] {min(i+32, len(todo))}/{len(todo)}", flush=True)
        mark("decomps")
    dmap = {}
    for r in jread(path):
        dmap.setdefault(r["id"], {})[r["m"]] = r["subquestions"]
    n_usable = sum(1 for pid in dmap if any(dmap[pid].values()))
    print(f"[decomps] problems with >=1 usable decomposition: {n_usable}/{len(probs)}",
          flush=True)

    # ---------------- stage 2: greedy node answers (deduped) ----------------
    path = os.path.join(OUT, f"nodeans.{tag}.jsonl")
    have = {(r["id"], r["qkey"]) for r in jread(path)}
    units, seen = [], set()
    for p in probs:
        for m, subs in sorted(dmap.get(p["id"], {}).items()):
            for q in (subs or []):
                k = qkey(q)
                if (p["id"], k) in have or (p["id"], k) in seen:
                    continue
                seen.add((p["id"], k))
                units.append((p["id"], k, p["problem"], q))
    print(f"[nodeans] {len(units)} unique subquestions to answer", flush=True)
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 64):
            batch = units[i : i + 64]
            ps = [P.SUBQ_USER.format(problem=pr, subquestion=q)
                  for (_, _, pr, q) in batch]
            outs = R.chat_batch(ps, max_new=400, bs=args.bs_sub)
            for (pid, k, _, q), o in zip(batch, outs):
                a = extract_boxed(o[0])
                w.write({"id": pid, "qkey": k, "subq": q,
                         "ans": a, "norm": normalize_answer(a)})
            print(f"[nodeans] {min(i+64, len(units))}/{len(units)}", flush=True)
        mark("nodeans")
    nmap = {}
    for r in jread(path):
        nmap.setdefault(r["id"], {})[r["qkey"]] = r

    # ---------------- stage 3: one aggregation per decomposition ----------------
    path = os.path.join(OUT, f"aggm.{tag}.jsonl")
    have = {(r["id"], r["m"]) for r in jread(path)}
    units = []
    for p in probs:
        for m, subs in sorted(dmap.get(p["id"], {}).items()):
            if not subs or (p["id"], m) in have:
                continue
            rows = [nmap.get(p["id"], {}).get(qkey(q)) for q in subs]
            if any(r is None or not r.get("ans") for r in rows):
                continue
            units.append({"id": p["id"], "m": m, "problem": p["problem"],
                          "subs": subs, "answers": [r["ans"] for r in rows],
                          "norms": [r["norm"] for r in rows]})
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 48):
            batch = units[i : i + 48]
            ps = [P.AGGREGATE_USER.format(
                      problem=u["problem"], facts=facts_str(u["subs"], u["answers"]))
                  for u in batch]
            outs = R.chat_batch(ps, max_new=400, bs=args.bs_agg)
            for u, o in zip(batch, outs):
                a = extract_boxed(o[0])
                w.write({"id": u["id"], "m": u["m"], "subs": u["subs"],
                         "sub_answers": u["answers"], "sub_norms": u["norms"],
                         "final_ans": a, "final_norm": normalize_answer(a)})
            print(f"[aggm] {min(i+48, len(units))}/{len(units)}", flush=True)
        mark("aggm")

    # ---------------- stage 4: extended root CoT for the SC@k curve ----------------
    path = os.path.join(OUT, f"rootext.{tag}.jsonl")
    have = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in have]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 4):
            batch = todo[i : i + 4]
            ps = [P.ROOT_COT_USER.format(problem=p["problem"]) for p in batch]
            greedy = R.chat_batch(ps, max_new=768, bs=args.bs_root)
            sampled = R.chat_batch(ps, max_new=768, temperature=TEMP_ROOT,
                                   n=N_ROOT_EXT, bs=args.bs_root)
            for p, g, ss in zip(batch, greedy, sampled):
                cands = []
                for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                    a = extract_boxed(text)
                    cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
                w.write({"id": p["id"], "cands": cands})
            print(f"[rootext] {min(i+4, len(todo))}/{len(todo)}", flush=True)
        mark("rootext")

    print(f"[shard {args.shard}] tokens: {toks}", flush=True)
    print(f"[shard {args.shard}] ALL STAGES DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bs-decompose", type=int, default=64)
    ap.add_argument("--bs-sub", type=int, default=64)
    ap.add_argument("--bs-agg", type=int, default=48)
    ap.add_argument("--bs-root", type=int, default=32)
    run(ap.parse_args())
