"""Generate labeled assignment data on GSM8K for training the semantic
interface scorer.

Per problem: decompose -> sub candidate domains -> enumerate assignments
(greedy + all single-node swaps + extra combos, capped) -> aggregate each
assignment -> label by exact-match against GSM8K gold answer.

Single-node swaps of the greedy assignment are natural hard negatives
("locally plausible but incompatible") when the swapped value is wrong,
and recovery positives when the greedy value was the wrong one.

Usage:
  python gsm_datagen.py --split train --limit 800 --shard 0 --num-shards 3
  python gsm_datagen.py --split test --limit 250 --with-root --shard 0 --num-shards 3
"""
import argparse
import itertools
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import extract_boxed, normalize_answer, answers_equal  # noqa: E402
from pilot import Runner, jread, JWriter, parse_decompose, facts_str  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_SUB_SAMPLED = 3
N_ROOT_SAMPLED = 4
MAX_SUB_DOMAIN = 3
MAX_ASSIGN = 12
TEMP = 0.8


def load_gsm(split, limit, shard, num_shards):
    data = jread(os.path.join(HERE, "data", f"gsm8k_{split}.jsonl"))
    seed = 1 if split == "train" else 2
    idxs = sorted(random.Random(seed).sample(range(len(data)), min(limit, len(data))))
    rows = []
    for rank, i in enumerate(idxs):
        if rank % num_shards != shard:
            continue
        d = data[i]
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"id": i, "problem": d["question"], "gold": gold})
    return rows


def load_math_complement(limit, shard, num_shards):
    """MATH-500 problems NOT used by the pilot eval (disjoint by construction)."""
    data = jread(os.path.join(HERE, "data", "math500_test.jsonl"))
    pilot_ids = set(random.Random(0).sample(range(len(data)), 200))
    idxs = [i for i in range(len(data)) if i not in pilot_ids][:limit]
    rows = []
    for rank, i in enumerate(idxs):
        if rank % num_shards != shard:
            continue
        d = data[i]
        rows.append({"id": i, "problem": d["problem"], "gold": d["answer"]})
    return rows


def build_assignments(domains, greedy_norms):
    """greedy combo + all single-node swaps + remaining product combos, cap."""
    base = []
    for si, dom in enumerate(domains):
        gi = next((k for k, c in enumerate(dom) if c["norm"] == greedy_norms[si]), 0)
        base.append(gi)
    combos = [tuple(base)]
    for si, dom in enumerate(domains):
        for k in range(len(dom)):
            if k != base[si]:
                c = list(base)
                c[si] = k
                combos.append(tuple(c))
    for c in itertools.product(*[range(len(d)) for d in domains]):
        if c not in combos:
            combos.append(c)
        if len(combos) >= MAX_ASSIGN:
            break
    return combos[:MAX_ASSIGN]


def run(args):
    tag = f"s{args.shard}"
    if args.dataset == "math":
        out = os.path.join(HERE, "outputs_math_train")
        probs = load_math_complement(args.limit, args.shard, args.num_shards)
    else:
        out = os.path.join(HERE, f"outputs_gsm_{args.split}")
        probs = load_gsm(args.split, args.limit, args.shard, args.num_shards)
    os.makedirs(out, exist_ok=True)
    print(f"[shard {args.shard}] {len(probs)} problems ({args.dataset}/{args.split})", flush=True)
    R = Runner(args.model)

    # ---- decompose ----
    path = os.path.join(out, f"decompose.{tag}.jsonl")
    done = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in done]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 48):
            batch = todo[i : i + 48]
            outs = R.chat_batch(
                [P.DECOMPOSE_USER.format(problem=p["problem"]) for p in batch],
                system=P.DECOMPOSE_SYSTEM, max_new=300, bs=24,
            )
            for p, o in zip(batch, outs):
                w.write({**p, "subquestions": parse_decompose(o[0])})
            print(f"[decompose] {min(i+48, len(todo))}/{len(todo)}", flush=True)
    dec = {r["id"]: r for r in jread(path)}
    valid = [p for p in probs if dec[p["id"]]["subquestions"]]
    print(f"[decompose] usable {len(valid)}/{len(probs)}", flush=True)

    # ---- sub candidates ----
    path = os.path.join(out, f"subcands.{tag}.jsonl")
    done = {(r["id"], r["sub_idx"]) for r in jread(path)}
    units = [
        (p["id"], si, p["problem"], q)
        for p in valid
        for si, q in enumerate(dec[p["id"]]["subquestions"])
        if (p["id"], si) not in done
    ]
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 24):
            batch = units[i : i + 24]
            ps = [P.SUBQ_USER.format(problem=pr, subquestion=q) for (_, _, pr, q) in batch]
            greedy = R.chat_batch(ps, max_new=320, bs=24)
            sampled = R.chat_batch(ps, max_new=320, temperature=TEMP, n=N_SUB_SAMPLED, bs=24)
            for (pid, si, _, q), g, ss in zip(batch, greedy, sampled):
                cands = []
                for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                    a = extract_boxed(text)
                    cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
                w.write({"id": pid, "sub_idx": si, "subq": q, "cands": cands})
            print(f"[subcands] {min(i+24, len(units))}/{len(units)}", flush=True)
    subc = {}
    for r in jread(path):
        subc.setdefault(r["id"], {})[r["sub_idx"]] = r

    # ---- root candidates (only for end-to-end eval split) ----
    if args.with_root:
        path = os.path.join(out, f"rootcands.{tag}.jsonl")
        done = {r["id"] for r in jread(path)}
        todo = [p for p in probs if p["id"] not in done]
        if todo:
            w = JWriter(path)
            for i in range(0, len(todo), 8):
                batch = todo[i : i + 8]
                ps = [P.ROOT_COT_USER.format(problem=p["problem"]) for p in batch]
                greedy = R.chat_batch(ps, max_new=512, bs=8)
                sampled = R.chat_batch(ps, max_new=512, temperature=TEMP, n=N_ROOT_SAMPLED, bs=8)
                for p, g, ss in zip(batch, greedy, sampled):
                    cands = []
                    for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                        a = extract_boxed(text)
                        cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
                    w.write({"id": p["id"], "cands": cands})
                print(f"[rootcands] {min(i+8, len(todo))}/{len(todo)}", flush=True)

    # ---- aggregate + label ----
    path = os.path.join(out, f"aggregate.{tag}.jsonl")
    done = {(r["id"], r["assign_idx"]) for r in jread(path)}
    units = []
    for p in valid:
        pid = p["id"]
        subs = dec[pid]["subquestions"]
        if pid not in subc or len(subc[pid]) != len(subs):
            continue
        domains, greedy_norms = [], []
        for si in range(len(subs)):
            cands = subc[pid][si]["cands"]
            g = next((c for c in cands if c["kind"] == "greedy" and c["norm"]), None)
            cnt = Counter(c["norm"] for c in cands if c["norm"] is not None)
            keys = sorted(cnt, key=lambda k: -cnt[k])
            gnorm = g["norm"] if g else None
            if gnorm in cnt:
                keys = [gnorm] + [k for k in keys if k != gnorm]
            dom = []
            for k in keys[:MAX_SUB_DOMAIN]:
                rep = next(c["ans"] for c in cands if c["norm"] == k and c["ans"])
                dom.append({"norm": k, "ans": rep, "freq": cnt[k] / max(1, sum(cnt.values()))})
            if not dom:
                dom = [{"norm": None, "ans": "(no answer found)", "freq": 0.0}]
            domains.append(dom)
            greedy_norms.append(gnorm)
        for ai, combo in enumerate(build_assignments(domains, greedy_norms)):
            if (pid, ai) in done:
                continue
            chosen = [domains[si][ci] for si, ci in enumerate(combo)]
            units.append(
                {"id": pid, "assign_idx": ai, "problem": p["problem"], "gold": p["gold"],
                 "subs": subs, "chosen": chosen,
                 "is_hardcommit": all(c["norm"] == g for c, g in zip(chosen, greedy_norms)),
                 "sub_freqs": [c["freq"] for c in chosen]}
            )
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 32):
            batch = units[i : i + 32]
            ps = [
                P.AGGREGATE_USER.format(
                    problem=u["problem"],
                    facts=facts_str(u["subs"], [c["ans"] for c in u["chosen"]]),
                )
                for u in batch
            ]
            outs = R.chat_batch(ps, max_new=320, bs=16)
            for u, o in zip(batch, outs):
                a = extract_boxed(o[0])
                w.write(
                    {"id": u["id"], "assign_idx": u["assign_idx"],
                     "sub_answers": [c["ans"] for c in u["chosen"]],
                     "sub_norms": [c["norm"] for c in u["chosen"]],
                     "sub_freqs": u["sub_freqs"], "is_hardcommit": u["is_hardcommit"],
                     "final_ans": a, "final_norm": normalize_answer(a),
                     "label": 1 if (a and answers_equal(a, u["gold"])) else 0}
                )
            print(f"[aggregate] {min(i+32, len(units))}/{len(units)}", flush=True)
    print(f"[shard {args.shard}] DATAGEN DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm", "math"], default="gsm")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--with-root", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    run(ap.parse_args())
