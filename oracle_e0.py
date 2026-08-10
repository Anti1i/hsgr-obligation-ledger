"""E0 pipeline with batched waves (faster than per-node bs=1)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, extract_boxed, normalize_answer  # noqa: E402
from oracle_hierarchy import serialize_node  # noqa: E402
from pilot import JWriter, Runner, jread  # noqa: E402
import prompts as P  # noqa: E402

N_ROOT = 20
TEMP_ROOT = 0.8
HERE = os.path.dirname(os.path.abspath(__file__))


def load_oracle_problems(data_path, limit, shard, num_shards):
    p = data_path if os.path.isabs(data_path) else os.path.join(HERE, data_path)
    data = jread(p)
    if not data:
        raise SystemExit(f"no problems in {p}")
    if limit and limit < len(data):
        import random
        idxs = sorted(random.Random(0).sample(range(len(data)), limit))
    else:
        idxs = list(range(len(data)))
    rows = []
    for rank, i in enumerate(idxs):
        if rank % num_shards != shard:
            continue
        d = data[i]
        rows.append({
            "id": i, "problem": d["problem"], "gold": str(d["answer"]),
            "nodes": d["nodes"], "n_steps": d["n_steps"],
        })
    return rows


def run_mode(R, probs, OUT, tag, mode, bs, toks, mark):
    path = os.path.join(OUT, f"nodes_{mode}.{tag}.jsonl")
    have = {(r["id"], r["node_idx"]) for r in jread(path)}
    # predicted values already written (for resume) + in-memory
    pred = defaultdict(dict)
    for r in jread(path):
        pred[r["id"]][r["node_idx"]] = r.get("norm")

    def feed_for(p, i):
        deps = p["nodes"][i]["depends_on"]
        if mode == "oracle":
            return {j: p["nodes"][j]["gold_value"] for j in deps}
        return {j: pred[p["id"]][j] for j in deps if j in pred[p["id"]]}

    def ready(p, i):
        if (p["id"], i) in have:
            return False
        if mode == "oracle":
            return True
        return all(j in pred[p["id"]] for j in p["nodes"][i]["depends_on"])

    w = JWriter(path)
    done_new = 0
    total_todo = sum(1 for p in probs for i in range(p["n_steps"])
                     if (p["id"], i) not in have)
    while True:
        units = []
        for p in probs:
            for i in range(p["n_steps"]):
                if ready(p, i):
                    units.append((p, i))
        if not units:
            break
        for i0 in range(0, len(units), bs):
            batch = units[i0: i0 + bs]
            structs = [serialize_node(p["problem"], p["nodes"], i,
                                      feed_for(p, i), mode=mode)
                       for p, i in batch]
            outs = R.chat_batch(
                [P.ORACLE_NODE_USER.format(structure=s) for s in structs],
                system=P.ORACLE_NODE_SYSTEM, max_new=256, bs=bs,
            )
            for (p, i), o in zip(batch, outs):
                a = extract_boxed(o[0])
                norm = normalize_answer(a)
                gold_n = p["nodes"][i]["gold_value"]
                feed = feed_for(p, i)
                w.write({"id": p["id"], "node_idx": i, "mode": mode,
                         "goal": p["nodes"][i]["goal"],
                         "depends_on": p["nodes"][i]["depends_on"],
                         "pred_feed": {str(k): v for k, v in feed.items()},
                         "ans": a, "norm": norm, "gold_value": gold_n,
                         "hit": bool(a and answers_equal(a, gold_n)),
                         "is_final": i == p["n_steps"] - 1})
                pred[p["id"]][i] = norm
                have.add((p["id"], i))
                done_new += 1
            print(f"[nodes_{mode}] {done_new}/{total_todo}", flush=True)
        if mode == "oracle":
            # oracle: all ready at once; one wave is enough
            break
    mark(f"nodes_{mode}")


def run(args):
    tag = f"s{args.shard}"
    OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    probs = load_oracle_problems(args.data, args.limit, args.shard, args.num_shards)
    print(f"[shard {args.shard}] {len(probs)} oracle problems", flush=True)
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

    if args.reuse_rootext:
        src = args.reuse_rootext
        if not os.path.isabs(src):
            src = os.path.join(HERE, src)
        dest = os.path.join(OUT, f"rootext.{tag}.jsonl")
        if not os.path.exists(dest):
            rows = []
            for p in sorted(glob.glob(os.path.join(src, "rootext.s*.jsonl"))):
                rows.extend(jread(p))
            want = {p["id"] for p in probs}
            kept = [r for r in rows if r["id"] in want]
            w = JWriter(dest)
            for r in kept:
                w.write(r)
            print(f"[rootext] reused {len(kept)}/{len(want)} from {src}", flush=True)

    run_mode(R, probs, OUT, tag, "oracle", args.bs_node, toks, mark)
    run_mode(R, probs, OUT, tag, "predicted", args.bs_node, toks, mark)

    path = os.path.join(OUT, f"rootext.{tag}.jsonl")
    have = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in have]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 4):
            batch = todo[i: i + 4]
            ps = [P.ROOT_COT_USER.format(problem=p["problem"]) for p in batch]
            greedy = R.chat_batch(ps, max_new=768, bs=args.bs_root)
            sampled = R.chat_batch(ps, max_new=768, temperature=TEMP_ROOT,
                                   n=N_ROOT, bs=args.bs_root)
            for p, g, ss in zip(batch, greedy, sampled):
                cands = []
                for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                    a = extract_boxed(text)
                    cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
                w.write({"id": p["id"], "cands": cands})
            print(f"[rootext] {min(i + 4, len(todo))}/{len(todo)}", flush=True)
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
    ap.add_argument("--reuse-rootext", default="")
    ap.add_argument("--bs-node", type=int, default=48)
    ap.add_argument("--bs-root", type=int, default=32)
    run(ap.parse_args())
