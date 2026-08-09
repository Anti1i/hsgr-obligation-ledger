"""Run the prompted verify scorer (recompute + VALID/INVALID vote) on any
outputs dir produced by gsm_datagen/pilot. Writes verify.s{shard}.jsonl."""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot import Runner, jread, JWriter, facts_str  # noqa: E402
from train_scorer import jread_glob  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(args):
    out = os.path.join(HERE, args.out_dir)
    dec = {r["id"]: r for r in jread_glob(os.path.join(out, "decompose.s*.jsonl"))}
    rows = jread_glob(os.path.join(out, "aggregate.s*.jsonl"))
    path = os.path.join(out, f"verify.s{args.shard}.jsonl")
    done = {(r["id"], r["assign_idx"]) for r in jread(path)}
    units = []
    for k, r in enumerate(rows):
        if k % args.num_shards != args.shard:
            continue
        if r["final_ans"] is None or (r["id"], r["assign_idx"]) in done:
            continue
        d = dec.get(r["id"])
        if not d or not d.get("subquestions"):
            continue
        units.append({**r, "problem": d["problem"], "subs": d["subquestions"]})
    print(f"[shard {args.shard}] {len(units)} to verify", flush=True)
    if not units:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(units), 16):
        batch = units[i : i + 16]
        ps = [
            P.VERIFY_USER.format(
                problem=u["problem"],
                facts=facts_str(u["subs"], u["sub_answers"]),
                final=u["final_ans"],
            )
            for u in batch
        ]
        greedy = R.chat_batch(ps, max_new=260, bs=8)
        sampled = R.chat_batch(ps, max_new=260, temperature=0.8, n=3, bs=8)
        for u, g, ss in zip(batch, greedy, sampled):
            verdicts = []
            for text in [g[0]] + ss:
                m = re.findall(r"VERDICT:\s*(VALID|INVALID)", text.upper())
                verdicts.append(1 if (m and m[-1] == "VALID") else 0)
            w.write({"id": u["id"], "assign_idx": u["assign_idx"],
                     "verify": sum(verdicts) / len(verdicts)})
        print(f"[verify] {min(i+16, len(units))}/{len(units)}", flush=True)
    print("VERIFY DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    main(ap.parse_args())
