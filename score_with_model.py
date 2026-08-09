"""Score assignments with the trained scorer: writes trained.<tag>.jsonl with
P(VALID)-based scores, same key schema as compat/verify maps.

Usage:
  python score_with_model.py --out-dir outputs           # MATH pilot assignments
  python score_with_model.py --out-dir outputs_gsm_test  # GSM8K test assignments
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scorer import SCORER_USER, facts_str, jread_glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = os.path.join(HERE, args.out_dir)
    dec = {r["id"]: r for r in jread_glob(os.path.join(out_dir, "decompose.s*.jsonl"))}
    rows = jread_glob(os.path.join(out_dir, "aggregate.s*.jsonl"))
    done = {
        (r["id"], r["assign_idx"])
        for r in jread_glob(os.path.join(out_dir, "trained.s*.jsonl"))
    }
    units = [
        r for r in rows
        if r["final_ans"] is not None and (r["id"], r["assign_idx"]) not in done
        and dec.get(r["id"], {}).get("subquestions")
    ]
    print(f"{len(units)} assignments to score", flush=True)
    if not units:
        return

    tok = AutoTokenizer.from_pretrained(args.ckpt, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    v_id = tok.encode("VALID", add_special_tokens=False)[0]
    i_id = tok.encode("INVALID", add_special_tokens=False)[0]

    out_path = os.path.join(out_dir, f"trained.s{args.shard}.jsonl")
    f = open(out_path, "a", buffering=1)
    my = [u for k, u in enumerate(units) if k % args.num_shards == args.shard]
    for i in range(0, len(my), args.bs):
        batch = my[i : i + args.bs]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": SCORER_USER.format(
                    problem=dec[u["id"]]["problem"],
                    facts=facts_str(dec[u["id"]]["subquestions"], u["sub_answers"]),
                    final=u["final_ans"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        p = torch.sigmoid(logits[:, v_id] - logits[:, i_id]).tolist()
        for u, s in zip(batch, p):
            f.write(json.dumps({"id": u["id"], "assign_idx": u["assign_idx"], "trained": s}) + "\n")
        if (i // args.bs) % 20 == 0:
            print(f"{min(i+args.bs, len(my))}/{len(my)}", flush=True)
    print("SCORING DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "scorer_ckpt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    main(ap.parse_args())
