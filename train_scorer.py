"""Train the semantic interface scorer: a small causal LM fine-tuned to emit
VALID / INVALID for (problem, sub-answers assignment, final answer) sketches.

Data: outputs_gsm_train/{decompose,aggregate}.s*.jsonl produced by gsm_datagen.
Split by problem id (95/5). Loss only on the verdict token. Score at inference
= P(VALID) / (P(VALID)+P(INVALID)).
"""
import argparse
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))

SCORER_USER = """Problem:
{problem}

Proposed intermediate results:
{facts}

Proposed final answer: {final}

Is this solution sketch correct and internally consistent? Answer VALID or INVALID."""


def facts_str(subs, answers):
    return "\n".join(f"{i+1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(zip(subs, answers)))


def jread_glob(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_examples(out_dir, max_pos=3, max_neg=5, max_syn_neg=3):
    dec = {r["id"]: r for r in jread_glob(os.path.join(out_dir, "decompose.s*.jsonl"))}
    byp = {}
    for r in jread_glob(os.path.join(out_dir, "aggregate.s*.jsonl")):
        byp.setdefault(r["id"], []).append(r)
    syn = {}
    for r in jread_glob(os.path.join(out_dir, "synthetic_negatives.jsonl")):
        syn.setdefault(r["id"], []).append(r)
    examples = []
    for pid, rows in byp.items():
        d = dec[pid]
        rng = random.Random(pid)
        seen, pos, neg = set(), [], []
        for r in rows:
            if r["final_ans"] is None:
                continue
            key = (tuple(r["sub_norms"]), r["final_norm"])
            if key in seen:
                continue
            seen.add(key)
            (pos if r["label"] == 1 else neg).append(r)
        rng.shuffle(pos)
        rng.shuffle(neg)
        picked = [(r, r["label"], "nat") for r in pos[:max_pos] + neg[:max_neg]]
        s = syn.get(pid, [])
        rng.shuffle(s)
        picked += [(r, 0, "syn") for r in s[:max_syn_neg]]
        for r, label, kind in picked:
            examples.append(
                {"id": f"{os.path.basename(out_dir)}:{pid}", "kind": kind,
                 "prompt": SCORER_USER.format(
                     problem=d["problem"],
                     facts=facts_str(d["subquestions"], r["sub_answers"]),
                     final=r["final_ans"]),
                 "label": label}
            )
    return examples


def main(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    examples = []
    for d in args.dirs.split(","):
        exs = build_examples(os.path.join(HERE, d.strip()))
        print(f"{d}: {len(exs)} examples", flush=True)
        examples += exs
    pids = sorted({e["id"] for e in examples})
    random.Random(42).shuffle(pids)
    val_pids = set(pids[: max(1, len(pids) // 20)])
    train = [e for e in examples if e["id"] not in val_pids]
    val = [e for e in examples if e["id"] in val_pids]
    npos = sum(e["label"] for e in train)
    print(f"train={len(train)} (pos={npos}, neg={len(train)-npos})  val={len(val)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    v_id = tok.encode("VALID", add_special_tokens=False)[0]
    i_id = tok.encode("INVALID", add_special_tokens=False)[0]
    print(f"verdict token ids: VALID={v_id} INVALID={i_id}", flush=True)

    def encode(batch):
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": e["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for e in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        labels = torch.tensor([v_id if e["label"] == 1 else i_id for e in batch])
        return enc, labels

    def val_auroc():
        model.eval()
        scores, ys, kinds = [], [], []
        with torch.no_grad():
            for i in range(0, len(val), args.bs * 2):
                batch = val[i : i + args.bs * 2]
                enc, _ = encode(batch)
                enc = {k: v.cuda() for k, v in enc.items()}
                logits = model(**enc).logits[:, -1, :]
                lv, li = logits[:, v_id], logits[:, i_id]
                p = torch.sigmoid(lv - li)
                scores += p.tolist()
                ys += [e["label"] for e in batch]
                kinds += [e.get("kind", "nat") for e in batch]
        model.train()

        def _auroc(sel):
            pos = [s for s, y, k in zip(scores, ys, kinds) if y == 1 and sel(k)]
            neg = [s for s, y, k in zip(scores, ys, kinds) if y == 0 and sel(k)]
            if not pos or not neg:
                return float("nan")
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            return wins / (len(pos) * len(neg))

        # natural subset uses natural negatives only; overall mixes both
        return _auroc(lambda k: True), _auroc(lambda k: k == "nat")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / args.bs) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.03), steps)
    print(f"total steps {steps}", flush=True)

    model.train()
    rng = random.Random(0)
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), args.bs):
            batch = train[i : i + args.bs]
            enc, labels = encode(batch)
            enc = {k: v.cuda() for k, v in enc.items()}
            logits = model(**enc).logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits.float(), labels.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
            if step % 400 == 0:
                a, an = val_auroc()
                print(f"  [val] AUROC={a:.4f}  natural-only={an:.4f}", flush=True)
    a, an = val_auroc()
    print(f"[final val] AUROC={a:.4f}  natural-only={an:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dirs", default="outputs_gsm_train")
    ap.add_argument("--out", default=os.path.join(HERE, "scorer_ckpt"))
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    main(ap.parse_args())
