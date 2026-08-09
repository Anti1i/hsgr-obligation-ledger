"""DCH-HSGR pilot (depth-2): end-to-end pipeline for one shard of problems.

Generalized over datasets: pass --data <jsonl> --out-dir <dir>. Accepts rows of
the form {"problem"|"question", "answer"} (extra fields are carried through).

Stages (all resumable, one process per GPU):
  1. decompose : problem -> 2-3 self-contained subquestions (greedy, JSON)
  2. subcands  : per subquestion, 1 greedy + N_SUB sampled candidate answers
  3. rootcands : per problem, 1 greedy + N_ROOT sampled direct CoT answers
  4. aggregate : for every assignment in the product of sub candidate domains,
                 derive a final answer. Includes the hard-commit assignment.
  5. compat    : LLM-judged compatibility score (0-10) per assignment
  6. verify    : recompute-then-vote VALID/INVALID scorer per assignment

Usage:
  python pilot.py --data data/gsm_chain_test.jsonl --out-dir outputs_chain \
                  --shard 0 --num-shards 4 --limit 300
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
from answer_check import extract_boxed, normalize_answer  # noqa: E402
import prompts as P  # noqa: E402

N_SUB_SAMPLED = 3
N_ROOT_SAMPLED = 4
MAX_SUB_DOMAIN = 3
MAX_ASSIGN = 32
TEMP = 0.8

HERE = os.path.dirname(os.path.abspath(__file__))


def jread(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


class JWriter:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class Runner:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Torch 2.13+cu130's cuDNN SDPA backend hits
        # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on this cluster's H100/H200
        # nodes; fall through to flash / mem-efficient / math.
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass

        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()
        self.torch = torch
        print(f"[model] loaded {model_id} in {time.time()-t0:.0f}s", flush=True)

    def chat_batch(self, users, system=None, max_new=512, temperature=None, n=1, bs=16):
        texts = []
        for u in users:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": u}
            ]
            texts.append(
                self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        flat = [t for t in texts for _ in range(n)]
        outs = []
        for i in range(0, len(flat), bs):
            chunk = flat[i : i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            kwargs = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id)
            if temperature is None:
                kwargs.update(do_sample=False)
            else:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, **kwargs)
            for j in range(len(chunk)):
                outs.append(
                    self.tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                )
            del enc, gen
            self.torch.cuda.empty_cache()
        return [outs[k * n : (k + 1) * n] for k in range(len(users))]


def load_problems(data_path, limit, shard, num_shards):
    p = data_path if os.path.isabs(data_path) else os.path.join(HERE, data_path)
    data = jread(p)
    if not data:
        raise SystemExit(f"no problems loaded from {p}")
    if limit and limit < len(data):
        idxs = sorted(random.Random(0).sample(range(len(data)), limit))
    else:
        idxs = list(range(len(data)))
    rows = []
    for rank, i in enumerate(idxs):
        if rank % num_shards != shard:
            continue
        d = data[i]
        rows.append({"id": i,
                     "problem": d.get("problem") or d.get("question"),
                     "gold": str(d["answer"]),
                     "level": d.get("level"), "subject": d.get("subject"),
                     "n_steps": d.get("n_steps")})
    return rows


def parse_decompose(text):
    """Extract subquestion strings. Models often wrap them in objects such as
    {"question": ..., "answer": ...}, so accept that shape too."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    subs = []
    for s in obj.get("subquestions", []):
        if isinstance(s, str):
            q = s
        elif isinstance(s, dict):
            q = next((s[k] for k in ("question", "subquestion", "q", "text")
                      if isinstance(s.get(k), str)), None)
        else:
            q = None
        if q and q.strip():
            subs.append(q.strip())
    return subs[:3] if len(subs) >= 2 else None


def facts_str(subs, answers):
    return "\n".join(f"{i+1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(zip(subs, answers)))


def run(args):
    tag = f"s{args.shard}"
    OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    probs = load_problems(args.data, args.limit, args.shard, args.num_shards)
    print(f"[shard {args.shard}] {len(probs)} problems from {args.data}", flush=True)
    R = Runner(args.model)

    # ---------------- stage 1: decompose ----------------
    path = os.path.join(OUT, f"decompose.{tag}.jsonl")
    done = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in done]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 48):
            batch = todo[i : i + 48]
            outs = R.chat_batch(
                [P.DECOMPOSE_USER.format(problem=p["problem"]) for p in batch],
                system=P.DECOMPOSE_SYSTEM, max_new=300, bs=args.bs_decompose,
            )
            for p, o in zip(batch, outs):
                w.write({**p, "subquestions": parse_decompose(o[0]), "raw": o[0][:500]})
            print(f"[decompose] {min(i+48, len(todo))}/{len(todo)}", flush=True)
    dec = {r["id"]: r for r in jread(path)}
    valid = [p for p in probs if dec[p["id"]]["subquestions"]]
    print(f"[decompose] usable: {len(valid)}/{len(probs)}", flush=True)

    # ---------------- stage 2: sub candidates ----------------
    path = os.path.join(OUT, f"subcands.{tag}.jsonl")
    done = {(r["id"], r["sub_idx"]) for r in jread(path)}
    units = [(p["id"], si, p["problem"], q)
             for p in valid
             for si, q in enumerate(dec[p["id"]]["subquestions"])
             if (p["id"], si) not in done]
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 24):
            batch = units[i : i + 24]
            ps = [P.SUBQ_USER.format(problem=pr, subquestion=q) for (_, _, pr, q) in batch]
            greedy = R.chat_batch(ps, max_new=400, bs=args.bs_sub)
            sampled = R.chat_batch(ps, max_new=400, temperature=TEMP,
                                   n=N_SUB_SAMPLED, bs=args.bs_sub)
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

    # ---------------- stage 3: root candidates ----------------
    path = os.path.join(OUT, f"rootcands.{tag}.jsonl")
    done = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in done]
    if todo:
        w = JWriter(path)
        for i in range(0, len(todo), 6):
            batch = todo[i : i + 6]
            ps = [P.ROOT_COT_USER.format(problem=p["problem"]) for p in batch]
            greedy = R.chat_batch(ps, max_new=768, bs=args.bs_root)
            sampled = R.chat_batch(ps, max_new=768, temperature=TEMP,
                                   n=N_ROOT_SAMPLED, bs=args.bs_root)
            for p, g, ss in zip(batch, greedy, sampled):
                cands = []
                for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                    a = extract_boxed(text)
                    cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
                w.write({"id": p["id"], "cands": cands})
            print(f"[rootcands] {min(i+6, len(todo))}/{len(todo)}", flush=True)

    # ---------------- stage 4: aggregate over assignments ----------------
    path = os.path.join(OUT, f"aggregate.{tag}.jsonl")
    done = {(r["id"], r["assign_idx"]) for r in jread(path)}
    units = []
    for p in valid:
        pid = p["id"]
        subs = dec[pid]["subquestions"]
        if pid not in subc or len(subc[pid]) != len(subs):
            continue
        domains, greedy_ans = [], []
        for si in range(len(subs)):
            cands = subc[pid][si]["cands"]
            g = next((c["ans"] for c in cands if c["kind"] == "greedy" and c["ans"]), None)
            cnt = Counter(c["norm"] for c in cands if c["norm"] is not None)
            keys = sorted(cnt, key=lambda k: -cnt[k])
            gnorm = normalize_answer(g) if g else None
            if gnorm in cnt:
                keys = [gnorm] + [k for k in keys if k != gnorm]
            ordered = []
            for k in keys[:MAX_SUB_DOMAIN]:
                rep = next(c["ans"] for c in cands if c["norm"] == k and c["ans"])
                ordered.append({"norm": k, "ans": rep,
                                "freq": cnt[k] / max(1, sum(cnt.values()))})
            if not ordered:
                ordered = [{"norm": None, "ans": "(no answer found)", "freq": 0.0}]
            domains.append(ordered)
            greedy_ans.append(gnorm)
        combos = list(itertools.product(*[range(len(d)) for d in domains]))[:MAX_ASSIGN]
        for ai, combo in enumerate(combos):
            if (pid, ai) in done:
                continue
            chosen = [domains[si][ci] for si, ci in enumerate(combo)]
            units.append({"id": pid, "assign_idx": ai, "problem": p["problem"],
                          "subs": subs, "chosen": chosen,
                          "is_hardcommit": all(c["norm"] == g for c, g in
                                               zip(chosen, greedy_ans)),
                          "sub_freqs": [c["freq"] for c in chosen]})
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 32):
            batch = units[i : i + 32]
            ps = [P.AGGREGATE_USER.format(
                      problem=u["problem"],
                      facts=facts_str(u["subs"], [c["ans"] for c in u["chosen"]]))
                  for u in batch]
            outs = R.chat_batch(ps, max_new=400, bs=args.bs_agg)
            for u, o in zip(batch, outs):
                a = extract_boxed(o[0])
                w.write({"id": u["id"], "assign_idx": u["assign_idx"],
                         "sub_answers": [c["ans"] for c in u["chosen"]],
                         "sub_norms": [c["norm"] for c in u["chosen"]],
                         "sub_freqs": u["sub_freqs"],
                         "is_hardcommit": u["is_hardcommit"],
                         "final_ans": a, "final_norm": normalize_answer(a)})
            print(f"[aggregate] {min(i+32, len(units))}/{len(units)}", flush=True)
    aggs = {}
    for r in jread(path):
        aggs.setdefault(r["id"], []).append(r)

    def pending(fname):
        p = os.path.join(OUT, fname)
        seen = {(r["id"], r["assign_idx"]) for r in jread(p)}
        out = []
        for pp in valid:
            for r in aggs.get(pp["id"], []):
                if (pp["id"], r["assign_idx"]) not in seen and r["final_ans"]:
                    out.append({**r, "problem": pp["problem"],
                                "subs": dec[pp["id"]]["subquestions"]})
        return p, out

    # ---------------- stage 5: compatibility scoring ----------------
    path, units = pending(f"compat.{tag}.jsonl")
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 48):
            batch = units[i : i + 48]
            ps = [P.COMPAT_USER.format(problem=u["problem"],
                                       facts=facts_str(u["subs"], u["sub_answers"]),
                                       final=u["final_ans"]) for u in batch]
            outs = R.chat_batch(ps, max_new=6, bs=args.bs_compat)
            for u, o in zip(batch, outs):
                m = re.search(r"\d+", o[0])
                w.write({"id": u["id"], "assign_idx": u["assign_idx"],
                         "compat": max(0, min(10, int(m.group(0)))) if m else None})
            print(f"[compat] {min(i+48, len(units))}/{len(units)}", flush=True)

    # ---------------- stage 6: verify scorer ----------------
    path, units = pending(f"verify.{tag}.jsonl")
    if units:
        w = JWriter(path)
        for i in range(0, len(units), 16):
            batch = units[i : i + 16]
            ps = [P.VERIFY_USER.format(problem=u["problem"],
                                        facts=facts_str(u["subs"], u["sub_answers"]),
                                        final=u["final_ans"]) for u in batch]
            greedy = R.chat_batch(ps, max_new=260, bs=args.bs_verify)
            sampled = R.chat_batch(ps, max_new=260, temperature=TEMP, n=3,
                                   bs=args.bs_verify)
            for u, g, ss in zip(batch, greedy, sampled):
                verdicts = []
                for text in [g[0]] + ss:
                    m = re.findall(r"VERDICT:\s*(VALID|INVALID)", text.upper())
                    verdicts.append(1 if (m and m[-1] == "VALID") else 0)
                w.write({"id": u["id"], "assign_idx": u["assign_idx"],
                         "verify": sum(verdicts) / len(verdicts), "verdicts": verdicts})
            print(f"[verify] {min(i+16, len(units))}/{len(units)}", flush=True)

    print(f"[shard {args.shard}] ALL STAGES DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    # H200 (141GB) can take far larger batches than the 4090 defaults
    ap.add_argument("--bs-decompose", type=int, default=64)
    ap.add_argument("--bs-sub", type=int, default=64)
    ap.add_argument("--bs-root", type=int, default=32)
    ap.add_argument("--bs-agg", type=int, default=48)
    ap.add_argument("--bs-compat", type=int, default=64)
    ap.add_argument("--bs-verify", type=int, default=32)
    run(ap.parse_args())
