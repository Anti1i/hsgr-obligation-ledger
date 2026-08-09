"""G1: extend direct-CoT root candidates to N samples for a budget-matched
self-consistency baseline.

V7 showed DCH+probe costs about as many generated tokens as SC@8 (MATH) / SC@7
(GSM), and DCH+verify as much as SC@13 / SC@9, while the pilot only sampled 5
root candidates. Without more samples no budget-matched claim is possible.

Writes <dir>/rootcands_ext.s0.jsonl with N_EXT additional sampled answers per
problem (resumable), then `--stage report` prints the SC@k accuracy curve
against the DCH operating points.

Usage:
  python g1_root_extend.py --stage gen --dir outputs --n-ext 15
  python g1_root_extend.py --stage report --dir outputs
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, extract_boxed, normalize_answer  # noqa: E402
from phase0_reward_audit import jread_glob  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = 0.8
CAP_ROOT = 768


def stage_gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = os.path.join(HERE, args.dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    path = os.path.join(d, "rootcands_ext.s0.jsonl")
    have = {}
    if os.path.exists(path):
        for r in (json.loads(l) for l in open(path)):
            have[r["id"]] = r
    todo = [pid for pid in sorted(dec)
            if len(have.get(pid, {}).get("cands", [])) < args.n_ext]
    print(f"[G1] {args.dir}: {len(dec)} problems, {len(todo)} need samples", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print("[G1] model loaded", flush=True)

    f = open(path, "a", buffering=1)
    for n_done, pid in enumerate(todo):
        user = P.ROOT_COT_USER.format(problem=dec[pid]["problem"])
        text = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        cands = list(have.get(pid, {}).get("cands", []))
        while len(cands) < args.n_ext:
            k = min(args.bs, args.n_ext - len(cands))
            enc = tok([text] * k, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=CAP_ROOT, do_sample=True,
                                     temperature=TEMP, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            for j in range(k):
                out = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                a = extract_boxed(out)
                cands.append({"ans": a, "norm": normalize_answer(a)})
            del enc, gen
            torch.cuda.empty_cache()
        f.write(json.dumps({"id": pid, "cands": cands}, ensure_ascii=False) + "\n")
        if n_done % 10 == 0:
            print(f"[G1] {n_done}/{len(todo)}", flush=True)
    print("[G1] gen done", flush=True)


def stage_report(args):
    d = os.path.join(HERE, args.dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    base = {r["id"]: r for r in jread_glob(d, "rootcands.s*.jsonl")}
    ext = {}
    p = os.path.join(d, "rootcands_ext.s0.jsonl")
    if os.path.exists(p):
        for r in (json.loads(l) for l in open(p)):
            ext[r["id"]] = r
    aggs = set(r["id"] for r in jread_glob(d, "aggregate.s*.jsonl"))
    ids = sorted(i for i in dec if dec[i].get("subquestions") and i in aggs)

    def pool(pid):
        cs = [c for c in base.get(pid, {}).get("cands", []) if c.get("norm")]
        cs += [c for c in ext.get(pid, {}).get("cands", []) if c.get("norm")]
        return cs

    print(f"== G1 {args.dir}: {len(ids)} usable problems ==")
    rows = []
    for k in (1, 3, 5, 8, 10, 13, 16, 20):
        ok, n = 0, 0
        for pid in ids:
            cs = pool(pid)
            if len(cs) < k:
                continue
            n += 1
            cnt = Counter(c["norm"] for c in cs[:k])
            top = cnt.most_common(1)[0][0]
            rep = next(c["ans"] for c in cs[:k] if c["norm"] == top)
            ok += bool(rep and answers_equal(rep, dec[pid]["gold"]))
        if n:
            rows.append({"k": k, "n": n, "acc": ok / n, "tokens": k * CAP_ROOT})
            print(f"  SC@{k:<3d} n={n:<4d} acc={ok/n:.3f}  tokens={k*CAP_ROOT}")
    with open(os.path.join(HERE, f"g1_report_{args.dir}.json"), "w") as fo:
        json.dump({"dir": args.dir, "n_ids": len(ids), "curve": rows}, fo, indent=1)
    print(f"saved g1_report_{args.dir}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "report"], required=True)
    ap.add_argument("--dir", default="outputs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-ext", type=int, default=15)
    ap.add_argument("--bs", type=int, default=5)
    a = ap.parse_args()
    stage_gen(a) if a.stage == "gen" else stage_report(a)
