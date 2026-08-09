"""G3: hidden-state probes on REAL reasoning traces (not synthetic corruptions).

V3 showed a probe separates fully-correct assignments from `wrong_value`
corruptions (same final answer, corrupted intermediate) with within-problem
AUROC 0.842. Those negatives are programmatic, so the probe may be detecting
surface arithmetic inconsistency. G3 tests the same hypothesis on natural
traces:

  stage gen    : re-sample subquestion answers for pilot problems, KEEPING the
                 full reasoning text (the pilot only stored extracted answers).
  stage extract: for each trace, read hidden states at the final token and at
                 fractional positions (25/50/75%) of the trace, so we can also
                 test mid-generation (step-level) readability, which is what an
                 ATLAS-style gated intervention would need.
  stage probe  : train probes to predict whether the trace's own answer matches
                 the value class that appears in a root-correct assignment
                 ("useful" label), and report AUROC at each fractional position.

The interesting comparison is early positions: if 25-50% into the trace already
predicts usefulness, latent gating can act before the answer exists.

Usage:
  python g3_trace_probe.py --stage gen     --dir outputs --limit 120 --n 4
  python g3_trace_probe.py --stage extract --dir outputs
  python g3_trace_probe.py --stage probe   --dir outputs
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, extract_boxed, normalize_answer  # noqa: E402
from phase0_reward_audit import jread_glob  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = 0.8
CAP = 400
FRACTIONS = [0.25, 0.50, 0.75, 1.0]
LAYERS = [14, 21, 28]


def useful_norms(out_dir):
    """Per (pid, sub_idx): value classes participating in a root-correct assignment."""
    d = os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    good = defaultdict(set)
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        pid = r["id"]
        if pid not in dec or not r.get("final_ans"):
            continue
        if answers_equal(r["final_ans"], dec[pid]["gold"]):
            for si, k in enumerate(r["sub_norms"]):
                good[(pid, si)].add(k)
    return dec, good


def stage_gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dec, good = useful_norms(args.dir)
    pids = [p for p in sorted(dec) if dec[p].get("subquestions")
            and any((p, si) in good for si in range(len(dec[p]["subquestions"])))]
    pids = pids[: args.limit]
    path = os.path.join(HERE, args.dir, "traces.jsonl")
    done = set()
    if os.path.exists(path):
        for r in (json.loads(l) for l in open(path)):
            done.add((r["id"], r["sub_idx"]))
    units = [(p, si, q) for p in pids
             for si, q in enumerate(dec[p]["subquestions"])
             if (p, si) not in done and (p, si) in good]
    print(f"[G3] {len(pids)} problems -> {len(units)} nodes to sample", flush=True)
    if not units:
        return

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print("[G3] model loaded", flush=True)

    f = open(path, "a", buffering=1)
    for i, (pid, si, q) in enumerate(units):
        user = P.SUBQ_USER.format(problem=dec[pid]["problem"], subquestion=q)
        text = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok([text] * args.n, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=CAP, do_sample=True,
                                 temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        traces = []
        for j in range(args.n):
            out = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            a = extract_boxed(out)
            traces.append({"text": out, "ans": a, "norm": normalize_answer(a),
                           "useful": int(normalize_answer(a) in good[(pid, si)])})
        f.write(json.dumps({"id": pid, "sub_idx": si, "subq": q, "traces": traces},
                           ensure_ascii=False) + "\n")
        del enc, gen
        torch.cuda.empty_cache()
        if i % 10 == 0:
            print(f"[G3] {i}/{len(units)}", flush=True)
    print("[G3] gen done", flush=True)


def stage_extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dec, _ = useful_norms(args.dir)
    rows = [json.loads(l) for l in open(os.path.join(HERE, args.dir, "traces.jsonl"))]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print(f"[G3] extract over {len(rows)} nodes", flush=True)

    feats = {(l, fr): [] for l in LAYERS for fr in FRACTIONS}
    metas = []
    for i, r in enumerate(rows):
        user = P.SUBQ_USER.format(problem=dec[r["id"]]["problem"], subquestion=r["subq"])
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        p_len = len(tok(prompt)["input_ids"])
        for t in r["traces"]:
            ids = tok(prompt + t["text"], return_tensors="pt",
                      truncation=True, max_length=1536).to("cuda")
            n_tok = ids["input_ids"].shape[1]
            gen_len = max(1, n_tok - p_len)
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            for l in LAYERS:
                h = out.hidden_states[l][0]
                for fr in FRACTIONS:
                    pos = min(n_tok - 1, p_len + int(gen_len * fr) - 1)
                    feats[(l, fr)].append(h[max(0, pos)].float().cpu())
            metas.append({"id": r["id"], "sub_idx": r["sub_idx"],
                          "useful": t["useful"], "norm": t["norm"]})
            del ids, out
            torch.cuda.empty_cache()
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    payload = {"layers": LAYERS, "fractions": FRACTIONS, "metas": metas,
               "feats": {f"{l}@{fr}": torch.stack(v).half()
                         for (l, fr), v in feats.items()}}
    out_p = os.path.join(HERE, args.dir, "trace_feats.pt")
    torch.save(payload, out_p)
    print(f"[G3] saved {out_p}", flush=True)


def stage_probe(args):
    import torch
    from verify_latent import auroc, fit_probe, within_problem_auroc

    pk = torch.load(os.path.join(HERE, args.dir, "trace_feats.pt"))
    metas = pk["metas"]
    y = torch.tensor([m["useful"] for m in metas], dtype=torch.float32)
    keys = sorted({m["id"] for m in metas})
    import random as _r

    _r.Random(4).shuffle(keys)
    hold = set(keys[: max(1, len(keys) // 3)])
    tr = [i for i, m in enumerate(metas) if m["id"] not in hold]
    te = [i for i, m in enumerate(metas) if m["id"] in hold]
    va = tr[: max(1, len(tr) // 6)]
    tr = tr[len(va):]
    print(f"[G3] rows={len(metas)} pos_rate={y.mean():.3f} "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    report = {}
    for name, X in pk["feats"].items():
        Xf = X.float()
        t, v, s = torch.tensor(tr), torch.tensor(va), torch.tensor(te)
        pt = [f"{metas[i]['id']}:{metas[i]['sub_idx']}" for i in tr]
        pv = [f"{metas[i]['id']}:{metas[i]['sub_idx']}" for i in va]
        sc, crit = fit_probe(Xf[t], y[t], pt, Xf[v], y[v], pv, use_rank=True)
        st = sc(Xf[s])
        lab = [int(y[i].item()) for i in te]
        pid = [f"{metas[i]['id']}:{metas[i]['sub_idx']}" for i in te]
        wp, nwp = within_problem_auroc(st, lab, pid)
        report[name] = {"val": crit, "auroc": auroc(st, lab), "wp_auroc": wp,
                        "n_nodes": nwp}
        print(f"  {name:10s} val={crit:.3f} test auroc={report[name]['auroc']:.3f} "
              f"wp={wp:.3f}")
    with open(os.path.join(HERE, f"g3_report_{args.dir}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved g3_report_{args.dir}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "extract", "probe"], required=True)
    ap.add_argument("--dir", default="outputs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    {"gen": stage_gen, "extract": stage_extract, "probe": stage_probe}[a.stage](a)
