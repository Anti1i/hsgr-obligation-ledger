"""Cheap SC@k ceiling on multi-hop QA (open-book, gold paragraphs only).

Goal: place SC@1 in the 0.3–0.6 band and measure SC@8−SC@1 gap.
Uses supporting paragraphs from the gold decomposition only (not full corpus
retrieval) so we isolate reasoning difficulty from retrieval.

Usage:
  python mh_ceiling.py --data data/musique_ans_val.jsonl --out-dir mh_ceil \\
      --limit 200 --n-samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot import JWriter, Runner, jread  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_RE = re.compile(r"#(\d+)")

SYSTEM = (
    "You answer multi-hop questions using ONLY the provided evidence paragraphs. "
    "Reason briefly, then put the final answer in \\boxed{}."
)

USER = """Evidence:
{evidence}

Question: {question}

Answer with \\boxed{{...}}."""


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;\"'`")
    return s


def answers_match(pred, gold, aliases=None):
    p = normalize(pred)
    cands = [gold] + list(aliases or [])
    return any(p == normalize(c) for c in cands if c)


def extract_boxed(text: str):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    if m:
        return m[-1].strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1] if lines else None


def evidence_from_row(r):
    """Prefer support paragraphs embedded in decomposition; else skip."""
    chunks = []
    for step in r.get("question_decomposition") or []:
        sp = step.get("support_paragraph")
        if isinstance(sp, dict) and sp.get("paragraph_text"):
            title = sp.get("title") or ""
            chunks.append(f"[{title}] {sp['paragraph_text']}".strip())
    # voidful mirror may only have paragraph_support_idx without text —
    # then we cannot do open-book without the paragraphs field.
    if not chunks and r.get("paragraphs"):
        idxs = sorted({
            step.get("paragraph_support_idx")
            for step in (r.get("question_decomposition") or [])
            if step.get("paragraph_support_idx") is not None
        })
        paras = {p.get("idx", i): p for i, p in enumerate(r["paragraphs"])}
        for i in idxs:
            p = paras.get(i)
            if not p:
                continue
            chunks.append(f"[{p.get('title','')}] {p.get('paragraph_text','')}".strip())
    return "\n\n".join(chunks)


def load_rows(path, limit, seed=0):
    rows = jread(path)
    usable = []
    for r in rows:
        ev = evidence_from_row(r)
        if not ev or not r.get("question") or not r.get("answer"):
            continue
        usable.append({**r, "_evidence": ev})
    if limit and limit < len(usable):
        idxs = sorted(random.Random(seed).sample(range(len(usable)), limit))
        usable = [usable[i] for i in idxs]
    return usable


def run(args):
    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(out, exist_ok=True)
    rows = load_rows(data, args.limit)
    print(f"[mh_ceiling] {len(rows)} usable problems", flush=True)
    if not rows:
        raise SystemExit("no usable rows (need support paragraph text)")

    tag = f"s{args.shard}"
    path = os.path.join(out, f"sc.{tag}.jsonl")
    have = {r["id"] for r in jread(path)}

    R = Runner(args.model)
    w = JWriter(path)
    n_new = 0
    for i, r in enumerate(rows):
        if i % args.num_shards != args.shard:
            continue
        rid = r.get("id") or f"row-{i}"
        if rid in have:
            continue
        prompts = [USER.format(evidence=r["_evidence"], question=r["question"])
                   for _ in range(args.n_samples)]
        outs = R.chat_batch(prompts, system=SYSTEM, max_new=256, bs=min(4, args.n_samples),
                            temperature=0.7 if args.n_samples > 1 else 0.0)
        cands = []
        for o in outs:
            text = o[0] if isinstance(o, (list, tuple)) else o
            ans = extract_boxed(text)
            cands.append({"ans": ans, "norm": normalize(ans) if ans else None,
                          "text": text[:500]})
        gold = r["answer"]
        aliases = r.get("answer_aliases") or []
        hit1 = bool(cands and cands[0]["ans"] and answers_match(cands[0]["ans"], gold, aliases))
        oracle_k = any(c["ans"] and answers_match(c["ans"], gold, aliases) for c in cands)
        vote = Counter(c["norm"] for c in cands if c["norm"])
        top = vote.most_common(1)[0][0] if vote else None
        sc = bool(top and answers_match(top, gold, aliases))
        w.write({
            "id": rid, "gold": gold, "aliases": aliases,
            "n_hops": len(r.get("question_decomposition") or []),
            "hit1": hit1, "sc": sc, "oracle": oracle_k,
            "cands": cands,
        })
        n_new += 1
        if n_new % 10 == 0:
            print(f"[mh_ceiling] {n_new} written", flush=True)
    print(f"[mh_ceiling] wrote {n_new} new rows -> {path}", flush=True)


def analyze(out_dir):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    rows = []
    import glob
    for p in sorted(glob.glob(os.path.join(d, "sc.s*.jsonl"))):
        rows += jread(p)
    n = len(rows)
    if not n:
        print("no rows")
        return
    hit1 = sum(r["hit1"] for r in rows) / n
    sc = sum(r["sc"] for r in rows) / n
    ora = sum(r["oracle"] for r in rows) / n
    print(f"== mh ceiling ({n}) ==")
    print(f"  greedy/SC@1≈ {hit1:.3f}")
    print(f"  SC@k        {sc:.3f}")
    print(f"  oracle@k    {ora:.3f}   gap(oracle-SC1)={ora-hit1:+.3f}")
    by = Counter()
    for r in rows:
        by[r.get("n_hops", 0)] += 1
    print("  hop mix:", dict(by))
    band = 0.30 <= hit1 <= 0.60
    print(f"  ceiling band 0.3–0.6: {'PASS' if band else 'FAIL'} "
          f"(hit1={hit1:.3f})")
    rep = {"n": n, "hit1": hit1, "sc": sc, "oracle": ora, "band_pass": band}
    with open(os.path.join(HERE, "mh_ceiling_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="mh_ceil")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only:
        analyze(a.out_dir)
    else:
        run(a)
        analyze(a.out_dir)
