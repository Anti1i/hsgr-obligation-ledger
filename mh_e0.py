"""E0-MH: oracle gold-decomposition execution on MuSiQue (decomposition tax).

Uses gold question_decomposition as the hierarchy. Two modes:
  predicted — feed predicted predecessor hop answers into later hops
  oracle    — feed gold predecessor hop answers

Baseline: open-book SC@k on the same gold support paragraphs (reuse mh_ceil
root samples when available, else resample).

Usage:
  python mh_e0.py --data data/musique_ans_val.jsonl --out-dir mh_e0 --limit 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import (  # noqa: E402
    SYSTEM as CEIL_SYSTEM,
    USER as CEIL_USER,
    answers_match,
    evidence_from_row,
    extract_boxed,
    normalize,
)
from pilot import JWriter, Runner, jread  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_RE = re.compile(r"#(\d+)")

HOP_SYSTEM = (
    "You answer one hop of a multi-hop question using ONLY the provided "
    "evidence and any given predecessor answers. Put the hop answer in \\boxed{}."
)

HOP_USER = """Evidence for this hop:
{evidence}

Original question: {question}

Structural state:
{structure}

Answer ONLY the current hop goal. \\boxed{{...}}."""


def hop_deps(decomp):
    deps = []
    for i, step in enumerate(decomp):
        q = step.get("question") or ""
        preds = sorted({int(x) - 1 for x in REF_RE.findall(q) if 0 < int(x) <= i})
        deps.append(preds)
    return deps


def resolve_goal(q, pred_vals):
    """Replace #k placeholders with predecessor answers."""
    def repl(m):
        k = int(m.group(1)) - 1
        return str(pred_vals.get(k, m.group(0)))
    return REF_RE.sub(repl, q)


def para_for_step(r, step):
    sp = step.get("support_paragraph")
    if isinstance(sp, dict) and sp.get("paragraph_text"):
        return f"[{sp.get('title','')}] {sp['paragraph_text']}".strip()
    idx = step.get("paragraph_support_idx")
    for i, p in enumerate(r.get("paragraphs") or []):
        if (p.get("idx", i) if isinstance(p, dict) else i) == idx:
            return f"[{p.get('title','')}] {p.get('paragraph_text','')}".strip()
    return evidence_from_row(r)


def load_rows(path, limit, seed=0):
    rows = jread(path)
    usable = []
    skip = {"short": 0, "no_ans": 0, "no_ev": 0}
    for r in rows:
        decomp = r.get("question_decomposition") or []
        if len(decomp) < 2:
            skip["short"] += 1
            continue
        if not r.get("answer"):
            skip["no_ans"] += 1
            continue
        if not evidence_from_row(r):
            skip["no_ev"] += 1
            continue
        usable.append(r)
    print(f"[mh_e0] loaded {len(rows)} raw → {len(usable)} usable  "
          f"(skip short={skip['short']} no_ans={skip['no_ans']} no_ev={skip['no_ev']})",
          flush=True)
    if not usable:
        raise SystemExit(
            "FATAL: 0 usable MuSiQue rows — jsonl likely missing support "
            "paragraphs; re-run: python fetch_data.py --which musique --force"
        )
    if limit and limit < len(usable):
        idxs = sorted(random.Random(seed).sample(range(len(usable)), limit))
        usable = [usable[i] for i in idxs]
    for i, r in enumerate(usable):
        r = dict(r)
        r["_uid"] = r.get("id") or f"row-{i}"
        usable[i] = r
    return usable


def run_hops(R, rows, OUT, mode, bs=8):
    tag = "s0"
    path = os.path.join(OUT, f"hops_{mode}.{tag}.jsonl")
    have = {(r["id"], r["hop"]) for r in jread(path)}
    pred = defaultdict(dict)
    for r in jread(path):
        pred[r["id"]][r["hop"]] = r.get("norm")

    w = JWriter(path)
    # topological waves
    while True:
        units = []
        for r in rows:
            decomp = r["question_decomposition"]
            deps = hop_deps(decomp)
            uid = r["_uid"]
            for i in range(len(decomp)):
                if (uid, i) in have:
                    continue
                if mode == "oracle":
                    units.append((r, i, deps[i]))
                elif all(j in pred[uid] for j in deps[i]):
                    units.append((r, i, deps[i]))
        if not units:
            break
        for i0 in range(0, len(units), bs):
            batch = units[i0:i0 + bs]
            users = []
            for r, i, deps in batch:
                decomp = r["question_decomposition"]
                if mode == "oracle":
                    feed = {j: decomp[j]["answer"] for j in deps}
                else:
                    feed = {j: pred[r["_uid"]][j] for j in deps}
                goal = resolve_goal(decomp[i]["question"], feed)
                lines = [
                    f"[CURRENT] hop {i+1}/{len(decomp)}",
                    f"[GOAL] {goal}",
                ]
                if feed:
                    lines.append("[DEPENDS_ON]")
                    for j, v in feed.items():
                        lines.append(f"  - hop {j+1}: {v}")
                else:
                    lines.append("[DEPENDS_ON] (none)")
                users.append(HOP_USER.format(
                    evidence=para_for_step(r, decomp[i]),
                    question=r["question"],
                    structure="\n".join(lines),
                ))
            outs = R.chat_batch(users, system=HOP_SYSTEM, max_new=128, bs=bs)
            for (r, i, deps), o in zip(batch, outs):
                text = o[0] if isinstance(o, (list, tuple)) else o
                ans = extract_boxed(text)
                gold = r["question_decomposition"][i]["answer"]
                norm = normalize(ans) if ans else None
                hit = bool(ans and answers_match(ans, gold))
                w.write({
                    "id": r["_uid"], "hop": i, "mode": mode,
                    "goal": r["question_decomposition"][i]["question"],
                    "ans": ans, "norm": norm, "gold": gold, "hit": hit,
                    "is_final": i == len(r["question_decomposition"]) - 1,
                })
                pred[r["_uid"]][i] = norm or ans
                have.add((r["_uid"], i))
            print(f"[hops_{mode}] +{len(batch)} (have {len(have)})", flush=True)
        if mode == "oracle":
            break


def run_sc(R, rows, OUT, n_samples=8, reuse_dir=None):
    path = os.path.join(OUT, "sc.s0.jsonl")
    if reuse_dir:
        srcs = sorted(glob.glob(os.path.join(reuse_dir, "sc.s*.jsonl")))
        if srcs:
            # copy matching ids
            want = {r["_uid"] for r in rows}
            kept = []
            for p in srcs:
                for r in jread(p):
                    if r["id"] in want:
                        kept.append(r)
            if len(kept) >= 0.8 * len(rows):
                w = JWriter(path)
                for r in kept:
                    w.write(r)
                print(f"[sc] reused {len(kept)} from {reuse_dir}", flush=True)
                return
    have = {r["id"] for r in jread(path)}
    w = JWriter(path)
    for r in rows:
        if r["_uid"] in have:
            continue
        ev = evidence_from_row(r)
        prompts = [CEIL_USER.format(evidence=ev, question=r["question"])
                   for _ in range(n_samples)]
        outs = R.chat_batch(prompts, system=CEIL_SYSTEM, max_new=256,
                            bs=min(4, n_samples), temperature=0.7)
        cands = []
        for o in outs:
            text = o[0] if isinstance(o, (list, tuple)) else o
            ans = extract_boxed(text)
            cands.append({"ans": ans, "norm": normalize(ans) if ans else None})
        gold, aliases = r["answer"], r.get("answer_aliases") or []
        hit1 = bool(cands and cands[0]["ans"] and answers_match(cands[0]["ans"], gold, aliases))
        ora = any(c["ans"] and answers_match(c["ans"], gold, aliases) for c in cands)
        vote = Counter(c["norm"] for c in cands if c["norm"])
        top = vote.most_common(1)[0][0] if vote else None
        sc = bool(top and answers_match(top, gold, aliases))
        w.write({"id": r["_uid"], "gold": gold, "aliases": aliases,
                 "hit1": hit1, "sc": sc, "oracle": ora, "cands": cands,
                 "n_hops": len(r["question_decomposition"])})
        print(f"[sc] {r['_uid']}", flush=True)


def analyze(OUT, rows):
    hops = {m: {} for m in ("predicted", "oracle")}
    for mode in hops:
        for r in jread(os.path.join(OUT, f"hops_{mode}.s0.jsonl")):
            hops[mode].setdefault(r["id"], {})[r["hop"]] = r
    sc = {r["id"]: r for r in jread(os.path.join(OUT, "sc.s0.jsonl"))}
    ids = [r["_uid"] for r in rows if r["_uid"] in sc]
    n = len(ids)
    print(f"== mh E0 ({n}) ==")
    by = {}
    for mode in ("predicted", "oracle"):
        final = hop_hit = hop_n = 0
        for uid in ids:
            row = next(r for r in rows if r["_uid"] == uid)
            gold, aliases = row["answer"], row.get("answer_aliases") or []
            nd = hops[mode].get(uid, {})
            for j, rr in nd.items():
                hop_n += 1
                hop_hit += bool(rr.get("hit"))
            last = nd.get(len(row["question_decomposition"]) - 1)
            if last and last.get("ans") and answers_match(last["ans"], gold, aliases):
                final += 1
        by[mode] = {"final": final / max(1, n), "hop_hit": hop_hit / max(1, hop_n)}
        print(f"  {mode}: final={by[mode]['final']:.3f}  hop_hit={by[mode]['hop_hit']:.3f}")
    hit1 = sum(sc[i]["hit1"] for i in ids) / max(1, n)
    sc8 = sum(sc[i]["sc"] for i in ids) / max(1, n)
    ora = sum(sc[i]["oracle"] for i in ids) / max(1, n)
    print(f"  SC@1={hit1:.3f}  SC@8={sc8:.3f}  oracle@8={ora:.3f}")
    tax = hit1 - by["oracle"]["final"]
    print(f"  decomposition tax (SC@1 - oracle-hops)={tax:+.3f}")
    delta = by["predicted"]["final"] - hit1
    print(f"  predicted vs SC@1 delta={delta:+.3f}")
    gate = by["predicted"]["final"] > hit1
    print(f"  HEADROOM gate (predicted > SC@1): {'PASS' if gate else 'FAIL'}")
    rep = {"n": n, "modes": by, "sc1": hit1, "sc8": sc8, "oracle8": ora,
           "tax": tax, "gate_pass": gate}
    with open(os.path.join(HERE, "mh_e0_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="mh_e0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--reuse-sc", default="")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    data = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    OUT = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(HERE, a.out_dir)
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows(data, a.limit)
    print(f"[mh_e0] {len(rows)} problems", flush=True)
    if a.analyze_only:
        analyze(OUT, rows)
        return
    R = Runner(a.model)
    run_hops(R, rows, OUT, "oracle")
    run_hops(R, rows, OUT, "predicted")
    run_sc(R, rows, OUT, reuse_dir=a.reuse_sc or None)
    analyze(OUT, rows)


if __name__ == "__main__":
    main()
