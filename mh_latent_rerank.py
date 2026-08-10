"""E2': latent verifier re-ranks the existing MuSiQue 8-sample pool.

Hypothesis under test (post §5.6 retraction of structure-as-control):
  Sampling already covers the answer (oracle@8 = 0.685). Majority vote
  (SC@8 = 0.500) leaves a +18.5pp selection gap. A linear probe on the
  frozen generator's own last-token hidden states can close a material
  fraction of that gap — without gold decomposition or gold paragraphs.

Pipeline:
  1. Load the cached SC@8 candidates (answer strings only is enough).
  2. Build a verification prompt per (problem, candidate): evidence +
     question + proposed answer. Extract last-token hidden states at
     layers {7,14,21,28} from frozen Qwen2.5-7B-Instruct.
  3. Problem-disjoint 5-fold CV: train a logistic probe (+ within-problem
     pairwise ranking loss) on 4 folds, score the held-out fold.
  4. Select argmax probe score among the 8 candidates; compare to SC@8
     majority vote with paired McNemar.

Gate (from BENCHMARK_DECISION.txt):
  close >= 1/3 of the 18.5pp gap  i.e.  >= +6pp over SC@8
  AND McNemar p < 0.05.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import (  # noqa: E402
    answers_match,
    evidence_from_row,
    normalize,
)
from mh_e0 import load_rows  # noqa: E402
from pilot import jread  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [7, 14, 21, 28]

VERIFY_USER = """Evidence:
{evidence}

Question: {question}

Proposed answer: {answer}

Is this answer correct given ONLY the evidence above?
Answer VALID or INVALID."""


# ---------------------------------------------------------------- metrics
def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
        # fraction of negatives strictly below + half the ties
        import bisect
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def within_problem_auroc(scores, labels, pids):
    byp = defaultdict(list)
    for s, y, p in zip(scores, labels, pids):
        byp[p].append((s, y))
    vals = []
    for items in byp.values():
        a = auroc([s for s, _ in items], [y for _, y in items])
        if a == a:
            vals.append(a)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return chi2, p


# ---------------------------------------------------------------- data
def find_sc(sc_dir):
    """Prefer an explicit --sc-dir; else discover the richest prior run."""
    if sc_dir:
        paths = sorted(glob.glob(os.path.join(sc_dir, "sc.s*.jsonl")))
        if paths:
            return paths
        raise SystemExit(f"no sc.s*.jsonl under {sc_dir}")
    # Discover under /mnt/scratch or local work dirs.
    cands = []
    for root in (HERE,
                 os.environ.get("SCR", ""),
                 "/mnt/scratch/z/" + os.environ.get("USER", "") + "/dch-hsgr"):
        if not root:
            continue
        for p in glob.glob(os.path.join(root, "work-*/mh_*/sc.s*.jsonl")):
            cands.append(p)
        for p in glob.glob(os.path.join(root, "mh_*/sc.s*.jsonl")):
            cands.append(p)
    if not cands:
        raise SystemExit("no cached SC samples found; pass --sc-dir")
    # Pick the directory with the most rows.
    by_dir = defaultdict(list)
    for p in cands:
        by_dir[os.path.dirname(p)].append(p)
    best, best_n = None, -1
    for d, ps in by_dir.items():
        n = sum(1 for p in ps for _ in open(p, encoding="utf-8") if _.strip())
        if n > best_n:
            best, best_n = d, n
    print(f"[sc] using {best} ({best_n} rows)", flush=True)
    return sorted(glob.glob(os.path.join(best, "sc.s*.jsonl")))


def build_units(rows, sc_by_id):
    """One unit per (problem, candidate index). Label = answer correctness."""
    units = []
    for r in rows:
        uid = r["_uid"]
        sc = sc_by_id.get(uid)
        if not sc:
            continue
        ev = evidence_from_row(r)
        gold, aliases = r["answer"], r.get("answer_aliases") or []
        for k, c in enumerate(sc["cands"]):
            ans = c.get("ans") or ""
            lab = int(bool(ans) and answers_match(ans, gold, aliases))
            units.append({
                "id": uid,
                "cand": k,
                "ans": ans,
                "norm": c.get("norm") or (normalize(ans) if ans else None),
                "label": lab,
                "prompt": VERIFY_USER.format(
                    evidence=ev, question=r["question"],
                    answer=ans if ans else "(empty)",
                ),
                "n_hops": len(r["question_decomposition"]),
                "gold": gold,
                "aliases": aliases,
            })
    return units


# ---------------------------------------------------------------- extract
def stage_extract(args, units, feat_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(feat_path) and not args.force_extract:
        print(f"[extract] exists {feat_path}, skip", flush=True)
        return

    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print(f"[extract] model={args.model}  units={len(units)}", flush=True)

    feats = {l: [] for l in LAYERS}
    metas = []
    bs = args.bs
    for i in range(0, len(units), bs):
        batch = units[i: i + bs]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": u["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for l in LAYERS:
            feats[l].append(out.hidden_states[l][:, -1, :].float().cpu())
        for u in batch:
            metas.append({k: u[k] for k in (
                "id", "cand", "ans", "norm", "label", "n_hops", "gold", "aliases"
            )})
        del enc, out
        torch.cuda.empty_cache()
        if (i // bs) % 10 == 0:
            print(f"  {min(i + bs, len(units))}/{len(units)}", flush=True)

    payload = {
        "layers": LAYERS,
        "feats": {l: torch.cat(feats[l]).half() for l in LAYERS},
        "metas": metas,
    }
    torch.save(payload, feat_path)
    print(f"[extract] saved {feat_path}", flush=True)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- probe
def fit_probe(X, y, pids, Xv, yv, pv, use_rank=True, epochs=400, lr=5e-3,
              wd=3e-3, rank_w=0.5, seed=0):
    import torch

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
    torch.manual_seed(seed)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=wd)

    pairs = None
    if use_rank:
        byp = defaultdict(lambda: ([], []))
        for i, yy in enumerate(y.tolist()):
            byp[pids[i]][int(yy)].append(i)
        rng = random.Random(seed)
        pp, nn = [], []
        for negs, poss in byp.values():
            if negs and poss:
                for _ in range(min(8, len(poss) * len(negs))):
                    pp.append(rng.choice(poss))
                    nn.append(rng.choice(negs))
        if pp:
            pairs = (torch.tensor(pp), torch.tensor(nn))

    best = (-1.0, None)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        logit = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        if pairs is not None:
            diff = logit[pairs[0]] - logit[pairs[1]]
            loss = loss + rank_w * torch.nn.functional.softplus(-diff).mean()
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == epochs:
            with torch.no_grad():
                sv = (Xvn @ w + b).tolist()
            crit, n = within_problem_auroc(sv, yv.tolist(), pv)
            if crit != crit or n < 5:
                crit = auroc(sv, yv.tolist())
            if crit == crit and crit > best[0]:
                best = (crit, (w.detach().clone(), b.detach().clone()))
    W, B = best[1] if best[1] is not None else (w.detach(), b.detach())

    def score(Xe):
        with torch.no_grad():
            return ((Xe - mu) / sd @ W + B).tolist()

    return score, best[0]


def stage_probe(feat_path, n_folds=5, seed=0):
    import torch

    pk = torch.load(feat_path, map_location="cpu")
    metas = pk["metas"]
    n = len(metas)
    print(f"[probe] {n} units  pos={sum(m['label'] for m in metas)} "
          f"problems={len({m['id'] for m in metas})}", flush=True)

    # Problem-level folds.
    pids = sorted({m["id"] for m in metas})
    rng = random.Random(seed)
    rng.shuffle(pids)
    folds = [pids[i::n_folds] for i in range(n_folds)]

    # Per-unit out-of-fold scores for every layer; pick best layer by OOF wp-AUROC.
    oof = {l: [None] * n for l in LAYERS}
    layer_crit = {}
    for l in LAYERS:
        X = pk["feats"][l].float()
        y = torch.tensor([m["label"] for m in metas], dtype=torch.float32)
        fold_crits = []
        for fi, hold in enumerate(folds):
            hold_set = set(hold)
            te = [i for i, m in enumerate(metas) if m["id"] in hold_set]
            # hold out 15% of train problems as early-stopping val
            tr_pids = [p for p in pids if p not in hold_set]
            rng2 = random.Random(seed + 17 * fi + l)
            rng2.shuffle(tr_pids)
            n_va = max(1, len(tr_pids) // 7)
            va_set = set(tr_pids[:n_va])
            tr = [i for i, m in enumerate(metas)
                  if m["id"] not in hold_set and m["id"] not in va_set]
            va = [i for i, m in enumerate(metas) if m["id"] in va_set]
            if len(tr) < 20 or not te:
                continue
            sc, crit = fit_probe(
                X[tr], y[tr], [metas[i]["id"] for i in tr],
                X[va], y[va], [metas[i]["id"] for i in va],
                use_rank=True, seed=seed + fi,
            )
            fold_crits.append(crit)
            scores = sc(X[te])
            for i, s in zip(te, scores):
                oof[l][i] = s
        layer_crit[l] = (sum(fold_crits) / len(fold_crits) if fold_crits
                         else float("nan"))
        print(f"  layer {l:2d}: mean val wp-AUROC={layer_crit[l]:.3f}", flush=True)

    bl = max(layer_crit, key=lambda k: layer_crit[k]
             if layer_crit[k] == layer_crit[k] else -1)
    print(f"[probe] best layer={bl}  val={layer_crit[bl]:.3f}", flush=True)
    scores = oof[bl]
    if any(s is None for s in scores):
        raise SystemExit("incomplete OOF scores")

    # Within-problem AUROC on OOF scores (the quantity selection consumes).
    wp, nwp = within_problem_auroc(
        scores, [m["label"] for m in metas], [m["id"] for m in metas]
    )
    pooled = auroc(scores, [m["label"] for m in metas])
    print(f"[probe] OOF pooled AUROC={pooled:.3f}  "
          f"within-problem AUROC={wp:.3f} (n={nwp})", flush=True)

    # Per-problem selection.
    by_p = defaultdict(list)
    for i, m in enumerate(metas):
        by_p[m["id"]].append((i, scores[i], m))

    rows_out = []
    for pid, items in by_p.items():
        items_sorted = sorted(items, key=lambda t: -t[1])
        best_i, best_s, best_m = items_sorted[0]
        # majority vote on norms (SC@8)
        vote = Counter(m["norm"] for _, _, m in items if m["norm"])
        top_norm = vote.most_common(1)[0][0] if vote else None
        gold, aliases = best_m["gold"], best_m["aliases"]
        sc8_ok = bool(top_norm and answers_match(top_norm, gold, aliases))
        # first candidate = SC@1 / greedy
        first = min(items, key=lambda t: t[2]["cand"])[2]
        sc1_ok = bool(first["label"])
        ora_ok = any(m["label"] for _, _, m in items)
        probe_ok = bool(best_m["label"])
        rows_out.append({
            "id": pid,
            "n_hops": best_m["n_hops"],
            "n_cands": len(items),
            "probe_ans": best_m["ans"],
            "probe_score": best_s,
            "probe_ok": probe_ok,
            "sc1_ok": sc1_ok,
            "sc8_ok": sc8_ok,
            "oracle_ok": ora_ok,
            "n_correct_cands": sum(m["label"] for _, _, m in items),
        })

    n_prob = len(rows_out)
    acc = {
        "sc1": sum(r["sc1_ok"] for r in rows_out) / n_prob,
        "sc8": sum(r["sc8_ok"] for r in rows_out) / n_prob,
        "probe": sum(r["probe_ok"] for r in rows_out) / n_prob,
        "oracle": sum(r["oracle_ok"] for r in rows_out) / n_prob,
    }
    d_sc8 = acc["probe"] - acc["sc8"]
    gap = acc["oracle"] - acc["sc8"]
    frac = d_sc8 / gap if gap > 0 else float("nan")

    # McNemar probe vs SC@8
    p_only = sum(1 for r in rows_out if r["probe_ok"] and not r["sc8_ok"])
    s_only = sum(1 for r in rows_out if r["sc8_ok"] and not r["probe_ok"])
    chi2, pval = mcnemar(p_only, s_only)

    print("\n== E2' latent verifier rerank ==")
    print(f"  n={n_prob}  layer={bl}  OOF wp-AUROC={wp:.3f}")
    print(f"  SC@1   = {acc['sc1']:.3f}")
    print(f"  SC@8   = {acc['sc8']:.3f}")
    print(f"  probe  = {acc['probe']:.3f}   "
          f"delta vs SC@8 = {d_sc8:+.3f}")
    print(f"  oracle = {acc['oracle']:.3f}   "
          f"gap(oracle-SC8) = {gap:+.3f}")
    print(f"  fraction of gap closed = {frac:.1%}" if frac == frac
          else "  fraction of gap closed = n/a")
    print(f"  McNemar probe vs SC@8: probe-only={p_only} sc8-only={s_only}  "
          f"chi2={chi2:.2f} p={pval:.4f}")

    # by hop count
    print("\n== by hop count ==")
    buckets = defaultdict(list)
    for r in rows_out:
        buckets[r["n_hops"]].append(r)
    by_hop = {}
    for h in sorted(buckets):
        rs = buckets[h]
        m = len(rs)
        row = {
            "n": m,
            "probe": sum(r["probe_ok"] for r in rs) / m,
            "sc8": sum(r["sc8_ok"] for r in rs) / m,
            "oracle": sum(r["oracle_ok"] for r in rs) / m,
        }
        by_hop[h] = row
        print(f"  {h}hop n={m:3d}  probe={row['probe']:.3f}  "
              f"sc8={row['sc8']:.3f}  ora={row['oracle']:.3f}  "
              f"d={row['probe']-row['sc8']:+.3f}")

    gate_pp = d_sc8 >= 0.06
    gate_p = pval < 0.05
    gate = gate_pp and gate_p
    print(f"\n  GATE (>=+6pp over SC@8 AND p<0.05): "
          f"{'PASS' if gate else 'FAIL'}  "
          f"(delta={d_sc8:+.3f} p={pval:.4f})")

    rep = {
        "n": n_prob,
        "layer": bl,
        "layer_crit": {str(k): v for k, v in layer_crit.items()},
        "oof_pooled_auroc": pooled,
        "oof_wp_auroc": wp,
        "acc": acc,
        "delta_sc8": d_sc8,
        "gap_oracle_sc8": gap,
        "frac_gap_closed": frac,
        "mcnemar": {"probe_only": p_only, "sc8_only": s_only,
                    "chi2": chi2, "p": pval},
        "by_hop": {str(k): v for k, v in by_hop.items()},
        "gate_pass": gate,
    }
    return rep, rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--sc-dir", default="",
                    help="directory containing sc.s*.jsonl; auto-discover if empty")
    ap.add_argument("--out-dir", default="mh_latent")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    a = ap.parse_args()

    data = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    OUT = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(HERE, a.out_dir)
    os.makedirs(OUT, exist_ok=True)
    feat_path = os.path.join(OUT, "hidden_feats.pt")

    rows = load_rows(data, a.limit)
    print(f"[mh_latent] {len(rows)} problems", flush=True)

    sc_paths = find_sc(a.sc_dir)
    sc_rows = []
    for p in sc_paths:
        sc_rows += jread(p)
    sc_by_id = {r["id"]: r for r in sc_rows}
    # Align to the same Random(0) subsample used by E0/ceiling.
    want = {r["_uid"] for r in rows}
    sc_by_id = {k: v for k, v in sc_by_id.items() if k in want}
    print(f"[mh_latent] SC rows matched: {len(sc_by_id)}/{len(rows)}", flush=True)
    if len(sc_by_id) < 0.8 * len(rows):
        raise SystemExit(
            f"FATAL: only {len(sc_by_id)}/{len(rows)} SC rows matched; "
            "pass --sc-dir pointing at the mh_ceil / mh_e0 workdir"
        )

    units = build_units(rows, sc_by_id)
    print(f"[mh_latent] {len(units)} units  "
          f"pos={sum(u['label'] for u in units)}", flush=True)

    if not a.probe_only:
        stage_extract(a, units, feat_path)

    rep, rows_out = stage_probe(feat_path, n_folds=a.folds)
    with open(os.path.join(OUT, "selections.jsonl"), "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "mh_latent_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(OUT, "mh_latent_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"saved mh_latent_report.json  gate={'PASS' if rep['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
