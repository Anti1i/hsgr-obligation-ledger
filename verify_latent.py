"""Verification suite for the latent semantic-interface potential (CPU only).

Runs three experiments on the cached hidden features from phase06:

  V1  probe architecture: best-single-layer vs naive-concat vs two-stage
      stacking, each with/without a within-problem pairwise ranking loss.
      Goal: fix the phase-0.6b negative result (naive concat < best layer).

  V2  cross-domain generalization: train GSM-only -> eval MATH, train
      MATH-only -> eval GSM, compared with joint training.

  V3  "right answer, wrong reasoning": AUROC of fully-correct assignments vs
      `wrong_value` synthetic negatives, which keep the SAME final answer but
      corrupt one intermediate result. A final-answer-only judge scores 0.5 by
      construction, so >0.5 is evidence the hidden state encodes reasoning
      validity rather than just answer correctness.

Usage: python verify_latent.py --stage {v1,v2,v3,all}
"""
import argparse
import bisect
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [7, 14, 21, 28]
TRAIN_DIRS = ["outputs_gsm_train", "outputs_math_train"]
EVAL_DIRS = ["outputs", "outputs_gsm_test"]


# ---------------------------------------------------------------- metrics
def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
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


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, dirs):
        import torch

        self.feats = {l: [] for l in LAYERS}
        self.meta = []
        for d in dirs:
            pk = torch.load(os.path.join(HERE, d, "hidden_feats.pt"))
            for l in LAYERS:
                self.feats[l].append(pk["feats"][l].float())
            for m in pk["metas"]:
                self.meta.append({**m, "dir": d, "key": f"{d}:{m['id']}"})
        for l in LAYERS:
            self.feats[l] = torch.cat(self.feats[l])
        self.y = torch.tensor([m["label"] for m in self.meta], dtype=torch.float32)

    def split_by_problem(self, frac, seed=42, subset=None):
        pool = range(len(self.meta)) if subset is None else subset
        pool = list(pool)
        keys = sorted({self.meta[i]["key"] for i in pool})
        random.Random(seed).shuffle(keys)
        hold = set(keys[: max(1, int(len(keys) * frac))])
        a = [i for i in pool if self.meta[i]["key"] not in hold]
        b = [i for i in pool if self.meta[i]["key"] in hold]
        return a, b


# ---------------------------------------------------------------- probe
def fit_probe(X, y, pids, Xv, yv, pv, use_rank=False, epochs=600, lr=5e-3,
              wd=3e-3, eval_every=25, rank_w=0.5, seed=0, vnat=None):
    """Logistic probe, optional within-problem pairwise ranking term.

    Early stopping on validation within-problem AUROC (what DCH selection
    actually consumes), falling back to pooled AUROC when too few problems.
    `vnat` restricts the criterion to natural units: synthetic negatives are
    far easier and otherwise dominate model selection.
    Returns a scorer that maps a design matrix to logits.
    """
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
        if ep % eval_every == 0 or ep == epochs:
            with torch.no_grad():
                sv = (Xvn @ w + b).tolist()
            yl = yv.tolist()
            if vnat is not None:
                sel = [i for i, ok in enumerate(vnat) if ok]
                sv_c = [sv[i] for i in sel]
                yl_c = [yl[i] for i in sel]
                pv_c = [pv[i] for i in sel]
            else:
                sv_c, yl_c, pv_c = sv, yl, pv
            crit, n = within_problem_auroc(sv_c, yl_c, pv_c)
            if crit != crit or n < 5:
                crit = auroc(sv_c, yl_c)
            if crit == crit and crit > best[0]:
                best = (crit, (w.detach().clone(), b.detach().clone()))
    W, B = best[1] if best[1] is not None else (w.detach(), b.detach())

    def score(Xe):
        with torch.no_grad():
            return ((Xe - mu) / sd @ W + B).tolist()

    return score, best[0]


# ------------------------------------------------- configs (featurizer + probe)
def featurizer_single(layer):
    return lambda feats, idx: feats[layer][idx]


def featurizer_concat():
    import torch

    return lambda feats, idx: torch.cat([feats[l][idx] for l in LAYERS], dim=1)


def make_stacking(data, inner_tr, inner_va, use_rank=True):
    """Stage 1: per-layer probes on inner-train. Stage 2 fitted by caller."""
    import torch

    itr, iva = torch.tensor(inner_tr), torch.tensor(inner_va)
    yi, yiv = data.y[itr], data.y[iva]
    pi = [data.meta[i]["key"] for i in inner_tr]
    piv = [data.meta[i]["key"] for i in inner_va]
    vnat = [data.meta[i]["kind"] == "nat" for i in inner_va]
    base = {}
    for l in LAYERS:
        Xl = data.feats[l]
        base[l] = fit_probe(Xl[itr], yi, pi, Xl[iva], yiv, piv,
                            use_rank=use_rank, vnat=vnat)[0]

    def feat(feats, idx):
        cols = [torch.tensor(base[l](feats[l][idx])).unsqueeze(1) for l in LAYERS]
        return torch.cat(cols, dim=1)

    return feat


def train_config(data, featurize, tr, va, use_rank, **kw):
    import torch

    tr_t, va_t = torch.tensor(tr), torch.tensor(va)
    X, Xv = featurize(data.feats, tr_t), featurize(data.feats, va_t)
    ptr = [data.meta[i]["key"] for i in tr]
    pva = [data.meta[i]["key"] for i in va]
    vnat = [data.meta[i]["kind"] == "nat" for i in va]
    return fit_probe(X, data.y[tr_t], ptr, Xv, data.y[va_t], pva,
                     use_rank=use_rank, vnat=vnat, **kw)


def eval_on(data, featurize, score, nat_only=True, subset=None):
    import torch

    keep = [i for i in (subset if subset is not None else range(len(data.meta)))
            if (not nat_only or data.meta[i]["kind"] == "nat")]
    if not keep:
        return {"n": 0, "auroc": float("nan"), "wp_auroc": float("nan")}
    idx = torch.tensor(keep)
    sc = score(featurize(data.feats, idx))
    lab = [int(data.y[i].item()) for i in keep]
    pid = [data.meta[i]["id"] for i in keep]
    wp, nwp = within_problem_auroc(sc, lab, pid)
    return {"n": len(keep), "auroc": auroc(sc, lab), "wp_auroc": wp,
            "n_problems": nwp}


# ---------------------------------------------------------------- V1
def stage_v1():
    data = Data(TRAIN_DIRS)
    tr, va = data.split_by_problem(0.15, seed=42)
    n_nat = sum(1 for m in data.meta if m["kind"] == "nat")
    print(f"[V1] rows={len(data.meta)} (nat={n_nat})  train={len(tr)} val={len(va)}")
    evals = {d: Data([d]) for d in EVAL_DIRS}

    configs = {}
    # single layer: pick best layer by val criterion, per loss variant
    for rank in (False, True):
        cand = {}
        for l in LAYERS:
            f = featurizer_single(l)
            sc, crit = train_config(data, f, tr, va, rank)
            cand[l] = (f, sc, crit)
        bl = max(cand, key=lambda l: cand[l][2])
        configs[f"single_best{'_rank' if rank else ''}"] = (*cand[bl], {"layer": bl})
    # naive concat
    for rank in (False, True):
        f = featurizer_concat()
        sc, crit = train_config(data, f, tr, va, rank)
        configs[f"concat{'_rank' if rank else ''}"] = (f, sc, crit, {})
    # two-stage stacking
    inner_tr, inner_va = data.split_by_problem(0.25, seed=7, subset=tr)
    if len(inner_va) > 20:
        f = make_stacking(data, inner_tr, inner_va)
        sc, crit = train_config(data, f, inner_va, va, True,
                                epochs=800, lr=2e-2, wd=1e-4)
        configs["stacking_rank"] = (f, sc, crit, {})

    report = {}
    for name, (f, sc, crit, extra) in configs.items():
        row = {"val_crit": crit, **extra}
        for d, ed in evals.items():
            row[d] = eval_on(ed, f, sc)
        report[name] = row
        print(f"[V1] {name:18s} val={crit:.3f}  " + "  ".join(
            f"{d}: auroc {row[d]['auroc']:.3f} wp {row[d]['wp_auroc']:.3f}"
            for d in EVAL_DIRS))
    with open(os.path.join(HERE, "verify_v1_report.json"), "w") as fo:
        json.dump(report, fo, indent=1)
    print("saved verify_v1_report.json")
    return report


# ---------------------------------------------------------------- V2
def stage_v2():
    evals = {d: Data([d]) for d in EVAL_DIRS}
    setups = {
        "joint": TRAIN_DIRS,
        "gsm_only": ["outputs_gsm_train"],
        "math_only": ["outputs_math_train"],
    }
    report = {}
    for name, dirs in setups.items():
        data = Data(dirs)
        tr, va = data.split_by_problem(0.15, seed=42)
        cand = {}
        for l in LAYERS:
            f = featurizer_single(l)
            sc, crit = train_config(data, f, tr, va, True)
            cand[l] = (f, sc, crit)
        bl = max(cand, key=lambda l: cand[l][2])
        f, sc, crit = cand[bl]
        row = {"val_crit": crit, "layer": bl, "n_rows": len(data.meta)}
        for d, ed in evals.items():
            row[d] = eval_on(ed, f, sc)
        report[name] = row
        print(f"[V2] {name:10s} layer={bl} val={crit:.3f}  " + "  ".join(
            f"{d}: auroc {row[d]['auroc']:.3f} wp {row[d]['wp_auroc']:.3f}"
            for d in EVAL_DIRS))
    with open(os.path.join(HERE, "verify_v2_report.json"), "w") as fo:
        json.dump(report, fo, indent=1)
    print("saved verify_v2_report.json")
    return report


# ---------------------------------------------------------------- V3
def synthetic_kinds(out_dir):
    """Replay phase06.build_units synthetic ordering to recover negative types."""
    from phase06_hidden_probe import MAX_SYN_PER_PROBLEM, jread_glob

    dec_ids = {r["id"] for r in
               jread_glob(os.path.join(HERE, out_dir, "decompose.s*.jsonl"))}
    syn_by_p = defaultdict(list)
    for r in jread_glob(os.path.join(HERE, out_dir, "synthetic_negatives.jsonl")):
        syn_by_p[r["id"]].append(r)
    kinds = []
    for pid, rows in syn_by_p.items():
        if pid not in dec_ids:
            continue
        rng = random.Random(pid)
        rng.shuffle(rows)
        for r in rows[:MAX_SYN_PER_PROBLEM]:
            kinds.append((pid, r["kind"]))
    return kinds


def stage_v3():
    import torch

    data = Data(TRAIN_DIRS)
    # attach synthetic subtypes: synthetic units follow all natural units per dir
    for d in TRAIN_DIRS:
        idx = [i for i, m in enumerate(data.meta) if m["dir"] == d and m["kind"] == "syn"]
        kinds = synthetic_kinds(d)
        if len(idx) != len(kinds):
            print(f"[V3] WARNING {d}: {len(idx)} syn feats vs {len(kinds)} replayed")
        for i, (pid, k) in zip(idx, kinds):
            if data.meta[i]["id"] != pid:
                print(f"[V3] WARNING {d}: id mismatch {data.meta[i]['id']} != {pid}")
            data.meta[i]["syn_kind"] = k
    counts = defaultdict(int)
    for m in data.meta:
        if m["kind"] == "syn":
            counts[m.get("syn_kind", "?")] += 1
    print(f"[V3] synthetic subtypes: {dict(counts)}")

    tr, te = data.split_by_problem(0.30, seed=11)
    cand = {}
    tr_in, va_in = data.split_by_problem(0.15, seed=5, subset=tr)
    for l in LAYERS:
        f = featurizer_single(l)
        sc, crit = train_config(data, f, tr_in, va_in, True)
        cand[l] = (f, sc, crit)
    bl = max(cand, key=lambda l: cand[l][2])
    f, sc, crit = cand[bl]
    print(f"[V3] probe layer={bl} val={crit:.3f} (train problems={len(set(data.meta[i]['key'] for i in tr))})")

    te_set = set(te)
    pos = [i for i in te_set if data.meta[i]["kind"] == "nat" and data.y[i] == 1]
    report = {"layer": bl, "val_crit": crit, "n_pos": len(pos), "results": {}}
    pos_sc = sc(f(data.feats, torch.tensor(pos)))
    for k in ("wrong_value", "wrong_final", "sibling_swap"):
        negs = [i for i in te_set if data.meta[i].get("syn_kind") == k]
        if not negs:
            continue
        neg_sc = sc(f(data.feats, torch.tensor(negs)))
        a = auroc(pos_sc + neg_sc, [1] * len(pos_sc) + [0] * len(neg_sc))
        # within-problem version: same problem, correct assignment vs its corruption
        pid = [data.meta[i]["key"] for i in pos] + [data.meta[i]["key"] for i in negs]
        wp, nwp = within_problem_auroc(pos_sc + neg_sc,
                                      [1] * len(pos_sc) + [0] * len(neg_sc), pid)
        report["results"][k] = {"n_neg": len(negs), "auroc": a, "wp_auroc": wp,
                                "n_problems": nwp}
        print(f"[V3] positives vs {k:13s}: auroc={a:.3f} wp={wp:.3f} (n_neg={len(negs)})")
    # natural negatives for reference (wrong final answer, real model errors)
    nat_neg = [i for i in te_set if data.meta[i]["kind"] == "nat" and data.y[i] == 0]
    if nat_neg:
        ns = sc(f(data.feats, torch.tensor(nat_neg)))
        report["results"]["natural_negative"] = {
            "n_neg": len(nat_neg),
            "auroc": auroc(pos_sc + ns, [1] * len(pos_sc) + [0] * len(ns)),
        }
        print(f"[V3] positives vs natural_neg : "
              f"auroc={report['results']['natural_negative']['auroc']:.3f} "
              f"(n_neg={len(nat_neg)})")
    with open(os.path.join(HERE, "verify_v3_report.json"), "w") as fo:
        json.dump(report, fo, indent=1)
    print("saved verify_v3_report.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if a.stage in ("v1", "all"):
        stage_v1()
    if a.stage in ("v2", "all"):
        stage_v2()
    if a.stage in ("v3", "all"):
        stage_v3()
