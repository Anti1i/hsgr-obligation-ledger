"""V5/V6/V7: structural verification of DCH components (zero GPU).

  V6  bidirectional message passing. The proposal claims top-down constraint
      feedback matters, but the pilot only ran one bottom-up pass. Here we run
      an explicit loopy-BP-style loop over the enumerated assignment space:
        w(a)   = prod_v q_v(a_v) * psi(a)^beta
        P(y)   ∝ rho(y)^gamma * sum_{a: final(a)=y} w(a)
        q_v(k) ← (1-eta) q_v(k) + eta * normalized top-down mass
      and compare round 1 (bottom-up only) against rounds 2-4.

  V5  node-level conformal candidate domains: calibrate a per-node frequency
      threshold so that "useful" value classes (those participating in at least
      one root-correct assignment) are retained at 1-alpha, and compare the
      resulting budget (average K) against the fixed K=1/2/3 policy.

  V7  budget accounting: generation calls and upper-bound generated tokens per
      problem for CoT / SC@k / DCH variants, to define the budget-matched
      comparison that the main experiments must run.

Usage: python verify_structure.py --dirs outputs,outputs_gsm_test --stage all
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402
from phase0_reward_audit import jread_glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-12

# generation caps from pilot.py, used for the token upper bound
CAP_DECOMPOSE, CAP_SUB, CAP_ROOT, CAP_AGG, CAP_VERIFY = 300, 400, 768, 400, 260
N_SUB_SAMPLED, N_ROOT_SAMPLED = 3, 4


def load(out_dir):
    d = os.path.join(HERE, out_dir)
    dec = {r["id"]: r for r in jread_glob(d, "decompose.s*.jsonl")}
    subc = defaultdict(dict)
    for r in jread_glob(d, "subcands.s*.jsonl"):
        subc[r["id"]][r["sub_idx"]] = r
    aggs = defaultdict(list)
    for r in jread_glob(d, "aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)
    roots = {r["id"]: r for r in jread_glob(d, "rootcands.s*.jsonl")}
    scorers = {}
    for name, pat, key in [("probe", "probe.s*.jsonl", "probe"),
                           ("trained", "trained.s*.jsonl", "trained"),
                           ("verify", "verify.s*.jsonl", "verify")]:
        m = {(r["id"], r["assign_idx"]): r[key] for r in jread_glob(d, pat) if key in r}
        if m:
            scorers[name] = m
    ids = sorted(i for i in dec if dec[i].get("subquestions") and i in subc
                 and i in aggs and len(subc[i]) == len(dec[i]["subquestions"]))
    return dec, subc, aggs, roots, scorers, ids


def node_domains(subc_p, n_sub, max_k=3):
    """Value classes per node with frequency, greedy value first (mirrors pilot)."""
    doms = []
    for si in range(n_sub):
        cands = subc_p[si]["cands"]
        g = next((c for c in cands if c["kind"] == "greedy" and c["ans"]), None)
        cnt = Counter(c["norm"] for c in cands if c["norm"] is not None)
        keys = sorted(cnt, key=lambda k: -cnt[k])
        gn = g["norm"] if g else None
        if gn in cnt:
            keys = [gn] + [k for k in keys if k != gn]
        tot = max(1, sum(cnt.values()))
        doms.append([{"norm": k, "freq": cnt[k] / tot} for k in keys[:max_k]])
    return doms


# ------------------------------------------------------------------ V6
def bp_predict(pid, dec, subc, aggs, roots, score_map, rounds, beta=1.0,
               gamma=1.0, eta=0.5):
    """Loopy BP over the enumerated assignment space. Returns per-round answers."""
    subs = dec[pid]["subquestions"]
    doms = node_domains(subc[pid], len(subs))
    rows = [r for r in aggs[pid] if r.get("final_norm") is not None]
    if not rows:
        return [], {"argmax_changed": False, "mean_belief_drift": 0.0}
    q = [{c["norm"]: max(c["freq"], 1e-3) for c in dom} for dom in doms]
    for qi in q:
        z = sum(qi.values())
        for k in qi:
            qi[k] /= z

    rc = roots.get(pid, {}).get("cands", [])
    rfreq = Counter(c["norm"] for c in rc if c["norm"] is not None)
    rtot = max(1, sum(rfreq.values()))
    rho = {v: c / rtot for v, c in rfreq.items()}

    preds, argmaxes, drifts = [], [], []
    for _ in range(rounds):
        w = []
        for r in rows:
            p = 1.0
            for si, k in enumerate(r["sub_norms"]):
                p *= q[si].get(k, 1e-4)
            if score_map is not None:
                s = score_map.get((pid, r["assign_idx"]))
                if s is not None:
                    p *= max(_norm_score(s), 1e-4) ** beta
            w.append(p)
        py = defaultdict(float)
        for r, wi in zip(rows, w):
            py[r["final_norm"]] += wi
        for v in list(py):
            py[v] *= (rho.get(v, 1e-3) ** gamma)
        if not py or max(py.values()) <= 0:
            preds.append(None)
            break
        best = max(py, key=py.get)
        argmaxes.append(best)
        preds.append(next(r["final_ans"] for r in rows if r["final_norm"] == best))

        # top-down: reweight node priors by posterior mass of consistent roots
        zt = sum(py.values()) or 1.0
        pyn = {v: m / zt for v, m in py.items()}
        drift = 0.0
        for si in range(len(q)):
            mass = defaultdict(float)
            for r, wi in zip(rows, w):
                mass[r["sub_norms"][si]] += wi * pyn.get(r["final_norm"], 0.0)
            tot = sum(mass.values())
            if tot <= EPS:
                continue
            before = dict(q[si])
            for k in q[si]:
                q[si][k] = (1 - eta) * q[si][k] + eta * (mass.get(k, 0.0) / tot)
            z = sum(q[si].values()) or 1.0
            for k in q[si]:
                q[si][k] /= z
            drift += sum(abs(q[si][k] - before[k]) for k in q[si])
        drifts.append(drift / max(1, len(q)))
    return preds, {"argmax_changed": len(set(argmaxes)) > 1,
                   "mean_belief_drift": sum(drifts) / max(1, len(drifts))}


def _norm_score(s):
    """Map heterogeneous scorer outputs to (0,1]."""
    if isinstance(s, (int, float)):
        return s / 10.0 if s > 1.0 else float(s)
    return 0.5


def stage_v6(out_dir, rounds=4):
    dec, subc, aggs, roots, scorers, ids = load(out_dir)
    variants = {"freq_only": None}
    for name in ("probe", "trained", "verify"):
        if name in scorers:
            variants[name] = scorers[name]
    print(f"== V6 {out_dir}: {len(ids)} problems, scorers={list(variants)} ==")
    rep = {}
    for name, sm in variants.items():
        correct = [0] * rounds
        n, n_argmax_changed, drift_sum = 0, 0, 0.0
        flips = Counter()
        for pid in ids:
            preds, diag = bp_predict(pid, dec, subc, aggs, roots, sm, rounds)
            if not preds:
                continue
            n += 1
            n_argmax_changed += int(diag["argmax_changed"])
            drift_sum += diag["mean_belief_drift"]
            gold = dec[pid]["gold"]
            oks = [bool(p and answers_equal(p, gold)) for p in preds]
            oks += [oks[-1]] * (rounds - len(oks))
            for t in range(rounds):
                correct[t] += oks[t]
            if oks[0] != oks[-1]:
                flips["fixed" if oks[-1] else "broke"] += 1
        accs = [c / max(1, n) for c in correct]
        rep[name] = {"n": n, "acc_by_round": accs, "flips": dict(flips),
                     "problems_where_root_argmax_changed": n_argmax_changed,
                     "mean_node_belief_drift_per_round": drift_sum / max(1, n)}
        print(f"  {name:10s} n={n}  " +
              "  ".join(f"r{t+1}={a:.3f}" for t, a in enumerate(accs)) +
              f"   root-argmax changed in {n_argmax_changed}/{n} problems, "
              f"mean node-belief drift/round={drift_sum/max(1,n):.3f}")
    return rep


# ------------------------------------------------------------------ V5
def stage_v5(out_dir, alphas=(0.1, 0.2), seed=3):
    dec, subc, aggs, roots, scorers, ids = load(out_dir)
    nodes = []
    for pid in ids:
        rows = aggs[pid]
        gold = dec[pid]["gold"]
        good = [r for r in rows if r.get("final_ans") and answers_equal(r["final_ans"], gold)]
        if not good:
            continue
        doms = node_domains(subc[pid], len(dec[pid]["subquestions"]))
        for si, dom in enumerate(doms):
            useful = {r["sub_norms"][si] for r in good}
            for rank, c in enumerate(dom):
                nodes.append({"id": pid, "sub_idx": si, "rank": rank,
                              "freq": c["freq"], "useful": c["norm"] in useful,
                              "n_classes": len(dom)})
    keys = sorted({n["id"] for n in nodes})
    random.Random(seed).shuffle(keys)
    cal_ids = set(keys[: len(keys) // 2])
    cal = [n for n in nodes if n["id"] in cal_ids]
    test = [n for n in nodes if n["id"] not in cal_ids]

    def group(rows):
        g = defaultdict(list)
        for n in rows:
            g[(n["id"], n["sub_idx"])].append(n)
        return g

    gt = group(test)
    rep = {"n_nodes_test": len(gt), "policies": {}}
    for K in (1, 2, 3):
        cov = sum(any(c["useful"] for c in sorted(cs, key=lambda x: x["rank"])[:K])
                  for cs in gt.values())
        size = sum(min(K, len(cs)) for cs in gt.values())
        rep["policies"][f"fixed_K{K}"] = {"coverage": cov / max(1, len(gt)),
                                          "avg_K": size / max(1, len(gt))}
    # conformal: nonconformity = 1 - freq of the useful class; keep all classes
    # whose freq >= 1 - qhat (always keeping the top-1 class)
    scores = sorted(1.0 - max((c["freq"] for c in cs if c["useful"]), default=1.0)
                    for cs in group(cal).values())
    m = len(scores)
    for alpha in alphas:
        k = min(m - 1, max(0, int(-(-((m + 1) * (1 - alpha)) // 1)) - 1))
        qhat = scores[k] if m else 1.0
        thr = 1.0 - qhat
        cov, size = 0, 0
        for cs in gt.values():
            ordered = sorted(cs, key=lambda x: x["rank"])
            keep = [c for c in ordered if c["freq"] >= thr] or ordered[:1]
            cov += any(c["useful"] for c in keep)
            size += len(keep)
        rep["policies"][f"conformal_a{alpha}"] = {
            "coverage": cov / max(1, len(gt)), "avg_K": size / max(1, len(gt)),
            "freq_threshold": thr}
    print(f"== V5 {out_dir}: nodes(test)={len(gt)} ==")
    for name, v in rep["policies"].items():
        print(f"  {name:16s} coverage={v['coverage']:.3f}  avg_K={v['avg_K']:.2f}")
    return rep


# ------------------------------------------------------------------ V7
def stage_v7(out_dir):
    dec, subc, aggs, roots, scorers, ids = load(out_dir)
    per = []
    for pid in ids:
        n_sub = len(dec[pid]["subquestions"])
        n_assign = len([r for r in aggs[pid] if r.get("final_ans")])
        sub_calls = n_sub * (1 + N_SUB_SAMPLED)
        dch_calls = 1 + sub_calls + n_assign
        dch_tok = (CAP_DECOMPOSE + sub_calls * CAP_SUB + n_assign * CAP_AGG)
        per.append({
            "cot_calls": 1, "cot_tok": CAP_ROOT,
            "sc5_calls": 5, "sc5_tok": 5 * CAP_ROOT,
            "dch_calls": dch_calls, "dch_tok": dch_tok,
            "dch_verify_calls": dch_calls + 4 * n_assign,
            "dch_verify_tok": dch_tok + 4 * n_assign * CAP_VERIFY,
            "dch_probe_calls": dch_calls, "dch_probe_tok": dch_tok,  # probe = 1 fwd, no gen
        })
    n = max(1, len(per))
    avg = {k: sum(p[k] for p in per) / n for k in per[0]}
    rep = {"n_problems": len(per), "avg": avg,
           "sc_equivalent_k_by_tokens": avg["dch_tok"] / CAP_ROOT,
           "sc_equivalent_k_by_tokens_verify": avg["dch_verify_tok"] / CAP_ROOT}
    print(f"== V7 {out_dir}: {len(per)} problems (upper-bound token caps) ==")
    print(f"  CoT       calls  1.0   tokens {avg['cot_tok']:.0f}")
    print(f"  SC@5      calls  5.0   tokens {avg['sc5_tok']:.0f}")
    print(f"  DCH+probe calls {avg['dch_probe_calls']:.1f}   tokens {avg['dch_probe_tok']:.0f}"
          f"  -> token-matched SC@k needs k≈{rep['sc_equivalent_k_by_tokens']:.1f}")
    print(f"  DCH+verify calls {avg['dch_verify_calls']:.1f}  tokens {avg['dch_verify_tok']:.0f}"
          f"  -> token-matched SC@k needs k≈{rep['sc_equivalent_k_by_tokens_verify']:.1f}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="outputs,outputs_gsm_test")
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = defaultdict(dict)
    for d in a.dirs.split(","):
        d = d.strip()
        if a.stage in ("v6", "all"):
            out[d]["v6"] = stage_v6(d)
        if a.stage in ("v5", "all"):
            out[d]["v5"] = stage_v5(d)
        if a.stage in ("v7", "all"):
            out[d]["v7"] = stage_v7(d)
    with open(os.path.join(HERE, "verify_structure_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved verify_structure_report.json")
