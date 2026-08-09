"""Merge shard outputs and compute the pilot metrics.

Answers the two case-study questions:
  Q1  How often is the correct answer recoverable from top-K candidate
      combinations when the hard-commit pipeline (and greedy CoT) fail?
  Q2  Can a prompted compatibility scorer + factorized selection (DCH-lite)
      actually recover it, compared to hard-commit / self-consistency?
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, normalize_answer  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", default="outputs")
_args, _ = _ap.parse_known_args()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), _args.out_dir)

# selection weights for DCH-lite (root evidence, sub-candidate evidence, compat)
W_ROOT, W_SUB, W_COMPAT = 0.4, 0.2, 0.4


def jread_glob(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, pattern))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    dec = {r["id"]: r for r in jread_glob("decompose.s*.jsonl")}
    roots = {r["id"]: r for r in jread_glob("rootcands.s*.jsonl")}
    aggs = defaultdict(list)
    for r in jread_glob("aggregate.s*.jsonl"):
        aggs[r["id"]].append(r)
    compat = {(r["id"], r["assign_idx"]): r["compat"] for r in jread_glob("compat.s*.jsonl")}
    verify = {(r["id"], r["assign_idx"]): r["verify"] for r in jread_glob("verify.s*.jsonl")}
    trained = {(r["id"], r["assign_idx"]): r["trained"] for r in jread_glob("trained.s*.jsonl")}
    probe = {(r["id"], r["assign_idx"]): r["probe"] for r in jread_glob("probe.s*.jsonl")}

    ids = sorted(set(dec) & set(roots))
    n_all = len(ids)
    usable = {i for i in ids if dec[i]["subquestions"] and aggs.get(i)}

    M = Counter()
    recovery_pool, scorer_labels = [], []
    per_problem = []

    for pid in ids:
        gold = dec[pid]["gold"]
        rcands = roots[pid]["cands"]
        r_greedy = next((c["ans"] for c in rcands if c["kind"] == "greedy"), None)
        cot_ok = answers_equal(r_greedy, gold) if r_greedy else False
        M["cot_greedy"] += cot_ok

        # self-consistency over root candidates (greedy + samples)
        cnt = Counter(c["norm"] for c in rcands if c["norm"] is not None)
        sc_ans = None
        if cnt:
            top = max(cnt.values())
            tied = [k for k, v in cnt.items() if v == top]
            gnorm = normalize_answer(r_greedy) if r_greedy else None
            sc_ans = gnorm if gnorm in tied else tied[0]
        sc_ok = answers_equal(sc_ans, gold) if sc_ans else False
        M["self_consistency"] += sc_ok

        if pid not in usable:
            per_problem.append({"id": pid, "usable": False, "cot": cot_ok, "sc": sc_ok})
            continue

        rows = aggs[pid]
        hc = next((r for r in rows if r["is_hardcommit"]), rows[0])
        hc_ok = answers_equal(hc["final_ans"], gold) if hc["final_ans"] else False
        M["hard_commit"] += hc_ok

        # oracle over enumerated candidate combinations
        agg_correct = [r for r in rows if r["final_ans"] and answers_equal(r["final_ans"], gold)]
        oracle_agg = bool(agg_correct)
        M["oracle_agg"] += oracle_agg
        root_vals_ok = any(c["ans"] and answers_equal(c["ans"], gold) for c in rcands)
        M["oracle_union"] += (oracle_agg or root_vals_ok)

        # scorer labels for AUROC (assignments with a compat score)
        for r in rows:
            s = compat.get((pid, r["assign_idx"]))
            if s is not None and r["final_ans"]:
                scorer_labels.append((s, 1 if answers_equal(r["final_ans"], gold) else 0))

        M["sc_usable"] += sc_ok
        M["cot_usable"] += cot_ok

        # ---- DCH-lite selection over the union value domain ----
        def select(w_root, w_sub, w_score, score_map, default_s=0.5):
            root_freq = Counter(c["norm"] for c in rcands if c["norm"] is not None)
            nroot = max(1, sum(root_freq.values()))
            belief = {}
            for v in root_freq:
                belief[v] = {"root": root_freq[v] / nroot, "assign": 0.0}
            for r in rows:
                v = r["final_norm"]
                if v is None:
                    continue
                s = score_map.get((pid, r["assign_idx"]))
                s = default_s if s is None else s
                sub_ev = sum(r["sub_freqs"]) / max(1, len(r["sub_freqs"]))
                a_score = w_sub * sub_ev + w_score * s
                b = belief.setdefault(v, {"root": 0.0, "assign": 0.0})
                b["assign"] = max(b["assign"], a_score)
            best_v, best_s = None, -1
            for v, b in belief.items():
                s = w_root * b["root"] + b["assign"]
                if s > best_s:
                    best_v, best_s = v, s
            return best_v

        compat01 = {k: (v / 10.0 if v is not None else None) for k, v in compat.items()}
        variants = {
            "dch_lite": select(W_ROOT, W_SUB, W_COMPAT, compat01),
            "dch_freq_only": select(0.5, 0.5, 0.0, {}),
            "dch_compat_only": select(0.0, 0.0, 1.0, compat01),
            "dch_verify": select(W_ROOT, W_SUB, W_COMPAT, verify),
            "dch_verify_only": select(0.0, 0.0, 1.0, verify),
        }
        if trained:
            variants["dch_trained"] = select(W_ROOT, W_SUB, W_COMPAT, trained)
            variants["dch_trained_only"] = select(0.0, 0.0, 1.0, trained)
        if probe:
            variants["dch_probe"] = select(W_ROOT, W_SUB, W_COMPAT, probe)
            variants["dch_probe_only"] = select(0.0, 0.0, 1.0, probe)
        vok = {}
        for name, v in variants.items():
            vok[name] = answers_equal(v, gold) if v else False
            M[name] += vok[name]
        dch_ok = vok["dch_lite"]

        if not hc_ok and oracle_agg:
            recovery_pool.append(
                (pid, dch_ok, sc_ok, vok["dch_verify"], vok["dch_freq_only"],
                 vok.get("dch_trained", False))
            )
        per_problem.append(
            {"id": pid, "usable": True, "cot": cot_ok, "sc": sc_ok, "hc": hc_ok,
             "oracle_agg": oracle_agg, "dch": dch_ok, "dch_verify": vok["dch_verify"],
             "n_assign": len(rows)}
        )

    n_use = len(usable)
    print(f"problems total={n_all}  decomposable+aggregated={n_use}")
    print(f"\n== accuracy (over all {n_all}) ==")
    print(f"  CoT greedy          : {M['cot_greedy']/n_all:.3f}")
    print(f"  Self-consistency@5  : {M['self_consistency']/n_all:.3f}")
    print(f"\n== accuracy (over usable {n_use}) ==")
    keys = ["cot_usable", "sc_usable", "hard_commit", "dch_freq_only", "dch_lite",
            "dch_verify", "dch_compat_only", "dch_verify_only"]
    if trained:
        keys += ["dch_trained", "dch_trained_only"]
    if probe:
        keys += ["dch_probe", "dch_probe_only"]
    keys += ["oracle_agg", "oracle_union"]
    for k in keys:
        print(f"  {k:<20}: {M[k]/n_use:.3f}")
    hc_wrong_recoverable = len(recovery_pool)
    print(f"\n== Q1: top-K oracle gap ==")
    print(f"  hard-commit wrong & correct answer in enumerated combinations: "
          f"{hc_wrong_recoverable}/{n_use} ({hc_wrong_recoverable/max(1,n_use):.3f})")
    if recovery_pool:
        n = len(recovery_pool)
        print(f"\n== Q2: Recovery@K on that pool (n={n}) ==")
        pairs = [("DCH-lite (compat)", 1), ("self-consistency", 2),
                 ("DCH-verify", 3), ("DCH-freq-only", 4)]
        if trained:
            pairs.append(("DCH-trained", 5))
        for label, idx in pairs:
            rec = sum(1 for t in recovery_pool if t[idx])
            print(f"  {label:<22}: {rec}/{n} ({rec/n:.3f})")

    # scorer AUROC
    def auroc_of(score_map, name):
        labels = []
        for pid in usable:
            gold = dec[pid]["gold"]
            for r in aggs[pid]:
                s = score_map.get((pid, r["assign_idx"]))
                if s is not None and r["final_ans"]:
                    labels.append((s, 1 if answers_equal(r["final_ans"], gold) else 0))
        pos = [s for s, y in labels if y == 1]
        neg = [s for s, y in labels if y == 0]
        if pos and neg:
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            print(f"  {name:<14} AUROC: {wins/(len(pos)*len(neg)):.3f}  "
                  f"(n={len(labels)}, pos mean {sum(pos)/len(pos):.2f}, neg mean {sum(neg)/len(neg):.2f})")

    print(f"\n== scorers ==")
    auroc_of(compat, "compat-0-10")
    auroc_of(verify, "verify-vote")
    auroc_of(trained, "trained-0.5B")
    auroc_of(probe, "probe-7B-latent")

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump({"metrics": dict(M), "n_all": n_all, "n_usable": n_use,
                   "recovery_pool": recovery_pool, "per_problem": per_problem}, f, indent=1)
    print(f"\nsaved outputs/report.json")


if __name__ == "__main__":
    main()
