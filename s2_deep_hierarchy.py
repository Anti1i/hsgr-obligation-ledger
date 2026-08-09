"""S2: depth-3 hierarchy pipeline + bidirectional message-passing retest.

Why: V6 showed top-down feedback is a no-op on the depth-2 star hierarchy. With
a deterministic aggregator and no shared parent constraints, the top-down update
only re-amplifies whichever assignment already won, so the root argmax cannot
move (accuracy identical across 4 rounds, root argmax changed in <=4/136
problems). The claim needs an intermediate layer that has BOTH a parent and
children.

Structure built here (per problem):
    root  ->  L1 nodes (2-3 subquestions)  ->  L2 leaves (2-3 sub-subquestions)
L2 leaf values come from sampling. L1 node values are DERIVED by aggregating a
choice of its children's values, so each L1 value carries a provenance set of
child assignments. That provenance is exactly the factor that lets top-down
information reach the leaves.

Stages (resumable):
  tree  : recursive decomposition to depth 3
  leaf  : candidate values for every L2 leaf
  mid   : aggregate child assignments -> L1 node value classes (with provenance)
  root  : aggregate L1 value assignments -> final answer
  bp    : 3-level belief propagation, round 1 (bottom-up) vs rounds 2-4

Usage:
  python s2_deep_hierarchy.py --stage tree --data data/gsm_deep_test.jsonl \
      --out-dir outputs_deep --limit 150
  python s2_deep_hierarchy.py --stage leaf --out-dir outputs_deep
  python s2_deep_hierarchy.py --stage mid  --out-dir outputs_deep
  python s2_deep_hierarchy.py --stage root --out-dir outputs_deep
  python s2_deep_hierarchy.py --stage bp   --out-dir outputs_deep
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal, extract_boxed, normalize_answer  # noqa: E402
from pilot import JWriter, Runner, facts_str, jread, load_problems, parse_decompose  # noqa: E402
import prompts as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

N_SAMPLED = 3          # sampled candidates per leaf (plus 1 greedy)
MAX_LEAF_DOMAIN = 3    # value classes kept per leaf
MAX_CHILD_ASSIGN = 12  # child assignments enumerated per L1 node
MAX_MID_DOMAIN = 3     # value classes kept per L1 node
MAX_ROOT_ASSIGN = 27
TEMP = 0.8
EPS = 1e-12


def out_path(args, name):
    d = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def value_classes(cands, max_k):
    """Dedup candidates into value classes ordered greedy-first then by frequency."""
    g = next((c for c in cands if c.get("kind") == "greedy" and c.get("ans")), None)
    cnt = Counter(c["norm"] for c in cands if c.get("norm") is not None)
    if not cnt:
        return []
    keys = sorted(cnt, key=lambda k: -cnt[k])
    if g and g.get("norm") in cnt:
        keys = [g["norm"]] + [k for k in keys if k != g["norm"]]
    tot = sum(cnt.values())
    out = []
    for k in keys[:max_k]:
        rep = next(c["ans"] for c in cands if c["norm"] == k and c.get("ans"))
        out.append({"norm": k, "ans": rep, "freq": cnt[k] / tot})
    return out


# ---------------------------------------------------------------- stage: tree
def stage_tree(args):
    probs = load_problems(args.data, args.limit, 0, 1)
    path = out_path(args, "tree.jsonl")
    done = {r["id"] for r in jread(path)}
    todo = [p for p in probs if p["id"] not in done]
    print(f"[tree] {len(todo)}/{len(probs)} problems to decompose", flush=True)
    if not todo:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(todo), 32):
        batch = todo[i : i + 32]
        l1_out = R.chat_batch([P.DECOMPOSE_USER.format(problem=p["problem"]) for p in batch],
                              system=P.DECOMPOSE_SYSTEM, max_new=300, bs=args.bs)
        l1 = [parse_decompose(o[0]) for o in l1_out]

        # second level: decompose each L1 subquestion in the context of the problem
        units = [(bi, si, p, q) for bi, (p, subs) in enumerate(zip(batch, l1))
                 if subs for si, q in enumerate(subs)]
        l2 = defaultdict(lambda: None)
        l2_raw = {}
        if units:
            outs = R.chat_batch(
                [P.DEEP_DECOMPOSE_USER.format(problem=p["problem"], subquestion=q)
                 for (_, _, p, q) in units],
                system=P.DECOMPOSE_SYSTEM, max_new=300, bs=args.bs)
            for (bi, si, _, q), o in zip(units, outs):
                l2_raw[(bi, si)] = o[0][:400]
                subs = parse_decompose(o[0])
                # reject degenerate splits that just restate the parent
                if subs and all(s.strip().lower() != q.strip().lower() for s in subs):
                    l2[(bi, si)] = subs
        for bi, (p, subs) in enumerate(zip(batch, l1)):
            nodes = []
            if subs:
                for si, q in enumerate(subs):
                    nodes.append({"subq": q, "children": l2[(bi, si)],
                                  "l2_raw": l2_raw.get((bi, si), "")})
            w.write({**p, "l1": nodes, "l1_raw": l1_out[bi][0][:400]})
        print(f"[tree] {min(i+32, len(todo))}/{len(todo)}", flush=True)
    trees = jread(path)
    no_l1 = sum(1 for t in trees if not t["l1"])
    depth3 = sum(1 for t in trees if any(nd.get("children") for nd in t["l1"]))
    print(f"[tree] {len(trees)} problems: no L1 split {no_l1}, "
          f">=1 depth-3 branch {depth3}", flush=True)
    if depth3 == 0 and trees:
        bad = next((nd for t in trees for nd in t["l1"] if not nd.get("children")), None)
        if bad:
            print("[tree] no depth-3 branch built; sample raw L2 output:\n"
                  + bad.get("l2_raw", "")[:400], flush=True)


# ---------------------------------------------------------------- stage: leaf
def leaf_units(trees):
    """(pid, l1_idx, leaf_idx, problem, question). Undecomposed L1 nodes are
    treated as their own leaf so every branch still yields a value."""
    units = []
    for t in trees:
        for mi, nd in enumerate(t["l1"]):
            kids = nd.get("children") or [nd["subq"]]
            for li, q in enumerate(kids):
                units.append((t["id"], mi, li, t["problem"], q))
    return units


def stage_leaf(args):
    trees = jread(out_path(args, "tree.jsonl"))
    trees = [t for t in trees if t.get("l1")]
    path = out_path(args, "leafcands.jsonl")
    done = {(r["id"], r["l1_idx"], r["leaf_idx"]) for r in jread(path)}
    units = [u for u in leaf_units(trees) if (u[0], u[1], u[2]) not in done]
    print(f"[leaf] {len(units)} leaf nodes to sample", flush=True)
    if not units:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(units), 24):
        batch = units[i : i + 24]
        ps = [P.SUBQ_USER.format(problem=pr, subquestion=q) for (_, _, _, pr, q) in batch]
        greedy = R.chat_batch(ps, max_new=400, bs=args.bs)
        sampled = R.chat_batch(ps, max_new=400, temperature=TEMP, n=N_SAMPLED, bs=args.bs)
        for (pid, mi, li, _, q), g, ss in zip(batch, greedy, sampled):
            cands = []
            for kind, text in [("greedy", g[0])] + [("sample", t) for t in ss]:
                a = extract_boxed(text)
                cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
            w.write({"id": pid, "l1_idx": mi, "leaf_idx": li, "q": q, "cands": cands})
        print(f"[leaf] {min(i+24, len(units))}/{len(units)}", flush=True)


# ---------------------------------------------------------------- stage: mid
def load_leaves(args):
    by = defaultdict(dict)
    for r in jread(out_path(args, "leafcands.jsonl")):
        by[(r["id"], r["l1_idx"])][r["leaf_idx"]] = r
    return by


def stage_mid(args):
    trees = {t["id"]: t for t in jread(out_path(args, "tree.jsonl")) if t.get("l1")}
    leaves = load_leaves(args)
    path = out_path(args, "midagg.jsonl")
    done = {(r["id"], r["l1_idx"], r["child_assign_idx"]) for r in jread(path)}

    units = []
    for pid, t in trees.items():
        for mi, nd in enumerate(t["l1"]):
            lv = leaves.get((pid, mi), {})
            if not lv:
                continue
            doms = [value_classes(lv[k]["cands"], MAX_LEAF_DOMAIN)
                    for k in sorted(lv)]
            if any(not d for d in doms):
                continue
            qs = [lv[k]["q"] for k in sorted(lv)]
            combos = list(itertools.product(*[range(len(d)) for d in doms]))[:MAX_CHILD_ASSIGN]
            for ci, combo in enumerate(combos):
                if (pid, mi, ci) in done:
                    continue
                chosen = [doms[j][c] for j, c in enumerate(combo)]
                units.append({"id": pid, "l1_idx": mi, "child_assign_idx": ci,
                              "problem": t["problem"], "subq": nd["subq"],
                              "child_qs": qs, "chosen": chosen,
                              "child_norms": [c["norm"] for c in chosen],
                              "child_freqs": [c["freq"] for c in chosen]})
    print(f"[mid] {len(units)} child assignments to aggregate", flush=True)
    if not units:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(units), 32):
        batch = units[i : i + 32]
        ps = [P.MID_AGGREGATE_USER.format(
                  problem=u["problem"], subquestion=u["subq"],
                  facts=facts_str(u["child_qs"], [c["ans"] for c in u["chosen"]]))
              for u in batch]
        outs = R.chat_batch(ps, max_new=350, bs=args.bs)
        for u, o in zip(batch, outs):
            a = extract_boxed(o[0])
            w.write({"id": u["id"], "l1_idx": u["l1_idx"],
                     "child_assign_idx": u["child_assign_idx"],
                     "child_norms": u["child_norms"], "child_freqs": u["child_freqs"],
                     "value_ans": a, "value_norm": normalize_answer(a)})
        print(f"[mid] {min(i+32, len(units))}/{len(units)}", flush=True)


# ---------------------------------------------------------------- stage: root
def mid_domains(args):
    """Per (pid, l1_idx): value classes with provenance = child assignments."""
    rows = defaultdict(list)
    for r in jread(out_path(args, "midagg.jsonl")):
        if r.get("value_norm") is not None:
            rows[(r["id"], r["l1_idx"])].append(r)
    doms = {}
    for key, rs in rows.items():
        by_val = defaultdict(list)
        for r in rs:
            by_val[r["value_norm"]].append(r)
        ranked = sorted(by_val.items(), key=lambda kv: -len(kv[1]))[:MAX_MID_DOMAIN]
        tot = sum(len(v) for v in by_val.values())
        doms[key] = [{"norm": k, "ans": v[0]["value_ans"],
                      "freq": len(v) / max(1, tot),
                      "provenance": [{"child_norms": r["child_norms"],
                                      "child_freqs": r["child_freqs"]} for r in v]}
                     for k, v in ranked]
    return doms


def stage_root(args):
    trees = {t["id"]: t for t in jread(out_path(args, "tree.jsonl")) if t.get("l1")}
    doms = mid_domains(args)
    path = out_path(args, "rootagg.jsonl")
    done = {(r["id"], r["assign_idx"]) for r in jread(path)}

    units = []
    for pid, t in trees.items():
        keys = [(pid, mi) for mi in range(len(t["l1"])) if (pid, mi) in doms]
        if len(keys) != len(t["l1"]) or not keys:
            continue
        dl = [doms[k] for k in keys]
        combos = list(itertools.product(*[range(len(d)) for d in dl]))[:MAX_ROOT_ASSIGN]
        for ai, combo in enumerate(combos):
            if (pid, ai) in done:
                continue
            chosen = [dl[j][c] for j, c in enumerate(combo)]
            units.append({"id": pid, "assign_idx": ai, "problem": t["problem"],
                          "subqs": [nd["subq"] for nd in t["l1"]], "chosen": chosen})
    print(f"[root] {len(units)} root assignments to aggregate", flush=True)
    if not units:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(units), 32):
        batch = units[i : i + 32]
        ps = [P.AGGREGATE_USER.format(
                  problem=u["problem"],
                  facts=facts_str(u["subqs"], [c["ans"] for c in u["chosen"]]))
              for u in batch]
        outs = R.chat_batch(ps, max_new=400, bs=args.bs)
        for u, o in zip(batch, outs):
            a = extract_boxed(o[0])
            w.write({"id": u["id"], "assign_idx": u["assign_idx"],
                     "mid_norms": [c["norm"] for c in u["chosen"]],
                     "mid_freqs": [c["freq"] for c in u["chosen"]],
                     "final_ans": a, "final_norm": normalize_answer(a)})
        print(f"[root] {min(i+32, len(units))}/{len(units)}", flush=True)


# ------------------------------------------------------------ stage: rootcot
def stage_rootcot(args):
    """Direct-CoT samples for the same problems, so the S1 gate math (unique
    coverage of the decomposition path) also applies to the deep pipeline."""
    trees = [t for t in jread(out_path(args, "tree.jsonl")) if t.get("l1")]
    path = out_path(args, "rootcands.jsonl")
    done = {r["id"] for r in jread(path)}
    todo = [t for t in trees if t["id"] not in done]
    print(f"[rootcot] {len(todo)} problems", flush=True)
    if not todo:
        return
    R = Runner(args.model)
    w = JWriter(path)
    for i in range(0, len(todo), 8):
        batch = todo[i : i + 8]
        ps = [P.ROOT_COT_USER.format(problem=t["problem"]) for t in batch]
        greedy = R.chat_batch(ps, max_new=768, bs=args.bs)
        sampled = R.chat_batch(ps, max_new=768, temperature=TEMP, n=4, bs=args.bs)
        for t, g, ss in zip(batch, greedy, sampled):
            cands = []
            for kind, text in [("greedy", g[0])] + [("sample", x) for x in ss]:
                a = extract_boxed(text)
                cands.append({"kind": kind, "ans": a, "norm": normalize_answer(a)})
            w.write({"id": t["id"], "cands": cands})
        print(f"[rootcot] {min(i+8, len(todo))}/{len(todo)}", flush=True)


# --------------------------------------------------------------- stage: gate
def stage_gate(args):
    """Same acceptance metrics as s1_gate.py, on the depth-3 pipeline."""
    trees = {t["id"]: t for t in jread(out_path(args, "tree.jsonl")) if t.get("l1")}
    roots = defaultdict(list)
    for r in jread(out_path(args, "rootagg.jsonl")):
        if r.get("final_norm") is not None:
            roots[r["id"]].append(r)
    rc = {r["id"]: r for r in jread(out_path(args, "rootcands.jsonl"))}
    ids = [pid for pid in roots if pid in trees]
    n = max(1, len(ids))
    c = Counter()
    for pid in ids:
        gold = trees[pid]["gold"]
        a_ok = any(answers_equal(r["final_ans"], gold) for r in roots[pid]
                   if r.get("final_ans"))
        cands = rc.get(pid, {}).get("cands", [])
        r_ok = any(answers_equal(x["ans"], gold) for x in cands if x.get("ans"))
        c["oracle_hier"] += a_ok
        c["oracle_root"] += r_ok
        c["oracle_union"] += (a_ok or r_ok)
        c["only_hier"] += (a_ok and not r_ok)
        c["only_root"] += (r_ok and not a_ok)
        greedy = next((x["ans"] for x in cands if x.get("kind") == "greedy"), None)
        c["cot_greedy"] += bool(greedy and answers_equal(greedy, gold))
        cnt = Counter(x["norm"] for x in cands if x.get("norm"))
        if cnt:
            top = cnt.most_common(1)[0][0]
            rep = next(x["ans"] for x in cands if x.get("norm") == top)
            c["sc5"] += bool(rep and answers_equal(rep, gold))
    rep = {k: v / n for k, v in c.items()}
    rep["n_problems"] = len(ids)
    rep["gate_pass"] = rep.get("only_hier", 0.0) > args.gate
    print(f"== S2 gate ({len(ids)} problems with a depth-3 hierarchy) ==")
    print(f"  CoT greedy {rep.get('cot_greedy',0):.3f}   SC@5 {rep.get('sc5',0):.3f}")
    print(f"  oracle: hierarchy {rep['oracle_hier']:.3f}  root CoT {rep['oracle_root']:.3f}"
          f"  union {rep['oracle_union']:.3f}")
    print(f"  gold ONLY via hierarchy: {rep.get('only_hier',0):.3f}"
          f"   ONLY via direct CoT: {rep.get('only_root',0):.3f}")
    print(f"  GATE (>{args.gate:.2f}): {'PASS' if rep['gate_pass'] else 'FAIL'}")
    with open(out_path(args, "s2_gate_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("saved s2_gate_report.json")


# ---------------------------------------------------------------- stage: bp
def stage_bp(args):
    """3-level BP. Round 1 is bottom-up only; later rounds add top-down."""
    trees = {t["id"]: t for t in jread(out_path(args, "tree.jsonl")) if t.get("l1")}
    doms = mid_domains(args)
    leaves = load_leaves(args)
    roots = defaultdict(list)
    for r in jread(out_path(args, "rootagg.jsonl")):
        if r.get("final_norm") is not None:
            roots[r["id"]].append(r)

    rounds, eta = args.rounds, args.eta
    correct = [0] * rounds
    n, changed, drift_tot = 0, 0, 0.0
    for pid, rows in roots.items():
        t = trees.get(pid)
        if not t:
            continue
        mids = [(pid, mi) for mi in range(len(t["l1"])) if (pid, mi) in doms]
        if len(mids) != len(t["l1"]):
            continue
        # leaf priors q[(mi, leaf_idx)][norm]
        q = {}
        for (p_, mi) in mids:
            lv = leaves.get((pid, mi), {})
            for li in sorted(lv):
                vc = value_classes(lv[li]["cands"], MAX_LEAF_DOMAIN)
                if vc:
                    z = sum(c["freq"] for c in vc) or 1.0
                    q[(mi, li)] = {c["norm"]: c["freq"] / z for c in vc}
        argmaxes, drifts, oks = [], [], []
        for _ in range(rounds):
            # bottom-up: mass of each L1 value class = sum over its provenance
            mid_mass = {}
            for (p_, mi) in mids:
                mm = {}
                for cls in doms[(pid, mi)]:
                    s = 0.0
                    for prov in cls["provenance"]:
                        pr = 1.0
                        for li, kn in enumerate(prov["child_norms"]):
                            pr *= q.get((mi, li), {}).get(kn, 1e-4)
                        s += pr
                    mm[cls["norm"]] = s
                z = sum(mm.values()) or 1.0
                mid_mass[mi] = {k: v / z for k, v in mm.items()}
            # root marginal
            py = defaultdict(float)
            wrow = []
            for r in rows:
                pr = 1.0
                for mi, kn in enumerate(r["mid_norms"]):
                    pr *= mid_mass.get(mi, {}).get(kn, 1e-4)
                wrow.append(pr)
                py[r["final_norm"]] += pr
            if not py or max(py.values()) <= 0:
                break
            best = max(py, key=py.get)
            argmaxes.append(best)
            pred = next(r["final_ans"] for r in rows if r["final_norm"] == best)
            oks.append(bool(pred and answers_equal(pred, t["gold"])))

            # top-down: root posterior -> L1 value beliefs -> leaf beliefs
            zt = sum(py.values()) or 1.0
            pyn = {k: v / zt for k, v in py.items()}
            mid_post = defaultdict(lambda: defaultdict(float))
            for r, wi in zip(rows, wrow):
                for mi, kn in enumerate(r["mid_norms"]):
                    mid_post[mi][kn] += wi * pyn.get(r["final_norm"], 0.0)
            d = 0.0
            for (p_, mi) in mids:
                leaf_mass = defaultdict(lambda: defaultdict(float))
                for cls in doms[(pid, mi)]:
                    post = mid_post[mi].get(cls["norm"], 0.0)
                    if post <= EPS:
                        continue
                    for prov in cls["provenance"]:
                        pr = 1.0
                        for li, kn in enumerate(prov["child_norms"]):
                            pr *= q.get((mi, li), {}).get(kn, 1e-4)
                        for li, kn in enumerate(prov["child_norms"]):
                            leaf_mass[li][kn] += post * pr
                for li, m in leaf_mass.items():
                    if (mi, li) not in q:
                        continue
                    tot = sum(m.values())
                    if tot <= EPS:
                        continue
                    before = dict(q[(mi, li)])
                    for k in q[(mi, li)]:
                        q[(mi, li)][k] = ((1 - eta) * before[k]
                                          + eta * (m.get(k, 0.0) / tot))
                    z = sum(q[(mi, li)].values()) or 1.0
                    for k in q[(mi, li)]:
                        q[(mi, li)][k] /= z
                    d += sum(abs(q[(mi, li)][k] - before[k]) for k in before)
            drifts.append(d / max(1, len(q)))
        if not oks:
            continue
        n += 1
        oks += [oks[-1]] * (rounds - len(oks))  # carry last decision forward
        for tt in range(rounds):
            correct[tt] += oks[tt]
        if len(set(argmaxes)) > 1:
            changed += 1
        drift_tot += sum(drifts) / max(1, len(drifts))

    accs = [c / max(1, n) for c in correct]
    rep = {"n": n, "acc_by_round": accs,
           "problems_where_root_argmax_changed": changed,
           "mean_leaf_belief_drift_per_round": drift_tot / max(1, n),
           "eta": eta}
    print(f"== S2 BP (depth-3): n={n} ==")
    print("  " + "  ".join(f"r{i+1}={a:.3f}" for i, a in enumerate(accs)))
    print(f"  root argmax changed in {changed}/{n} problems;"
          f" mean leaf-belief drift/round {rep['mean_leaf_belief_drift_per_round']:.3f}")
    with open(out_path(args, "s2_bp_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("saved s2_bp_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["tree", "leaf", "mid", "root", "rootcot", "gate", "bp"])
    ap.add_argument("--data", default="data/gsm_deep_test.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.10)
    a = ap.parse_args()
    {"tree": stage_tree, "leaf": stage_leaf, "mid": stage_mid,
     "root": stage_root, "rootcot": stage_rootcot, "gate": stage_gate,
     "bp": stage_bp}[a.stage](a)
