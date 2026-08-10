"""E0 analysis (CPU): oracle-hierarchy vs SC@k at matched measured tokens.

Primary quantity (after socratic-goal fix):
  decomposition tax = SC@1 − oracle-mode final accuracy
  (how much accuracy is lost by forcing node-local execution even when
   structure and predecessor values are perfect).

Reports:
  - DAG stats (edges, depth)
  - per-node hit rate under oracle vs predicted predecessors
  - final-answer accuracy for both modes
  - SC@k curve with measured tokens
  - matched-token comparison (diagnostic, not a hard gate)

Usage:
  python oracle_analyze.py --dir e0_soc --data data/gsm_oracle_soc.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_check import answers_equal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SC_KS = (1, 2, 3, 5, 9, 11, 21)


def jread_glob(out_dir, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, pattern))):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def analyze(out_dir, data_path):
    d = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    data = {}
    with open(data_path if os.path.isabs(data_path) else os.path.join(HERE, data_path),
              encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            data[i] = json.loads(line)

    nodes = {"predicted": defaultdict(dict), "oracle": defaultdict(dict)}
    for mode in nodes:
        for r in jread_glob(d, f"nodes_{mode}.s*.jsonl"):
            nodes[mode][r["id"]][r["node_idx"]] = r
    roots = {r["id"]: r["cands"] for r in jread_glob(d, "rootext.s*.jsonl")}
    toks = Counter()
    for p in sorted(glob.glob(os.path.join(d, "tokens.s*.json"))):
        with open(p) as f:
            toks.update(json.load(f))

    ids = sorted(i for i in data if i in roots and i in nodes["predicted"]
                 and i in nodes["oracle"])
    if not ids:
        # allow missing one mode
        ids = sorted(set(nodes["predicted"]) | set(nodes["oracle"]))
        ids = [i for i in ids if i in roots and i in data]
    if not ids:
        print(f"== E0: {out_dir}: no usable problems ==")
        return None

    n = len(ids)
    # token costs
    n_pred_calls = sum(len(nodes["predicted"][i]) for i in ids)
    n_ora_calls = sum(len(nodes["oracle"][i]) for i in ids)
    n_root_calls = sum(len(roots[i]) for i in ids)
    c_pred = toks.get("nodes_predicted", 0) / max(1, n_pred_calls)
    c_ora = toks.get("nodes_oracle", 0) / max(1, n_ora_calls)
    c_root = toks.get("rootext", 0) / max(1, n_root_calls)
    # if rootext was reused, tokens.s*.json may lack rootext — estimate from dd
    if c_root == 0 and n_root_calls:
        # fallback: use typical measured root cost from dd_deep (~331)
        c_root = 331.0
        print(f"  [warn] no rootext token meter; using fallback {c_root:.0f}/call")

    # DAG stats
    n_nodes = sum(data[i]["n_steps"] for i in ids)
    n_edges = sum(len(nd["depends_on"]) for i in ids for nd in data[i]["nodes"])
    depths = [data[i]["n_steps"] for i in ids]

    st = {m: Counter() for m in ("predicted", "oracle")}
    node_hit = {m: Counter() for m in ("predicted", "oracle")}
    for i in ids:
        gold = str(data[i]["answer"])
        for mode in ("predicted", "oracle"):
            nds = nodes[mode].get(i, {})
            if not nds:
                continue
            st[mode]["n"] += 1
            for j in range(data[i]["n_steps"]):
                r = nds.get(j)
                if not r:
                    continue
                node_hit[mode]["n"] += 1
                node_hit[mode]["hit"] += bool(r.get("hit"))
                if r.get("depends_on"):
                    node_hit[mode]["n_dep"] += 1
                    node_hit[mode]["hit_dep"] += bool(r.get("hit"))
                else:
                    node_hit[mode]["n_indep"] += 1
                    node_hit[mode]["hit_indep"] += bool(r.get("hit"))
            # final = last node
            last = nds.get(data[i]["n_steps"] - 1)
            if last and last.get("ans"):
                st[mode]["final"] += bool(answers_equal(last["ans"], gold))
            # all-nodes-correct
            if all(nds.get(j, {}).get("hit") for j in range(data[i]["n_steps"])):
                st[mode]["all_hit"] += 1
            # mean tokens for this problem under mode
            st[mode]["tokens"] += data[i]["n_steps"] * (c_pred if mode == "predicted" else c_ora)

    sc = Counter()
    sc_n = Counter()
    for i in ids:
        gold = str(data[i]["answer"])
        rc = roots[i]
        for k in SC_KS:
            sub = rc[:k]
            sc_n[k] += 1
            sc[f"oracle@{k}"] += any(x.get("ans") and answers_equal(x["ans"], gold)
                                     for x in sub)
            cnt = Counter(x["norm"] for x in sub if x.get("norm"))
            if cnt:
                top = cnt.most_common(1)[0][0]
                rep = next(x["ans"] for x in sub if x["norm"] == top)
                sc[f"sc@{k}"] += bool(rep and answers_equal(rep, gold))

    print(f"== E0 oracle-hierarchy ceiling: {out_dir} ({n} problems) ==")
    print(f"  DAG: {n_nodes} nodes, {n_edges} depends-on edges, "
          f"mean depth {sum(depths)/n:.2f}")
    print(f"  tokens/call: nodes_predicted {c_pred:.0f}  nodes_oracle {c_ora:.0f}  "
          f"rootCoT {c_root:.0f}")

    print("\n  -- local execution under oracle structure --")
    for mode in ("predicted", "oracle"):
        nh, ns = node_hit[mode], st[mode]
        if not ns["n"]:
            print(f"  {mode}: (no rows)")
            continue
        print(f"  {mode}:")
        print(f"    node hit           {nh['hit']/max(1,nh['n']):.3f}  "
              f"(dep {nh['hit_dep']/max(1,nh['n_dep']):.3f} / "
              f"indep {nh['hit_indep']/max(1,nh['n_indep']):.3f})")
        print(f"    all nodes correct  {ns['all_hit']/ns['n']:.3f}")
        print(f"    final-answer acc   {ns['final']/ns['n']:.3f}   "
              f"tokens/problem {ns['tokens']/ns['n']:.0f}  "
              f"(~SC@{ns['tokens']/ns['n']/max(1e-9,c_root):.1f})")

    print("\n  -- SC@k baseline --")
    print(f"  {'k':>2} {'SC@k':>7} {'oracle':>7} {'tokens':>7}")
    sc_rows = {}
    for k in SC_KS:
        if not sc_n[k]:
            continue
        row = {"sc": sc[f"sc@{k}"] / sc_n[k],
               "oracle": sc[f"oracle@{k}"] / sc_n[k],
               "tokens": k * c_root}
        sc_rows[k] = row
        print(f"  {k:>2} {row['sc']:>7.3f} {row['oracle']:>7.3f} {row['tokens']:>7.0f}")

    # Decomposition tax: even with perfect structure + gold predecessors,
    # how far below single-shot CoT do we land?
    print("\n  -- decomposition tax (oracle predecessors vs SC@1) --")
    tax = None
    if st["oracle"]["n"] and 1 in sc_rows:
        acc_ora = st["oracle"]["final"] / st["oracle"]["n"]
        sc1 = sc_rows[1]["sc"]
        tax = sc1 - acc_ora
        print(f"  oracle-mode final={acc_ora:.3f}  SC@1={sc1:.3f}  "
              f"tax={tax:+.3f}  "
              f"({'small — paradigm viable' if tax <= 0.05 else 'large — node-local execution costly'})")
        if tax > 0.15:
            print("  NOTE: large tax on GSM may still be OK if a harder multihop "
                  "benchmark has SC@1≪0.9; tax is a cost, not a GSM go/no-go.")
    else:
        print("  (need oracle-mode rows and SC@1)")

    print("\n  -- matched-token diagnostic (predicted vs SC) --")
    gate = False
    if st["predicted"]["n"]:
        t_pred = st["predicted"]["tokens"] / st["predicted"]["n"]
        acc_pred = st["predicted"]["final"] / st["predicted"]["n"]
        best_k = min(sc_rows, key=lambda k: abs(sc_rows[k]["tokens"] - t_pred))
        sc_acc = sc_rows[best_k]["sc"]
        delta = acc_pred - sc_acc
        gate = delta > 0.0
        print(f"  predicted final={acc_pred:.3f} @ {t_pred:.0f} tok  "
              f"vs SC@{best_k}={sc_acc:.3f} @ {sc_rows[best_k]['tokens']:.0f} tok  "
              f"delta={delta:+.3f}")
        print(f"  (diagnostic only; primary quantity is decomposition tax above)")
    else:
        print("  (no predicted-mode rows)")

    rep = {
        "n_problems": n,
        "dag": {"nodes": n_nodes, "edges": n_edges,
                "mean_depth": sum(depths) / n},
        "per_call_tokens": {"predicted": c_pred, "oracle": c_ora, "root": c_root},
        "modes": {},
        "sc": {str(k): v for k, v in sc_rows.items()},
        "decomposition_tax": tax,
        "matched_token_pred_beats_sc": gate,
    }
    for mode in ("predicted", "oracle"):
        ns, nh = st[mode], node_hit[mode]
        if not ns["n"]:
            continue
        rep["modes"][mode] = {
            "n": ns["n"],
            "final_acc": ns["final"] / ns["n"],
            "all_hit": ns["all_hit"] / ns["n"],
            "node_hit": nh["hit"] / max(1, nh["n"]),
            "node_hit_dep": nh["hit_dep"] / max(1, nh["n_dep"]),
            "tokens_per_problem": ns["tokens"] / ns["n"],
        }
    out_path = os.path.join(HERE, f"e0_report_{os.path.basename(d)}.json")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nsaved {out_path}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    for dd, dp in zip([x.strip() for x in a.dir.split(",")],
                      [x.strip() for x in a.data.split(",")]):
        analyze(dd, dp)
        print()
