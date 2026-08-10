"""Build an oracle typed dependency DAG from GSM8K calculator annotations.

Each <<expr=value>> step becomes a node. A depends-on edge i <- j exists when
step i's expression literally uses the numeric value produced by an earlier
step j (longest-match, same rule as data_prep.re_evaluate). This is the true
sequential structure the independent-subquestion prompt was forbidden to use.

Natural-language goals should come from GSM8K-socratic (human/model
subquestions before each `**`), not from scrubbing numbers out of the main
solution prose — that scrubbing produced unanswerable fragments and invalidated
the first E0 run.

Usage:
  python oracle_hierarchy.py --data data/gsm_deep_test.jsonl \\
      --out data/gsm_oracle_soc.jsonl --goal-source data/gsm8k_socratic_test.jsonl \\
      --goal-mode socratic
"""
from __future__ import annotations

import argparse
import json
import os
import re
from fractions import Fraction

from data_prep import NUM_RE, num_str, parse_steps, read_jsonl, safe_eval, write_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")


def _to_frac(s):
    v = safe_eval(str(s).replace(",", ""))
    return v


def socratic_goals(answer_text, n_steps):
    """Extract the Socratic subquestion before each <<expr=value>> annotation.

    Canonical format (openai/gsm8k socratic):
      How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 ...
    The goal is the text before `**` — no gold-value scrubbing required.
    """
    goals = []
    remaining = answer_text
    for _ in range(n_steps):
        m = STEP_RE.search(remaining)
        if not m:
            goals.append(None)
            continue
        block_start = remaining.rfind("\n", 0, m.start()) + 1
        block = remaining[block_start:m.start()]
        if "**" in block:
            goal = block.split("**", 1)[0].strip()
        else:
            # Rare non-socratic line: take leading prose, drop trailing arith.
            goal = re.sub(
                r"[\d.,]+\s*([+\-*/×x]\s*[\d.,]+\s*)*=?\s*\$?\s*$", "", block
            ).strip(" .:")
        goal = re.sub(r"\s+", " ", goal).strip()
        goals.append(goal or None)
        remaining = remaining[m.end():]
    return goals


def sentence_goals(answer_text, n_steps):
    """Legacy prose-scrub goals. Prefer socratic_goals; kept for comparison."""
    goals = []
    remaining = answer_text

    def strip_value_prefix(text, val):
        t = text.lstrip()
        if t.startswith(val):
            return t[len(val):].lstrip(" .")
        if val.isdigit():
            with_comma = f"{int(val):,}"
            if t.startswith(with_comma):
                return t[len(with_comma):].lstrip(" .")
        return t

    for _ in range(n_steps):
        m = STEP_RE.search(remaining)
        if not m:
            goals.append(None)
            continue
        val = m.group(2).strip()
        start = remaining.rfind("\n", 0, m.start()) + 1
        end = remaining.find("\n", m.end())
        if end == -1:
            end = remaining.find("####", m.end())
        if end == -1:
            end = len(remaining)
        before = remaining[start:m.start()]
        after = strip_value_prefix(remaining[m.end():end], val).strip(" .")
        before = re.sub(r"[\d.,]+\s*([+\-*/×x]\s*[\d.,]+\s*)*=?\s*\$?\s*$", "", before)
        before = before.strip(" .:")
        if after and len(after) >= 3 and not re.fullmatch(r"[\d.,]+", after):
            desc = after
        elif before and len(before) >= 3:
            desc = before
        else:
            desc = "the intermediate quantity required at this step"
        variants = {val}
        if val.isdigit():
            variants.add(f"{int(val):,}")
        for v in sorted(variants, key=len, reverse=True):
            desc = re.sub(r"(?<!\d)" + re.escape(v) + r"(?!\d)", "?", desc)
        desc = re.sub(r"\s+", " ", desc).strip(" .")
        goals.append(f"What is the value of: {desc}?")
        remaining = remaining[m.end():]
    return goals


def build_dag(steps):
    """steps: [(expr, Fraction)]. Return depends_on: list[list[int]] (pred indices)."""
    values = [v for _, v in steps]
    depends = [[] for _ in steps]
    for i, (expr, _) in enumerate(steps):
        # Collect earlier values that appear as literals in this expression.
        # Prefer later (closer) producers when the same number appears twice.
        producers = {}  # Fraction -> latest index < i
        for j in range(i):
            producers[values[j]] = j
        # Scan numbers in expr; match against known producer values.
        hits = set()
        for m in NUM_RE.finditer(expr):
            lit = _to_frac(m.group(0))
            if lit is None:
                continue
            if lit in producers:
                hits.add(producers[lit])
        depends[i] = sorted(hits)
    return depends


def serialize_node(problem, nodes, idx, pred_values, mode="predicted"):
    """Proposal §6 MVP text serialization of structural state for node idx."""
    n = nodes[idx]
    lines = [
        f"[ROOT] {problem.strip()}",
        f"[CURRENT] Intermediate step {idx + 1}/{len(nodes)}",
        f"[GOAL] {n['goal']}",
        f"[STATE] unresolved",
    ]
    deps = n["depends_on"]
    if deps:
        lines.append("[DEPENDS_ON]")
        for j in deps:
            val = pred_values.get(j)
            tag = nodes[j]["goal"][:80]
            if val is None:
                lines.append(f"  - step {j + 1}: (missing)  | {tag}")
            else:
                lines.append(f"  - step {j + 1}: {val}  | {tag}")
        lines.append("[CONSTRAINT] Use the DEPENDS_ON values; do not re-derive them from the problem alone.")
    else:
        lines.append("[DEPENDS_ON] (none — this step uses only quantities stated in the problem)")
        lines.append("[CONSTRAINT] Compute directly from the problem statement.")
    lines.append(f"[MODE] execute ({mode} predecessors)")
    return "\n".join(lines)


def row_to_oracle(r):
    """Convert a gsm8k / gsm_deep row into an oracle-hierarchy record."""
    problem = r.get("problem") or r.get("question")
    if "steps" in r and isinstance(r["steps"], list) and r["steps"]:
        # gsm_deep already parsed
        steps = [(s["expr"], _to_frac(s["value"])) for s in r["steps"]]
        if any(v is None for _, v in steps):
            return None
        gold = str(r["answer"])
        answer_text = None
        # Prefer reconstructing goals from raw gsm8k if available later; for
        # gsm_deep we only have expr/value — synthesize a neutral goal.
        goals = [f"Compute the intermediate quantity equal to evaluating `{e}` "
                 f"(use predecessor results when they appear)." for e, _ in steps]
    else:
        steps, g = parse_steps(r["answer"])
        if not steps or g is None:
            return None
        gold = num_str(g)
        answer_text = r["answer"]
        goals = sentence_goals(answer_text, len(steps))
        for i, (e, _) in enumerate(steps):
            if not goals[i]:
                goals[i] = (f"Compute the intermediate quantity for step {i + 1} "
                            f"of the solution.")
    depends = build_dag(steps)
    nodes = []
    for i, ((expr, val), goal) in enumerate(zip(steps, goals)):
        nodes.append({
            "idx": i,
            "expr": expr,
            "gold_value": num_str(val),
            "goal": goal,
            "depends_on": depends[i],
            "type": "compute",
        })
    return {
        "problem": problem,
        "answer": gold,
        "n_steps": len(nodes),
        "nodes": nodes,
        "source": r.get("source", "gsm_oracle"),
    }


def _goal_quality(goals):
    """Cheap diagnostics: are goals actually answerable questions?"""
    n = len(goals)
    if not n:
        return {}
    lens = [len(g or "") for g in goals]
    ends_q = sum(1 for g in goals if g and g.rstrip().endswith("?"))
    has_qmark_scrub = sum(1 for g in goals if g and re.search(r"\?\s*$", g) is None and "?" in (g or ""))
    short = sum(1 for g in goals if not g or len(g) < 20 or g.count(" ") < 3)
    return {
        "n": n,
        "mean_len": sum(lens) / n,
        "pct_endswith_?": ends_q / n,
        "pct_short": short / n,
        "pct_internal_?": has_qmark_scrub / n,
    }


def build(data_path, out_path, min_steps=1, limit=0, goal_source=None,
          goal_mode="socratic"):
    """Attach NL goals from goal_source by matching question/problem text.

    goal_mode:
      socratic — text before `**` (recommended; no answer scrubbing)
      prose    — legacy sentence_goals scrub (known to produce fragments)
    """
    goal_by_q = {}
    if goal_source:
        gs = goal_source if os.path.isabs(goal_source) else os.path.join(HERE, goal_source)
        extractor = socratic_goals if goal_mode == "socratic" else sentence_goals
        for r in read_jsonl(gs):
            q = (r.get("problem") or r.get("question") or "").strip()
            steps, _g = parse_steps(r["answer"])
            if not steps:
                continue
            goal_by_q[q] = extractor(r["answer"], len(steps))
        print(f"  goal_source={gs} mode={goal_mode} keyed={len(goal_by_q)}")

    rows = read_jsonl(data_path)
    out = []
    matched = 0
    all_goals = []
    for r in rows:
        o = row_to_oracle(r)
        if o is None or o["n_steps"] < min_steps:
            continue
        q = o["problem"].strip()
        if q in goal_by_q and len(goal_by_q[q]) == o["n_steps"]:
            for nd, g in zip(o["nodes"], goal_by_q[q]):
                if g:
                    nd["goal"] = g
            matched += 1
        all_goals.extend(nd["goal"] for nd in o["nodes"])
        out.append(o)
        if limit and len(out) >= limit:
            break
    write_jsonl(out_path, out)
    n_edges = sum(len(n["depends_on"]) for o in out for n in o["nodes"])
    n_nodes = sum(o["n_steps"] for o in out)
    n_with_dep = sum(1 for o in out for n in o["nodes"] if n["depends_on"])
    n_nl = sum(1 for o in out for n in o["nodes"]
               if not n["goal"].startswith("Compute the intermediate quantity"))
    print(f"  problems={len(out)} matched_goals={matched}/{len(out)}")
    print(f"  nodes={n_nodes}  edges={n_edges}  "
          f"nodes_with_deps={n_with_dep}/{n_nodes} "
          f"({n_with_dep / max(1, n_nodes):.1%})  "
          f"nl_goals={n_nl}/{n_nodes}")
    qstat = _goal_quality(all_goals)
    if qstat:
        print(f"  goal_quality: mean_len={qstat['mean_len']:.0f}  "
              f"endswith_?={qstat['pct_endswith_?']:.1%}  "
              f"short={qstat['pct_short']:.1%}  "
              f"internal_?={qstat['pct_internal_?']:.1%}")
        if goal_mode == "socratic" and qstat["pct_endswith_?"] < 0.7:
            print("  WARNING: socratic goals look weak; check --goal-source file")
        if goal_mode == "socratic" and qstat["pct_short"] > 0.25:
            print("  WARNING: many short goals; possible mismatch with gsm_deep")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--goal-source", default="",
                    help="gsm8k(-socratic) jsonl for NL goals (matched by question)")
    ap.add_argument("--goal-mode", default="socratic",
                    choices=["socratic", "prose"],
                    help="socratic=text before **; prose=legacy scrub (buggy)")
    a = ap.parse_args()
    data = a.data if os.path.isabs(a.data) else os.path.join(HERE, a.data)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    gs = a.goal_source or None
    build(data, out, min_steps=a.min_steps, limit=a.limit,
          goal_source=gs, goal_mode=a.goal_mode)
