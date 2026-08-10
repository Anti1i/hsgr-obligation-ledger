"""Oracle-gated ceiling for hierarchy-specific local repair.

This experiment is deliberately *not* an end-to-end method.  It asks a prior
reachability question before we invest in a hidden-state observer or EdgeGuide
editor: when a final-hop node is known to be wrong, can a typed dependency
failure certificate repair it better than a budget-matched generic retry?

The evaluation set excludes every item in the original Random(0), n=200
MuSiQue slice used by the preceding HSGR experiments.  The oracle correctness
gate is applied only for this ceiling: correct base outputs are retained and
only wrong outputs receive one repair call in each arm.

Pre-registered gates on the unseen set:

1. repairable: dependency repair improves full-set accuracy by >= 8pp,
   recovers >= 20% of base errors, and has exact paired McNemar p < .05;
2. hierarchy_specific: dependency repair beats the generic repair by >= 3pp
   and has exact paired McNemar p < .05.

Passing gate 1 alone establishes repair headroom, not an HSGR mechanism.
Passing both gates permits the next phase: learn a hidden-state observer and
replace the textual certificate with hierarchy-indexed latent information
routing.  Failure of gate 1 stops that route on this task.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import answers_match, evidence_from_row, extract_boxed  # noqa: E402
from mh_e0 import hop_deps, load_rows  # noqa: E402
from pilot import Runner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You execute exactly one node in a typed multi-hop reasoning hierarchy. "
    "Use only the supplied evidence and verified predecessor mappings. "
    "Return the current node answer in \\boxed{}."
)

BASE_USER = """Evidence:
{evidence}

Original question: {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[CURRENT GOAL] {goal}
[VERIFIED PREDECESSORS]
{dependencies}

Execute only the current node and put its answer in \\boxed{{}}."""

REPAIR_USER = """Evidence:
{evidence}

Original question: {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[CURRENT GOAL] {goal}
[VERIFIED PREDECESSORS]
{dependencies}

[PREVIOUS CANDIDATE]
{candidate}

{repair_instruction}
Return only the repaired current-node answer in \\boxed{{}}."""

GENERIC_REPAIR = (
    "[FAILURE CERTIFICATE] answer-consistency=FAILED. Re-execute the current "
    "node by carefully rechecking all supplied information. Do not defend or "
    "repeat the previous candidate without independently recomputing it."
)

EDGE_REPAIR = (
    "[FAILURE CERTIFICATE] dependency-consistency=FAILED. Re-execute the "
    "current node by explicitly substituting every verified predecessor "
    "mapping into the CURRENT GOAL before consulting the evidence. Do not "
    "bypass the mappings by solving the original question from scratch."
)


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test via a Binomial(n, .5) tail."""
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p": p}


def unit_id(row: dict) -> str:
    return str(row.get("id") or row.get("_uid"))


def make_unit(row: dict) -> dict | None:
    decomp = row.get("question_decomposition") or []
    if not decomp:
        return None
    hop = len(decomp) - 1
    deps = hop_deps(decomp)[hop]
    if not deps:
        return None
    dep_lines = [f"  - #{j + 1} = {decomp[j]['answer']} (verified)" for j in deps]
    fields = {
        "evidence": evidence_from_row(row),
        "question": row["question"],
        "hop": hop + 1,
        "n_hops": len(decomp),
        "goal": decomp[hop]["question"],
        "dependencies": "\n".join(dep_lines),
    }
    return {
        "id": unit_id(row),
        "n_hops": len(decomp),
        "n_dependencies": len(deps),
        "gold": str(row["answer"]),
        "aliases": list(row.get("answer_aliases") or []),
        "fields": fields,
        "base_user": BASE_USER.format(**fields),
    }


def unseen_units(data: str, exclude_limit: int, new_limit: int, seed: int):
    excluded = {unit_id(r) for r in load_rows(data, exclude_limit, seed=0)}
    pool = []
    for row in load_rows(data, 0, seed=0):
        if unit_id(row) in excluded:
            continue
        unit = make_unit(row)
        if unit is not None:
            pool.append(unit)
    if len(pool) < new_limit:
        raise SystemExit(f"need {new_limit} unseen dependency units; found {len(pool)}")
    chosen = random.Random(seed).sample(pool, new_limit)
    return chosen, len(pool), len(excluded)


def flatten(outputs: list[list[str]]) -> list[str]:
    return [row[0] for row in outputs]


def score_one(unit: dict, text: str) -> tuple[bool, str | None]:
    answer = extract_boxed(text)
    hit = bool(answer and answers_match(answer, unit["gold"], unit["aliases"]))
    return hit, answer


def prompt_tokens(runner: Runner, users: list[str]) -> int:
    return sum(len(runner.tok.encode(u, add_special_tokens=False)) for u in users)


def accuracy(hits: list[bool]) -> float:
    return sum(hits) / max(1, len(hits))


def main(args):
    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    units, pool_size, excluded_size = unseen_units(
        data, args.exclude_limit, args.new_limit, args.seed
    )
    print(
        f"[data] unseen_units={len(units)} eligible_pool={pool_size} "
        f"excluded_original_slice={excluded_size} "
        f"hop_counts={dict(Counter(u['n_hops'] for u in units))}",
        flush=True,
    )

    runner = Runner(args.model)
    token_counts = {}

    base_users = [u["base_user"] for u in units]
    before = runner.n_new_tokens
    base_texts = flatten(
        runner.chat_batch(
            base_users, system=SYSTEM, max_new=args.max_new, bs=args.bs
        )
    )
    token_counts["base_prompt"] = prompt_tokens(runner, base_users)
    token_counts["base_generated"] = runner.n_new_tokens - before

    base_hits, base_answers = [], []
    for unit, text in zip(units, base_texts):
        hit, answer = score_one(unit, text)
        base_hits.append(hit)
        base_answers.append(answer)
    wrong = [i for i, hit in enumerate(base_hits) if not hit]
    print(
        f"[base] acc={accuracy(base_hits):.4f} wrong={len(wrong)}/{len(units)}",
        flush=True,
    )

    arm_texts = {"generic": list(base_texts), "edge": list(base_texts)}
    arm_hits = {"generic": list(base_hits), "edge": list(base_hits)}
    arm_answers = {"generic": list(base_answers), "edge": list(base_answers)}

    for arm, instruction in (("generic", GENERIC_REPAIR), ("edge", EDGE_REPAIR)):
        users = []
        for i in wrong:
            fields = dict(units[i]["fields"])
            fields.update(candidate=base_texts[i][:800], repair_instruction=instruction)
            users.append(REPAIR_USER.format(**fields))
        before = runner.n_new_tokens
        repaired = flatten(
            runner.chat_batch(users, system=SYSTEM, max_new=args.max_new, bs=args.bs)
        )
        token_counts[f"{arm}_repair_prompt"] = prompt_tokens(runner, users)
        token_counts[f"{arm}_repair_generated"] = runner.n_new_tokens - before
        for idx, text in zip(wrong, repaired):
            hit, answer = score_one(units[idx], text)
            arm_texts[arm][idx] = text
            arm_hits[arm][idx] = hit
            arm_answers[arm][idx] = answer
        print(
            f"[{arm}] oracle-gated acc={accuracy(arm_hits[arm]):.4f} "
            f"recovered={sum(arm_hits[arm][i] for i in wrong)}/{len(wrong)}",
            flush=True,
        )

    acc = {
        "base": accuracy(base_hits),
        "oracle_generic": accuracy(arm_hits["generic"]),
        "oracle_edge": accuracy(arm_hits["edge"]),
    }
    recovered = {
        arm: sum(arm_hits[arm][i] for i in wrong) for arm in ("generic", "edge")
    }
    recovery_rate = {
        arm: recovered[arm] / max(1, len(wrong)) for arm in ("generic", "edge")
    }
    paired = {
        "edge_vs_base": exact_mcnemar(arm_hits["edge"], base_hits),
        "generic_vs_base": exact_mcnemar(arm_hits["generic"], base_hits),
        "edge_vs_generic": exact_mcnemar(arm_hits["edge"], arm_hits["generic"]),
    }
    delta_edge_base = acc["oracle_edge"] - acc["base"]
    delta_edge_generic = acc["oracle_edge"] - acc["oracle_generic"]
    gates = {
        "repairable": (
            delta_edge_base >= 0.08
            and recovery_rate["edge"] >= 0.20
            and paired["edge_vs_base"]["p"] < 0.05
        ),
        "hierarchy_specific": (
            delta_edge_generic >= 0.03
            and paired["edge_vs_generic"]["p"] < 0.05
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    cases_path = os.path.join(args.out_dir, "hsgr_edge_repair_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as f:
        for i, unit in enumerate(units):
            f.write(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "n_dependencies": unit["n_dependencies"],
                "goal": unit["fields"]["goal"],
                "gold": unit["gold"],
                "base": {
                    "answer": base_answers[i],
                    "hit": base_hits[i],
                    "text": base_texts[i][:800],
                },
                "oracle_generic": {
                    "answer": arm_answers["generic"][i],
                    "hit": arm_hits["generic"][i],
                    "text": arm_texts["generic"][i][:800],
                    "reused_base": base_hits[i],
                },
                "oracle_edge": {
                    "answer": arm_answers["edge"][i],
                    "hit": arm_hits["edge"][i],
                    "text": arm_texts["edge"][i][:800],
                    "reused_base": base_hits[i],
                },
            }, ensure_ascii=False) + "\n")

    report = {
        "experiment": "HSGR oracle-gated typed-edge repair ceiling",
        "claim_boundary": (
            "Oracle-gated text repair measures reachability only; it does not "
            "establish a usable hidden-state gate or latent intervention."
        ),
        "data": {
            "n": len(units),
            "eligible_unseen_pool": pool_size,
            "excluded_original_slice": excluded_size,
            "selection_seed": args.seed,
            "hop_counts": dict(Counter(u["n_hops"] for u in units)),
        },
        "accuracy": acc,
        "base_wrong": len(wrong),
        "recovered": recovered,
        "recovery_rate_among_base_wrong": recovery_rate,
        "delta": {
            "edge_vs_base": delta_edge_base,
            "generic_vs_base": acc["oracle_generic"] - acc["base"],
            "edge_vs_generic": delta_edge_generic,
        },
        "paired_exact_mcnemar": paired,
        "gates": gates,
        "advance_to_hidden_edgeguide": all(gates.values()),
        "token_counts": token_counts,
        "model": args.model,
    }
    report_path = os.path.join(args.out_dir, "hsgr_edge_repair_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print("\n== HSGR oracle-gated edge repair ceiling ==")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--out-dir", default="hsgr_edge_repair")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--exclude-limit", type=int, default=200)
    ap.add_argument("--new-limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    main(ap.parse_args())
