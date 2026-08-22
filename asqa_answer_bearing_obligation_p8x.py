"""Frozen answer-bearing obligation induction and direct-append screen P8x."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from asqa_auto_obligation_p7x import choose_oracle, random_index, relabel
from asqa_clean_fixed_support_p1x import aligned_clean_cases, render_documents, select_cases
from asqa_missing_selector_p6x import (
    EXPECTED_CASES,
    EXPECTED_ELIGIBLE,
    EXPECTED_P1X_ROWS,
    ModelRunner,
    SelectorCase,
    append_metrics,
    build_selector_cases,
    paired_append,
    score_append,
)
from asqa_set_guide_patch_p4x import load_p1x_rows


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_ANSWER_BEARING_OBLIGATION_P8X.md"


def render_claim_induction_prompt(item: SelectorCase) -> str:
    return (
        "Build the evidence-anchored answer obligations needed for a complete answer to the "
        "ambiguous question. Prioritize different meanings, entities, versions, events, or "
        "interpretations; do not split one interpretation into several background details. "
        "Return a JSON array with between two and five objects. Every object must have exactly "
        "two string fields: scope, a concise identifier of the interpretation; and claim, one "
        "self-contained factual sentence that directly answers the question for that scope using "
        "only the fixed documents. Each claim must be at most 45 words. Do not mention document "
        "numbers. Return the JSON array only.\n\n"
        f"Question: {item.case.question}\n\nFixed documents:\n{render_documents(item.case)}\n\n"
        "Answer obligations:"
    )


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'")


def valid_claim_nodes(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not 2 <= len(value) <= 5:
        return None
    nodes = []
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"scope", "claim"}:
            return None
        if not isinstance(item["scope"], str) or not isinstance(item["claim"], str):
            return None
        scope, claim = clean_text(item["scope"]), clean_text(item["claim"])
        key = (scope.casefold(), claim.casefold())
        if not scope or not claim or len(claim.split()) > 45 or key in seen:
            return None
        seen.add(key)
        nodes.append({"scope": scope, "claim": claim})
    return nodes


def parse_claim_nodes(text: str) -> tuple[list[dict[str, str]] | None, str]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    variants = [(stripped, "json")]
    if "\\'" in stripped:
        variants.append((stripped.replace("\\'", "'"), "json_apostrophe_repair"))
    for candidate, mode in variants:
        start = candidate.find("[")
        if start >= 0:
            try:
                value, _ = json.JSONDecoder().raw_decode(candidate[start:])
                nodes = valid_claim_nodes(value)
                if nodes is not None:
                    return nodes, mode
            except json.JSONDecodeError:
                pass
    values = []
    for line in [line.strip() for line in stripped.splitlines() if line.strip()]:
        try:
            value = json.loads(line.replace("\\'", "'"))
        except json.JSONDecodeError:
            return None, "invalid"
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            values.append(value[0])
        elif isinstance(value, dict):
            values.append(value)
        else:
            return None, "invalid"
    nodes = valid_claim_nodes(values)
    return (nodes, "multi_object") if nodes is not None else (None, "invalid")


def render_claim_coverage_prompt(item: SelectorCase, node: dict[str, str]) -> str:
    return (
        "Check whether a saved answer explicitly covers one answer obligation. Reply with exactly "
        "one label and nothing else: A means COVERED; B means MISSING.\n\n"
        f"Question: {item.case.question}\n\nSaved answer: {item.direct_answer}\n\n"
        f"Obligation scope: {node['scope']}\nObligation claim: {node['claim']}\n\nLabel:"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    eligible = aligned_clean_cases(args.alce, args.original)
    cases = select_cases(eligible, EXPECTED_CASES)
    p1x = load_p1x_rows(args.p1x_generations, cases)
    direct = {case.id: p1x[(case.id, "fixed_direct")] for case in cases}
    items, exact_rescore = build_selector_cases(cases, direct)
    repairs = [item for item in items if item.exactly_one_missing]
    frozen_p7r = load_jsonl(args.p7r_selections)
    if len(frozen_p7r) != 438:
        raise RuntimeError(f"expected 438 P7r rows, got {len(frozen_p7r)}")
    replay = {
        (str(row["id"]), str(row["arm"])): row
        for row in frozen_p7r
        if row["arm"] in {
            "induced_logit_append", "induced_oracle_append", "gold_oracle_append",
            "gold_logit_append", "p6x_generic_append",
        }
    }
    runner = ModelRunner(args.model)
    prompts = [render_claim_induction_prompt(item) for item in repairs]
    outputs = runner.generate(prompts, args.induction_batch_size, 512)
    nodes_by_id = {}
    induction_rows = []
    valid_count = 0
    for item, output in zip(repairs, outputs):
        nodes, mode = parse_claim_nodes(output)
        valid = nodes is not None
        valid_count += int(valid)
        used = nodes if nodes is not None else [{"scope": "general", "claim": item.case.question}]
        nodes_by_id[item.case.id] = used
        induction_rows.append(
            {
                "id": item.case.id,
                "raw_induction": output,
                "parse_mode": mode,
                "valid_claim_set": valid,
                "used_fallback": not valid,
                "nodes": used,
            }
        )
    print(f"[induction] valid={valid_count}/{len(repairs)}", flush=True)
    flat = [(item, index, node) for item in repairs for index, node in enumerate(nodes_by_id[item.case.id])]
    coverage_prompts = [render_claim_coverage_prompt(item, node) for item, _, node in flat]
    _, scores = runner.extract_selector(coverage_prompts, (27,), args.selector_batch_size)
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite P8x coverage score")

    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_rows = []
    for flat_index, (item, index, node) in enumerate(flat):
        row = score_append(item, "claim_candidate", None, node["claim"])
        row.update(
            {
                "candidate_index": index + 1,
                "scope": node["scope"],
                "claim": node["claim"],
                "coverage_score": float(scores[flat_index]),
            }
        )
        candidates_by_id[item.case.id].append(row)
        candidate_rows.append(row)

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in repairs:
        candidates = candidates_by_id[item.case.id]
        auto_index = min(range(len(candidates)), key=lambda index: (-candidates[index]["coverage_score"], index))
        selections = {
            "claim_logit_append": auto_index,
            "claim_random_append": random_index(item.case.id, len(candidates)),
            "claim_oracle_append": choose_oracle(candidates),
        }
        for arm, index in selections.items():
            row = candidates[index]
            rendered = f"{row['scope']}: {row['claim']}"
            by_arm[arm].append(
                relabel(row, arm, rendered, index, float(row["coverage_score"]) if arm != "claim_oracle_append" else None)
            )
        replay_map = {
            "p7r_question_logit_append": "induced_logit_append",
            "p7r_question_oracle_append": "induced_oracle_append",
            "gold_oracle_append": "gold_oracle_append",
            "gold_logit_append": "gold_logit_append",
            "p6x_generic_append": "p6x_generic_append",
        }
        for target_arm, source_arm in replay_map.items():
            row = dict(replay[(item.case.id, source_arm)])
            row["arm"] = target_arm
            by_arm[target_arm].append(row)

    arms = (
        "claim_logit_append", "claim_random_append", "claim_oracle_append",
        "p7r_question_logit_append", "p7r_question_oracle_append",
        "gold_oracle_append", "gold_logit_append", "p6x_generic_append",
    )
    absolute = {arm: append_metrics(by_arm[arm]) for arm in arms}
    paired = {
        "claim_logit_vs_random": paired_append(by_arm["claim_logit_append"], by_arm["claim_random_append"]),
        "claim_logit_vs_generic": paired_append(by_arm["claim_logit_append"], by_arm["p6x_generic_append"]),
        "claim_oracle_vs_question_oracle": paired_append(by_arm["claim_oracle_append"], by_arm["p7r_question_oracle_append"]),
        "claim_logit_vs_question_logit": paired_append(by_arm["claim_logit_append"], by_arm["p7r_question_logit_append"]),
    }
    node_counts = sorted(len(nodes_by_id[item.case.id]) for item in repairs)
    median_nodes = float(node_counts[len(node_counts) // 2])
    replay_exact = (
        math.isclose(absolute["p7r_question_oracle_append"]["str_hit"], 19 / 73, abs_tol=1e-12)
        and math.isclose(absolute["p7r_question_logit_append"]["str_hit"], 12 / 73, abs_tol=1e-12)
        and math.isclose(absolute["gold_oracle_append"]["str_hit"], 33 / 73, abs_tol=1e-12)
        and math.isclose(absolute["gold_logit_append"]["str_hit"], 29 / 73, abs_tol=1e-12)
        and math.isclose(absolute["p6x_generic_append"]["str_hit"], 7 / 73, abs_tol=1e-12)
    )
    valid_rate = valid_count / len(repairs)
    apparatus_gates = {
        "exact_counts_and_rescore": (
            len(eligible) == EXPECTED_ELIGIBLE and len(cases) == EXPECTED_CASES
            and len(p1x) == EXPECTED_P1X_ROWS and len(repairs) == 73 and exact_rescore
        ),
        "valid_rate_and_node_count": valid_rate >= 0.90 and 2.0 <= median_nodes <= 5.0,
        "finite_scores_and_full_denominator": all(len(by_arm[arm]) == 73 for arm in arms),
        "exact_frozen_replay": replay_exact,
        "automatic_preserves_prior_facets": absolute["claim_logit_append"]["all_present_preserved"] >= 0.98,
        "induction_prompt_excludes_saved_answer": all(item.direct_answer not in prompt for item, prompt in zip(repairs, prompts)),
    }
    claim_oracle = absolute["claim_oracle_append"]["str_hit"]
    gold_oracle = absolute["gold_oracle_append"]["str_hit"]
    claim_auto = absolute["claim_logit_append"]["str_hit"]
    gold_logit = absolute["gold_logit_append"]["str_hit"]
    auto_vs_generic = paired["claim_logit_vs_generic"]
    auto_vs_random = paired["claim_logit_vs_random"]
    ledger_gates = {
        "claim_action_oracle": claim_oracle >= 0.30 and claim_oracle >= 0.65 * gold_oracle,
        "automatic_action_absolute": claim_auto >= 0.20 and claim_auto >= 0.50 * gold_logit,
        "automatic_beats_generic": (
            auto_vs_generic["delta"] >= 0.10 and auto_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_generic["left_only_successes"] > auto_vs_generic["right_only_successes"]
        ),
        "automatic_beats_claim_random": (
            auto_vs_random["delta"] >= 0.10 and auto_vs_random["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_random["left_only_successes"] > auto_vs_random["right_only_successes"]
        ),
        "automatic_retains_60pct_claim_oracle": claim_oracle > 0 and claim_auto >= 0.60 * claim_oracle,
        "claim_oracle_improves_question_oracle_by_5_points": (
            paired["claim_oracle_vs_question_oracle"]["delta"] >= 0.05
        ),
    }
    if not all(apparatus_gates.values()):
        outcome = "APPARATUS_FAIL"
    elif all(ledger_gates.values()):
        outcome = "ANSWER_BEARING_OBLIGATION_PASS"
    elif ledger_gates["claim_action_oracle"]:
        outcome = "ANSWER_BEARING_ACTION_ONLY"
    else:
        outcome = "ANSWER_BEARING_OBLIGATION_FAIL"
    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "repair_cases": len(repairs), "valid_claim_sets": valid_count,
            "claim_candidates": len(candidate_rows),
        },
        "induction": {"valid_rate": valid_rate, "median_used_nodes": median_nodes},
        "absolute": absolute,
        "paired": paired,
        "apparatus_gates": apparatus_gates,
        "ledger_gates": ledger_gates,
        "interpretation_guard": (
            "P8x tests flat answer-bearing nodes and a direct append. It does not establish "
            "hierarchy, factual grounding, multi-step updates, end-to-end training, or novelty."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("asqa_answer_bearing_obligation_p8x_inductions.jsonl", induction_rows),
        ("asqa_answer_bearing_obligation_p8x_candidates.jsonl", candidate_rows),
    ]:
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_answer_bearing_obligation_p8x_selections.jsonl").open("w", encoding="utf-8") as handle:
        for arm in arms:
            for row in by_arm[arm]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_answer_bearing_obligation_p8x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--p7r-selections", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--induction-batch-size", type=int, default=4)
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
