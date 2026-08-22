"""Frozen automatic obligation-set induction and repair screen P7x."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from asqa_clean_fixed_support_p1x import aligned_clean_cases, hash_key, mean, render_documents, select_cases
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


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_AUTO_OBLIGATION_P7X.md"
RANDOM_SALT = "20260822-asqa-auto-obligation-p7x-random"
EXPECTED_P6X_ROWS = 438
REPLAY_ARMS = ("oracle_append", "logit_append", "generic_append")


def render_induction_prompt(item: SelectorCase) -> str:
    return (
        "Identify the distinct answer obligations needed for a complete, well-grounded answer "
        "to the ambiguous question below. Use the fixed documents to distinguish meanings, "
        "entities, or interpretations. Return exactly a JSON array of four concise standalone "
        "subquestions. Make them non-overlapping. Return the JSON array only.\n\n"
        f"Question: {item.case.question}\n\nFixed documents:\n{render_documents(item.case)}\n\n"
        "Four answer obligations:"
    )


def clean_obligation(value: str) -> str:
    value = re.sub(r"^\s*(?:\d+[\).:\-]|[-*])\s*", "", value.strip())
    return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'")


def valid_obligation_set(values: Any) -> list[str] | None:
    if not isinstance(values, list):
        return None
    cleaned = [clean_obligation(value) for value in values if isinstance(value, str)]
    unique = []
    seen = set()
    for value in cleaned:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            unique.append(value)
    if len(unique) != 4 or any(len(value.split()) > 35 for value in unique):
        return None
    return unique


def parse_obligations(text: str) -> tuple[list[str] | None, str]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = stripped.find("[")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[start:])
            parsed = valid_obligation_set(value)
            if parsed is not None:
                return parsed, "json"
        except json.JSONDecodeError:
            pass
    numbered = []
    for line in stripped.splitlines():
        match = re.match(r"^\s*(?:\d+[\).:\-]|[-*])\s+(.+?)\s*$", line)
        if match:
            numbered.append(match.group(1))
    parsed = valid_obligation_set(numbered)
    if parsed is not None:
        return parsed, "numbered"
    return None, "invalid"


def render_coverage_prompt(item: SelectorCase, obligation: str) -> str:
    return (
        "Check whether a saved answer explicitly covers one candidate interpretation. "
        "Reply with exactly one label and nothing else: A means COVERED; B means MISSING.\n\n"
        f"Question: {item.case.question}\n\nSaved answer: {item.direct_answer}\n\n"
        f"Candidate interpretation: {obligation}\n\nLabel:"
    )


def render_candidate_append_prompt(item: SelectorCase, obligation: str) -> str:
    return (
        f"Question: {item.case.question}\n\nFixed documents:\n{render_documents(item.case)}\n\n"
        f"Saved answer:\n{item.direct_answer}\n\nTask:\n"
        "Return only a short one- or two-sentence addition to append to the saved answer. "
        "Add the missing interpretation below using only the fixed documents. Do not repeat, "
        "rewrite, or contradict the saved answer, and do not add a preface.\n\n"
        f"Missing interpretation: {obligation}\n\nAddition:"
    )


def random_index(case_id: str, count: int) -> int:
    return int(hash_key(f"{RANDOM_SALT}|{case_id}")[:16], 16) % count


def load_p6x_generations(path: Path, case_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_P6X_ROWS:
        raise RuntimeError(f"expected {EXPECTED_P6X_ROWS} P6x rows, got {len(rows)}")
    selected = {
        (str(row["id"]), str(row["arm"])): row
        for row in rows
        if str(row["id"]) in case_ids and str(row["arm"]) in REPLAY_ARMS
    }
    expected = len(case_ids) * len(REPLAY_ARMS)
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} replay rows, got {len(selected)}")
    return selected


def choose_oracle(rows: list[dict[str, Any]]) -> int:
    return min(
        range(len(rows)),
        key=lambda index: (
            -int(bool(rows[index]["str_hit"])),
            -float(rows[index]["str_em"]),
            index,
        ),
    )


def relabel(row: dict[str, Any], arm: str, obligation: str, candidate_index: int, score: float | None):
    result = dict(row)
    result["arm"] = arm
    result["selected_obligation"] = obligation
    result["selected_candidate_index"] = candidate_index + 1
    result["selected_coverage_score"] = score
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    eligible = aligned_clean_cases(args.alce, args.original)
    eval_cases = select_cases(eligible, EXPECTED_CASES)
    p1x = load_p1x_rows(args.p1x_generations, eval_cases)
    direct = {case.id: p1x[(case.id, "fixed_direct")] for case in eval_cases}
    items, exact_rescore = build_selector_cases(eval_cases, direct)
    repairs = [item for item in items if item.exactly_one_missing]
    case_ids = {item.case.id for item in repairs}
    replay = load_p6x_generations(args.p6x_generations, case_ids)
    print(f"[apparatus] eligible={len(eligible)} eval={len(eval_cases)} repairs={len(repairs)}", flush=True)

    runner = ModelRunner(args.model)
    induction_prompts = [render_induction_prompt(item) for item in repairs]
    raw_inductions = runner.generate(induction_prompts, args.induction_batch_size, 256)
    obligation_sets: dict[str, list[str]] = {}
    induction_rows = []
    valid_count = 0
    for item, raw in zip(repairs, raw_inductions):
        parsed, parse_mode = parse_obligations(raw)
        valid = parsed is not None
        valid_count += int(valid)
        obligations = parsed if parsed is not None else [item.case.question]
        obligation_sets[item.case.id] = obligations
        induction_rows.append(
            {
                "id": item.case.id,
                "raw_induction": raw,
                "parse_mode": parse_mode,
                "valid_four_node_set": valid,
                "used_fallback": not valid,
                "obligations": obligations,
            }
        )
    print(f"[induction] valid={valid_count}/{len(repairs)}", flush=True)

    flat = [
        (item, index, obligation)
        for item in repairs
        for index, obligation in enumerate(obligation_sets[item.case.id])
    ]
    coverage_prompts = [render_coverage_prompt(item, obligation) for item, _, obligation in flat]
    _, coverage_scores = runner.extract_selector(coverage_prompts, (27,), args.selector_batch_size)
    if not np.isfinite(coverage_scores).all():
        raise RuntimeError("non-finite induced coverage score")
    append_prompts = [render_candidate_append_prompt(item, obligation) for item, _, obligation in flat]
    additions = runner.generate(append_prompts, args.generation_batch_size, 96)

    candidate_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_rows = []
    for flat_index, ((item, candidate_index, obligation), addition) in enumerate(zip(flat, additions)):
        scored = score_append(item, "induced_candidate", None, addition)
        scored.update(
            {
                "obligation": obligation,
                "candidate_index": candidate_index + 1,
                "coverage_score": float(coverage_scores[flat_index]),
            }
        )
        candidate_by_id[item.case.id].append(scored)
        candidate_rows.append(scored)

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in repairs:
        candidates = candidate_by_id[item.case.id]
        auto_index = min(range(len(candidates)), key=lambda index: (-candidates[index]["coverage_score"], index))
        random_choice = random_index(item.case.id, len(candidates))
        oracle_index = choose_oracle(candidates)
        selections = {
            "induced_logit_append": auto_index,
            "induced_random_append": random_choice,
            "induced_oracle_append": oracle_index,
        }
        for arm, index in selections.items():
            row = candidates[index]
            by_arm[arm].append(
                relabel(row, arm, row["obligation"], index, row["coverage_score"] if arm != "induced_oracle_append" else None)
            )
        for source_arm, target_arm in [
            ("oracle_append", "gold_oracle_append"),
            ("logit_append", "gold_logit_append"),
            ("generic_append", "p6x_generic_append"),
        ]:
            row = dict(replay[(item.case.id, source_arm)])
            row["arm"] = target_arm
            by_arm[target_arm].append(row)

    arms = (
        "induced_logit_append",
        "induced_random_append",
        "induced_oracle_append",
        "gold_oracle_append",
        "gold_logit_append",
        "p6x_generic_append",
    )
    absolute = {arm: append_metrics(by_arm[arm]) for arm in arms}
    paired = {
        "induced_logit_vs_random": paired_append(by_arm["induced_logit_append"], by_arm["induced_random_append"]),
        "induced_logit_vs_generic": paired_append(by_arm["induced_logit_append"], by_arm["p6x_generic_append"]),
        "induced_logit_vs_gold_logit": paired_append(by_arm["induced_logit_append"], by_arm["gold_logit_append"]),
        "induced_oracle_vs_gold_oracle": paired_append(by_arm["induced_oracle_append"], by_arm["gold_oracle_append"]),
    }
    valid_rate = valid_count / len(repairs)
    node_counts = sorted(len(obligation_sets[item.case.id]) for item in repairs)
    median_nodes = float(node_counts[len(node_counts) // 2])
    replay_exact = (
        math.isclose(absolute["gold_oracle_append"]["str_hit"], 33 / 73, abs_tol=1e-12)
        and math.isclose(absolute["gold_logit_append"]["str_hit"], 29 / 73, abs_tol=1e-12)
        and math.isclose(absolute["p6x_generic_append"]["str_hit"], 7 / 73, abs_tol=1e-12)
    )
    apparatus_gates = {
        "exact_counts_and_rescore": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(eval_cases) == EXPECTED_CASES
            and len(p1x) == EXPECTED_P1X_ROWS
            and len(repairs) == 73
            and len(flat) == len(candidate_rows)
            and exact_rescore
        ),
        "valid_induction_rate_and_median_nodes": valid_rate >= 0.90 and median_nodes == 4.0,
        "full_denominator_and_finite_scores": all(len(by_arm[arm]) == 73 for arm in arms),
        "exact_p6x_replay": replay_exact,
        "induction_prompt_excludes_saved_answer": all(item.direct_answer not in prompt for item, prompt in zip(repairs, induction_prompts)),
    }
    induced_oracle = absolute["induced_oracle_append"]["str_hit"]
    gold_oracle = absolute["gold_oracle_append"]["str_hit"]
    induced_auto = absolute["induced_logit_append"]["str_hit"]
    gold_logit = absolute["gold_logit_append"]["str_hit"]
    auto_vs_generic = paired["induced_logit_vs_generic"]
    auto_vs_random = paired["induced_logit_vs_random"]
    ledger_gates = {
        "induced_action_oracle": induced_oracle >= 0.30 and induced_oracle >= 0.65 * gold_oracle,
        "automatic_action_absolute": (
            induced_auto >= 0.20
            and induced_auto >= 0.50 * gold_logit
            and absolute["induced_logit_append"]["all_present_preserved"] >= 0.98
        ),
        "automatic_beats_generic": (
            auto_vs_generic["delta"] >= 0.10
            and auto_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_generic["left_only_successes"] > auto_vs_generic["right_only_successes"]
        ),
        "automatic_beats_induced_random": (
            auto_vs_random["delta"] >= 0.10
            and auto_vs_random["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_random["left_only_successes"] > auto_vs_random["right_only_successes"]
        ),
        "automatic_retains_60pct_induced_oracle": induced_oracle > 0 and induced_auto >= 0.60 * induced_oracle,
    }
    apparatus_pass = all(apparatus_gates.values())
    oracle_pass = ledger_gates["induced_action_oracle"]
    if not apparatus_pass:
        outcome = "APPARATUS_FAIL"
    elif all(ledger_gates.values()):
        outcome = "AUTO_OBLIGATION_LEDGER_PASS"
    elif oracle_pass:
        outcome = "INDUCED_ACTION_ONLY"
    else:
        outcome = "AUTO_OBLIGATION_FAIL"
    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "eligible": len(eligible),
            "eval_cases": len(eval_cases),
            "repair_cases": len(repairs),
            "valid_four_node_sets": valid_count,
            "induced_candidates": len(candidate_rows),
        },
        "induction": {"valid_rate": valid_rate, "median_used_nodes": median_nodes},
        "absolute": absolute,
        "paired": paired,
        "apparatus_gates": apparatus_gates,
        "ledger_gates": ledger_gates,
        "interpretation_guard": (
            "P7x tests automatically induced flat obligation sets and one bounded repair. It does "
            "not establish hierarchy, multi-step state updates, end-to-end training, or novelty."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_auto_obligation_p7x_inductions.jsonl").open("w", encoding="utf-8") as handle:
        for row in induction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_auto_obligation_p7x_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_auto_obligation_p7x_selections.jsonl").open("w", encoding="utf-8") as handle:
        for arm in arms:
            for row in by_arm[arm]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_auto_obligation_p7x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--p6x-generations", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--induction-batch-size", type=int, default=4)
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
