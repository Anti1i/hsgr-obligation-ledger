"""Calibrate relation judges before interpreting GAMUT process-repair results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import (
    ProcessCase,
    _score_prompts,
    build_process_cases,
    load_rows,
    negative_process_answer,
    positive_process_answer,
)
from gamut_process_repair_p0b import release_runner


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_RELATION_JUDGE_P0C.md"


def direct_prompt(case: ProcessCase, answer: str, *, swapped: bool, include_evidence: bool) -> str:
    mapping = (
        "A means NOT MET. B means MET."
        if swapped
        else "A means MET. B means NOT MET."
    )
    evidence = f"\n\nGold evidence:\n{case.evidence}" if include_evidence else ""
    return (
        "Judge whether the ANSWER meets the requirement. Judge only what the answer says; the "
        "requirement describes the expected relation. Reply with exactly A or B and nothing else. "
        f"{mapping}{evidence}\n\nAnswer:\n{answer}\n\n"
        f"Requirement: {case.target.ingredient}\n\nLabel:"
    )


def combine_counterbalanced(normal_score: float, swapped_score: float) -> float:
    """Positive means MET; inputs are the shared scorer's B-minus-A logits."""
    return (-normal_score + swapped_score) / 2.0


def extraction_prompt(case: ProcessCase, answer: str) -> str:
    steps = "\n".join(f"- P{index}: {step}" for index, step in enumerate(case.steps, 1))
    return (
        "Read only the answer below. Find which listed process steps it explicitly states, including "
        "clear paraphrases. Return their IDs in the order in which the answer states that they occur, "
        "not the order of this reference list. Do not infer a missing step from general knowledge. "
        "Return exactly one JSON object with one field named matched_order whose value is a list of "
        "step-ID strings. Return no markdown or explanation.\n\n"
        f"Reference steps:\n{steps}\n\nAnswer:\n{answer}\n\nJSON:"
    )


def parse_matched_order(text: str, step_count: int) -> tuple[list[str], bool, str]:
    stripped = text.strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return [], False, "no_json"
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return [], False, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"matched_order"}:
        return [], False, "wrong_fields"
    order = value["matched_order"]
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        return [], False, "not_string_list"
    allowed = {f"P{index}" for index in range(1, step_count + 1)}
    if any(item not in allowed for item in order):
        return [], False, "unknown_step"
    if len(order) != len(set(order)):
        return [], False, "duplicate_step"
    return order, True, "valid"


def relation_met(order: list[str]) -> bool:
    indices = [int(item[1:]) for item in order]
    return all(left < right for left, right in zip(indices, indices[1:]))


def summarize_controls(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    positive = [row for row in rows if row["answer_kind"] == "positive"]
    negative = [row for row in rows if row["answer_kind"] == "negative"]
    parse_validity = sum(row["valid"] for row in rows) / max(1, len(rows))
    positive_accuracy = sum(row["valid"] and row["relation_met"] for row in positive) / max(1, len(positive))
    negative_accuracy = sum(row["valid"] and not row["relation_met"] for row in negative) / max(1, len(negative))
    usable = positive_accuracy >= 0.95 and negative_accuracy >= 0.95
    if method == "extract_then_check":
        usable = usable and parse_validity >= 0.95
    return {
        "n_positive": len(positive),
        "n_negative": len(negative),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "parse_validity": parse_validity,
        "usable": usable,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_rows = [json.loads(line) for line in args.baseline_jsonl.read_text(encoding="utf-8").splitlines()]
    rows = load_rows(str(args.dataset), args.split)
    cases, _ = build_process_cases(rows, args.skip + args.n)
    case_by_id = {case.id: case for case in cases[args.skip : args.skip + args.n]}
    if set(case_by_id) != {row["id"] for row in baseline_rows} or len(baseline_rows) != args.n:
        raise RuntimeError("baseline IDs do not exactly match the frozen P0b slice")

    judge = ModelRunner(args.judge_model)
    controls_by_method: dict[str, list[dict[str, Any]]] = {}
    natural_by_method: dict[str, list[dict[str, Any]]] = {}

    for method, include_evidence in (
        ("counterbalanced_with_evidence", True),
        ("counterbalanced_answer_only", False),
    ):
        keys: list[tuple[str, str]] = []
        prompts: list[str] = []
        answers_by_id = {row["id"]: row["answer"] for row in baseline_rows}
        for case_id, case in case_by_id.items():
            for kind, answer in (
                ("positive", positive_process_answer(case.steps)),
                ("negative", negative_process_answer(case.steps)),
                ("natural", answers_by_id[case_id]),
            ):
                keys.append((case_id, kind))
                prompts.extend(
                    [
                        direct_prompt(case, answer, swapped=False, include_evidence=include_evidence),
                        direct_prompt(case, answer, swapped=True, include_evidence=include_evidence),
                    ]
                )
        scores = _score_prompts(judge, prompts, args.judge_batch_size)
        method_rows: list[dict[str, Any]] = []
        for index, (case_id, kind) in enumerate(keys):
            margin = combine_counterbalanced(scores[2 * index], scores[2 * index + 1])
            method_rows.append(
                {
                    "id": case_id,
                    "method": method,
                    "answer_kind": kind,
                    "valid": True,
                    "relation_met": margin > 0,
                    "semantic_pass_margin": margin,
                }
            )
        controls_by_method[method] = [row for row in method_rows if row["answer_kind"] != "natural"]
        natural_by_method[method] = [row for row in method_rows if row["answer_kind"] == "natural"]

    extraction_keys: list[tuple[str, str]] = []
    extraction_prompts: list[str] = []
    answers_by_id = {row["id"]: row["answer"] for row in baseline_rows}
    for case_id, case in case_by_id.items():
        for kind, answer in (
            ("positive", positive_process_answer(case.steps)),
            ("negative", negative_process_answer(case.steps)),
            ("natural", answers_by_id[case_id]),
        ):
            extraction_keys.append((case_id, kind))
            extraction_prompts.append(extraction_prompt(case, answer))
    outputs = judge.generate(extraction_prompts, args.generation_batch_size, 96)
    extraction_rows: list[dict[str, Any]] = []
    for (case_id, kind), raw in zip(extraction_keys, outputs):
        case = case_by_id[case_id]
        order, valid, parse_mode = parse_matched_order(raw, len(case.steps))
        extraction_rows.append(
            {
                "id": case_id,
                "method": "extract_then_check",
                "answer_kind": kind,
                "valid": valid,
                "parse_mode": parse_mode,
                "matched_order": order,
                "relation_met": valid and relation_met(order),
                "all_components_present": valid and len(order) == len(case.steps),
                "raw_output": raw,
            }
        )
    controls_by_method["extract_then_check"] = [row for row in extraction_rows if row["answer_kind"] != "natural"]
    natural_by_method["extract_then_check"] = [row for row in extraction_rows if row["answer_kind"] == "natural"]
    release_runner(judge)

    original_controls = {
        "positive_accuracy": sum(row["positive_control_score"] < 0 for row in baseline_rows) / len(baseline_rows),
        "negative_accuracy": sum(row["negative_control_score"] >= 0 for row in baseline_rows) / len(baseline_rows),
        "parse_validity": 1.0,
        "usable": False,
    }
    controls = {"original_direct": original_controls}
    for method, method_rows in controls_by_method.items():
        controls[method] = summarize_controls(method_rows, method)

    extraction_natural = natural_by_method["extract_then_check"]
    relation_only_ids = sorted(
        row["id"]
        for row in extraction_natural
        if row["valid"] and row["all_components_present"] and not row["relation_met"]
    )
    usable_methods = [method for method, values in controls.items() if values["usable"]]
    natural_predictions = {
        method: {row["id"]: row["relation_met"] for row in method_rows if row["valid"]}
        for method, method_rows in natural_by_method.items()
        if method in usable_methods
    }
    disagreement_ids: list[str] = []
    if len(natural_predictions) >= 2:
        for case_id in case_by_id:
            values = [predictions.get(case_id) for predictions in natural_predictions.values()]
            if None not in values and len(set(values)) > 1:
                disagreement_ids.append(case_id)

    report = {
        "protocol": PROTOCOL,
        "judge_model": args.judge_model,
        "n_cases": len(baseline_rows),
        "controls": controls,
        "usable_methods": usable_methods,
        "extract_then_check_natural": {
            "parse_validity": sum(row["valid"] for row in extraction_natural) / len(extraction_natural),
            "relation_failures": sum(row["valid"] and not row["relation_met"] for row in extraction_natural),
            "relation_only_failures": len(relation_only_ids),
            "relation_only_ids": relation_only_ids,
        },
        "usable_judge_disagreement_count": len(disagreement_ids),
        "usable_judge_disagreement_ids": disagreement_ids,
        "problem_gate_at_least_four_relation_only": len(relation_only_ids) >= 4,
        "interpretation_guard": (
            "P0c calibrates a same-family judge on oracle-step controls. It is not human validation "
            "and does not measure repair performance or hidden-state control."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for rows in controls_by_method.values() for row in rows]
    all_rows.extend(row for rows in natural_by_method.values() for row in rows)
    with (args.out_dir / "gamut_relation_judge_p0c_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_relation_judge_p0c_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="test_text_only")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--skip", type=int, default=48)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

