"""Audit whether P0b's 256-token cap confounded GAMUT process errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import baseline_prompt, build_process_cases, load_rows
from gamut_process_repair_p0b import release_runner
from gamut_relation_judge_p0c import extraction_prompt, parse_matched_order, relation_met


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_LONG_BASELINE_P0D.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summarize(rows: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    return {
        "n": len(rows),
        "parse_validity": len(valid) / max(1, len(rows)),
        "mean_output_tokens": mean(row["output_tokens"] for row in rows),
        "cap_hit_proxy_rate": mean(row["output_tokens"] >= cap - 2 for row in rows),
        "all_components_present_rate": mean(row["all_components_present"] for row in rows),
        "relation_failure_rate": mean(row["valid"] and not row["relation_met"] for row in rows),
        "relation_only_count": sum(
            row["valid"] and row["all_components_present"] and not row["relation_met"]
            for row in rows
        ),
        "relation_only_ids": sorted(
            row["id"]
            for row in rows
            if row["valid"] and row["all_components_present"] and not row["relation_met"]
        ),
    }


def paired_summary(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old = {row["id"]: row for row in old_rows}
    new = {row["id"]: row for row in new_rows}
    ids = sorted(old)
    if set(ids) != set(new):
        raise RuntimeError("old/new row IDs differ")
    return {
        "all_components_gained": sum(
            not old[case_id]["all_components_present"] and new[case_id]["all_components_present"]
            for case_id in ids
        ),
        "all_components_lost": sum(
            old[case_id]["all_components_present"] and not new[case_id]["all_components_present"]
            for case_id in ids
        ),
        "old_relation_failure_fixed": sum(
            old[case_id]["valid"] and not old[case_id]["relation_met"]
            and new[case_id]["valid"] and new[case_id]["relation_met"]
            for case_id in ids
        ),
        "new_relation_failure_created": sum(
            old[case_id]["valid"] and old[case_id]["relation_met"]
            and new[case_id]["valid"] and not new[case_id]["relation_met"]
            for case_id in ids
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_baselines = load_jsonl(args.old_baseline_jsonl)
    old_extractions = {
        row["id"]: row
        for row in load_jsonl(args.old_extraction_jsonl)
        if row["method"] == "extract_then_check" and row["answer_kind"] == "natural"
    }
    dataset_rows = load_rows(str(args.dataset), args.split)
    ordered, _ = build_process_cases(dataset_rows, args.skip + args.n)
    cases = ordered[args.skip : args.skip + args.n]
    case_by_id = {case.id: case for case in cases}
    old_answer_by_id = {row["id"]: row["answer"] for row in old_baselines}
    expected_ids = set(case_by_id)
    if set(old_answer_by_id) != expected_ids or set(old_extractions) != expected_ids:
        raise RuntimeError("frozen P0b/P0c IDs do not exactly match the requested slice")

    generator = ModelRunner(args.generator_model)
    new_answers = generator.generate(
        [baseline_prompt(case) for case in cases], args.generation_batch_size, args.new_cap
    )
    old_token_counts = {
        case_id: len(generator.tokenizer.encode(answer, add_special_tokens=False))
        for case_id, answer in old_answer_by_id.items()
    }
    new_token_counts = {
        case.id: len(generator.tokenizer.encode(answer, add_special_tokens=False))
        for case, answer in zip(cases, new_answers)
    }
    release_runner(generator)

    judge = ModelRunner(args.judge_model)
    outputs = judge.generate(
        [extraction_prompt(case, answer) for case, answer in zip(cases, new_answers)],
        args.judge_batch_size,
        96,
    )
    release_runner(judge)

    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    for case, new_answer, raw in zip(cases, new_answers, outputs):
        old_ext = old_extractions[case.id]
        old_rows.append(
            {
                "id": case.id,
                "arm": "old_256",
                "question": case.question,
                "steps": list(case.steps),
                "answer": old_answer_by_id[case.id],
                "output_tokens": old_token_counts[case.id],
                "valid": old_ext["valid"],
                "matched_order": old_ext["matched_order"],
                "relation_met": old_ext["relation_met"],
                "all_components_present": old_ext["all_components_present"],
            }
        )
        order, valid, parse_mode = parse_matched_order(raw, len(case.steps))
        new_rows.append(
            {
                "id": case.id,
                "arm": "new_768",
                "question": case.question,
                "steps": list(case.steps),
                "answer": new_answer.strip(),
                "output_tokens": new_token_counts[case.id],
                "valid": valid,
                "parse_mode": parse_mode,
                "matched_order": order,
                "relation_met": valid and relation_met(order),
                "all_components_present": valid and len(order) == len(case.steps),
                "raw_extraction": raw,
            }
        )

    old_summary = summarize(old_rows, args.old_cap)
    new_summary = summarize(new_rows, args.new_cap)
    all_steps_delta = new_summary["all_components_present_rate"] - old_summary["all_components_present_rate"]
    cap_hit_reduction = old_summary["cap_hit_proxy_rate"] - new_summary["cap_hit_proxy_rate"]
    report = {
        "protocol": PROTOCOL,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "old_256": old_summary,
        "new_768": new_summary,
        "paired": paired_summary(old_rows, new_rows),
        "deltas": {
            "all_components_present": all_steps_delta,
            "cap_hit_proxy_reduction": cap_hit_reduction,
        },
        "length_confound_gate": all_steps_delta >= 0.10 or cap_hit_reduction >= 0.20,
        "automatic_problem_gate_at_least_four_relation_only": new_summary["relation_only_count"] >= 4,
        "manual_review_required": new_summary["relation_only_ids"],
        "interpretation_guard": (
            "P0d audits answer-length confounding with oracle-step, same-family extraction. It does "
            "not measure repair quality, hidden-state control, automatic induction, or novelty."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "gamut_long_baseline_p0d_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in old_rows + new_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_long_baseline_p0d_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-baseline-jsonl", type=Path, required=True)
    parser.add_argument("--old-extraction-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="test_text_only")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--skip", type=int, default=48)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--old-cap", type=int, default=256)
    parser.add_argument("--new-cap", type=int, default=768)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

