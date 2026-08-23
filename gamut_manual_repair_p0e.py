"""Four-case GAMUT repair comparison after manual relation-error confirmation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import build_process_cases, edit_ratio, load_rows, typed_graph
from gamut_process_repair_p0b import (
    ARMS,
    numbered_sentences,
    parse_sentence_patch,
    release_runner,
)
from gamut_relation_judge_p0c import extraction_prompt, parse_matched_order, relation_met


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_MANUAL_REPAIR_P0E.md"
CONFIRMED_IDS = (
    "22cab127-2c4e-4b8e-9590-76d8d7ada2cd",
    "8dab7a6d-13e5-4081-884e-eced5b4cf615",
    "91ed57a0-bdf6-4111-a689-955e47280cb2",
    "afe5977f-01b0-4f5d-acea-842349f3d37b",
)


def flat_guide(steps: tuple[str, ...]) -> str:
    return "Required stages in canonical order:\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, 1)
    )


def repair_prompt(case: Any, answer: str, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    guide = typed_graph(case) if arm.startswith("typed_") else flat_guide(case.steps)
    if arm.endswith("sentence_patch"):
        saved = f"Saved answer split into numbered sentences:\n{numbered_sentences(answer)}"
        task = (
            "Return exactly one JSON object with start_sentence and end_sentence as one-based "
            "integers and replacement as a string. Replace at most four consecutive sentences. "
            "Make the smallest correction that makes the process follow the guide, while retaining "
            "all process stages and every unrelated fact. Return no markdown or explanation."
        )
    else:
        saved = f"Saved answer:\n{answer}"
        task = (
            "Return only the complete revised answer. Make the smallest correction that makes the "
            "process follow the guide, while retaining all process stages and every unrelated fact."
        )
    return (
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n{saved}\n\n"
        f"Process guide:\n{guide}\n\nTask:\n{task}"
    )


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "valid_extraction": mean(row["valid_extraction"] for row in rows),
        "all_components_present": mean(row["all_components_present"] for row in rows),
        "structural_safe_success": mean(row["structural_safe_success"] for row in rows),
        "patch_valid": mean(row["patch_valid"] for row in rows),
        "median_edit_ratio": median(row["edit_ratio"] for row in rows),
        "mean_edit_ratio": mean(row["edit_ratio"] for row in rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_rows(str(args.dataset), args.split)
    cases, _ = build_process_cases(source_rows, args.skip + args.n)
    case_by_id = {case.id: case for case in cases}
    if not set(CONFIRMED_IDS).issubset(case_by_id):
        raise RuntimeError("one or more frozen manual case IDs are absent")
    p0d_rows = [
        json.loads(line) for line in args.p0d_rows.read_text(encoding="utf-8").splitlines()
    ]
    baseline_by_id = {
        row["id"]: row["answer"]
        for row in p0d_rows
        if row["arm"] == "new_768" and row["id"] in CONFIRMED_IDS
    }
    if set(baseline_by_id) != set(CONFIRMED_IDS):
        raise RuntimeError("frozen P0d long answers are incomplete")

    generation_keys: list[tuple[str, str]] = []
    prompts: list[str] = []
    for case_id in CONFIRMED_IDS:
        case = case_by_id[case_id]
        for arm in ARMS:
            generation_keys.append((case_id, arm))
            prompts.append(repair_prompt(case, baseline_by_id[case_id], arm))
    generator = ModelRunner(args.generator_model)
    raw_outputs = generator.generate(prompts, args.generation_batch_size, 768)
    release_runner(generator)

    candidates: list[dict[str, Any]] = []
    for (case_id, arm), raw in zip(generation_keys, raw_outputs):
        original = baseline_by_id[case_id]
        if arm.endswith("sentence_patch"):
            answer, patch_valid, parse_mode, span = parse_sentence_patch(raw, original)
        else:
            answer, patch_valid, parse_mode, span = raw.strip(), bool(raw.strip()), "full_answer", None
        candidates.append(
            {
                "id": case_id,
                "arm": arm,
                "question": case_by_id[case_id].question,
                "steps": list(case_by_id[case_id].steps),
                "original_answer": original,
                "raw_output": raw,
                "answer": answer,
                "patch_valid": patch_valid,
                "parse_mode": parse_mode,
                "sentence_span": span,
                "edit_ratio": edit_ratio(original, answer),
            }
        )

    judge = ModelRunner(args.judge_model)
    extraction_outputs = judge.generate(
        [extraction_prompt(case_by_id[row["id"]], row["answer"]) for row in candidates],
        args.judge_batch_size,
        96,
    )
    release_runner(judge)
    for row, raw in zip(candidates, extraction_outputs):
        order, valid, mode = parse_matched_order(raw, len(row["steps"]))
        row.update(
            {
                "extraction_raw": raw,
                "extraction_parse_mode": mode,
                "valid_extraction": valid,
                "matched_order": order,
                "relation_met": valid and relation_met(order),
                "all_components_present": valid and len(order) == len(row["steps"]),
                "structural_safe_success": (
                    row["patch_valid"] and valid and len(order) == len(row["steps"])
                    and relation_met(order)
                ),
            }
        )

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_arm[row["arm"]].append(row)
    absolute = {arm: arm_metrics(by_arm[arm]) for arm in ARMS}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_case[row["id"]].append(row)
    oracle_rate = mean(any(row["structural_safe_success"] for row in rows) for rows in by_case.values())

    pairwise: dict[str, Any] = {}
    for form in ("full_rewrite", "sentence_patch"):
        flat = {row["id"]: row for row in by_arm[f"flat_{form}"]}
        typed = {row["id"]: row for row in by_arm[f"typed_{form}"]}
        pairwise[form] = {
            "typed_only_successes": sum(
                typed[case_id]["structural_safe_success"] and not flat[case_id]["structural_safe_success"]
                for case_id in CONFIRMED_IDS
            ),
            "flat_only_successes": sum(
                flat[case_id]["structural_safe_success"] and not typed[case_id]["structural_safe_success"]
                for case_id in CONFIRMED_IDS
            ),
        }
    typed_only_total = sum(values["typed_only_successes"] for values in pairwise.values())
    flat_only_total = sum(values["flat_only_successes"] for values in pairwise.values())
    minimal_success_cases = {
        row["id"] for row in candidates
        if row["arm"].endswith("sentence_patch")
        and row["structural_safe_success"] and row["edit_ratio"] <= 0.25
    }
    report = {
        "protocol": PROTOCOL,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "confirmed_case_ids": list(CONFIRMED_IDS),
        "absolute": absolute,
        "pairwise": pairwise,
        "action_oracle_structural_safe_success": oracle_rate,
        "automatic_gates_before_manual_preservation": {
            "action_space_viable_at_least_two_of_four": oracle_rate >= 0.5,
            "typed_representation_suggestive": typed_only_total >= 2 and typed_only_total > flat_only_total,
            "minimal_patch_viable_at_least_two_cases": len(minimal_success_cases) >= 2,
        },
        "minimal_success_case_ids": sorted(minimal_success_cases),
        "manual_review_required_for_all_16_outputs": True,
        "interpretation_guard": (
            "P0e is a four-case oracle-structure mechanism check. Automatic structural success is "
            "not a claim of factual preservation, significance, hidden control, or benchmark gain."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "gamut_manual_repair_p0e_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_manual_repair_p0e_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--p0d-rows", type=Path, required=True)
    parser.add_argument("--split", default="test_text_only")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--skip", type=int, default=48)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

