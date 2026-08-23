"""Fresh GAMUT process-repair replication with an independent 14B judge."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import (
    ProcessCase,
    Requirement,
    SELECTION_SALT,
    _score_prompts,
    baseline_prompt,
    build_process_cases,
    component_requirement,
    edit_ratio,
    hash_key,
    judge_prompt,
    load_rows,
    mean,
    metrics,
    negative_process_answer,
    positive_process_answer,
    preservation_text,
    typed_graph,
)


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_PROCESS_REPAIR_P0B.md"
ARMS = (
    "flat_full_rewrite",
    "typed_full_rewrite",
    "flat_sentence_patch",
    "typed_sentence_patch",
)


def release_runner(runner: ModelRunner) -> None:
    torch = runner.torch
    model = runner.model
    runner.model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()


def split_sentences(answer: str) -> list[str]:
    compact = re.sub(r"[ \t]+", " ", answer.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9]|\[))", compact)
    return [part.strip() for part in parts if part.strip()]


def numbered_sentences(answer: str) -> str:
    return "\n".join(f"[S{index}] {sentence}" for index, sentence in enumerate(split_sentences(answer), 1))


def parse_sentence_patch(text: str, answer: str) -> tuple[str, bool, str, tuple[int, int] | None]:
    stripped = text.strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return answer, False, "no_json_object", None
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return answer, False, "invalid_json", None
    expected = {"start_sentence", "end_sentence", "replacement"}
    if not isinstance(value, dict) or set(value) != expected:
        return answer, False, "wrong_fields", None
    first, last, replacement = (
        value["start_sentence"], value["end_sentence"], value["replacement"]
    )
    sentences = split_sentences(answer)
    if not isinstance(first, int) or isinstance(first, bool) or not isinstance(last, int) or isinstance(last, bool):
        return answer, False, "non_integer_index", None
    if not (1 <= first <= last <= len(sentences)):
        return answer, False, "index_out_of_range", None
    if last - first + 1 > 4:
        return answer, False, "span_over_four_sentences", None
    if not isinstance(replacement, str) or not replacement.strip():
        return answer, False, "empty_replacement", None
    if len(replacement.split()) > 180:
        return answer, False, "replacement_too_long", None
    revised = sentences[: first - 1] + [replacement.strip()] + sentences[last:]
    return " ".join(revised), True, "valid", (first, last)


def repair_prompt(case: ProcessCase, answer: str, arm: str, preserved: list[Requirement]) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    typed = arm.startswith("typed_")
    patch = arm.endswith("sentence_patch")
    target = (
        f"Failed requirement in ordinary text:\n{case.target.ingredient}"
        if not typed
        else f"Failed typed obligation:\n{typed_graph(case)}\n\nOriginal wording:\n{case.target.ingredient}"
    )
    if patch:
        saved = f"Saved answer split into numbered sentences:\n{numbered_sentences(answer)}"
        output = (
            "Return only one JSON object with exactly three fields: start_sentence and "
            "end_sentence as one-based integers, and replacement as a string. Select at most "
            "four consecutive sentences and replace them with the smallest passage that fixes "
            "the failed requirement while preserving every satisfied requirement. Do not return "
            "markdown or commentary."
        )
    else:
        saved = f"Saved answer:\n{answer}"
        output = (
            "Return only the complete revised answer. Make no unnecessary changes and do not "
            "mention these instructions."
        )
    return (
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n{saved}\n\n"
        f"{target}\n\nRequirements already satisfied in the saved answer and required to remain true:\n"
        f"{preservation_text(preserved)}\n\nTask:\n{output}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.dataset, args.split)
    ordered, source_audit = build_process_cases(rows, args.skip + args.n)
    p0_ids = {case.id for case in ordered[: args.skip]}
    cases = ordered[args.skip : args.skip + args.n]
    audit = dict(source_audit)
    audit.update(
        {
            "skip_p0_cases": args.skip,
            "fresh_selected_cases": len(cases),
            "p0_fresh_overlap": len(p0_ids & {case.id for case in cases}),
            "fresh_id_sha256": hash_key("\n".join(case.id for case in cases) + "\n"),
            "selection_salt": SELECTION_SALT,
        }
    )
    print(f"[audit] {json.dumps(audit, sort_keys=True)}", flush=True)

    generator = ModelRunner(args.generator_model)
    baseline_answers = generator.generate(
        [baseline_prompt(case) for case in cases], args.generation_batch_size, 256
    )
    baseline_by_id = {case.id: answer.strip() for case, answer in zip(cases, baseline_answers)}
    release_runner(generator)
    del generator

    judge = ModelRunner(args.judge_model)
    checks: list[tuple[str, str, str, str]] = []
    prompts: list[str] = []
    for case in cases:
        answer = baseline_by_id[case.id]
        for index, requirement in enumerate(case.answer_critical):
            checks.append((case.id, "ac", str(index), requirement.handle))
            prompts.append(judge_prompt(case, answer, requirement.handle, requirement.ingredient))
        for index, step in enumerate(case.steps):
            checks.append((case.id, "component", str(index), f"P{index + 1}"))
            prompts.append(judge_prompt(case, answer, f"P{index + 1}", component_requirement(step)))
        checks.extend(
            [
                (case.id, "control", "positive", case.target.handle),
                (case.id, "control", "negative", case.target.handle),
            ]
        )
        prompts.extend(
            [
                judge_prompt(case, positive_process_answer(case.steps), case.target.handle, case.target.ingredient),
                judge_prompt(case, negative_process_answer(case.steps), case.target.handle, case.target.ingredient),
            ]
        )
    baseline_scores = _score_prompts(judge, prompts, args.judge_batch_size)
    score_map = {key: value for key, value in zip(checks, baseline_scores)}
    release_runner(judge)
    del judge

    baseline_rows: list[dict[str, Any]] = []
    eligible: list[tuple[ProcessCase, list[int], bool]] = []
    positive_ok = negative_ok = 0
    for case in cases:
        target_index = case.answer_critical.index(case.target)
        ac_scores = [
            score_map[(case.id, "ac", str(index), requirement.handle)]
            for index, requirement in enumerate(case.answer_critical)
        ]
        component_scores = [
            score_map[(case.id, "component", str(index), f"P{index + 1}")]
            for index in range(len(case.steps))
        ]
        target_met = ac_scores[target_index] < 0
        component_met = [value < 0 for value in component_scores]
        met_indices = [
            index for index, value in enumerate(ac_scores) if value < 0 and index != target_index
        ]
        relation_only = not target_met and all(component_met)
        repair_eligible = (
            not target_met
            and sum(component_met) >= min(2, len(component_met))
            and sum(not value for value in component_met) <= 1
            and len(met_indices) >= 1
        )
        positive_score = score_map[(case.id, "control", "positive", case.target.handle)]
        negative_score = score_map[(case.id, "control", "negative", case.target.handle)]
        positive_ok += int(positive_score < 0)
        negative_ok += int(negative_score >= 0)
        if repair_eligible:
            eligible.append((case, met_indices, relation_only))
        baseline_rows.append(
            {
                "id": case.id,
                "question": case.question,
                "target_handle": case.target.handle,
                "target_requirement": case.target.ingredient,
                "steps": list(case.steps),
                "answer": baseline_by_id[case.id],
                "target_score": ac_scores[target_index],
                "target_met": target_met,
                "component_scores": component_scores,
                "component_met": component_met,
                "originally_met_indices": met_indices,
                "positive_control_score": positive_score,
                "negative_control_score": negative_score,
                "relation_only_failure": relation_only,
                "repair_eligible": repair_eligible,
            }
        )
    relation_only_count = sum(item[2] for item in eligible)
    print(
        f"[baseline] fresh={len(cases)} eligible={len(eligible)} relation_only={relation_only_count}",
        flush=True,
    )

    eligible_by_id = {case.id: (case, met_indices, relation_only) for case, met_indices, relation_only in eligible}
    generation_keys: list[tuple[str, str]] = []
    repair_prompts: list[str] = []
    for case, met_indices, _ in eligible:
        preserved = [case.answer_critical[index] for index in met_indices]
        for arm in ARMS:
            generation_keys.append((case.id, arm))
            repair_prompts.append(repair_prompt(case, baseline_by_id[case.id], arm, preserved))

    generator = ModelRunner(args.generator_model)
    raw_outputs = generator.generate(repair_prompts, args.generation_batch_size, 256) if repair_prompts else []
    release_runner(generator)
    del generator
    candidates: list[dict[str, Any]] = []
    for (case_id, arm), raw in zip(generation_keys, raw_outputs):
        case, met_indices, relation_only = eligible_by_id[case_id]
        baseline = baseline_by_id[case_id]
        if arm.endswith("sentence_patch"):
            answer, valid, parse_mode, span = parse_sentence_patch(raw, baseline)
        else:
            answer, valid, parse_mode, span = raw.strip(), bool(raw.strip()), "full_answer", None
            if not valid:
                answer = baseline
        candidates.append(
            {
                "id": case_id,
                "arm": arm,
                "relation_only_failure": relation_only,
                "raw_output": raw,
                "answer": answer,
                "patch_valid": valid,
                "parse_mode": parse_mode,
                "sentence_span": list(span) if span else None,
                "edit_ratio": edit_ratio(baseline, answer),
                "met_indices": met_indices,
            }
        )

    judge = ModelRunner(args.judge_model)
    final_specs: list[tuple[int, str, int]] = []
    final_prompts: list[str] = []
    for row_index, candidate in enumerate(candidates):
        case, met_indices, _ = eligible_by_id[candidate["id"]]
        target_index = case.answer_critical.index(case.target)
        final_specs.append((row_index, "target", target_index))
        final_prompts.append(judge_prompt(case, candidate["answer"], case.target.handle, case.target.ingredient))
        for index in met_indices:
            requirement = case.answer_critical[index]
            final_specs.append((row_index, "preserve", index))
            final_prompts.append(judge_prompt(case, candidate["answer"], requirement.handle, requirement.ingredient))
    final_scores = _score_prompts(judge, final_prompts, args.judge_batch_size)
    release_runner(judge)
    del judge
    scored: dict[int, dict[str, Any]] = defaultdict(lambda: {"preserve": []})
    for (row_index, kind, index), value in zip(final_specs, final_scores):
        if kind == "target":
            scored[row_index]["target"] = value
        else:
            scored[row_index]["preserve"].append((index, value))
    for row_index, candidate in enumerate(candidates):
        target_recovered = scored[row_index].get("target", 1.0) < 0
        preservation = [value < 0 for _, value in scored[row_index]["preserve"]]
        preservation_rate = mean(float(value) for value in preservation)
        no_regression = all(preservation)
        candidate.update(
            {
                "target_score": scored[row_index].get("target"),
                "target_recovered": target_recovered,
                "preservation_scores": [value for _, value in scored[row_index]["preserve"]],
                "preservation_rate": preservation_rate,
                "no_regression": no_regression,
                "safe_success": target_recovered and no_regression,
            }
        )

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_arm[candidate["arm"]].append(candidate)
        by_case[candidate["id"]].append(candidate)
    absolute = {arm: metrics(by_arm[arm]) for arm in ARMS}
    relation_only = {
        arm: metrics([row for row in by_arm[arm] if row["relation_only_failure"]])
        for arm in ARMS
    }
    oracle_hits = {
        case_id: any(row["safe_success"] for row in case_rows)
        for case_id, case_rows in by_case.items()
    }
    disagreement_cases = sum(
        len({bool(row["safe_success"]) for row in case_rows}) > 1
        for case_rows in by_case.values()
    )
    distinct_candidates = [
        len({re.sub(r"\s+", " ", row["answer"]).strip() for row in case_rows})
        for case_rows in by_case.values()
    ]
    oracle_rate = mean(float(value) for value in oracle_hits.values())
    best_fixed = max((absolute[arm]["safe_success"] for arm in ARMS), default=0.0)
    positive_accuracy = positive_ok / max(1, len(cases))
    negative_accuracy = negative_ok / max(1, len(cases))
    apparatus_gates = {
        "exactly_192_fresh_cases": len(cases) == 192,
        "zero_p0_overlap": audit["p0_fresh_overlap"] == 0,
        "positive_control_at_least_95pct": positive_accuracy >= 0.95,
        "negative_control_at_least_95pct": negative_accuracy >= 0.95,
        "complete_candidate_denominator": all(len(by_arm[arm]) == len(eligible) for arm in ARMS),
    }
    problem_gate = relation_only_count >= 4
    action_gate = oracle_rate >= 0.30
    flat_full_edit = absolute["flat_full_rewrite"]["median_edit_ratio"]
    typed_patch = absolute["typed_sentence_patch"]
    typed_gate = (
        typed_patch["safe_success"] >= 0.25
        and typed_patch["no_regression"] >= 0.85
        and typed_patch["safe_success"] - absolute["flat_sentence_patch"]["safe_success"] >= 0.05
        and (flat_full_edit == 0.0 or typed_patch["median_edit_ratio"] <= 0.50 * flat_full_edit)
    )
    hidden_gate = oracle_rate - best_fixed >= 0.10 and disagreement_cases >= 3
    if not all(apparatus_gates.values()):
        outcome = "APPARATUS_FAIL"
    elif not problem_gate:
        outcome = "PROBLEM_NOT_ESTABLISHED"
    elif not action_gate:
        outcome = "ACTION_SPACE_FAIL"
    elif typed_gate:
        outcome = "STRUCTURED_REPAIR_P0B_PASS"
    else:
        outcome = "REPAIR_WORKS_STRUCTURE_NOT_ADDED"

    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "audit": audit,
        "counts": {
            "baseline_cases": len(cases),
            "repair_cases": len(eligible),
            "relation_only_failures": relation_only_count,
            "disagreement_cases": disagreement_cases,
        },
        "controls": {"positive_accuracy": positive_accuracy, "negative_accuracy": negative_accuracy},
        "absolute": absolute,
        "relation_only": relation_only,
        "action_oracle": {
            "safe_success": oracle_rate,
            "best_fixed_safe_success": best_fixed,
            "oracle_minus_best_fixed": oracle_rate - best_fixed,
            "mean_distinct_candidates": mean(float(value) for value in distinct_candidates),
        },
        "apparatus_gates": apparatus_gates,
        "problem_gate_at_least_four_relation_only": problem_gate,
        "action_space_gate_oracle_at_least_30pct": action_gate,
        "typed_minimal_gate": typed_gate,
        "hidden_selector_gate": hidden_gate,
        "interpretation_guard": (
            "P0b is a fresh oracle-rubric case study with an independent same-family judge. It "
            "does not establish human validity, automatic induction, hidden control, or novelty."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, output_rows in [
        ("gamut_process_repair_p0b_baselines.jsonl", baseline_rows),
        ("gamut_process_repair_p0b_candidates.jsonl", candidates),
    ]:
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_process_repair_p0b_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test_text_only")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--skip", type=int, default=48)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
