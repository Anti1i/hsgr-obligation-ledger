"""Controlled witness-interference mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import edit_ratio
from gamut_process_repair_p0b import parse_sentence_patch, release_runner, split_sentences


PROTOCOL = "EXPERIMENT_PROTOCOL_WITNESS_INTERFERENCE_P0G.md"
OBLIGATION_IDS = ("O_ORDER", "O_ATTRIBUTION", "O_COVERAGE", "O_CONSISTENCY")
FAILURE_TYPES = ("ordering", "attribution", "coverage", "consistency")
TARGET_BY_FAILURE = dict(zip(FAILURE_TYPES, OBLIGATION_IDS))
ARMS = ("full_rewrite", "local_patch", "obligation_patch", "witness_patch")
SELECTION_SALT = "20260823-witness-interference-p0g"


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    first_event: str
    second_event: str
    primary_fact: str
    secondary_fact: str
    choice: str
    wrong_choice: str


@dataclass(frozen=True)
class ControlledCase:
    id: str
    scenario_id: str
    failure_type: str
    question: str
    evidence: str
    baseline_answer: str
    clean_answer: str
    obligations: dict[str, str]
    target_id: str
    expected_before: dict[str, bool]
    witness_sentences: dict[str, tuple[int, ...]]


SCENARIOS = (
    Scenario(
        "transit", "electric-bus pilot", "battery preconditioning at 06:30",
        "the first passenger run at 07:00", "electric buses used 31% less energy than the diesel comparison",
        "five-year operating costs were $1.8 million lower", "the electric-bus purchase",
        "the diesel-bus purchase",
    ),
    Scenario(
        "hospital", "hospital triage rollout", "the privacy review on 12 March",
        "the ward pilot on 18 March", "median triage time fell by 11 minutes",
        "the medication-error rate remained unchanged at 1.2%", "a phased hospital-wide rollout",
        "an immediate unmonitored rollout",
    ),
    Scenario(
        "river", "river-restoration trial", "sediment testing in April", "dredging in June",
        "nitrate concentration fell by 24%", "the wetland design protected 18 hectares of floodplain",
        "the wetland-restoration plan", "the concrete-channel plan",
    ),
    Scenario(
        "school", "school-menu evaluation", "the allergy audit on 3 August",
        "the menu launch on 20 August", "meal participation rose from 62% to 74%",
        "ingredient costs increased by only $0.18 per meal", "the revised fresh-food menu",
        "returning to the previous menu",
    ),
    Scenario(
        "datacenter", "data-center cooling trial", "the 72-hour load test", "production deployment",
        "cooling electricity use fell by 27%", "peak rack temperature stayed below 31 C",
        "the liquid-cooling design", "the legacy air-only design",
    ),
    Scenario(
        "housing", "housing-retrofit assessment", "the asbestos survey", "wall-insulation work",
        "winter heating demand fell by 22%", "indoor temperature stayed above 19 C during the coldest week",
        "the staged heat-pump and insulation retrofit", "replacing boilers without insulation",
    ),
)


def numbered(answer: str) -> str:
    return "\n".join(f"[S{i}] {text}" for i, text in enumerate(split_sentences(answer), 1))


def build_cases() -> list[ControlledCase]:
    cases: list[ControlledCase] = []
    for scenario in SCENARIOS:
        question = (
            f"Using reports A and B, give a concise recommendation about the {scenario.title}. "
            "State the event order, report both results with correct attribution, and give the "
            "evidence-backed recommendation."
        )
        evidence = (
            f"[A] The report states that {scenario.first_event} occurred before "
            f"{scenario.second_event}. It also reports that {scenario.primary_fact}.\n"
            f"[B] The evaluation reports that {scenario.secondary_fact}. It recommends "
            f"{scenario.choice} rather than {scenario.wrong_choice}."
        )
        obligations = {
            "O_ORDER": (
                f"The answer states that {scenario.first_event} happened before {scenario.second_event}."
            ),
            "O_ATTRIBUTION": (
                f"The answer states that {scenario.primary_fact} and attributes that statement to "
                "report [A], not report [B]."
            ),
            "O_COVERAGE": f"The answer includes the result that {scenario.secondary_fact}.",
            "O_CONSISTENCY": (
                f"The final recommendation is {scenario.choice}, not {scenario.wrong_choice}."
            ),
        }
        correct_s1 = (
            f"The report places {scenario.first_event} before {scenario.second_event}; it also "
            f"records that {scenario.primary_fact} [A]."
        )
        wrong_order_s1 = (
            f"The report places {scenario.second_event} before {scenario.first_event}; it also "
            f"records that {scenario.primary_fact} [A]."
        )
        wrong_attr_s1 = (
            f"The report places {scenario.first_event} before {scenario.second_event}; it also "
            f"records that {scenario.primary_fact} [B]."
        )
        s2 = f"It further finds that {scenario.secondary_fact} [B]."
        s3 = f"On that evidence, the recommended option is {scenario.choice}."
        wrong_s3 = f"On that evidence, the recommended option is {scenario.wrong_choice}."
        clean = " ".join((correct_s1, s2, s3))
        baselines = {
            "ordering": " ".join((wrong_order_s1, s2, s3)),
            "attribution": " ".join((wrong_attr_s1, s2, s3)),
            "coverage": " ".join((correct_s1, s3)),
            "consistency": " ".join((correct_s1, s2, wrong_s3)),
        }
        for failure_type in FAILURE_TYPES:
            baseline = baselines[failure_type]
            sentences = split_sentences(baseline)
            target_id = TARGET_BY_FAILURE[failure_type]
            expected = {oid: oid != target_id for oid in OBLIGATION_IDS}
            witnesses: dict[str, tuple[int, ...]] = {}
            if expected["O_ORDER"]:
                witnesses["O_ORDER"] = (1,)
            if expected["O_ATTRIBUTION"]:
                witnesses["O_ATTRIBUTION"] = (1,)
            if expected["O_COVERAGE"]:
                witnesses["O_COVERAGE"] = (2,)
            if expected["O_CONSISTENCY"]:
                witnesses["O_CONSISTENCY"] = (2 if failure_type == "coverage" else 3,)
            if len(sentences) not in (2, 3) or set(witnesses) != {oid for oid, met in expected.items() if met}:
                raise RuntimeError(f"invalid frozen case construction: {scenario.id}/{failure_type}")
            cases.append(
                ControlledCase(
                    id=f"{scenario.id}-{failure_type}", scenario_id=scenario.id,
                    failure_type=failure_type, question=question, evidence=evidence,
                    baseline_answer=baseline, clean_answer=clean, obligations=obligations,
                    target_id=target_id, expected_before=expected, witness_sentences=witnesses,
                )
            )
    return cases


def satisfied_text(case: ControlledCase, include_witnesses: bool) -> str:
    sentences = split_sentences(case.baseline_answer)
    lines: list[str] = []
    for oid in OBLIGATION_IDS:
        if not case.expected_before[oid]:
            continue
        lines.append(f"- {oid}: {case.obligations[oid]}")
        if include_witnesses:
            for sid in case.witness_sentences[oid]:
                lines.append(f"  Frozen witness [S{sid}]: {sentences[sid - 1]}")
    return "\n".join(lines)


def repair_prompt(case: ControlledCase, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    target = f"Failed target [{case.target_id}]: {case.obligations[case.target_id]}"
    if arm == "full_rewrite":
        saved = f"Saved answer:\n{case.baseline_answer}"
        extra = ""
        output = (
            "Return only the complete revised answer. Make the smallest change that fixes the "
            "failed target while preserving every other correct fact and requirement."
        )
    else:
        saved = f"Saved answer split into numbered sentences:\n{numbered(case.baseline_answer)}"
        extra = ""
        if arm == "obligation_patch":
            extra = "\n\nAlready-satisfied obligations whose meaning must remain true:\n" + satisfied_text(case, False)
        elif arm == "witness_patch":
            extra = (
                "\n\nAlready-satisfied obligations and their frozen supporting spans. Preserve each "
                "obligation's meaning; do not overwrite a witness unless the replacement retains "
                "that meaning:\n" + satisfied_text(case, True)
            )
        output = (
            "Return exactly one JSON object with start_sentence and end_sentence as one-based "
            "integers and replacement as a string. Replace at most two consecutive sentences. "
            "Fix the failed target with the smallest patch and preserve all other correct content. "
            "Return no markdown or explanation."
        )
    return (
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n{saved}\n\n"
        f"{target}{extra}\n\nTask:\n{output}"
    )


def parse_two_sentence_patch(text: str, answer: str) -> tuple[str, bool, str, tuple[int, int] | None]:
    revised, valid, mode, span = parse_sentence_patch(text, answer)
    if valid and span and span[1] - span[0] + 1 > 2:
        return answer, False, "span_over_two_sentences", None
    return revised, valid, mode, span


def judge_prompt(case: ControlledCase, answer: str) -> str:
    requirements = "\n".join(f"- {oid}: {case.obligations[oid]}" for oid in OBLIGATION_IDS)
    return (
        "Verify four independent requirements against the answer and fixed evidence. Judge only "
        "what the answer states. For each item, met must be true only when the complete requirement "
        "is satisfied. witness_sentences must list the one-based answer sentence IDs that directly "
        "support a true item, and must be empty for a false item. Return exactly one JSON object "
        "with field items, a list of four objects with fields id, met, witness_sentences. Preserve "
        "the requirement order below. Return no markdown or explanation.\n\n"
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n"
        f"Answer sentences:\n{numbered(answer)}\n\nRequirements:\n{requirements}\n\nJSON:"
    )


def parse_judgment(text: str, sentence_count: int) -> tuple[dict[str, dict[str, Any]], bool, str]:
    stripped = text.strip()
    if "```" in stripped:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return {}, False, "no_json"
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}, False, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"items"} or not isinstance(value["items"], list):
        return {}, False, "wrong_top_level"
    parsed: dict[str, dict[str, Any]] = {}
    for item in value["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "met", "witness_sentences"}:
            return {}, False, "wrong_item_fields"
        oid, met, witnesses = item["id"], item["met"], item["witness_sentences"]
        if oid not in OBLIGATION_IDS or oid in parsed or not isinstance(met, bool):
            return {}, False, "bad_item_value"
        if not isinstance(witnesses, list) or any(
            not isinstance(sid, int) or isinstance(sid, bool) or not 1 <= sid <= sentence_count
            for sid in witnesses
        ):
            return {}, False, "bad_witness"
        if (met and not witnesses) or (not met and witnesses) or len(witnesses) != len(set(witnesses)):
            return {}, False, "inconsistent_witness"
        parsed[oid] = {"met": met, "witness_sentences": witnesses}
    if tuple(parsed) != OBLIGATION_IDS:
        return {}, False, "wrong_item_order"
    return parsed, True, "valid"


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["judge_valid"]]
    successful = [row for row in valid if row["target_recovered"]]
    nodes = [met for row in valid for met in row["preserved_node_states"]]
    return {
        "n": len(rows),
        "judge_valid_rate": len(valid) / max(1, len(rows)),
        "target_repair_rate": mean(row["target_recovered"] for row in valid) if valid else 0.0,
        "successful_repairs": len(successful),
        "successful_repair_regression_rate": (
            mean(row["any_regression"] for row in successful) if successful else 0.0
        ),
        "successful_repairs_with_regression": sum(row["any_regression"] for row in successful),
        "preservation_rate": mean(nodes) if nodes else 0.0,
        "median_edit_ratio": median(row["edit_ratio"] for row in rows),
    }


def overlap_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    events = [
        (overlap, oid in row["regressed_obligations"])
        for row in rows
        for oid, overlap in row["overlap_by_obligation"].items()
    ]
    overlap_events = [regressed for overlap, regressed in events if overlap]
    no_overlap_events = [regressed for overlap, regressed in events if not overlap]
    p_overlap = mean(overlap_events) if overlap_events else 0.0
    p_no_overlap = mean(no_overlap_events) if no_overlap_events else 0.0
    enrichment = p_overlap / p_no_overlap if p_no_overlap else (float("inf") if p_overlap else 0.0)
    return {
        "overlap_node_events": len(overlap_events),
        "no_overlap_node_events": len(no_overlap_events),
        "regression_given_overlap": p_overlap,
        "regression_given_no_overlap": p_no_overlap,
        "enrichment": "inf" if enrichment == float("inf") else enrichment,
    }, enrichment


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases()
    case_by_id = {case.id: case for case in cases}
    generation_keys: list[tuple[str, str]] = []
    prompts: list[str] = []
    for case in cases:
        for arm in ARMS:
            generation_keys.append((case.id, arm))
            prompts.append(repair_prompt(case, arm))

    generator = ModelRunner(args.generator_model)
    raw_outputs = generator.generate(prompts, args.generation_batch_size, args.generation_cap)
    release_runner(generator)

    candidates: list[dict[str, Any]] = []
    for (case_id, arm), raw in zip(generation_keys, raw_outputs):
        case = case_by_id[case_id]
        if arm == "full_rewrite":
            answer, valid, mode, span = raw.strip(), bool(raw.strip()), "full_answer", None
            if not valid:
                answer = case.baseline_answer
            edit_ids = list(range(1, len(split_sentences(case.baseline_answer)) + 1))
        else:
            answer, valid, mode, span = parse_two_sentence_patch(raw, case.baseline_answer)
            edit_ids = list(range(span[0], span[1] + 1)) if span else []
        overlap_by_obligation = {
            oid: bool(set(edit_ids) & set(case.witness_sentences[oid]))
            for oid in OBLIGATION_IDS if case.expected_before[oid]
        }
        candidates.append(
            {
                "id": case_id, "scenario_id": case.scenario_id,
                "failure_type": case.failure_type, "arm": arm,
                "question": case.question, "evidence": case.evidence,
                "target_id": case.target_id, "obligations": case.obligations,
                "baseline_answer": case.baseline_answer, "raw_output": raw,
                "answer": answer, "patch_valid": valid, "parse_mode": mode,
                "sentence_span": list(span) if span else None,
                "edit_sentence_ids": edit_ids,
                "edit_ratio": edit_ratio(case.baseline_answer, answer),
                "frozen_witnesses": {k: list(v) for k, v in case.witness_sentences.items()},
                "overlap_by_obligation": overlap_by_obligation,
            }
        )

    control_specs: list[tuple[str, str]] = []
    judge_prompts: list[str] = []
    for case in cases:
        control_specs.append((case.id, "baseline"))
        judge_prompts.append(judge_prompt(case, case.baseline_answer))
    for scenario in SCENARIOS:
        case = next(case for case in cases if case.scenario_id == scenario.id)
        control_specs.append((case.id, "clean"))
        judge_prompts.append(judge_prompt(case, case.clean_answer))
    judge_prompts.extend(judge_prompt(case_by_id[row["id"]], row["answer"]) for row in candidates)

    judge = ModelRunner(args.judge_model)
    judge_outputs = judge.generate(judge_prompts, args.judge_batch_size, args.judge_cap)
    release_runner(judge)

    positive_total = positive_correct = negative_total = negative_correct = parse_ok = 0
    control_rows: list[dict[str, Any]] = []
    for (case_id, kind), raw in zip(control_specs, judge_outputs[: len(control_specs)]):
        case = case_by_id[case_id]
        answer = case.baseline_answer if kind == "baseline" else case.clean_answer
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(answer)))
        parse_ok += int(valid)
        expected = case.expected_before if kind == "baseline" else {oid: True for oid in OBLIGATION_IDS}
        for oid, expected_met in expected.items():
            if expected_met:
                positive_total += 1
                positive_correct += int(valid and parsed[oid]["met"])
            else:
                negative_total += 1
                negative_correct += int(valid and not parsed[oid]["met"])
        control_rows.append({
            "id": case_id, "kind": kind, "valid": valid, "parse_mode": mode,
            "expected": expected, "parsed": parsed, "raw": raw,
        })
    control_report = {
        "n_answers": len(control_specs),
        "parse_validity": parse_ok / len(control_specs),
        "positive_accuracy": positive_correct / positive_total,
        "negative_accuracy": negative_correct / negative_total,
    }
    control_report["usable"] = all(control_report[key] >= 0.95 for key in (
        "parse_validity", "positive_accuracy", "negative_accuracy"
    ))

    candidate_outputs = judge_outputs[len(control_specs):]
    for row, raw in zip(candidates, candidate_outputs):
        case = case_by_id[row["id"]]
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(row["answer"])))
        target_recovered = bool(valid and parsed[case.target_id]["met"])
        preserved_states = [
            bool(valid and parsed[oid]["met"])
            for oid in OBLIGATION_IDS if case.expected_before[oid]
        ]
        regressed = [
            oid for oid in OBLIGATION_IDS
            if case.expected_before[oid] and valid and not parsed[oid]["met"]
        ]
        row.update({
            "judge_raw": raw, "judge_valid": valid, "judge_parse_mode": mode,
            "final_ledger": parsed, "target_recovered": target_recovered,
            "preserved_node_states": preserved_states,
            "regressed_obligations": regressed,
            "any_regression": bool(regressed),
            "recheck_ids": [case.target_id] + [
                oid for oid, overlap in row["overlap_by_obligation"].items() if overlap
            ],
        })

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_arm[row["arm"]].append(row)
    summaries = {arm: arm_summary(by_arm[arm]) for arm in ARMS}

    successful = [row for row in candidates if row["judge_valid"] and row["target_recovered"]]
    overlap_all, enrichment_all = overlap_summary(successful)
    successful_patches = [row for row in successful if row["arm"] != "full_rewrite"]
    overlap_patch, enrichment_patch = overlap_summary(successful_patches)
    all_regressions = sum(len(row["regressed_obligations"]) for row in successful)
    caught_regressions = sum(
        oid in row["recheck_ids"] for row in successful for oid in row["regressed_obligations"]
    )
    recheck_total = sum(len(set(row["recheck_ids"])) for row in successful)
    possible_checks = len(successful) * len(OBLIGATION_IDS)

    obligation = summaries["obligation_patch"]
    witness = summaries["witness_patch"]
    obligation_rr = obligation["successful_repair_regression_rate"]
    witness_rr = witness["successful_repair_regression_rate"]
    relative_reduction = (
        (obligation_rr - witness_rr) / obligation_rr if obligation_rr > 0 else 0.0
    )
    target_delta = witness["target_repair_rate"] - obligation["target_repair_rate"]
    successful_with_regression = sum(row["any_regression"] for row in successful)

    all_gates = {
        "gate_1_at_least_15_successful_repairs": len(successful) >= 15,
        "gate_2_at_least_5_successful_repairs_with_regression_before_manual": (
            successful_with_regression >= 5
        ),
        "gate_3_patch_overlap_enrichment_at_least_3x": enrichment_patch >= 3.0,
        "gate_4_witness_intervention": target_delta >= -0.10 and relative_reduction >= 0.50,
    }
    all_gates["automatic_all_gates_before_manual"] = control_report["usable"] and all(all_gates.values())

    regression_review = [row for row in successful if row["any_regression"]]
    no_regression_success = [row for row in successful if not row["any_regression"]]
    sampled: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(
        no_regression_success,
        key=lambda value: hash_key(f"{SELECTION_SALT}|{value['id']}|{value['arm']}")
    ):
        key = (row["failure_type"], row["arm"])
        sampled.setdefault(key, row)
    review_rows = regression_review + list(sampled.values())

    report = {
        "protocol": PROTOCOL,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "n_scenarios": len(SCENARIOS), "n_instances": len(cases),
        "n_candidates": len(candidates), "control": control_report,
        "arm_summaries": summaries,
        "successful_repairs": len(successful),
        "successful_repairs_with_regression": successful_with_regression,
        "witness_overlap_all_successful_repairs": overlap_all,
        "witness_overlap_successful_patch_repairs": overlap_patch,
        "selective_reverification": {
            "regression_recall": caught_regressions / all_regressions if all_regressions else 1.0,
            "verification_saving": 1.0 - recheck_total / possible_checks if possible_checks else 0.0,
        },
        "witness_vs_obligation_patch": {
            "target_repair_delta": target_delta,
            "successful_repair_regression_relative_reduction": relative_reduction,
        },
        "automatic_gates_before_manual": all_gates,
        "manual_review_rows": len(review_rows),
        "interpretation_guard": (
            "P0g is a controlled six-block mechanism study. Gates require manual review and do not "
            "establish natural-data prevalence, planner benefit, or hidden-state signal."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates", candidates), ("controls", control_rows)):
        with (args.out_dir / f"witness_interference_p0g_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "witness_interference_p0g_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "witness_interference_p0g_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=512)
    parser.add_argument("--judge-cap", type=int, default=384)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
