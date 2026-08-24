"""P0m 2x2 diagnostic for cached-verdict anchoring and forced execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from dependency_recomputation_p0l import exact_mcnemar, paired_bootstrap_difference, parse_met
from gamut_process_repair_p0b import release_runner, split_sentences
from semantic_staleness_p0k import MECHANISMS, SemanticCase, build_cases, numbered


PROTOCOL = "EXPERIMENT_PROTOCOL_PERSISTENCE_RECOMPUTATION_P0M.md"
ARMS = ("fresh_free", "cached_free", "fresh_structured", "cached_structured")
STATES = ("dependency_edit", "harmless_edit")
MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct")
SEED = 20260824
OPERATORS = (
    "less_than", "claim_matches", "subtract_equals", "before", "above_threshold"
)
FRESH_CONTEXT = ""
CACHED_CONTEXT = "Earlier verification on an older version: SAT.\n"


def oracle_trace(case: SemanticCase, state: str) -> dict[str, Any]:
    metadata = case.oracle_metadata
    if case.mechanism == "comparison":
        operands = (
            [metadata["a_new"], metadata["b_old"]]
            if state == "dependency_edit"
            else [metadata["a_old"], metadata["b_harmless"]]
        )
        operator = "less_than"
    elif case.mechanism == "attribution":
        operands = [
            metadata["new_r1_claim"] if state == "dependency_edit" else metadata["old_r1_claim"],
            metadata["target_claim"],
        ]
        operator = "claim_matches"
    elif case.mechanism == "derived":
        operands = [
            metadata["revenue"] if state == "dependency_edit" else metadata["harmless_revenue"],
            metadata["new_cost"],
            metadata["stated_result"],
        ]
        operator = "subtract_equals"
    elif case.mechanism == "temporal":
        operands = (
            [metadata["a_new"], metadata["b_old"]]
            if state == "dependency_edit"
            else [metadata["a_old"], metadata["b_harmless"]]
        )
        operator = "before"
    elif case.mechanism == "definition":
        operands = (
            [metadata["old_score"], metadata["new_threshold"]]
            if state == "dependency_edit"
            else [metadata["harmless_score"], metadata["old_threshold"]]
        )
        operator = "above_threshold"
    else:
        raise ValueError(case.mechanism)
    expected_met = state == "harmless_edit"
    executed = execute_operator(operator, operands)
    if executed != expected_met:
        raise RuntimeError(f"oracle trace mismatch: {case.id} {state}")
    return {
        "source_ids": list(case.dependency_sentence_ids),
        "record_source_id": case.harmless_sentence_id,
        "operator": operator,
        "operands": operands,
        "expected_met": expected_met,
    }


def normalize_claim(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", value)
        if match:
            return float(value)
    return None


def execute_operator(operator: str, operands: list[Any]) -> bool | None:
    if operator == "claim_matches":
        if len(operands) != 2 or not all(isinstance(value, str) for value in operands):
            return None
        return normalize_claim(operands[0]) == normalize_claim(operands[1])
    expected_length = {"less_than": 2, "subtract_equals": 3, "before": 2, "above_threshold": 2}
    if operator not in expected_length or len(operands) != expected_length[operator]:
        return None
    values = [numeric(value) for value in operands]
    if any(value is None for value in values):
        return None
    if operator in ("less_than", "before"):
        return bool(values[0] < values[1])
    if operator == "above_threshold":
        return bool(values[0] > values[1])
    return bool(abs((values[0] - values[1]) - values[2]) <= 1e-8)


def context_for(cache: str) -> str:
    return CACHED_CONTEXT if cache == "cached" else FRESH_CONTEXT


def common_prompt(case: SemanticCase, document: str, cache: str) -> str:
    conclusion = split_sentences(document)[case.conclusion_sentence - 1]
    return (
        "Verify whether the audited conclusion is logically true under the current document.\n"
        f"{context_for(cache)}"
        "Use only the current document facts and do not accept the conclusion merely because "
        "its sentence is present.\n\n"
        f"Current document:\n{numbered(document)}\n\n"
        f"Audited conclusion [S{case.conclusion_sentence}]: {conclusion}\n"
        f"Criterion: {case.obligation}\n\n"
    )


def prompt_for(case: SemanticCase, document: str, cache: str, mode: str) -> str:
    base = common_prompt(case, document, cache)
    if mode == "free":
        return base + (
            "Return exactly one JSON object with only the field met. met must be a JSON Boolean. "
            "Return no markdown or explanation.\nJSON:"
        )
    operator_help = (
        "less_than(lhs,rhs); claim_matches(source_claim,asserted_claim); "
        "subtract_equals(revenue,cost,stated_result); before(first_event_year,second_event_year); "
        "above_threshold(score,threshold)"
    )
    return base + (
        "Externalize the verification record before the final verdict. Select the source sentence "
        "IDs, one operator, and the minimal operands in the operator's stated order. Claims must be "
        "minimal claim strings; numbers must be JSON numbers. Compute the relation yourself, then "
        "return the final verdict. Allowed operators: " + operator_help + ".\n"
        "Return exactly one JSON object with fields source_ids, operator, operands, computed_met, "
        "and met. source_ids and operands are lists; computed_met and met are JSON Booleans. "
        "Return no markdown or explanation.\nJSON:"
    )


def extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_source_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1 <= value <= 10 else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*[Ss]?(\d+)\s*", value)
        if match and 1 <= int(match.group(1)) <= 10:
            return int(match.group(1))
    return None


def parse_structured(raw: str) -> tuple[dict[str, Any], bool, str]:
    value = extract_json(raw)
    required = {"source_ids", "operator", "operands", "computed_met", "met"}
    if value is None or not required.issubset(value):
        return {}, False, "missing_json_or_fields"
    if not isinstance(value["source_ids"], list) or not isinstance(value["operands"], list):
        return {}, False, "bad_lists"
    source_ids = [parse_source_id(item) for item in value["source_ids"]]
    if any(item is None for item in source_ids) or len(source_ids) != len(set(source_ids)):
        return {}, False, "bad_source_ids"
    if not isinstance(value["operator"], str):
        return {}, False, "bad_operator"
    if any(isinstance(item, (dict, list, bool)) or item is None for item in value["operands"]):
        return {}, False, "bad_operands"
    if not isinstance(value["computed_met"], bool) or not isinstance(value["met"], bool):
        return {}, False, "bad_booleans"
    return {
        "source_ids": source_ids,
        "operator": value["operator"].strip().lower(),
        "operands": value["operands"],
        "computed_met": value["computed_met"],
        "met": value["met"],
    }, True, "valid"


def operands_match(predicted: list[Any], expected: list[Any]) -> bool:
    if len(predicted) != len(expected):
        return False
    for left, right in zip(predicted, expected):
        if isinstance(right, str):
            if not isinstance(left, str) or normalize_claim(left) != normalize_claim(right):
                return False
        else:
            left_number = numeric(left)
            if left_number is None or abs(left_number - float(right)) > 1e-8:
                return False
    return True


def assess_structured(parsed: dict[str, Any], valid: bool, oracle: dict[str, Any]) -> dict[str, Any]:
    if not valid:
        return {
            "checker_met": None, "source_record_found": False, "source_exact": False,
            "dependency_coverage": 0.0, "operator_correct": False, "operands_correct": False,
            "trace_complete": False, "executable": False, "checker_correct": False,
            "computed_matches_checker": False, "final_matches_checker": False,
            "final_override": False, "failure_stage": "invalid",
        }
    predicted_sources = set(parsed["source_ids"])
    expected_sources = set(oracle["source_ids"])
    checker_met = execute_operator(parsed["operator"], parsed["operands"])
    source_record_found = oracle["record_source_id"] in predicted_sources
    source_exact = predicted_sources == expected_sources
    dependency_coverage = len(predicted_sources & expected_sources) / len(expected_sources)
    operator_correct = parsed["operator"] == oracle["operator"]
    operand_correct = operands_match(parsed["operands"], oracle["operands"])
    trace_complete = source_exact and operator_correct and operand_correct
    executable = checker_met is not None
    checker_correct = checker_met == oracle["expected_met"] if executable else False
    computed_matches = checker_met == parsed["computed_met"] if executable else False
    final_matches = checker_met == parsed["met"] if executable else False
    final_override = bool(
        trace_complete and checker_correct and computed_matches and not final_matches
    )
    if not source_record_found:
        failure_stage = "source_localization"
    elif not source_exact:
        failure_stage = "dependency_coverage"
    elif not operator_correct:
        failure_stage = "operator_recognition"
    elif not operand_correct:
        failure_stage = "operand_extraction"
    elif not executable:
        failure_stage = "non_executable"
    elif not checker_correct:
        failure_stage = "checker_wrong"
    elif not computed_matches:
        failure_stage = "reported_computation"
    elif not final_matches:
        failure_stage = "final_override"
    else:
        failure_stage = "success"
    return {
        "checker_met": checker_met,
        "source_record_found": source_record_found,
        "source_exact": source_exact,
        "dependency_coverage": dependency_coverage,
        "operator_correct": operator_correct,
        "operands_correct": operand_correct,
        "trace_complete": trace_complete,
        "executable": executable,
        "checker_correct": checker_correct,
        "computed_matches_checker": computed_matches,
        "final_matches_checker": final_matches,
        "final_override": final_override,
        "failure_stage": failure_stage,
    }


def build_records() -> list[dict[str, Any]]:
    records = []
    for case in build_cases():
        for state, document in (
            ("dependency_edit", case.dependency_document),
            ("harmless_edit", case.harmless_document),
        ):
            oracle = oracle_trace(case, state)
            for cache in ("fresh", "cached"):
                for mode in ("free", "structured"):
                    arm = f"{cache}_{mode}"
                    records.append({
                        "record_id": f"{case.id}|{state}|{arm}",
                        "case_id": case.id,
                        "domain_id": case.domain_id,
                        "mechanism": case.mechanism,
                        "state": state,
                        "cache": cache,
                        "mode": mode,
                        "arm": arm,
                        "expected_met": oracle["expected_met"],
                        "oracle": oracle,
                        "prompt": prompt_for(case, document, cache, mode),
                    })
    return records


def arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    stale = [row for row in selected if row["state"] == "dependency_edit"]
    harmless = [row for row in selected if row["state"] == "harmless_edit"]
    result = {
        "n": len(selected),
        "parse_validity": mean(row["valid"] for row in selected),
        "stale_recall": mean(row["final_correct"] for row in stale),
        "harmless_specificity": mean(row["final_correct"] for row in harmless),
    }
    if arm.endswith("structured"):
        result.update({
            "checker_stale_recall": mean(row["assessment"]["checker_correct"] for row in stale),
            "checker_harmless_specificity": mean(row["assessment"]["checker_correct"] for row in harmless),
            "source_record_rate": mean(row["assessment"]["source_record_found"] for row in selected),
            "operator_accuracy": mean(row["assessment"]["operator_correct"] for row in selected),
            "operand_accuracy": mean(row["assessment"]["operands_correct"] for row in selected),
            "trace_complete_rate": mean(row["assessment"]["trace_complete"] for row in selected),
            "executable_rate": mean(row["assessment"]["executable"] for row in selected),
            "reported_computation_alignment": mean(row["assessment"]["computed_matches_checker"] for row in selected),
            "final_checker_alignment": mean(row["assessment"]["final_matches_checker"] for row in selected),
            "final_override_count": sum(row["assessment"]["final_override"] for row in selected),
            "stale_failure_stages": dict(Counter(row["assessment"]["failure_stage"] for row in stale)),
        })
    return result


def paired_outcome(
    rows: list[dict[str, Any]], left_arm: str, right_arm: str, left_field: str, right_field: str,
    seed_offset: int,
) -> dict[str, Any]:
    lookup = {(row["case_id"], row["state"], row["arm"]): row for row in rows}
    cases = sorted({row["case_id"] for row in rows})
    left = []
    right = []
    for case_id in cases:
        left_row = lookup[(case_id, "dependency_edit", left_arm)]
        right_row = lookup[(case_id, "dependency_edit", right_arm)]
        left.append(
            left_row["assessment"][left_field]
            if left_field in left_row.get("assessment", {}) else left_row[left_field]
        )
        right.append(
            right_row["assessment"][right_field]
            if right_field in right_row.get("assessment", {}) else right_row[right_field]
        )
    return {
        **exact_mcnemar(left, right),
        "bootstrap_difference": paired_bootstrap_difference(left, right, SEED + seed_offset),
    }


def summarize_model(rows: list[dict[str, Any]], model_index: int) -> dict[str, Any]:
    arms = {arm: arm_metrics(rows, arm) for arm in ARMS}
    effects = {
        "fresh_free_vs_cached_free": paired_outcome(
            rows, "fresh_free", "cached_free", "final_correct", "final_correct", 100 * model_index
        ),
        "fresh_structured_final_vs_cached_structured_final": paired_outcome(
            rows, "fresh_structured", "cached_structured", "final_correct", "final_correct", 100 * model_index + 1
        ),
        "fresh_structured_final_vs_fresh_free": paired_outcome(
            rows, "fresh_structured", "fresh_free", "final_correct", "final_correct", 100 * model_index + 2
        ),
        "cached_structured_final_vs_cached_free": paired_outcome(
            rows, "cached_structured", "cached_free", "final_correct", "final_correct", 100 * model_index + 3
        ),
        "fresh_structured_checker_vs_fresh_free": paired_outcome(
            rows, "fresh_structured", "fresh_free", "checker_correct", "final_correct", 100 * model_index + 4
        ),
        "cached_structured_checker_vs_cached_free": paired_outcome(
            rows, "cached_structured", "cached_free", "checker_correct", "final_correct", 100 * model_index + 5
        ),
    }
    anchoring = bool(
        arms["fresh_free"]["stale_recall"] - arms["cached_free"]["stale_recall"] >= 0.20
        and effects["fresh_free_vs_cached_free"]["two_sided_exact_p"] <= 0.05
        and arms["fresh_free"]["harmless_specificity"]
        >= arms["cached_free"]["harmless_specificity"] - 0.05
    )

    def rescue(cache: str, outcome_prefix: str) -> bool:
        structured = arms[f"{cache}_structured"]
        free = arms[f"{cache}_free"]
        metric = "stale_recall" if outcome_prefix == "final" else "checker_stale_recall"
        safe_metric = "harmless_specificity" if outcome_prefix == "final" else "checker_harmless_specificity"
        effect = effects[f"{cache}_structured_{outcome_prefix}_vs_{cache}_free"]
        return bool(
            structured[metric] >= 0.75
            and structured[metric] - free["stale_recall"] >= 0.20
            and effect["two_sided_exact_p"] <= 0.05
            and structured[safe_metric] >= 0.95
        )

    model_rescue = {cache: rescue(cache, "final") for cache in ("fresh", "cached")}
    checker_rescue = {cache: rescue(cache, "checker") for cache in ("fresh", "cached")}
    by_mechanism: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        cells = {}
        for arm in ARMS:
            selected = [
                row for row in rows
                if row["mechanism"] == mechanism and row["arm"] == arm
            ]
            stale = [row for row in selected if row["state"] == "dependency_edit"]
            harmless = [row for row in selected if row["state"] == "harmless_edit"]
            cells[arm] = {
                "final_stale_correct": sum(row["final_correct"] for row in stale),
                "final_harmless_correct": sum(row["final_correct"] for row in harmless),
            }
            if arm.endswith("structured"):
                cells[arm]["checker_stale_correct"] = sum(row["assessment"]["checker_correct"] for row in stale)
                cells[arm]["trace_complete_stale"] = sum(row["assessment"]["trace_complete"] for row in stale)
                cells[arm]["final_override_stale"] = sum(row["assessment"]["final_override"] for row in stale)
        by_mechanism[mechanism] = cells
    return {
        "arms": arms,
        "paired_effects": effects,
        "gates": {
            "d1_apparatus_validity": all(arms[arm]["parse_validity"] >= 0.95 for arm in ARMS),
            "d2_cache_anchoring": anchoring,
            "d3_model_execution_rescue": model_rescue,
            "d4_checker_execution_rescue": checker_rescue,
            "d5_final_override_counts": {
                cache: arms[f"{cache}_structured"]["final_override_count"]
                for cache in ("fresh", "cached")
            },
        },
        "by_mechanism": by_mechanism,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_records = build_records()
    all_rows: list[dict[str, Any]] = []
    summaries = {}
    for model_index, model_name in enumerate(args.models):
        records = [dict(record) for record in base_records]
        random.Random(SEED + model_index).shuffle(records)
        order_hash = hashlib.sha256(
            "\n".join(record["record_id"] for record in records).encode("utf-8")
        ).hexdigest()
        runner = ModelRunner(model_name)
        outputs = runner.generate([record["prompt"] for record in records], args.batch_size, args.cap)
        release_runner(runner)
        model_rows = []
        for record, raw in zip(records, outputs):
            if record["mode"] == "free":
                predicted_met, valid, parse_mode = parse_met(raw)
                parsed = {"met": predicted_met} if valid else {}
                assessment = {}
            else:
                parsed, valid, parse_mode = parse_structured(raw)
                predicted_met = parsed.get("met") if valid else None
                assessment = assess_structured(parsed, valid, record["oracle"])
            model_rows.append({
                **record,
                "model": model_name,
                "valid": valid,
                "parse_mode": parse_mode,
                "parsed": parsed,
                "predicted_met": predicted_met,
                "final_correct": bool(valid and predicted_met == record["expected_met"]),
                "assessment": assessment,
                "raw": raw,
            })
        summary = summarize_model(model_rows, model_index)
        summary["generation_order_sha256"] = order_hash
        summaries[model_name] = summary
        all_rows.extend(model_rows)

    d1 = all(summary["gates"]["d1_apparatus_validity"] for summary in summaries.values())
    anchoring_robust = d1 and all(summary["gates"]["d2_cache_anchoring"] for summary in summaries.values())
    model_rescue_robust = d1 and all(
        all(summary["gates"]["d3_model_execution_rescue"].values()) for summary in summaries.values()
    )
    checker_rescue_robust = d1 and all(
        all(summary["gates"]["d4_checker_execution_rescue"].values()) for summary in summaries.values()
    )
    if not d1:
        branch = "APPARATUS_FAILURE"
    elif anchoring_robust:
        branch = "A_CACHE_ANCHORING"
    elif model_rescue_robust or checker_rescue_robust:
        branch = "B_RECOMPUTATION_EXECUTION"
    else:
        branch = "C_STOP_CONTROLLED_LINE"

    review = []
    by_model = defaultdict(list)
    for row in all_rows:
        by_model[row["model"]].append(row)
        reasons = []
        if not row["valid"]:
            reasons.append("invalid")
        if row["mode"] == "structured" and row["state"] == "dependency_edit":
            if row["assessment"].get("failure_stage") != "success":
                reasons.append(row["assessment"].get("failure_stage", "unknown_stage"))
        if row["assessment"].get("final_override"):
            reasons.append("final_override")
        if reasons:
            review.append({"reasons": sorted(set(reasons)), **row})
    for model_name, rows in by_model.items():
        lookup = {(row["case_id"], row["state"], row["arm"]): row for row in rows}
        for case_id in sorted({row["case_id"] for row in rows}):
            fresh = lookup[(case_id, "dependency_edit", "fresh_free")]
            cached = lookup[(case_id, "dependency_edit", "cached_free")]
            if fresh["predicted_met"] != cached["predicted_met"]:
                review.append({
                    "reasons": ["free_cache_discordance"], "model": model_name, "case_id": case_id,
                    "fresh": fresh, "cached": cached,
                })

    report = {
        "protocol": PROTOCOL,
        "models": list(args.models),
        "n_base_cases": 40,
        "n_states": 2,
        "n_arms": 4,
        "n_generations": len(all_rows),
        "model_summaries": summaries,
        "robust_diagnostics": {
            "apparatus_valid": d1,
            "cache_anchoring": anchoring_robust,
            "model_execution_rescue": model_rescue_robust,
            "checker_execution_rescue": checker_rescue_robust,
        },
        "decision_branch": branch,
        "manual_review_items": len(review),
        "interpretation_guard": (
            "Structured execution is a diagnostic and overlaps current structured-verification methods."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "persistence_recomputation_p0m_rows.jsonl", all_rows)
    write_jsonl(args.out_dir / "persistence_recomputation_p0m_review.jsonl", review)
    (args.out_dir / "persistence_recomputation_p0m_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"result_dir={args.out_dir}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cap", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
