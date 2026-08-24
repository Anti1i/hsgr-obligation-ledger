"""Five-arm causal screen for relation-aware dependency recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0b import release_runner, split_sentences
from semantic_staleness_p0k import MECHANISMS, SemanticCase, build_cases, numbered, tokens


PROTOCOL = "EXPERIMENT_PROTOCOL_DEPENDENCY_RECOMPUTATION_P0L.md"
ARMS = ("flat", "source_only", "relation_only", "source_relation", "shuffled_guide")
STATES = ("dependency_edit", "harmless_edit")
PRIMARY_ARM = "source_relation"
SHUFFLE_SEED = 20260824

RELATION_GUIDES = {
    "comparison": "Compare the first current reported value with the second using the strict less-than relation.",
    "attribution": "Match the currently named source claim against the exact claim asserted by the conclusion.",
    "derived": "Subtract current operating cost from current revenue and compare the result with the stated balance.",
    "temporal": "Compare the two current event dates and test the conclusion's claimed before relation.",
    "definition": "Apply the current classification threshold to the subject's current recorded score.",
}


def closest_wrong_mechanism(mechanism: str) -> str:
    correct_length = len(tokens(RELATION_GUIDES[mechanism]))
    candidates = [name for name in MECHANISMS if name != mechanism]
    return min(
        candidates,
        key=lambda name: (abs(len(tokens(RELATION_GUIDES[name])) - correct_length), name),
    )


def irrelevant_source_ids(case: SemanticCase) -> tuple[int, ...]:
    forbidden = set(case.dependency_sentence_ids) | {case.conclusion_sentence}
    candidates = [index for index in (4, 5, 7, 8, 1, 10) if index not in forbidden]
    return tuple(candidates[: len(case.dependency_sentence_ids)])


def source_text(source_ids: tuple[int, ...]) -> str:
    return ", ".join(f"S{index}" for index in source_ids)


def guide_for(case: SemanticCase, arm: str) -> dict[str, Any]:
    if arm == "flat":
        text = "No external dependency map is supplied; independently recompute from the current document."
        return {"text": text, "source_ids": [], "relation_mechanism": None}
    if arm == "source_only":
        text = f"Priority evidence locations: {source_text(case.dependency_sentence_ids)}. Re-read their current contents before deciding."
        return {"text": text, "source_ids": list(case.dependency_sentence_ids), "relation_mechanism": None}
    if arm == "relation_only":
        text = f"Dependency operation: {RELATION_GUIDES[case.mechanism]}"
        return {"text": text, "source_ids": [], "relation_mechanism": case.mechanism}
    if arm == "source_relation":
        text = (
            f"Priority evidence locations: {source_text(case.dependency_sentence_ids)}. "
            f"Dependency operation: {RELATION_GUIDES[case.mechanism]}"
        )
        return {
            "text": text,
            "source_ids": list(case.dependency_sentence_ids),
            "relation_mechanism": case.mechanism,
        }
    if arm == "shuffled_guide":
        wrong_mechanism = closest_wrong_mechanism(case.mechanism)
        wrong_sources = irrelevant_source_ids(case)
        text = (
            f"Priority evidence locations: {source_text(wrong_sources)}. "
            f"Dependency operation: {RELATION_GUIDES[wrong_mechanism]}"
        )
        return {
            "text": text,
            "source_ids": list(wrong_sources),
            "relation_mechanism": wrong_mechanism,
        }
    raise ValueError(f"unknown arm: {arm}")


def prompt_for(case: SemanticCase, document: str, guide_text: str) -> str:
    conclusion = split_sentences(document)[case.conclusion_sentence - 1]
    return (
        "A verifier marked this conclusion SAT on an older document version. The cached verdict "
        "is not evidence and may now be wrong. Decide whether the conclusion is logically true "
        "under the current document below. Recompute from current facts rather than accepting the "
        "conclusion because its sentence is still present.\n\n"
        f"Current document:\n{numbered(document)}\n\n"
        f"Audited conclusion [S{case.conclusion_sentence}]: {conclusion}\n"
        f"Criterion: {case.obligation}\n\n"
        f"Guide:\n{guide_text}\n\n"
        "Return exactly one JSON object with only the field met. The value of met must be the JSON "
        "Boolean true or false. Return no markdown or explanation.\nJSON:"
    )


def parse_met(raw: str) -> tuple[bool | None, bool, str]:
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and set(value) == {"met"} and isinstance(value["met"], bool):
            return value["met"], True, "json"
    matches = re.findall(r'["\']?met["\']?\s*:\s*(true|false)', text, re.IGNORECASE)
    if len(matches) == 1:
        return matches[0].lower() == "true", True, "explicit_boolean_recovery"
    return None, False, "invalid"


def build_records() -> list[dict[str, Any]]:
    records = []
    for case in build_cases():
        for state, document, expected_met in (
            ("dependency_edit", case.dependency_document, False),
            ("harmless_edit", case.harmless_document, True),
        ):
            for arm in ARMS:
                guide = guide_for(case, arm)
                records.append({
                    "record_id": f"{case.id}|{state}|{arm}",
                    "case_id": case.id,
                    "domain_id": case.domain_id,
                    "mechanism": case.mechanism,
                    "state": state,
                    "arm": arm,
                    "expected_met": expected_met,
                    "guide": guide,
                    "prompt": prompt_for(case, document, guide["text"]),
                })
    return records


def exact_mcnemar(left_correct: list[bool], right_correct: list[bool]) -> dict[str, Any]:
    left_wins = sum(left and not right for left, right in zip(left_correct, right_correct))
    right_wins = sum(right and not left for left, right in zip(left_correct, right_correct))
    discordant = left_wins + right_wins
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_wins, right_wins) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "left_wins": left_wins,
        "right_wins": right_wins,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_difference(
    left_correct: list[bool], right_correct: list[bool], seed: int, draws: int = 5000
) -> dict[str, float]:
    rng = random.Random(seed)
    differences = []
    size = len(left_correct)
    for _ in range(draws):
        indices = [rng.randrange(size) for _ in range(size)]
        differences.append(mean(int(left_correct[index]) - int(right_correct[index]) for index in indices))
    differences.sort()
    return {
        "mean": mean(differences),
        "ci95_low": differences[int(0.025 * draws)],
        "ci95_high": differences[int(0.975 * draws) - 1],
    }


def summarize_model(rows: list[dict[str, Any]], model_index: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"arms": {}, "paired_dependency_effects": {}, "mechanisms": {}}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        dependency = [row for row in selected if row["state"] == "dependency_edit"]
        harmless = [row for row in selected if row["state"] == "harmless_edit"]
        summary["arms"][arm] = {
            "n": len(selected),
            "parse_validity": mean(row["valid"] for row in selected),
            "stale_recall": mean(row["correct"] for row in dependency),
            "harmless_specificity": mean(row["correct"] for row in harmless),
            "balanced_accuracy": (
                mean(row["correct"] for row in dependency)
                + mean(row["correct"] for row in harmless)
            ) / 2,
        }

    dependency_by_arm = {
        arm: sorted(
            (row for row in rows if row["arm"] == arm and row["state"] == "dependency_edit"),
            key=lambda row: row["case_id"],
        )
        for arm in ARMS
    }
    for baseline_index, baseline in enumerate(("flat", "shuffled_guide")):
        left = [row["correct"] for row in dependency_by_arm[PRIMARY_ARM]]
        right = [row["correct"] for row in dependency_by_arm[baseline]]
        summary["paired_dependency_effects"][f"{PRIMARY_ARM}_vs_{baseline}"] = {
            **exact_mcnemar(left, right),
            "bootstrap_difference": paired_bootstrap_difference(
                left, right, SHUFFLE_SEED + 100 * model_index + baseline_index
            ),
        }

    for mechanism in MECHANISMS:
        mechanism_cells = {}
        for arm in ARMS:
            dependency = [
                row for row in rows
                if row["mechanism"] == mechanism and row["arm"] == arm and row["state"] == "dependency_edit"
            ]
            harmless = [
                row for row in rows
                if row["mechanism"] == mechanism and row["arm"] == arm and row["state"] == "harmless_edit"
            ]
            mechanism_cells[arm] = {
                "dependency_correct": sum(row["correct"] for row in dependency),
                "dependency_total": len(dependency),
                "harmless_correct": sum(row["correct"] for row in harmless),
                "harmless_total": len(harmless),
            }
        summary["mechanisms"][mechanism] = mechanism_cells

    arm = summary["arms"]
    vs_flat = summary["paired_dependency_effects"][f"{PRIMARY_ARM}_vs_flat"]
    vs_shuffle = summary["paired_dependency_effects"][f"{PRIMARY_ARM}_vs_shuffled_guide"]
    relation_contribution = max(
        arm["relation_only"]["stale_recall"], arm[PRIMARY_ARM]["stale_recall"]
    ) - arm["source_only"]["stale_recall"]
    breadth = sum(
        cells[PRIMARY_ARM]["dependency_correct"] >= 6
        for cells in summary["mechanisms"].values()
    )
    summary["gates"] = {
        "g1_output_validity": all(arm[name]["parse_validity"] >= 0.98 for name in ARMS),
        "g2_primary_causal_rescue": bool(
            arm[PRIMARY_ARM]["stale_recall"] >= 0.75
            and arm[PRIMARY_ARM]["stale_recall"] - arm["flat"]["stale_recall"] >= 0.20
            and arm[PRIMARY_ARM]["stale_recall"] - arm["shuffled_guide"]["stale_recall"] >= 0.15
            and vs_flat["two_sided_exact_p"] <= 0.05
            and vs_shuffle["two_sided_exact_p"] <= 0.05
        ),
        "g3_harmless_specificity": bool(
            arm[PRIMARY_ARM]["harmless_specificity"] >= 0.95
            and arm["flat"]["harmless_specificity"] - arm[PRIMARY_ARM]["harmless_specificity"] <= 0.05
        ),
        "g4_relation_contribution": relation_contribution >= 0.10,
        "g5_mechanism_breadth": breadth >= 4,
        "relation_contribution_over_source_only": relation_contribution,
        "mechanisms_with_at_least_6_of_8": breadth,
    }
    summary["passes_all_model_gates"] = all(
        summary["gates"][name]
        for name in (
            "g1_output_validity", "g2_primary_causal_rescue", "g3_harmless_specificity",
            "g4_relation_contribution", "g5_mechanism_breadth",
        )
    )
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_records = build_records()
    all_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}

    for model_index, model_name in enumerate(args.models):
        records = [dict(record) for record in base_records]
        random.Random(SHUFFLE_SEED + model_index).shuffle(records)
        order_hash = hashlib.sha256(
            "\n".join(record["record_id"] for record in records).encode("utf-8")
        ).hexdigest()
        runner = ModelRunner(model_name)
        raw_outputs = runner.generate(
            [record["prompt"] for record in records], args.batch_size, args.cap
        )
        release_runner(runner)
        model_rows = []
        for record, raw in zip(records, raw_outputs):
            predicted, valid, parse_mode = parse_met(raw)
            correct = bool(valid and predicted == record["expected_met"])
            model_rows.append({
                **record,
                "model": model_name,
                "predicted_met": predicted,
                "valid": valid,
                "parse_mode": parse_mode,
                "correct": correct,
                "raw": raw,
            })
        summary = summarize_model(model_rows, model_index)
        summary["generation_order_sha256"] = order_hash
        model_summaries[model_name] = summary
        all_rows.extend(model_rows)

    review = []
    for model_name in args.models:
        rows = [row for row in all_rows if row["model"] == model_name]
        lookup = {(row["case_id"], row["state"], row["arm"]): row for row in rows}
        for case in build_cases():
            primary_dep = lookup[(case.id, "dependency_edit", PRIMARY_ARM)]
            shuffled_dep = lookup[(case.id, "dependency_edit", "shuffled_guide")]
            primary_safe = lookup[(case.id, "harmless_edit", PRIMARY_ARM)]
            flat_safe = lookup[(case.id, "harmless_edit", "flat")]
            reasons = []
            if not primary_dep["correct"]:
                reasons.append("primary_missed_dependency_edit")
            if shuffled_dep["correct"] and not primary_dep["correct"]:
                reasons.append("shuffled_correct_primary_wrong")
            if flat_safe["correct"] and not primary_safe["correct"]:
                reasons.append("primary_harmless_regression")
            if reasons:
                review.append({
                    "model": model_name,
                    "case_id": case.id,
                    "mechanism": case.mechanism,
                    "reasons": reasons,
                    "primary_dependency": primary_dep,
                    "shuffled_dependency": shuffled_dep,
                    "primary_harmless": primary_safe,
                    "flat_harmless": flat_safe,
                })
        for row in rows:
            if not row["valid"]:
                review.append({"model": model_name, "case_id": row["case_id"], "reasons": ["invalid_generation"], "row": row})

    report = {
        "protocol": PROTOCOL,
        "models": list(args.models),
        "n_base_cases": 40,
        "n_states_per_arm": 80,
        "n_arms": len(ARMS),
        "n_generations": len(all_rows),
        "cached_old_verdict": "SAT",
        "model_summaries": model_summaries,
        "full_p0l_pass": all(summary["passes_all_model_gates"] for summary in model_summaries.values()),
        "manual_review_items": len(review),
        "interpretation_guard": (
            "A controlled Guide effect does not establish natural prevalence or hidden-state predictability."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "dependency_recomputation_p0l_rows.jsonl", all_rows)
    write_jsonl(args.out_dir / "dependency_recomputation_p0l_review.jsonl", review)
    (args.out_dir / "dependency_recomputation_p0l_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"result_dir={args.out_dir}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
