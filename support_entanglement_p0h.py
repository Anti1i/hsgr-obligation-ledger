"""Paired causal study of support co-location under sentence-level repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import edit_ratio
from gamut_process_repair_p0b import release_runner, split_sentences


PROTOCOL = "EXPERIMENT_PROTOCOL_SUPPORT_ENTANGLEMENT_P0H.md"
OBLIGATION_IDS = ("O_LEFT", "O_TARGET", "O_RIGHT")
LAYOUTS = ("entangled", "disentangled")
ARMS = ("sentence_patch", "full_rewrite")
SELECTION_SALT = "20260823-support-entanglement-p0h"


@dataclass(frozen=True)
class ContentBlock:
    id: str
    target_type: str
    title: str
    left: str
    target_correct: str
    target_wrong: str
    right: str
    evidence: str


@dataclass(frozen=True)
class PairedCase:
    id: str
    block_id: str
    target_type: str
    layout: str
    question: str
    evidence: str
    baseline_answer: str
    clean_answer: str
    obligations: dict[str, str]
    expected_before: dict[str, bool]
    witness_sentences: dict[str, tuple[int, ...]]
    semantic_atoms: tuple[str, str, str]


BLOCKS = (
    ContentBlock(
        "transit", "numeric", "electric-bus pilot",
        "battery preconditioning began at 06:30 [A]",
        "energy use was 31% lower than the diesel comparison [B]",
        "energy use was 13% lower than the diesel comparison [B]",
        "five-year operating cost was $1.8 million lower [C]",
        "[A] Battery preconditioning began at 06:30.\n[B] Energy use was 31% lower than the diesel comparison.\n[C] Five-year operating cost was $1.8 million lower.",
    ),
    ContentBlock(
        "hospital", "numeric", "hospital triage rollout",
        "the privacy review finished on 12 March [A]",
        "median triage time fell by 11 minutes [B]",
        "median triage time fell by 1 minute [B]",
        "the medication-error rate remained 1.2% [C]",
        "[A] The privacy review finished on 12 March.\n[B] Median triage time fell by 11 minutes.\n[C] The medication-error rate remained 1.2%.",
    ),
    ContentBlock(
        "river", "numeric", "river-restoration trial",
        "sediment testing was completed in April [A]",
        "nitrate concentration fell by 24% [B]",
        "nitrate concentration fell by 42% [B]",
        "the wetland protected 18 hectares of floodplain [C]",
        "[A] Sediment testing was completed in April.\n[B] Nitrate concentration fell by 24%.\n[C] The wetland protected 18 hectares of floodplain.",
    ),
    ContentBlock(
        "school", "numeric", "school-menu evaluation",
        "the allergy audit finished on 3 August [A]",
        "meal participation rose from 62% to 74% [B]",
        "meal participation rose from 62% to 64% [B]",
        "ingredient cost increased by $0.18 per meal [C]",
        "[A] The allergy audit finished on 3 August.\n[B] Meal participation rose from 62% to 74%.\n[C] Ingredient cost increased by $0.18 per meal.",
    ),
    ContentBlock(
        "datacenter", "numeric", "data-center cooling trial",
        "the load test lasted 72 hours [A]",
        "cooling electricity use fell by 27% [B]",
        "cooling electricity use fell by 17% [B]",
        "peak rack temperature stayed below 31 C [C]",
        "[A] The load test lasted 72 hours.\n[B] Cooling electricity use fell by 27%.\n[C] Peak rack temperature stayed below 31 C.",
    ),
    ContentBlock(
        "housing", "numeric", "housing-retrofit assessment",
        "an asbestos survey preceded the retrofit [A]",
        "winter heating demand fell by 22% [B]",
        "winter heating demand fell by 12% [B]",
        "indoor temperature stayed above 19 C [C]",
        "[A] An asbestos survey preceded the retrofit.\n[B] Winter heating demand fell by 22%.\n[C] Indoor temperature stayed above 19 C.",
    ),
    ContentBlock(
        "library", "attribution", "library digitization review",
        "the archive scan covered 48,000 pages [A]",
        "source [B] reports that retrieval time fell by 40%",
        "source [C] reports that retrieval time fell by 40%",
        "98% of catalog records passed validation [C]",
        "[A] The archive scan covered 48,000 pages.\n[B] Retrieval time fell by 40%.\n[C] 98% of catalog records passed validation.",
    ),
    ContentBlock(
        "water", "attribution", "water-treatment evaluation",
        "the calibration used 24 reference samples [A]",
        "source [B] reports that turbidity fell by 36%",
        "source [A] reports that turbidity fell by 36%",
        "residual chlorine stayed within the safety limit [C]",
        "[A] The calibration used 24 reference samples.\n[B] Turbidity fell by 36%.\n[C] Residual chlorine stayed within the safety limit.",
    ),
    ContentBlock(
        "wildfire", "attribution", "wildfire-sensor trial",
        "the network used 63 ridge sensors [A]",
        "source [B] reports a 14-minute alert lead",
        "source [C] reports a 14-minute alert lead",
        "the false-alarm rate was 3.1% [C]",
        "[A] The network used 63 ridge sensors.\n[B] The alert lead was 14 minutes.\n[C] The false-alarm rate was 3.1%.",
    ),
    ContentBlock(
        "bridge", "attribution", "rail-bridge inspection",
        "the inspection covered all 16 support joints [A]",
        "source [B] reports that vibration fell by 19%",
        "source [A] reports that vibration fell by 19%",
        "the certified load capacity was unchanged [C]",
        "[A] The inspection covered all 16 support joints.\n[B] Vibration fell by 19%.\n[C] The certified load capacity was unchanged.",
    ),
    ContentBlock(
        "farm", "attribution", "precision-irrigation trial",
        "the soil survey sampled 52 plots [A]",
        "source [B] reports that water use fell by 28%",
        "source [C] reports that water use fell by 28%",
        "crop yield increased by 12% [C]",
        "[A] The soil survey sampled 52 plots.\n[B] Water use fell by 28%.\n[C] Crop yield increased by 12%.",
    ),
    ContentBlock(
        "satellite", "attribution", "satellite-imaging calibration",
        "the calibration used 35 ground markers [A]",
        "source [B] reports a geolocation error of 0.4 metres",
        "source [A] reports a geolocation error of 0.4 metres",
        "cloud-free coverage reached 97% [C]",
        "[A] The calibration used 35 ground markers.\n[B] Geolocation error was 0.4 metres.\n[C] Cloud-free coverage reached 97%.",
    ),
    ContentBlock(
        "factory", "ordering", "factory-line qualification",
        "the pressure test covered all six vessels [A]",
        "the pressure test occurred before production deployment [B]",
        "production deployment occurred before the pressure test [B]",
        "the final defect rate was 0.8% [C]",
        "[A] The pressure test covered all six vessels.\n[B] The pressure test occurred before production deployment.\n[C] The final defect rate was 0.8%.",
    ),
    ContentBlock(
        "coast", "ordering", "coastal-barrier project",
        "the mapping covered 4.2 kilometres of shoreline [A]",
        "flood mapping occurred before barrier construction [B]",
        "barrier construction occurred before flood mapping [B]",
        "the modeled flood depth fell by 32% [C]",
        "[A] The mapping covered 4.2 kilometres of shoreline.\n[B] Flood mapping occurred before barrier construction.\n[C] The modeled flood depth fell by 32%.",
    ),
    ContentBlock(
        "vaccine", "ordering", "vaccine cold-chain rollout",
        "the audit examined 28 storage sites [A]",
        "the cold-chain audit occurred before regional rollout [B]",
        "regional rollout occurred before the cold-chain audit [B]",
        "spoilage fell from 6.4% to 1.1% [C]",
        "[A] The audit examined 28 storage sites.\n[B] The cold-chain audit occurred before regional rollout.\n[C] Spoilage fell from 6.4% to 1.1%.",
    ),
    ContentBlock(
        "museum", "ordering", "museum display-case upgrade",
        "the baseline logged humidity for 30 days [A]",
        "the humidity baseline was collected before case sealing [B]",
        "case sealing occurred before the humidity baseline [B]",
        "no object damage was recorded [C]",
        "[A] The baseline logged humidity for 30 days.\n[B] The humidity baseline was collected before case sealing.\n[C] No object damage was recorded.",
    ),
    ContentBlock(
        "airport", "ordering", "airport flight-path review",
        "the noise survey used 11 monitoring stations [A]",
        "the noise survey occurred before the flight-path change [B]",
        "the flight-path change occurred before the noise survey [B]",
        "average departure delay was unchanged [C]",
        "[A] The noise survey used 11 monitoring stations.\n[B] The noise survey occurred before the flight-path change.\n[C] Average departure delay was unchanged.",
    ),
    ContentBlock(
        "recycling", "ordering", "recycling-line upgrade",
        "the audit sampled 76 material batches [A]",
        "the contamination audit occurred before the sorting upgrade [B]",
        "the sorting upgrade occurred before the contamination audit [B]",
        "material recovery increased by 21% [C]",
        "[A] The audit sampled 76 material batches.\n[B] The contamination audit occurred before the sorting upgrade.\n[C] Material recovery increased by 21%.",
    ),
)


def render(block: ContentBlock, layout: str, clean: bool) -> str:
    target = block.target_correct if clean else block.target_wrong
    if layout == "entangled":
        return f"The record shows that {block.left}; {target}; and {block.right}."
    if layout == "disentangled":
        return (
            f"The record shows that {block.left}. "
            f"It also shows that {target}. "
            f"It further shows that {block.right}."
        )
    raise ValueError(f"unknown layout: {layout}")


def build_cases() -> list[PairedCase]:
    cases: list[PairedCase] = []
    for block in BLOCKS:
        obligations = {
            "O_LEFT": f"The answer states that {block.left}.",
            "O_TARGET": f"The answer states that {block.target_correct}.",
            "O_RIGHT": f"The answer states that {block.right}.",
        }
        for layout in LAYOUTS:
            witnesses = (
                {"O_LEFT": (1,), "O_RIGHT": (1,)}
                if layout == "entangled"
                else {"O_LEFT": (1,), "O_RIGHT": (3,)}
            )
            case = PairedCase(
                id=f"{block.id}-{layout}", block_id=block.id,
                target_type=block.target_type, layout=layout,
                question=(
                    f"Using sources A, B, and C, report the three specified findings about the "
                    f"{block.title} with correct values, source attribution, and event order."
                ),
                evidence=block.evidence,
                baseline_answer=render(block, layout, False),
                clean_answer=render(block, layout, True),
                obligations=obligations,
                expected_before={"O_LEFT": True, "O_TARGET": False, "O_RIGHT": True},
                witness_sentences=witnesses,
                semantic_atoms=(block.left, block.target_wrong, block.right),
            )
            expected_sentences = 1 if layout == "entangled" else 3
            if len(split_sentences(case.baseline_answer)) != expected_sentences:
                raise RuntimeError(f"sentence construction failed: {case.id}")
            cases.append(case)
    return cases


def numbered(answer: str) -> str:
    return "\n".join(f"[S{i}] {text}" for i, text in enumerate(split_sentences(answer), 1))


def repair_prompt(case: PairedCase, arm: str) -> str:
    preserved = "\n".join(
        f"- {oid}: {case.obligations[oid]}" for oid in ("O_LEFT", "O_RIGHT")
    )
    common = (
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n"
        f"Failed target:\n- O_TARGET: {case.obligations['O_TARGET']}\n\n"
        f"Already-satisfied requirements that must remain true:\n{preserved}\n\n"
    )
    if arm == "sentence_patch":
        task = (
            "Saved answer split into numbered sentences:\n"
            f"{numbered(case.baseline_answer)}\n\nTask:\n"
            "Return exactly one JSON object with start_sentence, end_sentence, and replacement. "
            "Replace exactly one source sentence: start_sentence and end_sentence must be the "
            "same one-based integer. Fix O_TARGET with the smallest replacement while preserving "
            "O_LEFT and O_RIGHT. Return no markdown or explanation."
        )
    elif arm == "full_rewrite":
        task = (
            f"Saved answer:\n{case.baseline_answer}\n\nTask:\n"
            "Return only the complete revised answer. Make the smallest change that fixes O_TARGET "
            "while preserving O_LEFT and O_RIGHT. Return no markdown or explanation."
        )
    else:
        raise ValueError(f"unknown arm: {arm}")
    return common + task


def parse_one_sentence_patch(
    text: str, answer: str
) -> tuple[str, bool, str, tuple[int, int] | None]:
    stripped = text.strip()
    if "```" in stripped:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return answer, False, "no_json", None
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return answer, False, "invalid_json", None
    if not isinstance(value, dict) or set(value) != {
        "start_sentence", "end_sentence", "replacement"
    }:
        return answer, False, "wrong_fields", None
    first, last, replacement = (
        value["start_sentence"], value["end_sentence"], value["replacement"]
    )
    sentences = split_sentences(answer)
    if (
        not isinstance(first, int) or isinstance(first, bool)
        or not isinstance(last, int) or isinstance(last, bool)
    ):
        return answer, False, "non_integer_index", None
    if first != last:
        return answer, False, "not_one_sentence", None
    if not 1 <= first <= len(sentences):
        return answer, False, "index_out_of_range", None
    if not isinstance(replacement, str) or not replacement.strip():
        return answer, False, "empty_replacement", None
    if len(replacement.split()) > 120:
        return answer, False, "replacement_too_long", None
    revised = sentences[: first - 1] + [replacement.strip()] + sentences[first:]
    return " ".join(revised), True, "valid", (first, last)


def judge_prompt(case: PairedCase, answer: str) -> str:
    requirements = "\n".join(
        f"- {oid}: {case.obligations[oid]}" for oid in OBLIGATION_IDS
    )
    return (
        "Judge what the answer visibly states against the fixed evidence. Do not infer or silently "
        "correct a wrong number, citation, or event order from the evidence. Return exactly one "
        "JSON object with field items. items must contain three objects in the requirement order, "
        "each with fields id, met, and witness_sentences. A true item needs one or more one-based "
        "answer sentence IDs that directly state it; a false item must have an empty witness list. "
        "Return no markdown or explanation.\n\n"
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n"
        f"Answer sentences:\n{numbered(answer)}\n\nRequirements:\n{requirements}\n\nJSON:"
    )


def parse_judgment(
    text: str, sentence_count: int
) -> tuple[dict[str, dict[str, Any]], bool, str]:
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


def exact_one_sided_sign_p(entangled_only: int, disentangled_only: int) -> float:
    discordant = entangled_only + disentangled_only
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(entangled_only, discordant + 1)) / (2 ** discordant)


def paired_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and row["judge_valid"]]
    by_block: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_block[row["block_id"]][row["layout"]] = row
    eligible = [
        pair for pair in by_block.values()
        if set(pair) == set(LAYOUTS)
        and pair["entangled"]["target_recovered"]
        and pair["disentangled"]["target_recovered"]
    ]
    entangled_only = sum(
        pair["entangled"]["any_regression"] and not pair["disentangled"]["any_regression"]
        for pair in eligible
    )
    disentangled_only = sum(
        pair["disentangled"]["any_regression"] and not pair["entangled"]["any_regression"]
        for pair in eligible
    )
    entangled_rate = mean(pair["entangled"]["any_regression"] for pair in eligible) if eligible else 0.0
    disentangled_rate = mean(pair["disentangled"]["any_regression"] for pair in eligible) if eligible else 0.0
    return {
        "valid_rows": len(selected),
        "joint_target_success_pairs": len(eligible),
        "entangled_regression_rate_joint_success": entangled_rate,
        "disentangled_regression_rate_joint_success": disentangled_rate,
        "paired_regression_risk_difference": entangled_rate - disentangled_rate,
        "entangled_only_discordant": entangled_only,
        "disentangled_only_discordant": disentangled_only,
        "exact_one_sided_sign_p": exact_one_sided_sign_p(entangled_only, disentangled_only),
    }


def arm_layout_summary(rows: list[dict[str, Any]], arm: str, layout: str) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["arm"] == arm and row["layout"] == layout and row["judge_valid"]
    ]
    successful = [row for row in selected if row["target_recovered"]]
    return {
        "n": len(selected),
        "target_repair_rate": mean(row["target_recovered"] for row in selected) if selected else 0.0,
        "safe_repair_rate": mean(
            row["target_recovered"] and not row["any_regression"] for row in selected
        ) if selected else 0.0,
        "regression_rate_among_successful": mean(
            row["any_regression"] for row in successful
        ) if successful else 0.0,
        "median_edit_ratio": median(row["edit_ratio"] for row in selected) if selected else 0.0,
        "median_source_character_share": median(
            row["source_character_share"] for row in selected
        ) if selected else 0.0,
    }


def operator_interaction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_block: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["judge_valid"]:
            by_block[row["block_id"]][(row["arm"], row["layout"])] = row
    required = {(arm, layout) for arm in ARMS for layout in LAYOUTS}
    common = [
        cells for cells in by_block.values()
        if set(cells) == required and all(cells[key]["target_recovered"] for key in required)
    ]
    effects: dict[str, float] = {}
    for arm in ARMS:
        entangled = mean(
            cells[(arm, "entangled")]["any_regression"] for cells in common
        ) if common else 0.0
        disentangled = mean(
            cells[(arm, "disentangled")]["any_regression"] for cells in common
        ) if common else 0.0
        effects[arm] = entangled - disentangled
    return {
        "common_four_cell_target_success_blocks": len(common),
        "sentence_patch_layout_effect": effects["sentence_patch"],
        "full_rewrite_layout_effect": effects["full_rewrite"],
        "difference_in_layout_effects": effects["sentence_patch"] - effects["full_rewrite"],
    }


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases()
    case_by_id = {case.id: case for case in cases}
    generation_keys = [(case.id, arm) for case in cases for arm in ARMS]
    prompts = [repair_prompt(case_by_id[case_id], arm) for case_id, arm in generation_keys]

    generator = ModelRunner(args.generator_model)
    raw_outputs = generator.generate(prompts, args.generation_batch_size, args.generation_cap)
    release_runner(generator)

    candidates: list[dict[str, Any]] = []
    for (case_id, arm), raw in zip(generation_keys, raw_outputs):
        case = case_by_id[case_id]
        source_sentences = split_sentences(case.baseline_answer)
        if arm == "sentence_patch":
            answer, valid, mode, span = parse_one_sentence_patch(raw, case.baseline_answer)
            edit_ids = [span[0]] if span else []
        else:
            answer, valid, mode, span = raw.strip(), bool(raw.strip()), "full_answer", None
            if not valid:
                answer = case.baseline_answer
            edit_ids = list(range(1, len(source_sentences) + 1))
        source_characters = sum(len(source_sentences[sid - 1]) for sid in edit_ids)
        overlap = {
            oid: bool(set(ids) & set(edit_ids)) for oid, ids in case.witness_sentences.items()
        }
        candidates.append({
            "id": case.id, "block_id": case.block_id, "target_type": case.target_type,
            "layout": case.layout, "arm": arm, "question": case.question,
            "evidence": case.evidence, "obligations": case.obligations,
            "semantic_atoms": list(case.semantic_atoms), "baseline_answer": case.baseline_answer,
            "raw_output": raw, "answer": answer, "patch_valid": valid, "parse_mode": mode,
            "sentence_span": list(span) if span else None, "edit_sentence_ids": edit_ids,
            "source_character_share": source_characters / max(1, len(case.baseline_answer)),
            "edit_ratio": edit_ratio(case.baseline_answer, answer),
            "frozen_witnesses": {oid: list(ids) for oid, ids in case.witness_sentences.items()},
            "overlap_by_obligation": overlap,
            "selected_unit_preserved_obligation_degree": sum(overlap.values()),
        })

    control_specs: list[tuple[str, str]] = []
    judge_prompts: list[str] = []
    for case in cases:
        for kind, answer in (("baseline", case.baseline_answer), ("clean", case.clean_answer)):
            control_specs.append((case.id, kind))
            judge_prompts.append(judge_prompt(case, answer))
    judge_prompts.extend(judge_prompt(case_by_id[row["id"]], row["answer"]) for row in candidates)

    judge = ModelRunner(args.judge_model)
    raw_judgments = judge.generate(judge_prompts, args.judge_batch_size, args.judge_cap)
    release_runner(judge)

    positive_total = positive_correct = negative_total = negative_correct = parse_ok = 0
    controls: list[dict[str, Any]] = []
    for (case_id, kind), raw in zip(control_specs, raw_judgments[: len(control_specs)]):
        case = case_by_id[case_id]
        answer = case.baseline_answer if kind == "baseline" else case.clean_answer
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(answer)))
        expected = case.expected_before if kind == "baseline" else {oid: True for oid in OBLIGATION_IDS}
        parse_ok += int(valid)
        for oid, value in expected.items():
            if value:
                positive_total += 1
                positive_correct += int(valid and parsed[oid]["met"])
            else:
                negative_total += 1
                negative_correct += int(valid and not parsed[oid]["met"])
        controls.append({
            "id": case_id, "kind": kind, "expected": expected, "valid": valid,
            "parse_mode": mode, "parsed": parsed, "raw": raw,
        })
    control = {
        "n_answers": len(control_specs),
        "parse_validity": parse_ok / len(control_specs),
        "positive_accuracy": positive_correct / positive_total,
        "negative_accuracy": negative_correct / negative_total,
    }
    control["usable"] = all(control[key] >= 0.95 for key in (
        "parse_validity", "positive_accuracy", "negative_accuracy"
    ))

    for row, raw in zip(candidates, raw_judgments[len(control_specs):]):
        case = case_by_id[row["id"]]
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(row["answer"])))
        regressed = [
            oid for oid in ("O_LEFT", "O_RIGHT") if valid and not parsed[oid]["met"]
        ]
        row.update({
            "judge_raw": raw, "judge_valid": valid, "judge_parse_mode": mode,
            "final_ledger": parsed, "target_recovered": bool(valid and parsed["O_TARGET"]["met"]),
            "regressed_obligations": regressed, "any_regression": bool(regressed),
        })

    layout_summaries = {
        arm: {layout: arm_layout_summary(candidates, arm, layout) for layout in LAYOUTS}
        for arm in ARMS
    }
    paired = {arm: paired_summary(candidates, arm) for arm in ARMS}
    patch = paired["sentence_patch"]
    interaction = operator_interaction_summary(candidates)
    gates = {
        "gate_1_controls": control["usable"],
        "gate_2_at_least_12_joint_patch_success_pairs": patch["joint_target_success_pairs"] >= 12,
        "gate_3_direction_and_exact_test": (
            patch["entangled_only_discordant"] >= 5
            and patch["disentangled_only_discordant"] <= 1
            and patch["exact_one_sided_sign_p"] < 0.05
        ),
        "gate_4_patch_risk_difference_at_least_20pp": (
            patch["paired_regression_risk_difference"] >= 0.20
        ),
        "gate_5_common_operator_interaction": (
            interaction["common_four_cell_target_success_blocks"] >= 10
            and interaction["difference_in_layout_effects"] >= 0.15
        ),
    }
    gates["automatic_all_gates_before_manual"] = all(gates.values())

    regressions = [
        row for row in candidates if row["judge_valid"] and row["target_recovered"]
        and row["any_regression"]
    ]
    clean_successes = [
        row for row in candidates if row["judge_valid"] and row["target_recovered"]
        and not row["any_regression"]
    ]
    sampled = sorted(
        clean_successes,
        key=lambda row: hash_key(f"{SELECTION_SALT}|{row['id']}|{row['arm']}")
    )[:12]
    review_rows = regressions + sampled

    report = {
        "protocol": PROTOCOL, "generator_model": args.generator_model,
        "judge_model": args.judge_model, "n_content_blocks": len(BLOCKS),
        "n_layout_cases": len(cases), "n_candidates": len(candidates),
        "control": control, "layout_summaries": layout_summaries,
        "paired_summaries": paired, "operator_interaction": interaction,
        "automatic_gates_before_manual": gates, "manual_review_rows": len(review_rows),
        "interpretation_guard": (
            "P0h manipulates support co-location under a sentence-replacement interface. It does "
            "not estimate natural prevalence or establish a general semantic entanglement metric."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates", candidates), ("controls", controls), ("review", review_rows)):
        with (args.out_dir / f"support_entanglement_p0h_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "support_entanglement_p0h_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--generation-cap", type=int, default=384)
    parser.add_argument("--judge-cap", type=int, default=320)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
