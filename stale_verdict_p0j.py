"""Controlled study of selective obligation-verdict invalidation after revision."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0 import edit_ratio
from gamut_process_repair_p0b import parse_sentence_patch, release_runner, split_sentences
from support_entanglement_p0h import parse_forced_sentence_replacement


PROTOCOL = "EXPERIMENT_PROTOCOL_STALE_VERDICT_P0J.md"
OBLIGATION_IDS = tuple(f"O{i:02d}" for i in range(1, 13))
TARGET_TYPES = ("numeric", "attribution", "ordering")
OPERATORS = ("target_sentence", "section_patch", "free_patch", "full_rewrite")
REVIEW_SALT = "20260824-stale-verdict-p0j"
NEGATIVE_REVIEW_N = 36
TARGET_RECALL = 0.90


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    facts: tuple[str, ...]
    wrong_by_type: dict[str, tuple[int, str]]


@dataclass(frozen=True)
class ControlledCase:
    id: str
    scenario_id: str
    target_type: str
    title: str
    question: str
    evidence: str
    clean_answer: str
    baseline_answer: str
    obligations: dict[str, str]
    target_id: str
    target_sentence: int
    expected_before: dict[str, bool]
    witness_sentences: dict[str, tuple[int, ...]]


def rotate_scenario(
    scenario_id: str,
    title: str,
    facts: tuple[str, ...],
    wrong: dict[str, tuple[int, str]],
    offset: int,
) -> Scenario:
    """Rotate sentence order so target positions cannot be learned globally."""
    if len(facts) != 12 or set(wrong) != set(TARGET_TYPES):
        raise ValueError(f"invalid scenario specification: {scenario_id}")
    offset %= len(facts)
    rotated = facts[offset:] + facts[:offset]
    mapped: dict[str, tuple[int, str]] = {}
    for target_type, (old_index, wrong_text) in wrong.items():
        mapped[target_type] = ((old_index - offset) % len(facts), wrong_text)
    return Scenario(scenario_id, title, rotated, mapped)


def build_scenarios() -> tuple[Scenario, ...]:
    specs = (
        (
            "transit", "electric-bus pilot",
            (
                "The pilot operated 42 electric buses [E1].",
                "The depot installed 18 overnight chargers [E2].",
                "Battery preconditioning began at 06:30 [E3].",
                "Electric buses used 31% less energy than the diesel comparison [E4].",
                "Five-year operating cost was $1.8 million lower [E5].",
                "Tailpipe carbon emissions were eliminated on the trial routes [E6].",
                "Report E7 states that winter range retained 82% of its mild-weather level [E7].",
                "Peak charging demand stayed below 4.5 MW [E8].",
                "Driver training was completed for 96 operators [E9].",
                "Battery preconditioning occurred before the first passenger run [E10].",
                "The report recommends a phased electric-bus purchase [E11].",
                "Maintenance downtime fell by 14% [E12].",
            ),
            {
                "numeric": (3, "Electric buses used 13% less energy than the diesel comparison [E4]."),
                "attribution": (6, "Report E8 states that winter range retained 82% of its mild-weather level [E8]."),
                "ordering": (9, "The first passenger run occurred before battery preconditioning [E10]."),
            },
        ),
        (
            "hospital", "hospital triage rollout",
            (
                "The evaluation covered six hospital wards [E1].",
                "The system processed 18,400 triage records [E2].",
                "The privacy review finished on 12 March [E3].",
                "Median triage time fell by 11 minutes [E4].",
                "Implementation cost was $640,000 [E5].",
                "The medication-error rate remained at 1.2% [E6].",
                "Report E7 attributes the 9% reduction in repeat assessments to the nurse-led protocol [E7].",
                "No patient record left the hospital network [E8].",
                "Training was completed by 214 clinicians [E9].",
                "The privacy review occurred before the ward pilot [E10].",
                "The evaluation recommends a phased hospital-wide rollout [E11].",
                "Patient satisfaction increased by seven points [E12].",
            ),
            {
                "numeric": (3, "Median triage time fell by 1 minute [E4]."),
                "attribution": (6, "Report E8 attributes the 9% reduction in repeat assessments to the nurse-led protocol [E8]."),
                "ordering": (9, "The ward pilot occurred before the privacy review [E10]."),
            },
        ),
        (
            "river", "river-restoration trial",
            (
                "The trial monitored 14 river sites [E1].",
                "The restored wetland covered 18 hectares [E2].",
                "Sediment testing finished in April [E3].",
                "Nitrate concentration fell by 24% [E4].",
                "Annual maintenance cost fell by $310,000 [E5].",
                "The design protected 18 hectares of floodplain [E6].",
                "Report E7 attributes the improved fish count to restored side channels [E7].",
                "Peak modeled flood depth fell by 0.6 metres [E8].",
                "Local crews planted 36,000 native reeds [E9].",
                "Sediment testing occurred before dredging [E10].",
                "The report recommends expanding the wetland-restoration plan [E11].",
                "Dissolved oxygen rose by 13% [E12].",
            ),
            {
                "numeric": (3, "Nitrate concentration fell by 42% [E4]."),
                "attribution": (6, "Report E8 attributes the improved fish count to restored side channels [E8]."),
                "ordering": (9, "Dredging occurred before sediment testing [E10]."),
            },
        ),
        (
            "school", "school-menu evaluation",
            (
                "The evaluation included 23 schools [E1].",
                "The survey recorded 8,600 student meals [E2].",
                "The allergy audit finished on 3 August [E3].",
                "Meal participation rose from 62% to 74% [E4].",
                "Ingredient cost increased by $0.18 per meal [E5].",
                "Food waste fell by 16% [E6].",
                "Report E7 attributes the participation increase to the revised fresh-food menu [E7].",
                "All kitchens passed the final safety inspection [E8].",
                "Training was completed by 117 kitchen staff [E9].",
                "The allergy audit occurred before the menu launch [E10].",
                "The evaluation recommends retaining the revised menu [E11].",
                "Average fruit consumption rose by 0.7 servings [E12].",
            ),
            {
                "numeric": (3, "Meal participation rose from 62% to 64% [E4]."),
                "attribution": (6, "Report E8 attributes the participation increase to the revised fresh-food menu [E8]."),
                "ordering": (9, "The menu launch occurred before the allergy audit [E10]."),
            },
        ),
        (
            "datacenter", "data-center cooling trial",
            (
                "The trial covered 320 server racks [E1].",
                "The load test lasted 72 hours [E2].",
                "Leak testing finished on 8 May [E3].",
                "Cooling electricity use fell by 27% [E4].",
                "Projected annual savings were $2.4 million [E5].",
                "Peak rack temperature stayed below 31 C [E6].",
                "Report E7 attributes the energy reduction to the liquid-cooling loop [E7].",
                "Peak water use stayed below 44 cubic metres per hour [E8].",
                "Training was completed by 38 facility engineers [E9].",
                "Leak testing occurred before production deployment [E10].",
                "The report recommends a staged liquid-cooling rollout [E11].",
                "Unplanned shutdown time fell by 19% [E12].",
            ),
            {
                "numeric": (3, "Cooling electricity use fell by 17% [E4]."),
                "attribution": (6, "Report E8 attributes the energy reduction to the liquid-cooling loop [E8]."),
                "ordering": (9, "Production deployment occurred before leak testing [E10]."),
            },
        ),
        (
            "housing", "housing-retrofit assessment",
            (
                "The assessment covered 164 apartments [E1].",
                "The retrofit installed 82 heat pumps [E2].",
                "The asbestos survey finished in September [E3].",
                "Winter heating demand fell by 22% [E4].",
                "Average annual household cost fell by $460 [E5].",
                "Indoor temperature stayed above 19 C during the coldest week [E6].",
                "Report E7 attributes most heat savings to the wall insulation [E7].",
                "Grid demand stayed below the agreed 1.8 MW limit [E8].",
                "Training was completed by 24 maintenance workers [E9].",
                "The asbestos survey occurred before wall-insulation work [E10].",
                "The assessment recommends the staged heat-pump and insulation retrofit [E11].",
                "Reported condensation problems fell by 35% [E12].",
            ),
            {
                "numeric": (3, "Winter heating demand fell by 12% [E4]."),
                "attribution": (6, "Report E8 attributes most heat savings to the wall insulation [E8]."),
                "ordering": (9, "Wall-insulation work occurred before the asbestos survey [E10]."),
            },
        ),
        (
            "library", "library digitization review",
            (
                "The project scanned 48,000 archive pages [E1].",
                "The index covered 12 document collections [E2].",
                "Copyright clearance finished in January [E3].",
                "Median retrieval time fell by 40% [E4].",
                "Annual storage cost fell by $86,000 [E5].",
                "Ninety-eight percent of catalog records passed validation [E6].",
                "Report E7 attributes the faster retrieval to the new semantic index [E7].",
                "The public portal met its accessibility target [E8].",
                "Training was completed by 53 librarians [E9].",
                "Copyright clearance occurred before public release [E10].",
                "The review recommends expanding digitization to the remaining collections [E11].",
                "User search abandonment fell by 18% [E12].",
            ),
            {
                "numeric": (3, "Median retrieval time fell by 14% [E4]."),
                "attribution": (6, "Report E8 attributes the faster retrieval to the new semantic index [E8]."),
                "ordering": (9, "Public release occurred before copyright clearance [E10]."),
            },
        ),
        (
            "wildfire", "wildfire-sensor trial",
            (
                "The network used 63 ridge sensors [E1].",
                "The trial covered 1,900 square kilometres [E2].",
                "Radio calibration finished on 2 June [E3].",
                "The network provided a 14-minute alert lead [E4].",
                "Projected annual operating cost was $740,000 [E5].",
                "The false-alarm rate was 3.1% [E6].",
                "Report E7 attributes the alert improvement to ridge-to-ridge relays [E7].",
                "Battery availability remained above 97% [E8].",
                "Training was completed by 76 dispatchers [E9].",
                "Radio calibration occurred before the fire-season activation [E10].",
                "The report recommends permanent deployment in the northern zone [E11].",
                "Average incident-location error fell to 0.8 kilometres [E12].",
            ),
            {
                "numeric": (3, "The network provided a 4-minute alert lead [E4]."),
                "attribution": (6, "Report E8 attributes the alert improvement to ridge-to-ridge relays [E8]."),
                "ordering": (9, "The fire-season activation occurred before radio calibration [E10]."),
            },
        ),
    )
    return tuple(
        rotate_scenario(sid, title, facts, wrong, offset)
        for offset, (sid, title, facts, wrong) in enumerate(specs)
    )


def build_cases() -> list[ControlledCase]:
    cases: list[ControlledCase] = []
    for scenario in build_scenarios():
        clean = " ".join(scenario.facts)
        evidence = "\n".join(scenario.facts)
        obligations = {
            oid: f"The answer states this report finding correctly: {fact}"
            for oid, fact in zip(OBLIGATION_IDS, scenario.facts)
        }
        for target_type in TARGET_TYPES:
            target_index, wrong = scenario.wrong_by_type[target_type]
            baseline_facts = list(scenario.facts)
            baseline_facts[target_index] = wrong
            target_id = OBLIGATION_IDS[target_index]
            expected = {oid: oid != target_id for oid in OBLIGATION_IDS}
            witnesses = {
                oid: (index + 1,)
                for index, oid in enumerate(OBLIGATION_IDS)
                if oid != target_id
            }
            case = ControlledCase(
                id=f"{scenario.id}-{target_type}",
                scenario_id=scenario.id,
                target_type=target_type,
                title=scenario.title,
                question=(
                    f"Using the fixed report, write a complete 12-finding summary of the "
                    f"{scenario.title}. Preserve the reported values, source attribution, "
                    "event order, operational findings, and recommendation."
                ),
                evidence=evidence,
                clean_answer=clean,
                baseline_answer=" ".join(baseline_facts),
                obligations=obligations,
                target_id=target_id,
                target_sentence=target_index + 1,
                expected_before=expected,
                witness_sentences=witnesses,
            )
            if len(split_sentences(case.clean_answer)) != 12:
                raise RuntimeError(f"clean sentence construction failed: {case.id}")
            if len(split_sentences(case.baseline_answer)) != 12:
                raise RuntimeError(f"baseline sentence construction failed: {case.id}")
            cases.append(case)
    return cases


def numbered(answer: str) -> str:
    return "\n".join(
        f"[S{index}] {sentence}"
        for index, sentence in enumerate(split_sentences(answer), 1)
    )


def preserved_requirements(case: ControlledCase) -> str:
    return "\n".join(
        f"- {oid}: {case.obligations[oid]}"
        for oid in OBLIGATION_IDS
        if oid != case.target_id
    )


def section_span(target_sentence: int) -> tuple[int, int]:
    start = ((target_sentence - 1) // 4) * 4 + 1
    return start, start + 3


def repair_prompt(case: ControlledCase, operator: str) -> str:
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator}")
    common = (
        f"Question: {case.question}\n\nFixed report evidence:\n{case.evidence}\n\n"
        f"Failed repair target:\n- {case.target_id}: {case.obligations[case.target_id]}\n\n"
        "Already-correct requirements whose meaning must remain true:\n"
        f"{preserved_requirements(case)}\n\n"
    )
    if operator == "target_sentence":
        task = (
            f"Saved answer split into numbered sentences:\n{numbered(case.baseline_answer)}\n\n"
            f"The program will replace sentence [S{case.target_sentence}]. Return exactly one "
            "JSON object with the single field replacement. replacement must be the complete "
            "revised sentence. Fix the target and preserve all other findings. Return no markdown."
        )
    elif operator == "section_patch":
        first, last = section_span(case.target_sentence)
        task = (
            f"Saved answer split into numbered sentences:\n{numbered(case.baseline_answer)}\n\n"
            f"The program will replace the complete section [S{first}]-[S{last}]. Return exactly "
            "one JSON object with the single field replacement. replacement must be the complete "
            "revised section, including every still-correct finding from that section. Fix the "
            "target and preserve all other findings. Return no markdown."
        )
    elif operator == "free_patch":
        task = (
            f"Saved answer split into numbered sentences:\n{numbered(case.baseline_answer)}\n\n"
            "Return exactly one JSON object with fields start_sentence, end_sentence, and "
            "replacement. Replace at most four consecutive source sentences with the smallest "
            "patch that fixes the target while preserving every other finding. Return no markdown."
        )
    else:
        task = (
            f"Saved answer:\n{case.baseline_answer}\n\nReturn only the complete revised answer. "
            "Fix the failed target, retain all 12 findings, and preserve every already-correct "
            "value, attribution, event order, and recommendation. Return no commentary."
        )
    return common + "Task:\n" + task


def parse_forced_span_replacement(
    text: str, answer: str, first: int, last: int
) -> tuple[str, bool, str, tuple[int, int] | None]:
    stripped = text.strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    left, right = stripped.find("{"), stripped.rfind("}")
    if left < 0 or right < left:
        return answer, False, "no_json", None
    try:
        value = json.loads(stripped[left : right + 1])
    except json.JSONDecodeError:
        return answer, False, "invalid_json", None
    if not isinstance(value, dict) or set(value) != {"replacement"}:
        return answer, False, "wrong_fields", None
    replacement = value["replacement"]
    sentences = split_sentences(answer)
    if not 1 <= first <= last <= len(sentences):
        raise ValueError("frozen section out of range")
    if not isinstance(replacement, str) or not replacement.strip():
        return answer, False, "empty_replacement", None
    if len(replacement.split()) > 240:
        return answer, False, "replacement_too_long", None
    revised = sentences[: first - 1] + [replacement.strip()] + sentences[last:]
    return " ".join(revised), True, "valid", (first, last)


def apply_candidate(
    case: ControlledCase, operator: str, raw: str
) -> tuple[str, bool, str, tuple[int, int] | None]:
    if operator == "target_sentence":
        return parse_forced_sentence_replacement(raw, case.baseline_answer, case.target_sentence)
    if operator == "section_patch":
        return parse_forced_span_replacement(
            raw, case.baseline_answer, *section_span(case.target_sentence)
        )
    if operator == "free_patch":
        return parse_sentence_patch(raw, case.baseline_answer)
    answer = raw.strip()
    return (
        (answer, True, "full_answer", None)
        if answer
        else (case.baseline_answer, False, "empty_answer", None)
    )


def judge_prompt(case: ControlledCase, answer: str) -> str:
    requirements = "\n".join(
        f"- {oid}: {case.obligations[oid]}" for oid in OBLIGATION_IDS
    )
    return (
        "Judge only what the ANSWER visibly states against the fixed report. Do not use the report "
        "to silently correct a wrong value, citation, event order, or omitted finding. Return "
        "exactly one JSON object with field items. items must contain 12 objects in the given "
        "requirement order. Every object has exactly id, met, and witness_sentences. id is the "
        "exact requirement ID. met is a JSON boolean. A true item needs one or more one-based "
        "ANSWER sentence IDs that directly state the finding; a false item has an empty list. "
        "Sentence IDs are integers, never strings. Return no markdown or explanation.\n\n"
        f"Question: {case.question}\n\nFixed report:\n{case.evidence}\n\n"
        f"ANSWER sentences:\n{numbered(answer)}\n\nRequirements:\n{requirements}\n\nJSON:"
    )


def parse_judgment(
    text: str, sentence_count: int
) -> tuple[dict[str, dict[str, Any]], bool, str]:
    stripped = text.strip()
    if "```" in stripped:
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
    if not isinstance(value, dict) or set(value) != {"items"}:
        return {}, False, "wrong_top_level"
    if not isinstance(value["items"], list):
        return {}, False, "items_not_list"
    parsed: dict[str, dict[str, Any]] = {}
    for item in value["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "met", "witness_sentences"}:
            return {}, False, "wrong_item_fields"
        oid, met, witnesses = item["id"], item["met"], item["witness_sentences"]
        if oid not in OBLIGATION_IDS or oid in parsed or not isinstance(met, bool):
            return {}, False, "bad_item_value"
        if not isinstance(witnesses, list) or any(
            not isinstance(sid, int)
            or isinstance(sid, bool)
            or not 1 <= sid <= sentence_count
            for sid in witnesses
        ):
            return {}, False, "bad_witness"
        if (met and not witnesses) or (not met and witnesses):
            return {}, False, "inconsistent_witness"
        if len(witnesses) != len(set(witnesses)):
            return {}, False, "duplicate_witness"
        parsed[oid] = {"met": met, "witness_sentences": witnesses}
    if tuple(parsed) != OBLIGATION_IDS:
        return {}, False, "wrong_item_order"
    return parsed, True, "valid"


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")
CITATION_RE = re.compile(r"\[[A-Za-z]+\d+\]")
NEGATIONS = frozenset(("no", "not", "never", "neither", "nor", "without", "unchanged"))
FEATURE_NAMES = (
    "witness_touched",
    "distance_to_change",
    "witness_exactly_preserved",
    "max_witness_similarity",
    "obligation_changed_lexical_overlap",
    "global_edit_ratio",
    "changed_old_share",
    "changed_new_share",
    "number_symmetric_difference",
    "citation_symmetric_difference",
    "negation_symmetric_difference",
    "old_witness_position",
    "patch_valid",
    *(f"operator_{operator}" for operator in OPERATORS),
    *(f"target_{target_type}" for target_type in TARGET_TYPES),
    "generator_qwen3",
)


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def sentence_diff(old_answer: str, new_answer: str) -> dict[str, Any]:
    old = split_sentences(old_answer)
    new = split_sentences(new_answer)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    changed_old: set[int] = set()
    changed_new: set[int] = set()
    opcodes: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_old.update(range(i1 + 1, i2 + 1))
        changed_new.update(range(j1 + 1, j2 + 1))
        if tag == "insert":
            if i1 > 0:
                changed_old.add(i1)
            elif old:
                changed_old.add(1)
        opcodes.append({
            "tag": tag,
            "old_start": i1 + 1,
            "old_end": i2,
            "new_start": j1 + 1,
            "new_end": j2,
        })
    return {
        "old_sentences": old,
        "new_sentences": new,
        "changed_old_sentence_ids": sorted(changed_old),
        "changed_new_sentence_ids": sorted(changed_new),
        "opcodes": opcodes,
    }


def symmetric_count(pattern: re.Pattern[str], old_text: str, new_text: str) -> int:
    old = Counter(match.lower() for match in pattern.findall(old_text))
    new = Counter(match.lower() for match in pattern.findall(new_text))
    return sum((old - new).values()) + sum((new - old).values())


def negation_count(old_text: str, new_text: str) -> int:
    old = Counter(token for token in tokens(old_text) if token in NEGATIONS)
    new = Counter(token for token in tokens(new_text) if token in NEGATIONS)
    return sum((old - new).values()) + sum((new - old).values())


def transition_features(
    case: ControlledCase,
    oid: str,
    answer: str,
    operator: str,
    generator_model: str,
    patch_valid: bool,
    diff: dict[str, Any] | None = None,
) -> dict[str, float]:
    if oid == case.target_id or not case.expected_before[oid]:
        raise ValueError("prediction unit must be an old-SAT non-target obligation")
    diff = diff or sentence_diff(case.baseline_answer, answer)
    witness_id = case.witness_sentences[oid][0]
    old_sentences = diff["old_sentences"]
    new_sentences = diff["new_sentences"]
    changed_old = set(diff["changed_old_sentence_ids"])
    changed_new = set(diff["changed_new_sentence_ids"])
    old_witness = old_sentences[witness_id - 1]
    changed_text = " ".join(
        new_sentences[index - 1] for index in sorted(changed_new) if index <= len(new_sentences)
    )
    if changed_old:
        distance = min(abs(witness_id - index) for index in changed_old)
        distance_norm = min(distance, 12) / 12
    else:
        distance_norm = 1.0
    max_similarity = max(
        (
            difflib.SequenceMatcher(None, old_witness, sentence, autojunk=False).ratio()
            for sentence in new_sentences
        ),
        default=0.0,
    )
    values: dict[str, float] = {
        "witness_touched": float(witness_id in changed_old),
        "distance_to_change": float(distance_norm),
        "witness_exactly_preserved": float(old_witness in new_sentences),
        "max_witness_similarity": float(max_similarity),
        "obligation_changed_lexical_overlap": jaccard(tokens(case.obligations[oid]), tokens(changed_text)),
        "global_edit_ratio": float(edit_ratio(case.baseline_answer, answer)),
        "changed_old_share": len(changed_old) / max(1, len(old_sentences)),
        "changed_new_share": len(changed_new) / max(1, len(new_sentences)),
        "number_symmetric_difference": float(symmetric_count(NUMBER_RE, old_witness, changed_text)),
        "citation_symmetric_difference": float(symmetric_count(CITATION_RE, old_witness, changed_text)),
        "negation_symmetric_difference": float(negation_count(old_witness, changed_text)),
        "old_witness_position": witness_id / 12,
        "patch_valid": float(patch_valid),
    }
    values.update({f"operator_{name}": float(operator == name) for name in OPERATORS})
    values.update({f"target_{name}": float(case.target_type == name) for name in TARGET_TYPES})
    values["generator_qwen3"] = float("Qwen3" in generator_model)
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError(f"feature order mismatch: {tuple(values)}")
    return values


def policy_metrics(
    transitions: list[dict[str, Any]], selected: list[bool], revision_count: int
) -> dict[str, Any]:
    if len(transitions) != len(selected):
        raise ValueError("selection length mismatch")
    stale_total = sum(row["stale"] for row in transitions)
    caught = sum(row["stale"] and choose for row, choose in zip(transitions, selected))
    selected_count = sum(selected)
    checks = revision_count + selected_count
    total_verdicts = revision_count * 12
    return {
        "stale_total": stale_total,
        "stale_caught": caught,
        "stale_recall": caught / stale_total if stale_total else 0.0,
        "non_target_rechecks": selected_count,
        "total_rechecks_including_target": checks,
        "verification_saving": 1 - checks / total_verdicts if total_verdicts else 0.0,
    }


def frozen_policy_selections(
    transitions: list[dict[str, Any]], name: str
) -> list[bool]:
    selections: list[bool] = []
    for row in transitions:
        feature = row["features"]
        if name == "full":
            select = True
        elif name == "target_only":
            select = False
        elif name == "witness_overlap":
            select = bool(feature["witness_touched"])
        elif name == "proximity":
            select = feature["distance_to_change"] <= (1 / 12)
        elif name == "witness_similarity":
            select = feature["max_witness_similarity"] < 0.90
        elif name == "union":
            select = bool(
                feature["witness_touched"]
                or feature["max_witness_similarity"] < 0.90
                or (
                    feature["obligation_changed_lexical_overlap"] >= 0.20
                    and feature["distance_to_change"] <= (2 / 12)
                )
            )
        else:
            raise ValueError(f"unknown policy: {name}")
        selections.append(select)
    return selections


def threshold_for_recall(scores: list[float], labels: list[int], target: float) -> float:
    positives = sum(labels)
    if not scores or not positives:
        return math.inf
    candidates = sorted(set(scores), reverse=True)
    candidates.append(min(scores) - 1e-12)
    best = candidates[-1]
    best_selected = len(scores) + 1
    for threshold in candidates:
        selected = [score >= threshold for score in scores]
        recall = sum(label and keep for label, keep in zip(labels, selected)) / positives
        count = sum(selected)
        if recall >= target and count < best_selected:
            best, best_selected = threshold, count
    return best


def nested_grouped_predictions(
    transitions: list[dict[str, Any]], target_recall: float = TARGET_RECALL
) -> tuple[list[bool], list[float], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = sorted({row["scenario_id"] for row in transitions})
    selections = [False] * len(transitions)
    scores = [0.0] * len(transitions)
    folds: list[dict[str, Any]] = []

    def matrix(indices: list[int]):
        return np.asarray(
            [[transitions[index]["features"][name] for name in FEATURE_NAMES] for index in indices],
            dtype=np.float64,
        )

    def labels(indices: list[int]):
        return np.asarray([int(transitions[index]["stale"]) for index in indices], dtype=np.int64)

    def make_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="liblinear", random_state=0
            ),
        )

    for outer_group in groups:
        test_indices = [
            index for index, row in enumerate(transitions) if row["scenario_id"] == outer_group
        ]
        train_indices = [
            index for index, row in enumerate(transitions) if row["scenario_id"] != outer_group
        ]
        train_groups = sorted({transitions[index]["scenario_id"] for index in train_indices})
        y_train = labels(train_indices)
        if len(set(y_train.tolist())) < 2 or len(train_groups) < 3:
            raise RuntimeError(f"insufficient outer training labels for {outer_group}")
        inner_scores: dict[int, float] = {}
        for inner_group in train_groups:
            inner_valid = [
                index for index in train_indices
                if transitions[index]["scenario_id"] == inner_group
            ]
            inner_train = [index for index in train_indices if index not in set(inner_valid)]
            y_inner = labels(inner_train)
            if len(set(y_inner.tolist())) < 2:
                continue
            model = make_model()
            model.fit(matrix(inner_train), y_inner)
            predicted = model.predict_proba(matrix(inner_valid))[:, 1]
            inner_scores.update({index: float(score) for index, score in zip(inner_valid, predicted)})
        if set(inner_scores) != set(train_indices):
            raise RuntimeError(f"incomplete inner OOF predictions for {outer_group}")
        ordered_inner = [inner_scores[index] for index in train_indices]
        threshold = threshold_for_recall(
            ordered_inner, labels(train_indices).tolist(), target_recall
        )
        model = make_model()
        model.fit(matrix(train_indices), y_train)
        predicted = model.predict_proba(matrix(test_indices))[:, 1]
        for index, score in zip(test_indices, predicted):
            scores[index] = float(score)
            selections[index] = bool(score >= threshold)
        folds.append({
            "held_out_scenario": outer_group,
            "train_n": len(train_indices),
            "test_n": len(test_indices),
            "inner_threshold": threshold,
            "inner_positive_n": int(sum(labels(train_indices))),
            "test_positive_n": int(sum(labels(test_indices))),
        })
    return selections, scores, folds


def matched_random(
    transitions: list[dict[str, Any]], learned_selected: list[bool], draws: int = 1000
) -> dict[str, float]:
    by_revision: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(transitions):
        by_revision[row["revision_id"]].append(index)
    stale_total = sum(row["stale"] for row in transitions)
    if not stale_total:
        return {"draws": draws, "mean_stale_recall": 0.0, "p95_stale_recall": 0.0}
    rng = random.Random(20260824)
    recalls: list[float] = []
    for _ in range(draws):
        caught = 0
        for indices in by_revision.values():
            budget = sum(learned_selected[index] for index in indices)
            chosen = set(rng.sample(indices, budget)) if budget else set()
            caught += sum(transitions[index]["stale"] for index in chosen)
        recalls.append(caught / stale_total)
    ordered = sorted(recalls)
    return {
        "draws": draws,
        "mean_stale_recall": mean(recalls),
        "p95_stale_recall": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def analyze_transitions(
    transitions: list[dict[str, Any]], revision_count: int
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for name in ("full", "target_only", "witness_overlap", "proximity", "witness_similarity", "union"):
        policies[name] = policy_metrics(
            transitions, frozen_policy_selections(transitions, name), revision_count
        )
    positive_scenarios = {row["scenario_id"] for row in transitions if row["stale"]}
    learnable = sum(row["stale"] for row in transitions) >= 10 and len(positive_scenarios) >= 4
    learned: dict[str, Any] = {"available": False}
    if learnable:
        selected, scores, folds = nested_grouped_predictions(transitions)
        for row, score, choose in zip(transitions, scores, selected):
            row["learned_oof_score"] = score
            row["learned_selected"] = choose
        learned = policy_metrics(transitions, selected, revision_count)
        learned.update({"available": True, "outer_folds": folds})
        learned["matched_random"] = matched_random(transitions, selected)
    policies["learned"] = learned
    stale_total = sum(row["stale"] for row in transitions)
    policies["oracle_upper_bound"] = {
        "stale_recall": 1.0 if stale_total else 0.0,
        "verification_saving": (
            1 - (revision_count + stale_total) / (revision_count * 12)
            if revision_count else 0.0
        ),
        "note": "Uses labels and is only the maximum saving at 100% stale recall.",
    }
    return policies


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rate_by(
    candidates: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row[field])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        valid_judged = [row for row in rows if row["judge_valid"]]
        eligible = [row for row in rows if row["eligible_transition"]]
        result[key] = {
            "attempts": len(rows),
            "patch_valid_rate": mean(row["patch_valid"] for row in rows),
            "judge_valid_rate": mean(row["judge_valid"] for row in rows),
            "target_repair_rate_among_valid_judgments": (
                mean(row["target_recovered"] for row in valid_judged) if valid_judged else 0.0
            ),
            "eligible_revisions": len(eligible),
            "stale_non_target_verdicts": sum(len(row["stale_ids"]) for row in eligible),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases()
    case_by_id = {case.id: case for case in cases}
    generation_keys = [
        (case.id, operator) for case in cases for operator in OPERATORS
    ]
    candidates: list[dict[str, Any]] = []

    for generator_model in args.generator_models:
        prompts = [
            repair_prompt(case_by_id[case_id], operator)
            for case_id, operator in generation_keys
        ]
        generator = ModelRunner(generator_model)
        raw_outputs = generator.generate(
            prompts, args.generation_batch_size, args.generation_cap
        )
        release_runner(generator)
        for (case_id, operator), raw in zip(generation_keys, raw_outputs):
            case = case_by_id[case_id]
            answer, valid, mode, span = apply_candidate(case, operator, raw)
            diff = sentence_diff(case.baseline_answer, answer)
            generator_slug = "qwen3_8b" if "Qwen3" in generator_model else "qwen2_5_7b"
            revision_id = f"{case.id}|{operator}|{generator_slug}"
            candidates.append({
                "revision_id": revision_id,
                "case_id": case.id,
                "scenario_id": case.scenario_id,
                "target_type": case.target_type,
                "target_id": case.target_id,
                "target_sentence": case.target_sentence,
                "operator": operator,
                "generator_model": generator_model,
                "question": case.question,
                "evidence": case.evidence,
                "obligations": case.obligations,
                "baseline_answer": case.baseline_answer,
                "answer": answer,
                "raw_generation": raw,
                "patch_valid": valid,
                "parse_mode": mode,
                "declared_span": list(span) if span else None,
                "actual_diff": {
                    key: value for key, value in diff.items()
                    if key not in ("old_sentences", "new_sentences")
                },
                "edit_ratio": float(edit_ratio(case.baseline_answer, answer)),
            })

    control_specs: list[tuple[str, str]] = []
    judge_prompts: list[str] = []
    for case in cases:
        control_specs.append((case.id, "baseline"))
        judge_prompts.append(judge_prompt(case, case.baseline_answer))
    for scenario in build_scenarios():
        clean_case = next(case for case in cases if case.scenario_id == scenario.id)
        control_specs.append((clean_case.id, "clean"))
        judge_prompts.append(judge_prompt(clean_case, clean_case.clean_answer))
    judge_prompts.extend(
        judge_prompt(case_by_id[row["case_id"]], row["answer"])
        for row in candidates
    )

    judge = ModelRunner(args.judge_model)
    raw_judgments = judge.generate(judge_prompts, args.judge_batch_size, args.judge_cap)
    release_runner(judge)

    controls: list[dict[str, Any]] = []
    parse_correct = positive_correct = negative_correct = 0
    positive_total = negative_total = 0
    for (case_id, kind), raw in zip(
        control_specs, raw_judgments[: len(control_specs)]
    ):
        case = case_by_id[case_id]
        answer = case.baseline_answer if kind == "baseline" else case.clean_answer
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(answer)))
        expected = (
            case.expected_before
            if kind == "baseline"
            else {oid: True for oid in OBLIGATION_IDS}
        )
        parse_correct += int(valid)
        for oid, expected_met in expected.items():
            if expected_met:
                positive_total += 1
                positive_correct += int(valid and parsed[oid]["met"])
            else:
                negative_total += 1
                negative_correct += int(valid and not parsed[oid]["met"])
        controls.append({
            "case_id": case_id,
            "kind": kind,
            "answer": answer,
            "expected": expected,
            "judge_valid": valid,
            "parse_mode": mode,
            "parsed": parsed,
            "raw_judgment": raw,
        })
    control = {
        "n_answers": len(control_specs),
        "parse_validity": parse_correct / len(control_specs),
        "positive_accuracy": positive_correct / positive_total,
        "negative_accuracy": negative_correct / negative_total,
    }
    control["usable"] = all(
        control[name] >= 0.95
        for name in ("parse_validity", "positive_accuracy", "negative_accuracy")
    )

    candidate_raw = raw_judgments[len(control_specs):]
    for row, raw in zip(candidates, candidate_raw):
        case = case_by_id[row["case_id"]]
        parsed, valid, mode = parse_judgment(raw, len(split_sentences(row["answer"])))
        target_recovered = bool(valid and parsed[case.target_id]["met"])
        stale_ids = [
            oid for oid in OBLIGATION_IDS
            if oid != case.target_id and valid and not parsed[oid]["met"]
        ]
        eligible = bool(row["patch_valid"] and valid and target_recovered)
        row.update({
            "judge_raw": raw,
            "judge_valid": valid,
            "judge_parse_mode": mode,
            "final_ledger": parsed,
            "target_recovered": target_recovered,
            "eligible_transition": eligible,
            "stale_ids": stale_ids if eligible else [],
        })

    transitions: list[dict[str, Any]] = []
    eligible_candidates = [row for row in candidates if row["eligible_transition"]]
    for row in eligible_candidates:
        case = case_by_id[row["case_id"]]
        diff = sentence_diff(case.baseline_answer, row["answer"])
        for oid in OBLIGATION_IDS:
            if oid == case.target_id:
                continue
            features = transition_features(
                case,
                oid,
                row["answer"],
                row["operator"],
                row["generator_model"],
                row["patch_valid"],
                diff,
            )
            transitions.append({
                "transition_id": f"{row['revision_id']}|{oid}",
                "revision_id": row["revision_id"],
                "case_id": row["case_id"],
                "scenario_id": row["scenario_id"],
                "target_type": row["target_type"],
                "target_id": row["target_id"],
                "obligation_id": oid,
                "obligation": case.obligations[oid],
                "old_witness_sentences": list(case.witness_sentences[oid]),
                "operator": row["operator"],
                "generator_model": row["generator_model"],
                "baseline_answer": case.baseline_answer,
                "revised_answer": row["answer"],
                "evidence": case.evidence,
                "stale": oid in row["stale_ids"],
                "candidate_label_source": args.judge_model,
                "manual_label": None,
                "features": features,
            })

    policies = analyze_transitions(transitions, len(eligible_candidates))
    stale_rows = [row for row in transitions if row["stale"]]
    stale_scenarios = {row["scenario_id"] for row in stale_rows}
    stale_operators = {row["operator"] for row in stale_rows}
    stale_generators = {row["generator_model"] for row in stale_rows}
    phenomenon_candidate = (
        len(stale_rows) >= 20
        and len(stale_scenarios) >= 5
        and len(stale_operators) >= 3
    )
    learned = policies["learned"]
    single_heuristics = [
        policies[name] for name in ("witness_overlap", "proximity", "witness_similarity")
        if policies[name]["stale_recall"] >= TARGET_RECALL
    ]
    best_single_saving = max(
        (metric["verification_saving"] for metric in single_heuristics), default=0.0
    )
    predictability_candidate = bool(
        learned.get("available")
        and learned["stale_recall"] >= TARGET_RECALL
        and learned["verification_saving"] >= 0.25
        and learned["stale_recall"]
            - learned["matched_random"]["mean_stale_recall"] >= 0.15
        and learned["verification_saving"] - best_single_saving >= 0.05
    )

    negative_rows = [row for row in transitions if not row["stale"]]
    sampled_negative = sorted(
        negative_rows,
        key=lambda row: hash_key(f"{REVIEW_SALT}|{row['transition_id']}"),
    )[:NEGATIVE_REVIEW_N]
    review_rows: list[dict[str, Any]] = []
    for row in stale_rows + sampled_negative:
        review = dict(row)
        review["review_reason"] = "candidate_stale" if row["stale"] else "frozen_negative_sample"
        review["manual_questions"] = {
            "old_answer_met_obligation": None,
            "revised_answer_met_obligation": None,
            "candidate_stale_label_correct": None,
            "evidence_leakage_or_ambiguity": None,
            "notes": "",
        }
        review_rows.append(review)

    report = {
        "protocol": PROTOCOL,
        "generator_models": args.generator_models,
        "judge_model": args.judge_model,
        "n_scenarios": len(build_scenarios()),
        "n_cases": len(cases),
        "n_attempted_revisions": len(candidates),
        "n_eligible_target_recovered_revisions": len(eligible_candidates),
        "n_prediction_transitions": len(transitions),
        "control": control,
        "candidate_stale_non_target_verdicts": len(stale_rows),
        "candidate_positive_scenarios": sorted(stale_scenarios),
        "candidate_positive_operators": sorted(stale_operators),
        "candidate_positive_generators": sorted(stale_generators),
        "by_operator": rate_by(candidates, "operator"),
        "by_generator": rate_by(candidates, "generator_model"),
        "by_target_type": rate_by(candidates, "target_type"),
        "policies_using_candidate_labels": policies,
        "automatic_candidate_gates_before_manual": {
            "gate_1_verifier_apparatus": control["usable"],
            "gate_2_candidate_phenomenon_support": phenomenon_candidate,
            "gate_3_candidate_useful_predictability": predictability_candidate,
            "scientific_all_gates": False,
            "scientific_status": "pending mandatory manual audit",
        },
        "manual_review": {
            "all_candidate_positives": len(stale_rows),
            "frozen_candidate_negatives": len(sampled_negative),
            "total_rows": len(review_rows),
        },
        "interpretation_guard": (
            "Candidate labels are verifier outputs on a controlled synthetic matrix. They do not "
            "establish natural prevalence, a witness-overlap mechanism, or a scientific gate pass "
            "until the frozen manual audit is complete."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "stale_verdict_p0j_candidates.jsonl", candidates)
    write_jsonl(args.out_dir / "stale_verdict_p0j_controls.jsonl", controls)
    write_jsonl(args.out_dir / "stale_verdict_p0j_transitions.jsonl", transitions)
    write_jsonl(args.out_dir / "stale_verdict_p0j_review.jsonl", review_rows)
    (args.out_dir / "stale_verdict_p0j_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator-models",
        nargs="+",
        default=("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"),
    )
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=2)
    parser.add_argument("--generation-cap", type=int, default=640)
    parser.add_argument("--judge-cap", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
