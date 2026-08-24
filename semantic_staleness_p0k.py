"""Controlled semantic-staleness screen with unchanged conclusion witnesses."""

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
from gamut_process_repair_p0b import release_runner, split_sentences


PROTOCOL = "EXPERIMENT_PROTOCOL_SEMANTIC_STALENESS_P0K.md"
MECHANISMS = ("comparison", "attribution", "derived", "temporal", "definition")
ARMS = ("dependency_edit", "harmless_edit")
TARGET_RECALL = 0.90


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    option_a: str
    option_b: str
    metric: str
    unit: str
    claim: str
    alternate_claim: str
    event_a: str
    event_b: str
    subject: str
    class_name: str
    hard_alias: bool
    index: int


@dataclass(frozen=True)
class SemanticCase:
    id: str
    domain_id: str
    mechanism: str
    hard_alias: bool
    old_document: str
    dependency_document: str
    harmless_document: str
    obligation: str
    conclusion_sentence: int
    dependency_sentence_ids: tuple[int, ...]
    harmless_sentence_id: int
    oracle_old_met: bool
    oracle_dependency_met: bool
    oracle_harmless_met: bool
    oracle_metadata: dict[str, Any]


DOMAINS = (
    Domain("transit", "regional transit report", "Amber route", "Blue route", "energy use", "kWh/km", "winter reliability improved", "summer ridership improved", "battery validation", "passenger launch", "Depot 14", "priority depot", False, 0),
    Domain("hospital", "hospital operations report", "North ward", "South ward", "triage delay", "minutes", "repeat assessments declined", "bed turnover improved", "privacy review", "ward rollout", "Patient K", "high-risk patient", False, 1),
    Domain("river", "river restoration report", "Upper reach", "Lower reach", "nitrate level", "mg/L", "fish abundance increased", "reed coverage increased", "sediment audit", "dredging", "Site R", "priority site", False, 2),
    Domain("school", "school nutrition report", "Fresh menu", "Legacy menu", "food waste", "kg/day", "meal participation increased", "fruit intake increased", "allergy audit", "menu launch", "School M", "priority school", False, 3),
    Domain("datacenter", "data-center cooling report", "Liquid loop", "Air system", "cooling demand", "MWh", "rack stability improved", "water demand declined", "leak test", "production deployment", "Rack Q", "high-risk rack", True, 4),
    Domain("housing", "housing retrofit report", "Insulated block", "Control block", "heating demand", "MWh", "winter comfort improved", "condensation declined", "asbestos survey", "insulation work", "Building H", "priority building", True, 5),
    Domain("library", "library digitization report", "Semantic index", "Keyword index", "retrieval time", "seconds", "search abandonment declined", "catalog coverage increased", "copyright clearance", "public release", "Collection L", "high-risk collection", True, 6),
    Domain("wildfire", "wildfire sensor report", "Ridge network", "Valley network", "alert delay", "minutes", "location accuracy improved", "battery availability increased", "radio calibration", "season activation", "Zone W", "priority zone", True, 7),
)


def render(sentences: list[str]) -> str:
    document = " ".join(sentences)
    if len(split_sentences(document)) != 10:
        raise RuntimeError(f"document did not retain ten sentences: {document}")
    return document


def place_record(context: list[str], record: str, flip: bool) -> tuple[list[str], int]:
    """Place one composite record in a counterbalanced sentence slot."""
    record_slot = 6 if flip else 3
    sentences = list(context)
    sentences[record_slot - 1] = record
    return sentences, record_slot


def base_context(domain: Domain, glossary: str, conclusion: str) -> list[str]:
    return [
        f"This is the frozen {domain.title}.",
        glossary,
        "The primary register was locked before the review.",
        "The review applied one reporting convention throughout.",
        "The supplementary appendix was checked separately.",
        "The consolidated registered values are reported here.",
        f"The audit covers {40 + domain.index} recorded units.",
        "All comparisons use the same reporting window.",
        conclusion,
        "The document was approved for internal review.",
    ]


def comparison_case(domain: Domain) -> SemanticCase:
    a_old, b_value, a_new = 20 + domain.index, 30 + domain.index, 40 + domain.index
    if domain.hard_alias:
        glossary = f"In conclusions, the reference option means {domain.option_a}, and the benchmark means {domain.option_b}."
        conclusion = f"The reference option recorded lower {domain.metric} than the benchmark."
    else:
        glossary = "No aliases are used in the comparison conclusion."
        conclusion = f"{domain.option_a} recorded lower {domain.metric} than {domain.option_b}."
    old_record = (
        f"The paired register records {domain.option_a} at {a_old} {domain.unit} and "
        f"{domain.option_b} at {b_value} {domain.unit} for {domain.metric}."
    )
    dependency_record = (
        f"The paired register records {domain.option_a} at {a_new} {domain.unit} and "
        f"{domain.option_b} at {b_value} {domain.unit} for {domain.metric}."
    )
    harmless_b = b_value + 20
    harmless_record = (
        f"The paired register records {domain.option_a} at {a_old} {domain.unit} and "
        f"{domain.option_b} at {harmless_b} {domain.unit} for {domain.metric}."
    )
    sentences, record_id = place_record(
        base_context(domain, glossary, conclusion), old_record, (domain.index + 0) % 2 == 1,
    )
    dependency = list(sentences)
    dependency[record_id - 1] = dependency_record
    harmless = list(sentences)
    harmless[record_id - 1] = harmless_record
    dependency_ids = ((2, record_id) if domain.hard_alias else (record_id,))
    return SemanticCase(
        f"{domain.id}-comparison", domain.id, "comparison", domain.hard_alias,
        render(sentences), render(dependency), render(harmless),
        f"The comparison conclusion in sentence 9 is numerically correct under the document's reported {domain.metric} values.",
        9, dependency_ids, record_id, a_old < b_value, a_new < b_value, a_old < harmless_b,
        {"a_old": a_old, "a_new": a_new, "b_old": b_value, "b_harmless": harmless_b, "relation": "a_less_than_b"},
    )


def attribution_case(domain: Domain) -> SemanticCase:
    if domain.hard_alias:
        glossary = "In conclusions, the primary source means report R1, and the secondary source means report R2."
        conclusion = f"The claim that {domain.claim} is supported by the primary source."
    else:
        glossary = "The report labels R1 and R2 are used directly in the conclusion."
        conclusion = f"The claim that {domain.claim} is supported by report R1."
    old_record = f"Report R1 concludes that {domain.claim}, while report R2 concludes that {domain.alternate_claim}."
    dependency_record = f"Report R1 concludes that {domain.alternate_claim}, while report R2 concludes that {domain.alternate_claim}."
    harmless_record = f"Report R1 concludes that {domain.claim}, while report R2 concludes that {domain.claim}."
    sentences, record_id = place_record(
        base_context(domain, glossary, conclusion), old_record, (domain.index + 1) % 2 == 1,
    )
    dependency = list(sentences)
    dependency[record_id - 1] = dependency_record
    harmless = list(sentences)
    harmless[record_id - 1] = harmless_record
    dependency_ids = ((2, record_id) if domain.hard_alias else (record_id,))
    return SemanticCase(
        f"{domain.id}-attribution", domain.id, "attribution", domain.hard_alias,
        render(sentences), render(dependency), render(harmless),
        "The evidence-attribution conclusion in sentence 9 correctly identifies a source that supports the stated claim.",
        9, dependency_ids, record_id, True, False, True,
        {"old_r1_claim": domain.claim, "new_r1_claim": domain.alternate_claim, "target_claim": domain.claim},
    )


def derived_case(domain: Domain) -> SemanticCase:
    revenue = 130 + 5 * domain.index
    old_cost = 100 + 5 * domain.index
    new_cost = old_cost + 20
    profit = revenue - old_cost
    if domain.hard_alias:
        glossary = "In conclusions, net balance means reported revenue minus reported operating cost."
        conclusion = f"The net balance is {profit} million dollars."
    else:
        glossary = "Profit is defined as reported revenue minus reported operating cost."
        conclusion = f"The reported profit is {profit} million dollars."
    old_staff = 24 + domain.index
    new_staff = old_staff + 1
    harmless_revenue = revenue + 20
    old_record = f"The ledger records revenue of {revenue} million dollars, operating cost of {old_cost} million dollars, and {old_staff} staff positions."
    dependency_record = f"The ledger records revenue of {revenue} million dollars, operating cost of {new_cost} million dollars, and {new_staff} staff positions."
    harmless_record = f"The ledger records revenue of {harmless_revenue} million dollars, operating cost of {new_cost} million dollars, and {old_staff} staff positions."
    sentences, record_id = place_record(
        base_context(domain, glossary, conclusion), old_record, (domain.index + 2) % 2 == 1,
    )
    dependency = list(sentences)
    dependency[record_id - 1] = dependency_record
    harmless = list(sentences)
    harmless[record_id - 1] = harmless_record
    dependency_ids = (2, record_id)
    return SemanticCase(
        f"{domain.id}-derived", domain.id, "derived", domain.hard_alias,
        render(sentences), render(dependency), render(harmless),
        "The derived financial conclusion in sentence 9 equals revenue minus operating cost under the current document version.",
        9, dependency_ids, record_id, revenue - old_cost == profit,
        revenue - new_cost == profit, harmless_revenue - new_cost == profit,
        {"revenue": revenue, "harmless_revenue": harmless_revenue, "old_cost": old_cost, "new_cost": new_cost, "stated_result": profit},
    )


def temporal_case(domain: Domain) -> SemanticCase:
    year_a, year_b, year_a_new = 2010 + domain.index, 2015 + domain.index, 2020 + domain.index
    if domain.hard_alias:
        glossary = f"In conclusions, the validation milestone means {domain.event_a}, and the deployment milestone means {domain.event_b}."
        conclusion = "The validation milestone occurred before the deployment milestone."
    else:
        glossary = "Event names are used directly in the ordering conclusion."
        conclusion = f"{domain.event_a.capitalize()} occurred before {domain.event_b}."
    harmless_b = year_b + 10
    old_record = f"The milestone register dates {domain.event_a} to {year_a} and {domain.event_b} to {year_b}."
    dependency_record = f"The milestone register dates {domain.event_a} to {year_a_new} and {domain.event_b} to {year_b}."
    harmless_record = f"The milestone register dates {domain.event_a} to {year_a} and {domain.event_b} to {harmless_b}."
    sentences, record_id = place_record(
        base_context(domain, glossary, conclusion), old_record, (domain.index + 3) % 2 == 1,
    )
    dependency = list(sentences)
    dependency[record_id - 1] = dependency_record
    harmless = list(sentences)
    harmless[record_id - 1] = harmless_record
    dependency_ids = ((2, record_id) if domain.hard_alias else (record_id,))
    return SemanticCase(
        f"{domain.id}-temporal", domain.id, "temporal", domain.hard_alias,
        render(sentences), render(dependency), render(harmless),
        "The temporal-order conclusion in sentence 9 is consistent with the event dates in the current document version.",
        9, dependency_ids, record_id, year_a < year_b, year_a_new < year_b, year_a < harmless_b,
        {"a_old": year_a, "a_new": year_a_new, "b_old": year_b, "b_harmless": harmless_b, "relation": "a_before_b"},
    )


def definition_case(domain: Domain) -> SemanticCase:
    old_threshold = 60 + domain.index
    score = 75 + domain.index
    new_threshold = 85 + domain.index
    glossary = f"The label {domain.class_name} is assigned only when a recorded score exceeds the current threshold."
    conclusion = f"{domain.subject} belongs to the {domain.class_name} category."
    harmless_score = score + 20
    old_record = f"The classification register sets the {domain.class_name} threshold above {old_threshold} and records {domain.subject} at score {score}."
    dependency_record = f"The classification register sets the {domain.class_name} threshold above {new_threshold} and records {domain.subject} at score {score}."
    harmless_record = f"The classification register sets the {domain.class_name} threshold above {old_threshold} and records {domain.subject} at score {harmless_score}."
    sentences, record_id = place_record(
        base_context(domain, glossary, conclusion), old_record, (domain.index + 4) % 2 == 1,
    )
    dependency = list(sentences)
    dependency[record_id - 1] = dependency_record
    harmless = list(sentences)
    harmless[record_id - 1] = harmless_record
    dependency_ids = (2, record_id)
    return SemanticCase(
        f"{domain.id}-definition", domain.id, "definition", domain.hard_alias,
        render(sentences), render(dependency), render(harmless),
        f"The classification conclusion in sentence 9 follows the current threshold definition and {domain.subject}'s recorded score.",
        9, dependency_ids, record_id, score > old_threshold, score > new_threshold, harmless_score > old_threshold,
        {"old_score": score, "harmless_score": harmless_score, "old_threshold": old_threshold, "new_threshold": new_threshold, "relation": "score_above_threshold"},
    )


BUILDERS = (comparison_case, attribution_case, derived_case, temporal_case, definition_case)


def build_cases() -> list[SemanticCase]:
    cases = [builder(domain) for domain in DOMAINS for builder in BUILDERS]
    for case in cases:
        if not case.oracle_old_met or case.oracle_dependency_met or not case.oracle_harmless_met:
            raise RuntimeError(f"oracle construction failed: {case.id}")
        old = split_sentences(case.old_document)
        dependency = split_sentences(case.dependency_document)
        harmless = split_sentences(case.harmless_document)
        target = case.conclusion_sentence - 1
        if old[target] != dependency[target] or old[target] != harmless[target]:
            raise RuntimeError(f"conclusion text changed: {case.id}")
        if case.conclusion_sentence in case.dependency_sentence_ids or case.conclusion_sentence == case.harmless_sentence_id:
            raise RuntimeError(f"edit overlaps conclusion witness: {case.id}")
        if sum(a != b for a, b in zip(old, dependency)) != 1:
            raise RuntimeError(f"dependency edit is not one sentence: {case.id}")
        if sum(a != b for a, b in zip(old, harmless)) != 1:
            raise RuntimeError(f"harmless edit is not one sentence: {case.id}")
    return cases


def numbered(document: str) -> str:
    return "\n".join(
        f"[S{index}] {sentence}"
        for index, sentence in enumerate(split_sentences(document), 1)
    )


def judge_prompt(case: SemanticCase, document: str) -> str:
    return (
        "Evaluate whether the criterion is logically true under this document version. The "
        "criterion concerns the validity of a conclusion, not merely whether its words appear. "
        "An unchanged conclusion can be false if an upstream value, source, date, definition, or "
        "premise changed. Resolve any explicit glossary aliases. Return exactly one JSON object "
        "with fields met, conclusion_sentence, and dependency_sentences. met is a JSON Boolean. "
        f"conclusion_sentence must identify the sentence making the audited conclusion. "
        "dependency_sentences must be a non-empty list of integer sentence IDs containing the "
        "facts needed to check it. Return no markdown or explanation.\n\n"
        f"Document:\n{numbered(document)}\n\nCriterion:\n{case.obligation}\n\nJSON:"
    )


def parse_judgment(
    raw: str, sentence_count: int
) -> tuple[dict[str, Any], bool, str]:
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}, False, "no_json"
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}, False, "invalid_json"
    if not isinstance(value, dict) or set(value) != {
        "met", "conclusion_sentence", "dependency_sentences"
    }:
        return {}, False, "wrong_fields"
    met = value["met"]
    target = value["conclusion_sentence"]
    dependencies = value["dependency_sentences"]
    if not isinstance(met, bool):
        return {}, False, "bad_met"
    if not isinstance(target, int) or isinstance(target, bool) or not 1 <= target <= sentence_count:
        return {}, False, "bad_conclusion"
    if not isinstance(dependencies, list) or not dependencies:
        return {}, False, "bad_dependencies"
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 1 <= index <= sentence_count
        for index in dependencies
    ):
        return {}, False, "bad_dependency_index"
    if len(dependencies) != len(set(dependencies)):
        return {}, False, "duplicate_dependency"
    return {
        "met": met,
        "conclusion_sentence": target,
        "dependency_sentences": dependencies,
    }, True, "valid"


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")
CITATION_RE = re.compile(r"\bR\d+\b", re.IGNORECASE)
STOP_ENTITIES = frozenset(("The", "This", "In", "No", "All", "Report", "Records"))
FEATURE_NAMES = (
    "token_jaccard_witness",
    "token_jaccard_obligation",
    "character_similarity_witness",
    "character_similarity_obligation",
    "entity_overlap_witness",
    "entity_overlap_obligation",
    "citation_overlap",
    "number_overlap",
    "distance_to_conclusion",
    "source_position",
    "changed_token_count",
    "edit_ratio",
)


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def entities(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"\b[A-Z][A-Za-z0-9-]*\b", text)
        if token not in STOP_ENTITIES
    }


def changed_sentence(old_document: str, new_document: str) -> tuple[int, str, str]:
    old = split_sentences(old_document)
    new = split_sentences(new_document)
    changed = [index for index, (left, right) in enumerate(zip(old, new), 1) if left != right]
    if len(old) != len(new) or len(changed) != 1:
        raise ValueError("P0k requires exactly one changed sentence")
    index = changed[0]
    return index, old[index - 1], new[index - 1]


def surface_features(
    case: SemanticCase, revised_document: str
) -> dict[str, float]:
    source_id, old_source, new_source = changed_sentence(case.old_document, revised_document)
    witness = split_sentences(case.old_document)[case.conclusion_sentence - 1]
    change = old_source + " " + new_source
    change_tokens = tokens(change)
    witness_tokens = tokens(witness)
    obligation_tokens = tokens(case.obligation)
    values = {
        "token_jaccard_witness": jaccard(change_tokens, witness_tokens),
        "token_jaccard_obligation": jaccard(change_tokens, obligation_tokens),
        "character_similarity_witness": difflib.SequenceMatcher(None, change, witness, autojunk=False).ratio(),
        "character_similarity_obligation": difflib.SequenceMatcher(None, change, case.obligation, autojunk=False).ratio(),
        "entity_overlap_witness": float(len(entities(change) & entities(witness))),
        "entity_overlap_obligation": float(len(entities(change) & entities(case.obligation))),
        "citation_overlap": float(bool(set(CITATION_RE.findall(change)) & set(CITATION_RE.findall(witness + " " + case.obligation)))),
        "number_overlap": float(bool(set(NUMBER_RE.findall(change)) & set(NUMBER_RE.findall(witness + " " + case.obligation)))),
        "distance_to_conclusion": abs(case.conclusion_sentence - source_id) / 10,
        "source_position": source_id / 10,
        "changed_token_count": float(len(change_tokens)),
        "edit_ratio": float(edit_ratio(case.old_document, revised_document)),
    }
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError("feature order drift")
    return values


def heuristic_selection(rows: list[dict[str, Any]], name: str) -> list[bool]:
    selected: list[bool] = []
    for row in rows:
        feature = row["features"]
        if name == "full":
            choose = True
        elif name in ("none", "witness_overlap"):
            choose = False
        elif name == "token_jaccard":
            choose = max(feature["token_jaccard_witness"], feature["token_jaccard_obligation"]) >= 0.10
        elif name == "character_similarity":
            choose = max(feature["character_similarity_witness"], feature["character_similarity_obligation"]) >= 0.35
        elif name == "entity_overlap":
            choose = max(feature["entity_overlap_witness"], feature["entity_overlap_obligation"]) >= 1
        elif name == "citation_overlap":
            choose = bool(feature["citation_overlap"])
        elif name == "surface_union":
            choose = bool(
                max(feature["token_jaccard_witness"], feature["token_jaccard_obligation"]) >= 0.10
                or max(feature["character_similarity_witness"], feature["character_similarity_obligation"]) >= 0.35
                or max(feature["entity_overlap_witness"], feature["entity_overlap_obligation"]) >= 1
                or feature["citation_overlap"]
                or feature["number_overlap"]
            )
        else:
            raise ValueError(f"unknown heuristic: {name}")
        selected.append(choose)
    return selected


def policy_metrics(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    stale_total = sum(row["stale"] for row in rows)
    caught = sum(row["stale"] and choose for row, choose in zip(rows, selected))
    checks = sum(selected)
    return {
        "stale_total": stale_total,
        "stale_caught": caught,
        "stale_recall": caught / stale_total if stale_total else 0.0,
        "rechecks": checks,
        "verification_saving": 1 - checks / len(rows) if rows else 0.0,
    }


def threshold_for_recall(scores: list[float], labels: list[int], target: float) -> float:
    positives = sum(labels)
    if not positives:
        return math.inf
    best, best_count = min(scores) - 1e-12, len(scores) + 1
    for threshold in sorted(set(scores), reverse=True) + [min(scores) - 1e-12]:
        chosen = [score >= threshold for score in scores]
        recall = sum(label and select for label, select in zip(labels, chosen)) / positives
        if recall >= target and sum(chosen) < best_count:
            best, best_count = threshold, sum(chosen)
    return best


def nested_surface_predictions(
    rows: list[dict[str, Any]]
) -> tuple[list[bool], list[float], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = sorted({row["domain_id"] for row in rows})
    selected = [False] * len(rows)
    scores = [0.0] * len(rows)
    folds: list[dict[str, Any]] = []

    def matrix(indices: list[int]):
        return np.asarray([[rows[i]["features"][name] for name in FEATURE_NAMES] for i in indices])

    def labels(indices: list[int]):
        return np.asarray([int(rows[i]["stale"]) for i in indices])

    def model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=0),
        )

    for outer in groups:
        test = [i for i, row in enumerate(rows) if row["domain_id"] == outer]
        train = [i for i, row in enumerate(rows) if row["domain_id"] != outer]
        inner_scores: dict[int, float] = {}
        for inner in sorted({rows[i]["domain_id"] for i in train}):
            valid = [i for i in train if rows[i]["domain_id"] == inner]
            valid_set = set(valid)
            fit = [i for i in train if i not in valid_set]
            estimator = model()
            estimator.fit(matrix(fit), labels(fit))
            predicted = estimator.predict_proba(matrix(valid))[:, 1]
            inner_scores.update({i: float(score) for i, score in zip(valid, predicted)})
        ordered_scores = [inner_scores[i] for i in train]
        threshold = threshold_for_recall(ordered_scores, labels(train).tolist(), TARGET_RECALL)
        estimator = model()
        estimator.fit(matrix(train), labels(train))
        predicted = estimator.predict_proba(matrix(test))[:, 1]
        for index, score in zip(test, predicted):
            scores[index] = float(score)
            selected[index] = bool(score >= threshold)
        folds.append({"held_out_domain": outer, "threshold": threshold, "test_stale": int(sum(labels(test)))})
    return selected, scores, folds


def matched_random(rows: list[dict[str, Any]], selected: list[bool], draws: int = 1000) -> dict[str, float]:
    by_case: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_case[row["case_id"]].append(index)
    stale_total = sum(row["stale"] for row in rows)
    rng = random.Random(20260824)
    recalls: list[float] = []
    for _ in range(draws):
        caught = 0
        for indices in by_case.values():
            budget = sum(selected[i] for i in indices)
            chosen = set(rng.sample(indices, budget)) if budget else set()
            caught += sum(rows[i]["stale"] for i in chosen)
        recalls.append(caught / stale_total)
    ordered = sorted(recalls)
    return {
        "draws": draws,
        "mean_stale_recall": mean(recalls),
        "p95_stale_recall": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases()
    specs = [
        (case.id, state, document, expected)
        for case in cases
        for state, document, expected in (
            ("old", case.old_document, True),
            ("dependency_edit", case.dependency_document, False),
            ("harmless_edit", case.harmless_document, True),
        )
    ]
    judgments: list[dict[str, Any]] = []
    judge_summaries: dict[str, Any] = {}
    judge_labels: dict[str, dict[tuple[str, str], bool | None]] = {}

    for judge_model in args.judge_models:
        runner = ModelRunner(judge_model)
        raw_outputs = runner.generate(
            [judge_prompt(next(case for case in cases if case.id == case_id), document) for case_id, _, document, _ in specs],
            args.judge_batch_size,
            args.judge_cap,
        )
        release_runner(runner)
        parsed_count = conclusion_correct = dependency_overlap = 0
        state_correct = Counter()
        state_total = Counter()
        mechanism_cells: dict[str, Counter[str]] = defaultdict(Counter)
        labels: dict[tuple[str, str], bool | None] = {}
        for (case_id, state, document, expected), raw in zip(specs, raw_outputs):
            case = next(case for case in cases if case.id == case_id)
            parsed, valid, mode = parse_judgment(raw, 10)
            predicted = parsed["met"] if valid else None
            labels[(case_id, state)] = predicted
            parsed_count += int(valid)
            conclusion_correct += int(valid and parsed["conclusion_sentence"] == case.conclusion_sentence)
            dependency_overlap += int(
                valid and bool(set(parsed["dependency_sentences"]) & set(case.dependency_sentence_ids))
            )
            state_total[state] += 1
            state_correct[state] += int(valid and predicted == expected)
            mechanism_cells[case.mechanism][f"{state}_total"] += 1
            mechanism_cells[case.mechanism][f"{state}_correct"] += int(valid and predicted == expected)
            judgments.append({
                "judge_model": judge_model,
                "case_id": case_id,
                "domain_id": case.domain_id,
                "mechanism": case.mechanism,
                "hard_alias": case.hard_alias,
                "state": state,
                "expected_met": expected,
                "document": document,
                "obligation": case.obligation,
                "valid": valid,
                "parse_mode": mode,
                "parsed": parsed,
                "correct": bool(valid and predicted == expected),
                "raw": raw,
            })
        summary = {
            "n": len(specs),
            "parse_validity": parsed_count / len(specs),
            "old_sat_accuracy": state_correct["old"] / state_total["old"],
            "dependency_fail_accuracy": state_correct["dependency_edit"] / state_total["dependency_edit"],
            "harmless_sat_accuracy": state_correct["harmless_edit"] / state_total["harmless_edit"],
            "conclusion_sentence_accuracy": conclusion_correct / len(specs),
            "dependency_source_overlap_rate": dependency_overlap / len(specs),
            "by_mechanism": {
                mechanism: {
                    state: cells[f"{state}_correct"] / cells[f"{state}_total"]
                    for state in ("old", "dependency_edit", "harmless_edit")
                }
                for mechanism, cells in sorted(mechanism_cells.items())
            },
        }
        summary["passes_g1_individual"] = all(
            summary[name] >= 0.95
            for name in ("parse_validity", "old_sat_accuracy", "dependency_fail_accuracy", "harmless_sat_accuracy")
        )
        judge_summaries[judge_model] = summary
        judge_labels[judge_model] = labels

    models = list(args.judge_models)
    agreement_total = agreement = 0
    if len(models) >= 2:
        for key in judge_labels[models[0]]:
            left, right = judge_labels[models[0]][key], judge_labels[models[1]][key]
            agreement_total += 1
            agreement += int(left is not None and left == right)
    dual_agreement = agreement / agreement_total if agreement_total else 0.0
    gate_1 = all(summary["passes_g1_individual"] for summary in judge_summaries.values()) and dual_agreement >= 0.95

    mechanism_pass: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        dependency_correct = min(
            sum(
                row["correct"] for row in judgments
                if row["judge_model"] == model and row["mechanism"] == mechanism and row["state"] == "dependency_edit"
            )
            for model in models
        )
        harmless_correct = min(
            sum(
                row["correct"] for row in judgments
                if row["judge_model"] == model and row["mechanism"] == mechanism and row["state"] == "harmless_edit"
            )
            for model in models
        )
        mechanism_pass[mechanism] = {
            "minimum_dependency_correct_across_judges": dependency_correct,
            "minimum_harmless_correct_across_judges": harmless_correct,
            "passes": dependency_correct >= 7 and harmless_correct >= 7,
        }
    gate_3 = sum(cell["passes"] for cell in mechanism_pass.values()) >= 4

    rows: list[dict[str, Any]] = []
    case_by_id = {case.id: case for case in cases}
    for case in cases:
        for arm, document, stale in (
            ("dependency_edit", case.dependency_document, True),
            ("harmless_edit", case.harmless_document, False),
        ):
            source_id, old_source, new_source = changed_sentence(case.old_document, document)
            rows.append({
                "row_id": f"{case.id}|{arm}",
                "case_id": case.id,
                "domain_id": case.domain_id,
                "mechanism": case.mechanism,
                "hard_alias": case.hard_alias,
                "arm": arm,
                "stale": stale,
                "old_document": case.old_document,
                "revised_document": document,
                "obligation": case.obligation,
                "conclusion_sentence": case.conclusion_sentence,
                "changed_sentence": source_id,
                "old_source": old_source,
                "new_source": new_source,
                "witness_text": split_sentences(case.old_document)[case.conclusion_sentence - 1],
                "witness_unchanged": (
                    split_sentences(case.old_document)[case.conclusion_sentence - 1]
                    == split_sentences(document)[case.conclusion_sentence - 1]
                ),
                "features": surface_features(case, document),
            })

    policy_names = (
        "full", "none", "witness_overlap", "token_jaccard", "character_similarity",
        "entity_overlap", "citation_overlap", "surface_union",
    )
    policies = {
        name: policy_metrics(rows, heuristic_selection(rows, name))
        for name in policy_names
    }
    learned_selected, learned_scores, outer_folds = nested_surface_predictions(rows)
    for row, score, choose in zip(rows, learned_scores, learned_selected):
        row["learned_oof_score"] = score
        row["learned_selected"] = choose
    learned = policy_metrics(rows, learned_selected)
    learned.update({
        "available": True,
        "outer_folds": outer_folds,
        "matched_random": matched_random(rows, learned_selected),
    })
    policies["learned_surface"] = learned

    frozen_heuristics = [
        name for name in policy_names
        if name not in ("full", "none", "witness_overlap")
    ]
    surface_sufficient = any(
        policies[name]["stale_recall"] >= 0.90
        and policies[name]["verification_saving"] >= 0.25
        for name in frozen_heuristics
    )
    gate_2 = not surface_sufficient
    eligible_simple = [
        policies[name] for name in frozen_heuristics
        if policies[name]["stale_recall"] >= 0.90
    ]
    best_simple_saving = max((metric["verification_saving"] for metric in eligible_simple), default=0.0)
    gate_4 = bool(
        learned["stale_recall"] >= 0.90
        and learned["verification_saving"] >= 0.25
        and learned["stale_recall"] - learned["matched_random"]["mean_stale_recall"] >= 0.15
        and learned["verification_saving"] - best_simple_saving >= 0.10
    )

    strongest_name = min(
        frozen_heuristics,
        key=lambda name: (-policies[name]["stale_recall"], -policies[name]["verification_saving"], name),
    )
    strongest_selected = heuristic_selection(rows, strongest_name)
    disagreement_cases = {
        case_id for case_id in case_by_id
        if any(
            judge_labels[models[0]][(case_id, state)] != judge_labels[models[1]][(case_id, state)]
            for state in ("old", "dependency_edit", "harmless_edit")
        )
    }
    review_rows = [
        row for row, selected in zip(rows, strongest_selected)
        if (row["stale"] and not selected) or row["case_id"] in disagreement_cases
    ]
    for row in review_rows:
        row["review_reason"] = {
            "strongest_surface_false_negative": bool(row["stale"] and not strongest_selected[rows.index(row)]),
            "dual_judge_disagreement": row["case_id"] in disagreement_cases,
        }

    case_rows = [
        {
            **case.__dict__,
            "dependency_sentence_ids": list(case.dependency_sentence_ids),
        }
        for case in cases
    ]
    report = {
        "protocol": PROTOCOL,
        "judge_models": models,
        "n_domains": len(DOMAINS),
        "n_mechanisms": len(MECHANISMS),
        "n_base_cases": len(cases),
        "n_revision_rows": len(rows),
        "oracle_stale": sum(row["stale"] for row in rows),
        "all_witnesses_byte_unchanged": all(row["witness_unchanged"] for row in rows),
        "all_edits_disjoint_from_conclusion": all(row["changed_sentence"] != row["conclusion_sentence"] for row in rows),
        "judge_summaries": judge_summaries,
        "dual_judge_label_agreement": dual_agreement,
        "mechanism_coverage": mechanism_pass,
        "surface_policies": policies,
        "strongest_frozen_surface_heuristic": strongest_name,
        "gates_before_manual": {
            "g1_semantic_apparatus": gate_1,
            "g2_surface_baselines_insufficient": gate_2,
            "g3_mechanism_coverage": gate_3,
            "g4_learned_surface_screen": gate_4,
            "p0k_hard_case_gate_before_manual": bool(gate_1 and gate_2 and gate_3),
        },
        "manual_review_rows": len(review_rows),
        "interpretation_guard": (
            "P0k is a controlled paired existence screen. Passing does not estimate natural "
            "semantic-staleness prevalence or justify RL. Hidden-state work is allowed only after "
            "manual review confirms G1-G3."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "semantic_staleness_p0k_cases.jsonl", case_rows)
    write_jsonl(args.out_dir / "semantic_staleness_p0k_judgments.jsonl", judgments)
    write_jsonl(args.out_dir / "semantic_staleness_p0k_rows.jsonl", rows)
    write_jsonl(args.out_dir / "semantic_staleness_p0k_review.jsonl", review_rows)
    (args.out_dir / "semantic_staleness_p0k_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-models", nargs="+",
        default=("Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct"),
    )
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-cap", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
