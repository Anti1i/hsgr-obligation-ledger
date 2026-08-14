"""Frozen zero-model audit for ASQA with ALCE's five oracle-reranked docs.

This script implements EXPERIMENT_PROTOCOL_ASQA_FIXED_SUPPORT_P0.md.  It uses
released labels and passages only; it neither retrieves evidence nor calls a
language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any


ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
SELECTION_SALT = "20260815"


def normalize_answer(text: str) -> str:
    """Match the normalization used by the released ALCE evaluation code."""
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = ARTICLE_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def normalize_whitespace(text: str) -> str:
    """Whitespace-only normalization for the verbatim-leakage guard."""
    return " ".join(text.split())


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def normalized_alias_group(qa_pair: dict[str, Any]) -> tuple[str, ...]:
    aliases = qa_pair.get("short_answers", [])
    if not isinstance(aliases, list):
        return ()
    normalized = {
        normalize_answer(value)
        for value in aliases
        if isinstance(value, str) and normalize_answer(value)
    }
    return tuple(sorted(normalized))


def facet_groups(record: dict[str, Any]) -> list[tuple[str, ...]]:
    qa_pairs = record.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        return []
    return [
        normalized_alias_group(pair)
        for pair in qa_pairs
        if isinstance(pair, dict)
    ]


def exact_presence(aliases: tuple[str, ...], text: str) -> bool:
    normalized_text = normalize_answer(text)
    return bool(aliases) and any(alias in normalized_text for alias in aliases)


def facet_score(groups: list[tuple[str, ...]], text: str) -> tuple[float, bool, list[bool]]:
    present = [exact_presence(group, text) for group in groups]
    coverage = sum(present) / len(present) if present else 0.0
    return coverage, bool(present) and all(present), present


def long_answers(record: dict[str, Any]) -> list[str]:
    annotations = record.get("annotations", [])
    if not isinstance(annotations, list):
        return []
    answers: list[str] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        answer = annotation.get("long_answer")
        if isinstance(answer, str) and answer.strip():
            answers.append(answer)
    return answers


def fixed_context(record: dict[str, Any]) -> tuple[str, int, int, int]:
    docs = record.get("docs", [])
    if not isinstance(docs, list):
        docs = []
    fixed_docs = docs[:5]
    rendered: list[str] = []
    nonempty = 0
    for doc in fixed_docs:
        if not isinstance(doc, dict):
            continue
        title = doc.get("title", "")
        text = doc.get("text", "")
        title = title if isinstance(title, str) else ""
        text = text if isinstance(text, str) else ""
        if text.strip():
            nonempty += 1
        rendered.append(f"Title: {title}\n{text}".strip())
    return "\n\n".join(rendered), len(docs), len(fixed_docs), nonempty


def alignment_signature(record: dict[str, Any]) -> tuple[str, tuple[tuple[str, ...], ...]]:
    question = record.get("ambiguous_question", record.get("question", ""))
    normalized_question = normalize_whitespace(question) if isinstance(question, str) else ""
    return normalized_question, tuple(sorted(facet_groups(record)))


def load_original_dev(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    dev = data.get("dev") if isinstance(data, dict) else None
    if not isinstance(dev, dict):
        raise ValueError("original ASQA JSON must contain a dev object")
    if not all(isinstance(value, dict) for value in dev.values()):
        raise ValueError("original ASQA dev values must be objects")
    return {str(key): value for key, value in dev.items()}


def load_alce(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ALCE ASQA JSON must be a list")
    if not all(isinstance(value, dict) for value in data):
        raise ValueError("ALCE ASQA records must be objects")
    return data


def extract_row(record_id: str, record: dict[str, Any]) -> dict[str, Any]:
    raw_groups = facet_groups(record)
    invalid_groups = sum(not group for group in raw_groups)
    valid_groups = [group for group in raw_groups if group]
    unique_groups = list(dict.fromkeys(valid_groups))
    duplicate_groups = len(valid_groups) - len(unique_groups)

    context, total_doc_count, fixed_doc_count, nonempty_doc_count = fixed_context(record)
    passage_coverage, passage_strict, passage_present = facet_score(unique_groups, context)

    answers = long_answers(record)
    answer_scores = [facet_score(unique_groups, answer)[:2] for answer in answers]
    best_human_coverage = max((score[0] for score in answer_scores), default=0.0)
    best_human_strict = any(score[1] for score in answer_scores)
    annotation_mean_coverage = (
        statistics.fmean(score[0] for score in answer_scores) if answer_scores else 0.0
    )
    annotation_strict_rate = (
        statistics.fmean(float(score[1]) for score in answer_scores) if answer_scores else 0.0
    )

    normalized_context_verbatim = normalize_whitespace(context)
    leaked_answers = [
        answer
        for answer in answers
        if normalize_whitespace(answer)
        and normalize_whitespace(answer) in normalized_context_verbatim
    ]
    leaked = bool(leaked_answers)

    structurally_valid = bool(unique_groups) and invalid_groups == 0
    eligible = (
        structurally_valid
        and 2 <= len(unique_groups) <= 6
        and total_doc_count == 5
        and fixed_doc_count == 5
        and nonempty_doc_count == 5
        and passage_strict
        and best_human_strict
        and not leaked
    )
    return {
        "id": record_id,
        "raw_facet_count": len(raw_groups),
        "facet_count": len(unique_groups),
        "invalid_facet_groups": invalid_groups,
        "duplicate_facet_groups": duplicate_groups,
        "has_duplicate_facets": duplicate_groups > 0,
        "total_doc_count": total_doc_count,
        "fixed_doc_count": fixed_doc_count,
        "nonempty_fixed_doc_count": nonempty_doc_count,
        "context_word_count": len(context.split()),
        "passage_coverage": passage_coverage,
        "passage_strict": passage_strict,
        "passage_present_facets": sum(passage_present),
        "human_answer_count": len(answers),
        "best_human_coverage": best_human_coverage,
        "best_human_strict": best_human_strict,
        "annotation_mean_coverage": annotation_mean_coverage,
        "annotation_strict_rate": annotation_strict_rate,
        "verbatim_human_answer_leak": leaked,
        "eligible": eligible,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_asqa(alce_path: Path, original_path: Path) -> dict[str, Any]:
    original = load_original_dev(original_path)
    alce = load_alce(alce_path)

    alce_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in alce:
        record_id = str(record.get("sample_id", ""))
        alce_by_id.setdefault(record_id, []).append(record)

    mismatch_reasons: Counter[str] = Counter()
    aligned: list[tuple[str, dict[str, Any]]] = []
    for record_id, original_record in original.items():
        matches = alce_by_id.get(record_id, [])
        if not matches:
            mismatch_reasons["missing_alce_id"] += 1
            continue
        if len(matches) != 1:
            mismatch_reasons["nonunique_alce_id"] += 1
            continue
        alce_record = matches[0]
        if alignment_signature(original_record) != alignment_signature(alce_record):
            mismatch_reasons["question_or_qa_pair_mismatch"] += 1
            continue
        aligned.append((record_id, alce_record))

    rows = [extract_row(record_id, record) for record_id, record in aligned]
    n = len(rows)
    original_n = len(original)

    def rate(predicate: Any) -> float:
        return sum(bool(predicate(row)) for row in rows) / n if n else 0.0

    alignment_rate = n / original_n if original_n else 0.0
    multi_facet_rate = rate(lambda row: row["facet_count"] >= 2)
    three_plus_rate = rate(lambda row: row["facet_count"] >= 3)
    duplicate_example_rate = rate(lambda row: row["has_duplicate_facets"])
    five_nonempty_docs_rate = rate(lambda row: row["nonempty_fixed_doc_count"] == 5)
    median_context_words = percentile([row["context_word_count"] for row in rows], 0.5) or 0.0
    passage_str_em = statistics.fmean(row["passage_coverage"] for row in rows) if rows else 0.0
    passage_str_hit = rate(lambda row: row["passage_strict"])
    human_best_str_em = statistics.fmean(row["best_human_coverage"] for row in rows) if rows else 0.0
    human_best_str_hit = rate(lambda row: row["best_human_strict"])
    human_annotation_str_em = (
        statistics.fmean(row["annotation_mean_coverage"] for row in rows) if rows else 0.0
    )
    human_annotation_str_hit = (
        statistics.fmean(row["annotation_strict_rate"] for row in rows) if rows else 0.0
    )
    leakage_rate = rate(lambda row: row["verbatim_human_answer_leak"])
    eligible_ids = [row["id"] for row in rows if row["eligible"]]
    selected_ids = sorted(
        eligible_ids,
        key=lambda record_id: hashlib.sha256(
            f"{SELECTION_SALT}|{record_id}".encode("utf-8")
        ).hexdigest(),
    )[:192]

    gates = {
        "g1_at_least_900_aligned": n >= 900,
        "g2_alignment_rate_at_least_99pct": alignment_rate >= 0.99,
        "g3_two_plus_unique_facets_at_least_90pct": multi_facet_rate >= 0.90,
        "g4_three_plus_unique_facets_at_least_30pct": three_plus_rate >= 0.30,
        "g5_duplicate_facet_examples_below_2pct": duplicate_example_rate < 0.02,
        "g6_five_nonempty_fixed_docs_at_least_95pct": five_nonempty_docs_rate >= 0.95,
        "g7_median_fixed_context_words_at_least_300": median_context_words >= 300,
        "g8_passage_str_em_at_least_80pct_and_hit_at_least_50pct": (
            passage_str_em >= 0.80 and passage_str_hit >= 0.50
        ),
        "g9_human_best_str_em_at_least_80pct_and_hit_at_least_50pct": (
            human_best_str_em >= 0.80 and human_best_str_hit >= 0.50
        ),
        "g10_verbatim_long_answer_leakage_below_1pct": leakage_rate < 0.01,
    }
    eligibility_gate = len(eligible_ids) >= 192
    critical_gate_names = {
        "g1_at_least_900_aligned",
        "g2_alignment_rate_at_least_99pct",
        "g6_five_nonempty_fixed_docs_at_least_95pct",
        "g7_median_fixed_context_words_at_least_300",
        "g10_verbatim_long_answer_leakage_below_1pct",
    }
    if all(gates.values()) and eligibility_gate:
        decision = "PASS"
    elif not all(gates[name] for name in critical_gate_names) or not eligibility_gate:
        decision = "FAIL"
    else:
        decision = "BORDERLINE"

    return {
        "protocol": "EXPERIMENT_PROTOCOL_ASQA_FIXED_SUPPORT_P0.md",
        "dataset": "ASQA dev + ALCE oracle-reranked top-5 docs",
        "decision": decision,
        "source": {
            "original_path": str(original_path),
            "original_sha256": sha256_file(original_path),
            "alce_path": str(alce_path),
            "alce_sha256": sha256_file(alce_path),
        },
        "counts": {
            "original_dev_examples": original_n,
            "alce_examples": len(alce),
            "unique_alce_ids": len(alce_by_id),
            "aligned_examples": n,
            "unmatched_or_inconsistent_examples": original_n - n,
            "eligible_examples": len(eligible_ids),
            "selected_p1_examples": len(selected_ids),
        },
        "alignment_mismatch_reasons": dict(mismatch_reasons),
        "rates": {
            "alignment": alignment_rate,
            "two_plus_unique_facets": multi_facet_rate,
            "three_plus_unique_facets": three_plus_rate,
            "examples_with_duplicate_facets": duplicate_example_rate,
            "five_nonempty_fixed_docs": five_nonempty_docs_rate,
            "passage_str_em": passage_str_em,
            "passage_str_hit": passage_str_hit,
            "human_best_of_annotations_str_em": human_best_str_em,
            "human_best_of_annotations_str_hit": human_best_str_hit,
            "human_all_annotations_str_em": human_annotation_str_em,
            "human_all_annotations_str_hit": human_annotation_str_hit,
            "verbatim_human_answer_leakage": leakage_rate,
            "eligible": len(eligible_ids) / n if n else 0.0,
        },
        "distributions": {
            "unique_facets": distribution([row["facet_count"] for row in rows]),
            "fixed_context_words": distribution([row["context_word_count"] for row in rows]),
            "passage_facet_coverage": distribution([row["passage_coverage"] for row in rows]),
            "best_human_facet_coverage": distribution([row["best_human_coverage"] for row in rows]),
            "human_answers_per_example": distribution([row["human_answer_count"] for row in rows]),
        },
        "facet_count_histogram": dict(sorted(Counter(row["facet_count"] for row in rows).items())),
        "invalid_facet_group_examples": sum(row["invalid_facet_groups"] > 0 for row in rows),
        "gates": gates,
        "eligibility_gate_at_least_192": eligibility_gate,
        "selection_salt": SELECTION_SALT,
        "selected_p1_ids": selected_ids,
        "metric_note": (
            "Gate 9 uses the best released human annotation per example, matching the frozen "
            "eligibility rule that at least one human long answer must cover every facet. "
            "Scores across all annotations are also reported as a stricter diagnostic."
        ),
        "interpretation_guard": (
            "PASS establishes a viable fixed-support structural apparatus only; it does not "
            "establish model difficulty, hidden-state encoding, causal Guide utility, or gains."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-output", type=Path)
    args = parser.parse_args()

    for path in (args.alce, args.original):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    report = audit_asqa(args.alce, args.original)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.selected_output:
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_output.write_text(
            "\n".join(report["selected_p1_ids"]) + "\n", encoding="utf-8"
        )
    print(rendered)


if __name__ == "__main__":
    main()
