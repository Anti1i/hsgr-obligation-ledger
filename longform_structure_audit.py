"""Zero-model structural audit for fixed-support long-form QA datasets.

P0 currently implements the official CLAPnQ GOLD answerable schema.  It uses
only released fields and does not call a language model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")


def percentile(values: list[float], q: float) -> float | None:
    """Return a linearly interpolated percentile for q in [0, 1]."""
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


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    count = len(SENTENCE_END_RE.findall(text))
    return max(1, count)


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"invalid_json:{exc.msg}"
                continue
            if not isinstance(value, dict):
                yield line_number, None, "record_not_object"
                continue
            yield line_number, value, None


def extract_clapnq(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    passages = record.get("passages")
    outputs = record.get("output")
    if not isinstance(passages, list) or not passages or not isinstance(passages[0], dict):
        return None, "missing_passage"
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        return None, "missing_output"

    passage = passages[0]
    output = outputs[0]
    passage_text = passage.get("text")
    passage_sentences = passage.get("sentences")
    answer = output.get("answer")
    selected = output.get("selected_sentences")
    meta = output.get("meta", {})

    if not isinstance(passage_text, str) or not passage_text.strip():
        return None, "empty_passage_text"
    if not isinstance(passage_sentences, list) or not all(
        isinstance(item, str) for item in passage_sentences
    ):
        return None, "invalid_passage_sentences"
    if not isinstance(answer, str) or not answer.strip():
        return None, "empty_answer"
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        return None, "invalid_selected_sentences"

    unique_selected = list(dict.fromkeys(selected))
    passage_sentence_set = set(passage_sentences)
    missing = [item for item in unique_selected if item not in passage_sentence_set]
    pair_overlaps = [jaccard(a, b) for a, b in combinations(unique_selected, 2)]
    non_consecutive = bool(meta.get("non_consecutive", False)) if isinstance(meta, dict) else False
    has_minimal_answer = bool(meta.get("has_minimal_answer", False)) if isinstance(meta, dict) else False
    support_count = len(unique_selected)

    return {
        "id": str(record.get("id", "")),
        "passage_sentence_count": len(passage_sentences),
        "passage_word_count": len(passage_text.split()),
        "answer_sentence_count": sentence_count(answer),
        "answer_word_count": len(answer.split()),
        "raw_support_count": len(selected),
        "support_count": support_count,
        "duplicate_support_count": len(selected) - support_count,
        "missing_support_count": len(missing),
        "non_consecutive": non_consecutive,
        "has_minimal_answer": has_minimal_answer,
        "pair_jaccards": pair_overlaps,
        "structural_headroom": support_count >= 3 or (support_count >= 2 and non_consecutive),
        "subset_state_capacity": 2**min(support_count, 20),
        "pilot_eligible": (
            len(answer.split()) >= 30
            and support_count >= 2
            and not missing
            and (non_consecutive or support_count >= 3)
        ),
    }, None


def audit_clapnq(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    malformed = Counter()
    split_counts: dict[str, dict[str, int]] = {}

    for path in paths:
        valid = 0
        invalid = 0
        for line_number, record, load_error in load_jsonl(path):
            if load_error:
                malformed[load_error] += 1
                invalid += 1
                continue
            assert record is not None
            row, error = extract_clapnq(record)
            if error:
                malformed[error] += 1
                invalid += 1
                continue
            assert row is not None
            row["source_file"] = path.name
            row["line_number"] = line_number
            rows.append(row)
            valid += 1
        split_counts[path.name] = {"valid": valid, "malformed": invalid}

    n = len(rows)
    support_counts = [row["support_count"] for row in rows]
    total_selected = sum(row["raw_support_count"] for row in rows)
    total_duplicates = sum(row["duplicate_support_count"] for row in rows)
    missing_records = sum(row["missing_support_count"] > 0 for row in rows)
    pair_jaccards = [value for row in rows for value in row["pair_jaccards"]]

    def rate(predicate: Any) -> float:
        return sum(bool(predicate(row)) for row in rows) / n if n else 0.0

    fixed_gold_rate = rate(
        lambda row: row["passage_sentence_count"] > 0
        and row["answer_word_count"] > 0
        and row["raw_support_count"] >= 0
    )
    multi_support_rate = rate(lambda row: row["support_count"] >= 2)
    three_plus_rate = rate(lambda row: row["support_count"] >= 3)
    nonconsecutive_multi_rate = rate(
        lambda row: row["support_count"] >= 2 and row["non_consecutive"]
    )
    headroom_rate = rate(lambda row: row["structural_headroom"])
    pilot_eligible_rate = rate(lambda row: row["pilot_eligible"])
    missing_record_rate = missing_records / n if n else 1.0

    gates = {
        "at_least_500_valid_examples": n >= 500,
        "all_have_fixed_passage_answer_and_support_list": fixed_gold_rate == 1.0,
        "multi_support_rate_at_least_50pct": multi_support_rate >= 0.50,
        "three_plus_support_rate_at_least_20pct": three_plus_rate >= 0.20,
        "nonconsecutive_multi_rate_at_least_25pct": nonconsecutive_multi_rate >= 0.25,
        "structural_headroom_rate_at_least_30pct": headroom_rate >= 0.30,
        "median_answer_words_at_least_30": (percentile(
            [row["answer_word_count"] for row in rows], 0.50
        ) or 0) >= 30,
        "missing_support_record_rate_below_1pct": missing_record_rate < 0.01,
    }
    integrity_gate_names = {
        "at_least_500_valid_examples",
        "all_have_fixed_passage_answer_and_support_list",
        "missing_support_record_rate_below_1pct",
    }
    if all(gates.values()):
        decision = "PASS"
    elif all(gates[name] for name in integrity_gate_names):
        decision = "BORDERLINE"
    else:
        decision = "FAIL"

    return {
        "protocol": "EXPERIMENT_PROTOCOL_LONGFORM_STRUCTURE_AUDIT_P0.md",
        "dataset": "CLAPnQ GOLD answerable train+dev",
        "decision": decision,
        "valid_examples": n,
        "malformed_examples": sum(malformed.values()),
        "malformed_reasons": dict(malformed),
        "split_counts": split_counts,
        "rates": {
            "fixed_passage_answer_support_list": fixed_gold_rate,
            "multi_support": multi_support_rate,
            "three_plus_support": three_plus_rate,
            "nonconsecutive_multi_support": nonconsecutive_multi_rate,
            "structural_headroom": headroom_rate,
            "pilot_eligible": pilot_eligible_rate,
            "annotated_minimal_answer": rate(lambda row: row["has_minimal_answer"]),
            "records_with_missing_support": missing_record_rate,
            "duplicate_selected_support_items": (
                total_duplicates / total_selected if total_selected else 0.0
            ),
        },
        "distributions": {
            "support_sentences": distribution(support_counts),
            "subset_state_capacity_capped_at_2pow20": distribution(
                [row["subset_state_capacity"] for row in rows]
            ),
            "passage_sentences": distribution(
                [row["passage_sentence_count"] for row in rows]
            ),
            "passage_words": distribution([row["passage_word_count"] for row in rows]),
            "answer_sentences": distribution(
                [row["answer_sentence_count"] for row in rows]
            ),
            "answer_words": distribution([row["answer_word_count"] for row in rows]),
            "selected_sentence_pair_token_jaccard": distribution(pair_jaccards),
        },
        "support_count_histogram": dict(sorted(Counter(support_counts).items())),
        "gates": gates,
        "interpretation_guard": (
            "A PASS establishes non-collapsed annotated structure only; it does not establish "
            "model headroom, hidden-state encoding, causal guide utility, or end-task gains."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("data/longform_cache/clapnq_train_answerable.jsonl"),
            Path("data/longform_cache/clapnq_dev_answerable.jsonl"),
        ],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    missing = [str(path) for path in args.paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")
    report = audit_clapnq(args.paths)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
