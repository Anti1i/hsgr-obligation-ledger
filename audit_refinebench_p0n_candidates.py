#!/usr/bin/env python3
"""Print compact, criterion-focused evidence for the P0n manual audit.

Raw RefineBench text remains in the scratch result directory.  This helper only
reads the review bundle and prints selected evidence to the terminal; it does
not copy benchmark content into the repository.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


STOP = {
    "a", "all", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "including", "is", "it", "of", "on", "or", "response",
    "should", "that", "the", "their", "to", "whether", "with",
}


def units(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n\s*\n|(?=^#{1,6}\s)", text, flags=re.M)
    return [re.sub(r"\s+", " ", piece).strip() for piece in pieces if piece.strip()]


def tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
        if len(token) > 2 and token not in STOP
    }


def top_units(text: str, criterion: str, n: int) -> list[tuple[float, str]]:
    query = tokens(criterion)
    ranked = []
    for index, unit in enumerate(units(text)):
        words = tokens(unit)
        overlap = len(query & words)
        coverage = overlap / len(query) if query else 0.0
        density = overlap / max(len(words), 1)
        score = coverage + 0.25 * density - index * 1e-8
        ranked.append((score, unit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_jsonl", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--arm")
    parser.add_argument(
        "--review-type",
        choices=("candidate_yes_to_no", "yes_to_yes_control", "all"),
        default="candidate_yes_to_no",
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=700)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.review_jsonl.open(encoding="utf-8")]
    candidates = [
        row for row in rows
        if args.review_type == "all" or row["review_type"] == args.review_type
    ]
    if args.case_id:
        candidates = [row for row in candidates if row["case_id"] == args.case_id]
    if args.arm:
        candidates = [row for row in candidates if row["arm"] == args.arm]

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["case_id"], row["arm"])].append(row)

    for (case_id, arm), group in sorted(grouped.items()):
        first = group[0]
        diff = first["diff_aids"]
        print("=" * 100)
        print(
            f"{case_id} | {arm} | {first['field']} | {first['review_type']}={len(group)} | "
            f"chars={len(first['old_answer'])}->{len(first['new_answer'])} | "
            f"similarity={diff['sequence_similarity']:.3f} | "
            f"exact_sentence_preserved={diff['preserved_old_sentence_fraction']:.3f}"
        )
        print("QUESTION:", re.sub(r"\s+", " ", first["question"])[: args.max_chars])
        for row in sorted(group, key=lambda item: item["criterion"]["criterion_id"]):
            criterion = row["criterion"]
            print("-" * 100)
            print(f"C{criterion['criterion_id']} (targeted={criterion['targeted']}): {criterion['criterion']}")
            print("  OLD TOP EVIDENCE:")
            for score, unit in top_units(row["old_answer"], criterion["criterion"], args.top):
                print(f"    [{score:.3f}] {unit[: args.max_chars]}")
            print("  NEW TOP EVIDENCE:")
            for score, unit in top_units(row["new_answer"], criterion["criterion"], args.top):
                print(f"    [{score:.3f}] {unit[: args.max_chars]}")


if __name__ == "__main__":
    main()
