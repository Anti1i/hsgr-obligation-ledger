"""Small read-only diagnostic for frozen P7x outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def run(args: argparse.Namespace) -> dict[str, Any]:
    inductions = load_jsonl(args.inductions)
    candidates = load_jsonl(args.candidates)
    selections = load_jsonl(args.selections)
    induction_by_id = {str(row["id"]): row for row in inductions}
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_id[str(row["id"])].append(row)
    for rows in candidates_by_id.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))
    arms = ("induced_logit_append", "induced_random_append", "induced_oracle_append")
    subset_metrics = {}
    for arm in arms:
        arm_rows = [row for row in selections if row["arm"] == arm]
        subset_metrics[arm] = {}
        for subset, wanted in (("valid", True), ("invalid", False), ("all", None)):
            rows = [
                row for row in arm_rows
                if wanted is None or bool(induction_by_id[str(row["id"])]["valid_four_node_set"]) == wanted
            ]
            subset_metrics[arm][subset] = {
                "n": len(rows),
                "str_hit": rate(rows, "str_hit"),
                "target_recovery": rate(rows, "target_recovered"),
            }
    oracle = {
        str(row["id"]): bool(row["str_hit"])
        for row in selections if row["arm"] == "induced_oracle_append"
    }
    auto = {
        str(row["id"]): bool(row["str_hit"])
        for row in selections if row["arm"] == "induced_logit_append"
    }
    categories = {
        "oracle_success_auto_fail": [case_id for case_id in oracle if oracle[case_id] and not auto[case_id]],
        "valid_oracle_fail": [
            case_id for case_id in oracle
            if induction_by_id[case_id]["valid_four_node_set"] and not oracle[case_id]
        ],
        "invalid": [case_id for case_id in oracle if not induction_by_id[case_id]["valid_four_node_set"]],
    }
    samples = {}
    for category, ids in categories.items():
        samples[category] = []
        for case_id in ids[:3]:
            induction = induction_by_id[case_id]
            samples[category].append(
                {
                    "id": case_id,
                    "parse_mode": induction["parse_mode"],
                    "raw_induction": str(induction["raw_induction"])[:800],
                    "obligations": induction["obligations"],
                    "candidates": [
                        {
                            "index": row["candidate_index"],
                            "obligation": row["obligation"],
                            "coverage_score": row["coverage_score"],
                            "str_hit": row["str_hit"],
                            "target_recovered": row["target_recovered"],
                            "generated": str(row["generated"])[:300],
                        }
                        for row in candidates_by_id[case_id]
                    ],
                }
            )
    result = {
        "counts": {
            "inductions": len(inductions),
            "candidates": len(candidates),
            "selections": len(selections),
            "parse_modes": dict(Counter(row["parse_mode"] for row in inductions)),
            "category_sizes": {name: len(ids) for name, ids in categories.items()},
        },
        "subset_metrics": subset_metrics,
        "samples": samples,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inductions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
