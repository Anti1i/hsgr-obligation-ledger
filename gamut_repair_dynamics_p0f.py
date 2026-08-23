"""Turn P0e outputs into a frozen repair cross-effect audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_REPAIR_DYNAMICS_P0F.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transition_row(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_order = baseline["matched_order"]
    if not baseline["valid"] or baseline["relation_met"] or not baseline["all_components_present"]:
        raise ValueError(f"case {baseline['id']} does not satisfy the frozen relation-only precondition")
    baseline_nodes = set(baseline_order)
    final_known = bool(candidate["valid_extraction"])
    final_order = candidate["matched_order"] if final_known else []
    final_nodes = set(final_order)
    regressions = sorted(baseline_nodes - final_nodes) if final_known else None
    gains = sorted(final_nodes - baseline_nodes) if final_known else None
    relation_sorted = bool(final_known and candidate["relation_met"])
    any_component_regression = bool(regressions) if regressions is not None else None
    target_recovered = bool(
        candidate["patch_valid"] and relation_sorted and not any_component_regression
    )
    return {
        "id": candidate["id"],
        "arm": candidate["arm"],
        "question": candidate["question"],
        "steps": candidate["steps"],
        "original_answer": candidate["original_answer"],
        "candidate_answer": candidate["answer"],
        "patch_valid": candidate["patch_valid"],
        "sentence_span": candidate["sentence_span"],
        "edit_ratio": candidate["edit_ratio"],
        "baseline_order": baseline_order,
        "final_order": final_order if final_known else None,
        "outcome_known": final_known,
        "relation_sorted": relation_sorted,
        "complete_target_recovered": target_recovered,
        "component_regressions": regressions,
        "component_gains": gains,
        "any_component_regression": any_component_regression,
        "destructive_repair_attempt": any_component_regression,
        "automatic_monotonic_success": target_recovered,
        "manual_factual_preservation": "PENDING",
        "manual_fix_one_break_another": "PENDING",
    }


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["outcome_known"]]
    return {
        "n": len(rows),
        "known_outcomes": len(known),
        "relation_sorted_rate": mean(row["relation_sorted"] for row in known) if known else 0.0,
        "complete_target_recovery_rate": (
            mean(row["complete_target_recovered"] for row in known) if known else 0.0
        ),
        "component_regression_attempt_rate": (
            mean(row["any_component_regression"] for row in known) if known else 0.0
        ),
        "destructive_repair_attempt_count": sum(row["destructive_repair_attempt"] for row in known),
        "automatic_monotonic_success_rate": (
            mean(row["automatic_monotonic_success"] for row in known) if known else 0.0
        ),
        "median_edit_ratio": median(row["edit_ratio"] for row in rows),
    }


def review_packet(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# P0f manual preservation review",
        "",
        "For every output, mark factual preservation PASS/FAIL/UNCERTAIN and name any changed fact.",
        "Do not infer safety from the automatic order/component fields.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']} — {row['arm']}",
                "",
                f"- baseline order: `{row['baseline_order']}`",
                f"- final order: `{row['final_order']}`",
                f"- component regressions: `{row['component_regressions']}`",
                f"- relation sorted: `{row['relation_sorted']}`",
                f"- complete target recovered: `{row['complete_target_recovered']}`",
                f"- edit ratio: `{row['edit_ratio']:.4f}`",
                "- manual factual preservation: **PENDING**",
                "- did a recovered target break a separate fact?: **PENDING**",
                "",
                "### Required process steps",
                "",
                *[f"- P{index}: {step}" for index, step in enumerate(row["steps"], 1)],
                "",
                "### Original",
                "",
                row["original_answer"],
                "",
                "### Candidate",
                "",
                row["candidate_answer"],
                "",
            ]
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_jsonl(args.p0e_candidates)
    p0d_rows = load_jsonl(args.p0d_rows)
    baseline_by_id = {
        row["id"]: row for row in p0d_rows if row["arm"] == "new_768"
    }
    candidate_ids = {row["id"] for row in candidates}
    if len(candidates) != 16 or len(candidate_ids) != 4:
        raise RuntimeError("expected the frozen four cases and sixteen P0e outputs")
    if not candidate_ids.issubset(baseline_by_id):
        raise RuntimeError("one or more P0e cases are missing from P0d")

    rows = [transition_row(candidate, baseline_by_id[candidate["id"]]) for candidate in candidates]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    edge_counts: Counter[str] = Counter()
    for row in rows:
        if row["component_regressions"] is not None:
            for node in row["component_regressions"]:
                edge_counts[f"R_ORDER->{node}"] += 1
    destructive_count = sum(
        bool(row["destructive_repair_attempt"])
        for row in rows
        if row["destructive_repair_attempt"] is not None
    )
    report = {
        "protocol": PROTOCOL,
        "n_cases": len(candidate_ids),
        "n_repair_attempts": len(rows),
        "arm_summaries": {arm: arm_summary(arm_rows) for arm, arm_rows in sorted(by_arm.items())},
        "repair_influence_edge_counts": dict(sorted(edge_counts.items())),
        "destructive_repair_attempt_count": destructive_count,
        "repair_attempt_component_regression_exists": destructive_count >= 1,
        "strict_fix_one_break_another_gate": "PENDING_MANUAL_FACTUAL_REVIEW",
        "manual_factual_review_pending": len(rows),
        "claims_not_tested": [
            "repair-target asymmetry",
            "influence predictability or learnability",
            "planner superiority",
            "hidden-state prediction",
            "population prevalence",
        ],
        "interpretation_guard": (
            "P0f is a four-case one-target intervention audit. Component loss is a negative side "
            "effect of an attempted repair, not proof that the composite target was fixed. The "
            "strict fix-one-break-another claim requires manual factual review of a completely "
            "recovered target; benchmark-level and hidden-state claims remain out of scope."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "gamut_repair_dynamics_p0f_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_repair_dynamics_p0f_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "gamut_repair_dynamics_p0f_review.md").write_text(
        review_packet(rows), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0e-candidates", type=Path, required=True)
    parser.add_argument("--p0d-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
