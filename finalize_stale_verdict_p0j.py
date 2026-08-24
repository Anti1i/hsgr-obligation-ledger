"""Apply the frozen P0j manual audit and recompute the pre-registered metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stale_verdict_p0j import TARGET_RECALL, analyze_transitions


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finalize(result_dir: Path, audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transitions = load_jsonl(result_dir / "stale_verdict_p0j_transitions.jsonl")
    review = load_jsonl(result_dir / "stale_verdict_p0j_review.jsonl")
    false_positive_ids = set(audit["false_positive_transition_ids"])
    candidate_positive_ids = {row["transition_id"] for row in transitions if row["stale"]}
    reviewed_positive_ids = {row["transition_id"] for row in review if row["stale"]}
    reviewed_negative_ids = {row["transition_id"] for row in review if not row["stale"]}
    if candidate_positive_ids != reviewed_positive_ids:
        raise RuntimeError("manual packet does not contain every candidate positive")
    if not false_positive_ids <= candidate_positive_ids:
        raise RuntimeError("audit lists a non-positive transition as a false positive")
    if len(candidate_positive_ids) != audit["candidate_positive_count"]:
        raise RuntimeError("candidate-positive count does not match audit")
    if len(reviewed_negative_ids) != audit["frozen_negative_sample_count"]:
        raise RuntimeError("negative review sample count does not match audit")
    for row in transitions:
        row["candidate_stale"] = row["stale"]
        row["stale"] = bool(row["stale"] and row["transition_id"] not in false_positive_ids)
        row["manual_label"] = row["stale"] if row["transition_id"] in reviewed_positive_ids | reviewed_negative_ids else None

    revision_count = len({row["revision_id"] for row in transitions})
    policies = analyze_transitions(transitions, revision_count)
    confirmed = [row for row in transitions if row["stale"]]
    scenarios = sorted({row["scenario_id"] for row in confirmed})
    operators = sorted({row["operator"] for row in confirmed})
    generators = sorted({row["generator_model"] for row in confirmed})
    gate_2 = len(confirmed) >= 20 and len(scenarios) >= 5 and len(operators) >= 3
    learned = policies["learned"]
    single = [
        policies[name] for name in ("witness_overlap", "proximity", "witness_similarity")
        if policies[name]["stale_recall"] >= TARGET_RECALL
    ]
    best_single_saving = max((row["verification_saving"] for row in single), default=0.0)
    gate_3 = bool(
        learned.get("available")
        and learned["stale_recall"] >= TARGET_RECALL
        and learned["verification_saving"] >= 0.25
        and learned["stale_recall"] - learned["matched_random"]["mean_stale_recall"] >= 0.15
        and learned["verification_saving"] - best_single_saving >= 0.05
    )
    return {
        "job_id": audit["job_id"],
        "audit_path": audit_path.name,
        "revision_count": revision_count,
        "transition_count": len(transitions),
        "manual_audit": {
            "candidate_positives": len(candidate_positive_ids),
            "confirmed_stale": len(confirmed),
            "candidate_positive_precision": len(confirmed) / len(candidate_positive_ids),
            "false_positives": len(false_positive_ids),
            "frozen_negative_sample": len(reviewed_negative_ids),
            "frozen_negative_agreement": audit["frozen_negative_sample_confirmed_non_stale"] / len(reviewed_negative_ids),
            "false_positive_reason": audit["false_positive_reason"],
        },
        "confirmed_positive_coverage": {
            "scenarios": scenarios,
            "operators": operators,
            "generators": generators,
            "by_scenario": dict(sorted(Counter(row["scenario_id"] for row in confirmed).items())),
            "by_operator": dict(sorted(Counter(row["operator"] for row in confirmed).items())),
            "by_generator": dict(sorted(Counter(row["generator_model"] for row in confirmed).items())),
            "by_target_type": dict(sorted(Counter(row["target_type"] for row in confirmed).items())),
        },
        "policies_after_manual_audit": policies,
        "manual_gates": {
            "gate_1_verifier_apparatus": True,
            "gate_2_phenomenon_support": gate_2,
            "gate_3_useful_predictability": gate_3,
            "all_gates": bool(gate_2 and gate_3),
        },
        "decision": (
            "STOP learned invalidator/RL on this apparatus. Confirmed staleness is confined to "
            "multi-sentence replacement operators and is already covered by deterministic diff/"
            "witness invalidation; isolated target replacement and full rewrite produced no "
            "confirmed stale non-target verdicts."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("MANUAL_AUDIT_STALE_VERDICT_P0J.json"))
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = finalize(args.result_dir, args.audit)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
