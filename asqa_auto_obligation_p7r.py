"""Parser-only replay of frozen P7x automatic obligation outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from asqa_auto_obligation_p7x import (
    choose_oracle,
    parse_obligations,
    random_index,
    relabel,
    render_candidate_append_prompt,
    render_coverage_prompt,
    valid_obligation_set,
)
from asqa_clean_fixed_support_p1x import aligned_clean_cases, select_cases
from asqa_missing_selector_p6x import (
    EXPECTED_CASES,
    EXPECTED_ELIGIBLE,
    EXPECTED_P1X_ROWS,
    ModelRunner,
    append_metrics,
    build_selector_cases,
    paired_append,
    score_append,
)
from asqa_set_guide_patch_p4x import load_p1x_rows


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_AUTO_OBLIGATION_P7R.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_obligations_repaired(text: str) -> tuple[list[str] | None, str]:
    parsed, mode = parse_obligations(text)
    if parsed is not None:
        return parsed, mode
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    values = []
    for line in [line.strip() for line in stripped.splitlines() if line.strip()]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None, "invalid"
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
            return None, "invalid"
        values.append(value[0])
    repaired = valid_obligation_set(values)
    return (repaired, "multi_array") if repaired is not None else (None, "invalid")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    eligible = aligned_clean_cases(args.alce, args.original)
    cases = select_cases(eligible, EXPECTED_CASES)
    p1x = load_p1x_rows(args.p1x_generations, cases)
    direct = {case.id: p1x[(case.id, "fixed_direct")] for case in cases}
    items, exact_rescore = build_selector_cases(cases, direct)
    repairs = [item for item in items if item.exactly_one_missing]
    item_by_id = {item.case.id: item for item in repairs}

    induction_rows = load_jsonl(args.p7x_inductions)
    frozen_candidates = load_jsonl(args.p7x_candidates)
    frozen_selections = load_jsonl(args.p7x_selections)
    if len(induction_rows) != 73 or len(frozen_candidates) != 238 or len(frozen_selections) != 438:
        raise RuntimeError("frozen P7x row-count mismatch")
    induction_by_id = {str(row["id"]): row for row in induction_rows}
    repaired_sets = {}
    parse_modes = {}
    for case_id, row in induction_by_id.items():
        parsed, mode = parse_obligations_repaired(str(row["raw_induction"]))
        if parsed is None:
            raise RuntimeError(f"P7r could not repair induction {case_id}")
        repaired_sets[case_id] = parsed
        parse_modes[case_id] = mode

    originally_valid = {
        case_id for case_id, row in induction_by_id.items() if bool(row["valid_four_node_set"])
    }
    recovered_ids = set(induction_by_id) - originally_valid
    reused_candidates = [row for row in frozen_candidates if str(row["id"]) in originally_valid]
    if len(originally_valid) != 55 or len(recovered_ids) != 18 or len(reused_candidates) != 220:
        raise RuntimeError("P7r frozen valid/invalid partition mismatch")
    reused_match = all(
        repaired_sets[str(row["id"])][int(row["candidate_index"]) - 1] == str(row["obligation"])
        for row in reused_candidates
    )
    if not reused_match:
        raise RuntimeError("reparsed valid obligations do not align with frozen candidates")

    recovered_flat = [
        (item_by_id[case_id], index, obligation)
        for case_id in sorted(recovered_ids)
        for index, obligation in enumerate(repaired_sets[case_id])
    ]
    runner = ModelRunner(args.model)
    coverage_prompts = [render_coverage_prompt(item, obligation) for item, _, obligation in recovered_flat]
    _, coverage_scores = runner.extract_selector(coverage_prompts, (27,), args.selector_batch_size)
    if not np.isfinite(coverage_scores).all():
        raise RuntimeError("non-finite P7r coverage scores")
    append_prompts = [render_candidate_append_prompt(item, obligation) for item, _, obligation in recovered_flat]
    additions = runner.generate(append_prompts, args.generation_batch_size, 96)
    new_candidates = []
    for flat_index, ((item, candidate_index, obligation), addition) in enumerate(zip(recovered_flat, additions)):
        row = score_append(item, "induced_candidate", None, addition)
        row.update(
            {
                "obligation": obligation,
                "candidate_index": candidate_index + 1,
                "coverage_score": float(coverage_scores[flat_index]),
                "p7r_recovered": True,
            }
        )
        new_candidates.append(row)
    all_candidates = [dict(row, p7r_recovered=False) for row in reused_candidates] + new_candidates
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_candidates:
        candidates_by_id[str(row["id"])].append(row)
    for rows in candidates_by_id.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))

    frozen_replay = {
        (str(row["id"]), str(row["arm"])): row
        for row in frozen_selections
        if row["arm"] in {"gold_oracle_append", "gold_logit_append", "p6x_generic_append"}
    }
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in repairs:
        candidates = candidates_by_id[item.case.id]
        if len(candidates) != 4:
            raise RuntimeError(f"P7r case does not have four candidates: {item.case.id}")
        auto_index = min(range(4), key=lambda index: (-float(candidates[index]["coverage_score"]), index))
        selections = {
            "induced_logit_append": auto_index,
            "induced_random_append": random_index(item.case.id, 4),
            "induced_oracle_append": choose_oracle(candidates),
        }
        for arm, index in selections.items():
            row = candidates[index]
            by_arm[arm].append(
                relabel(row, arm, str(row["obligation"]), index, float(row["coverage_score"]) if arm != "induced_oracle_append" else None)
            )
        for arm in ("gold_oracle_append", "gold_logit_append", "p6x_generic_append"):
            by_arm[arm].append(dict(frozen_replay[(item.case.id, arm)]))

    arms = (
        "induced_logit_append", "induced_random_append", "induced_oracle_append",
        "gold_oracle_append", "gold_logit_append", "p6x_generic_append",
    )
    absolute = {arm: append_metrics(by_arm[arm]) for arm in arms}
    paired = {
        "induced_logit_vs_random": paired_append(by_arm["induced_logit_append"], by_arm["induced_random_append"]),
        "induced_logit_vs_generic": paired_append(by_arm["induced_logit_append"], by_arm["p6x_generic_append"]),
        "induced_logit_vs_gold_logit": paired_append(by_arm["induced_logit_append"], by_arm["gold_logit_append"]),
        "induced_oracle_vs_gold_oracle": paired_append(by_arm["induced_oracle_append"], by_arm["gold_oracle_append"]),
    }
    replay_exact = (
        math.isclose(absolute["gold_oracle_append"]["str_hit"], 33 / 73, abs_tol=1e-12)
        and math.isclose(absolute["gold_logit_append"]["str_hit"], 29 / 73, abs_tol=1e-12)
        and math.isclose(absolute["p6x_generic_append"]["str_hit"], 7 / 73, abs_tol=1e-12)
    )
    apparatus_gates = {
        "exact_counts_and_rescore": (
            len(eligible) == EXPECTED_ELIGIBLE and len(cases) == EXPECTED_CASES
            and len(p1x) == EXPECTED_P1X_ROWS and len(repairs) == 73 and exact_rescore
        ),
        "all_73_exact_four_node_sets": len(repaired_sets) == 73 and all(len(values) == 4 for values in repaired_sets.values()),
        "exact_reuse_and_recovery_counts": reused_match and len(reused_candidates) == 220 and len(new_candidates) == 72 and len(all_candidates) == 292,
        "full_denominator_and_finite_scores": all(len(by_arm[arm]) == 73 for arm in arms),
        "exact_p6x_replay": replay_exact,
    }
    induced_oracle = absolute["induced_oracle_append"]["str_hit"]
    gold_oracle = absolute["gold_oracle_append"]["str_hit"]
    induced_auto = absolute["induced_logit_append"]["str_hit"]
    gold_logit = absolute["gold_logit_append"]["str_hit"]
    auto_vs_generic = paired["induced_logit_vs_generic"]
    auto_vs_random = paired["induced_logit_vs_random"]
    ledger_gates = {
        "induced_action_oracle": induced_oracle >= 0.30 and induced_oracle >= 0.65 * gold_oracle,
        "automatic_action_absolute": (
            induced_auto >= 0.20 and induced_auto >= 0.50 * gold_logit
            and absolute["induced_logit_append"]["all_present_preserved"] >= 0.98
        ),
        "automatic_beats_generic": (
            auto_vs_generic["delta"] >= 0.10 and auto_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_generic["left_only_successes"] > auto_vs_generic["right_only_successes"]
        ),
        "automatic_beats_induced_random": (
            auto_vs_random["delta"] >= 0.10 and auto_vs_random["mcnemar_p_exact_two_sided"] < 0.05
            and auto_vs_random["left_only_successes"] > auto_vs_random["right_only_successes"]
        ),
        "automatic_retains_60pct_induced_oracle": induced_oracle > 0 and induced_auto >= 0.60 * induced_oracle,
    }
    apparatus_pass = all(apparatus_gates.values())
    if not apparatus_pass:
        outcome = "APPARATUS_FAIL"
    elif all(ledger_gates.values()):
        outcome = "AUTO_OBLIGATION_LEDGER_PASS"
    elif ledger_gates["induced_action_oracle"]:
        outcome = "INDUCED_ACTION_ONLY"
    else:
        outcome = "AUTO_OBLIGATION_FAIL"
    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "repair_cases": len(repairs),
            "originally_valid_sets": len(originally_valid),
            "recovered_multi_array_sets": sum(mode == "multi_array" for mode in parse_modes.values()),
            "reused_candidates": len(reused_candidates),
            "new_candidates": len(new_candidates),
            "total_candidates": len(all_candidates),
        },
        "absolute": absolute,
        "paired": paired,
        "apparatus_gates": apparatus_gates,
        "ledger_gates": ledger_gates,
        "interpretation_guard": "P7r repairs only a frozen serialization mismatch and reuses every valid P7x candidate unchanged.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_auto_obligation_p7r_new_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in new_candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_auto_obligation_p7r_selections.jsonl").open("w", encoding="utf-8") as handle:
        for arm in arms:
            for row in by_arm[arm]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_auto_obligation_p7r_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--p7x-inductions", type=Path, required=True)
    parser.add_argument("--p7x-candidates", type=Path, required=True)
    parser.add_argument("--p7x-selections", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
