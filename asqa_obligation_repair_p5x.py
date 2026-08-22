"""Frozen ASQA obligation-preserving local repair Oracle P5x."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from asqa_clean_fixed_support_p1x import (
    Case,
    ModelRunner,
    aligned_clean_cases,
    exact_mcnemar_p,
    hash_key,
    mean,
    median,
    render_documents,
)
from asqa_fixed_support_audit import facet_score
from asqa_single_node_intervention_p3x import select_fresh_cases


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_OBLIGATION_REPAIR_P5X.md"
WRONG_SALT = "20260822-asqa-obligation-repair-p5x-swap"
EXPECTED_ELIGIBLE = 427
EXPECTED_P3X_CASES = 192
EXPECTED_FRESH_POOL = 235
EXPECTED_P3X_ROWS = 1108
ARMS = (
    "target_append",
    "generic_append",
    "target_rewrite",
    "correct_ledger_rewrite",
    "swapped_ledger_rewrite",
)


@dataclass(frozen=True)
class RepairCase:
    case: Case
    direct_answer: str
    original_present: tuple[bool, ...]
    target_index: int
    swap_index: int
    all_true_row: dict[str, Any]


def load_p3x_rows(path: Path, cases: list[Case]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    case_by_id = {case.id: case for case in cases}
    if len(rows) != EXPECTED_P3X_ROWS:
        raise RuntimeError(f"expected {EXPECTED_P3X_ROWS} P3x rows, found {len(rows)}")
    if {str(row.get("id")) for row in rows} != set(case_by_id):
        raise RuntimeError("P3x generation IDs do not match reconstructed cases")

    counts = Counter((row.get("id"), row.get("arm"), row.get("candidate_index")) for row in rows)
    expected: list[tuple[str, str, int | None]] = []
    for case in cases:
        expected.extend(
            [
                (case.id, "fixed_direct", None),
                (case.id, "all_true", None),
                (case.id, "single_decoy", None),
            ]
        )
        expected.extend(
            (case.id, "single_true", index)
            for index in range(1, len(case.facet_questions) + 1)
        )
    if len(expected) != len(rows) or any(counts[key] != 1 for key in expected):
        raise RuntimeError("P3x row apparatus is incomplete or duplicated")
    return rows


def choose_swap_index(case: Case, present: tuple[bool, ...]) -> int:
    candidates = [index for index, value in enumerate(present) if value]
    if not candidates:
        raise ValueError("cannot choose status swap without a present facet")
    return min(
        candidates,
        key=lambda index: hash_key(f"{WRONG_SALT}|{case.id}|{index}"),
    )


def build_repair_cases(cases: list[Case], rows: list[dict[str, Any]]) -> tuple[list[RepairCase], bool]:
    direct = {row["id"]: row for row in rows if row["arm"] == "fixed_direct"}
    all_true = {row["id"]: row for row in rows if row["arm"] == "all_true"}
    repair_cases: list[RepairCase] = []
    exact_rescore = True
    for case in cases:
        row = direct[case.id]
        _, strict, rescored = facet_score(list(case.alias_groups), str(row["answer"]))
        present = tuple(bool(value) for value in rescored)
        saved_present = tuple(bool(value) for value in row["present_vector"])
        exact_rescore = exact_rescore and present == saved_present and bool(strict) == bool(row["str_hit"])
        missing = [index for index, value in enumerate(present) if not value]
        if strict or len(missing) != 1 or sum(present) < 1:
            continue
        repair_cases.append(
            RepairCase(
                case=case,
                direct_answer=str(row["answer"]),
                original_present=present,
                target_index=missing[0],
                swap_index=choose_swap_index(case, present),
                all_true_row=all_true[case.id],
            )
        )
    return repair_cases, exact_rescore


def ledger_lines(repair: RepairCase, swapped: bool) -> list[str]:
    lines = []
    for index, question in enumerate(repair.case.facet_questions):
        missing_label = index == (repair.swap_index if swapped else repair.target_index)
        label = "UNSATISFIED — ADD" if missing_label else "SATISFIED — PRESERVE"
        lines.append(f"- [{label}] {question}")
    return lines


def common_context(repair: RepairCase) -> str:
    return (
        f"Question: {repair.case.question}\n\n"
        f"Fixed documents:\n{render_documents(repair.case)}\n\n"
        f"Saved answer:\n{repair.direct_answer}"
    )


def render_prompt(repair: RepairCase, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    common = common_context(repair)
    target = repair.case.facet_questions[repair.target_index]
    if arm == "target_append":
        instruction = (
            "Return only a short one- or two-sentence addition to append to the saved "
            "answer. Add the missing interpretation below using only the fixed documents. "
            "Do not repeat, rewrite, or contradict the saved answer, and do not add a preface.\n\n"
            f"Missing interpretation: {target}\n\nAddition:"
        )
    elif arm == "generic_append":
        instruction = (
            "Return only a short one- or two-sentence addition to append to the saved "
            "answer. Add one important, factually supported interpretation that the saved "
            "answer does not yet cover, using only the fixed documents. Do not repeat, "
            "rewrite, or contradict the saved answer, and do not add a preface.\n\nAddition:"
        )
    elif arm == "target_rewrite":
        instruction = (
            "Rewrite the saved answer as one coherent answer. Add the missing interpretation "
            "below using only the fixed documents while preserving every correct fact and "
            "interpretation already expressed. Do not mention these instructions.\n\n"
            f"Missing interpretation: {target}\n\nRewritten answer:"
        )
    else:
        swapped = arm == "swapped_ledger_rewrite"
        ledger = "\n".join(ledger_lines(repair, swapped))
        instruction = (
            "Rewrite the saved answer as one coherent answer using only the fixed documents. "
            "Follow the obligation ledger exactly: add the UNSATISFIED item and preserve every "
            "SATISFIED item without spending unnecessary extra detail on it. Do not mention "
            "the ledger or these instructions.\n\n"
            f"Obligation ledger:\n{ledger}\n\nRewritten answer:"
        )
    return f"{common}\n\nTask:\n{instruction}"


def final_answer(repair: RepairCase, arm: str, generated: str) -> str:
    if arm in {"target_append", "generic_append"}:
        return f"{repair.direct_answer.rstrip()} {generated.strip()}".strip()
    return generated.strip()


def score_arm(repair: RepairCase, arm: str, generated: str) -> dict[str, Any]:
    answer = final_answer(repair, arm, generated)
    coverage, strict, present_raw = facet_score(list(repair.case.alias_groups), answer)
    present = tuple(bool(value) for value in present_raw)
    original_indices = [index for index, value in enumerate(repair.original_present) if value]
    preserved = sum(present[index] for index in original_indices)
    lost = sum(not present[index] for index in original_indices)
    return {
        "id": repair.case.id,
        "arm": arm,
        "target_index": repair.target_index + 1,
        "swap_index": repair.swap_index + 1,
        "generated": generated,
        "answer": answer,
        "str_em": coverage,
        "str_hit": bool(strict),
        "target_recovered": present[repair.target_index],
        "present_vector": list(present),
        "original_present_vector": list(repair.original_present),
        "preservation_rate": preserved / len(original_indices),
        "all_original_present_preserved": lost == 0,
        "newly_lost_facets": lost,
        "generated_word_count": len(generated.split()),
        "answer_word_count": len(answer.split()),
        "facet_count": len(present),
    }


def metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "str_em": mean(float(row["str_em"]) for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
        "target_recovery": mean(float(row["target_recovered"]) for row in rows),
        "mean_preservation": mean(float(row["preservation_rate"]) for row in rows),
        "all_present_preserved": mean(float(row["all_original_present_preserved"]) for row in rows),
        "mean_newly_lost": mean(float(row["newly_lost_facets"]) for row in rows),
        "median_generated_words": median(float(row["generated_word_count"]) for row in rows),
        "median_answer_words": median(float(row["answer_word_count"]) for row in rows),
    }


def paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("paired IDs differ")
    ids = sorted(left_by_id)
    left_hits = [bool(left_by_id[case_id]["str_hit"]) for case_id in ids]
    right_hits = [bool(right_by_id[case_id]["str_hit"]) for case_id in ids]
    p_value, left_only, right_only = exact_mcnemar_p(left_hits, right_hits)
    left_metrics, right_metrics = metrics(left), metrics(right)
    return {
        "left": left[0]["arm"],
        "right": right[0]["arm"],
        "str_hit_delta": left_metrics["str_hit"] - right_metrics["str_hit"],
        "str_em_delta": left_metrics["str_em"] - right_metrics["str_em"],
        "all_present_preserved_delta": (
            left_metrics["all_present_preserved"] - right_metrics["all_present_preserved"]
        ),
        "mcnemar_p_exact_two_sided": p_value,
        "left_only_successes": left_only,
        "right_only_successes": right_only,
    }


def prompt_apparatus_valid(repairs: list[RepairCase]) -> bool:
    for repair in repairs:
        correct = render_prompt(repair, "correct_ledger_rewrite")
        swapped = render_prompt(repair, "swapped_ledger_rewrite")
        for question in repair.case.facet_questions:
            if correct.count(question) != 1 or swapped.count(question) != 1:
                return False
        if repair.target_index == repair.swap_index:
            return False
        correct_lines = ledger_lines(repair, False)
        swapped_lines = ledger_lines(repair, True)
        if sum("UNSATISFIED" in line for line in correct_lines) != 1:
            return False
        if sum("UNSATISFIED" in line for line in swapped_lines) != 1:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    eligible = aligned_clean_cases(args.alce, args.original)
    old_cases, fresh_pool, cases = select_fresh_cases(eligible, EXPECTED_P3X_CASES)
    p3x_rows = load_p3x_rows(args.p3x_generations, cases)
    repairs, exact_rescore = build_repair_cases(cases, p3x_rows)
    print(
        f"[apparatus] eligible={len(eligible)} old={len(old_cases)} "
        f"fresh_pool={len(fresh_pool)} p3x={len(cases)} rows={len(p3x_rows)} "
        f"repair_eligible={len(repairs)}",
        flush=True,
    )

    runner = ModelRunner(args.model)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        max_new_tokens = 96 if arm.endswith("append") else 192
        prompts = [render_prompt(repair, arm) for repair in repairs]
        generated = runner.generate(prompts, args.batch_size, max_new_tokens)
        if len(generated) != len(repairs):
            raise RuntimeError(f"{arm}: generated {len(generated)} outputs for {len(repairs)} cases")
        rows.extend(score_arm(repair, arm, output) for repair, output in zip(repairs, generated))
        print(f"[arm] {arm} complete", flush=True)

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    absolute = {arm: metrics(by_arm[arm]) for arm in ARMS}
    target_vs_generic = paired(by_arm["target_append"], by_arm["generic_append"])
    correct_vs_swapped = paired(
        by_arm["correct_ledger_rewrite"], by_arm["swapped_ledger_rewrite"]
    )
    correct_vs_target = paired(
        by_arm["correct_ledger_rewrite"], by_arm["target_rewrite"]
    )

    all_true_selected = [repair.all_true_row for repair in repairs]
    all_true_context = {
        "n": len(all_true_selected),
        "str_em": mean(float(row["str_em"]) for row in all_true_selected),
        "str_hit": mean(float(row["str_hit"]) for row in all_true_selected),
        "median_words": median(float(row["word_count"]) for row in all_true_selected),
    }

    apparatus_gates = {
        "g1_exact_source_counts_rows_and_zero_old_overlap": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(old_cases) == EXPECTED_P3X_CASES
            and len(fresh_pool) == EXPECTED_FRESH_POOL
            and len(cases) == EXPECTED_P3X_CASES
            and len(p3x_rows) == EXPECTED_P3X_ROWS
            and not ({case.id for case in old_cases} & {case.id for case in cases})
        ),
        "g2_at_least_40_valid_exactly_one_missing_cases": (
            len(repairs) >= 40
            and all(
                sum(not value for value in repair.original_present) == 1
                and sum(repair.original_present) >= 1
                and 2 <= len(repair.case.facet_questions) <= 6
                and len(repair.case.documents) == 5
                for repair in repairs
            )
        ),
        "g3_saved_present_vectors_exactly_rescore": exact_rescore,
        "g4_identical_facet_sets_and_single_status_swap": prompt_apparatus_valid(repairs),
    }
    local_gates = {
        "g5_target_append_hit_at_least_30pct": absolute["target_append"]["str_hit"] >= 0.30,
        "g6_target_beats_generic_append_by_10_points": target_vs_generic["str_hit_delta"] >= 0.10,
        "g7_target_vs_generic_significant_and_target_wins": (
            target_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and target_vs_generic["left_only_successes"] > target_vs_generic["right_only_successes"]
        ),
        "g8_target_append_preserves_all_present_in_98pct": (
            absolute["target_append"]["all_present_preserved"] >= 0.98
        ),
        "g9_target_append_median_generated_words_5_to_60": (
            5 <= absolute["target_append"]["median_generated_words"] <= 60
        ),
    }
    state_gates = {
        "g10_correct_ledger_hit_at_least_35pct": absolute["correct_ledger_rewrite"]["str_hit"] >= 0.35,
        "g11_correct_beats_swapped_by_5_points_significantly": (
            correct_vs_swapped["str_hit_delta"] >= 0.05
            and correct_vs_swapped["mcnemar_p_exact_two_sided"] < 0.05
            and correct_vs_swapped["left_only_successes"] > correct_vs_swapped["right_only_successes"]
        ),
        "g12_correct_beats_target_rewrite_by_5_points": correct_vs_target["str_hit_delta"] >= 0.05,
        "g13_correct_ledger_preserves_all_present_in_90pct": (
            absolute["correct_ledger_rewrite"]["all_present_preserved"] >= 0.90
        ),
        "g14_correct_preservation_beats_target_by_5_points": (
            correct_vs_target["all_present_preserved_delta"] >= 0.05
        ),
    }
    apparatus_pass = all(apparatus_gates.values())
    local_pass = apparatus_pass and all(local_gates.values())
    state_pass = apparatus_pass and all(state_gates.values())
    if not apparatus_pass:
        outcome = "APPARATUS_FAIL"
    elif local_pass and state_pass:
        outcome = "BOTH_PASS"
    elif local_pass:
        outcome = "LOCAL_ONLY_PASS"
    elif state_pass:
        outcome = "OBLIGATION_STATE_ONLY_PASS"
    else:
        outcome = "BOTH_FAIL"

    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "eligible": len(eligible),
            "old_p1x": len(old_cases),
            "fresh_pool": len(fresh_pool),
            "p3x_cases": len(cases),
            "p3x_rows": len(p3x_rows),
            "repair_eligible": len(repairs),
            "generated_rows": len(rows),
        },
        "selection": {
            "rule": "saved direct has exactly one missing and at least one present facet",
            "wrong_salt": WRONG_SALT,
            "repair_id_sha256": hash_key("\n".join(repair.case.id for repair in repairs) + "\n"),
            "repair_facet_histogram": dict(sorted(Counter(len(repair.case.facet_questions) for repair in repairs).items())),
        },
        "absolute": absolute,
        "paired": {
            "target_append_vs_generic_append": target_vs_generic,
            "correct_ledger_vs_swapped_ledger": correct_vs_swapped,
            "correct_ledger_vs_target_rewrite": correct_vs_target,
        },
        "existing_all_true_context": all_true_context,
        "apparatus_gates": apparatus_gates,
        "local_action_gates": local_gates,
        "obligation_state_gates": state_gates,
        "generation": {
            "greedy": True,
            "append_max_new_tokens": 96,
            "rewrite_max_new_tokens": 192,
            "batch_size": args.batch_size,
        },
        "protocol_match": {
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            "arms": list(ARMS),
            "wrong_salt": WRONG_SALT == "20260822-asqa-obligation-repair-p5x-swap",
        },
        "interpretation_guard": (
            "P5x is a gold-facet textual Oracle action/state screen. It does not establish "
            "automatic state induction, hidden-state control, hierarchy, or novelty."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_obligation_repair_p5x_generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_obligation_repair_p5x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "repair_ids.txt").write_text(
        "\n".join(repair.case.id for repair in repairs) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p3x-generations", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
