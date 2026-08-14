"""Fresh ASQA single-node intervention Oracle P3x."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
    render_checklist,
    render_user,
    select_cases as select_p1x_cases,
)
from asqa_fixed_support_audit import facet_score


SELECTION_SALT = "20260815-asqa-single-node-p3x"
HALF_SALT = "20260815-asqa-single-node-p3x-half"
EXPECTED_ELIGIBLE = 427
EXPECTED_OLD = 192
EXPECTED_FRESH_POOL = 235


@dataclass(frozen=True)
class Decoy:
    case_id: str
    facet_index: int
    question: str


def fixed_half(record_id: str) -> int:
    return int(hash_key(f"{HALF_SALT}|{record_id}"), 16) % 2


def select_fresh_cases(
    eligible: list[Case], n: int = 192
) -> tuple[list[Case], list[Case], list[Case]]:
    old_cases = select_p1x_cases(eligible, EXPECTED_OLD)
    old_ids = {case.id for case in old_cases}
    fresh_pool = [case for case in eligible if case.id not in old_ids]
    ordered = sorted(
        fresh_pool, key=lambda case: hash_key(f"{SELECTION_SALT}|{case.id}")
    )
    if len(ordered) < n:
        raise RuntimeError(f"need {n} fresh cases, found {len(ordered)}")
    return old_cases, fresh_pool, ordered[:n]


def build_single_decoys(cases: list[Case]) -> dict[str, Decoy]:
    candidates = [
        Decoy(case.id, index, question)
        for case in cases
        for index, question in enumerate(case.facet_questions, 1)
    ]
    mapping: dict[str, Decoy] = {}
    for source in cases:
        target_words = mean(len(question.split()) for question in source.facet_questions)
        valid = [candidate for candidate in candidates if candidate.case_id != source.id]
        if not valid:
            raise RuntimeError(f"no decoy candidate for {source.id}")
        mapping[source.id] = min(
            valid,
            key=lambda candidate: (
                abs(len(candidate.question.split()) - target_words),
                hash_key(
                    f"asqa-p3x-decoy|{source.id}|{candidate.case_id}|"
                    f"{candidate.facet_index}"
                ),
            ),
        )
    return mapping


def render_single_user(case: Case, question: str) -> str:
    direct = render_user(case, "fixed_direct")
    suffix = "\n\nAnswer:"
    if not direct.endswith(suffix):
        raise RuntimeError("unexpected P1x direct prompt format")
    return (
        direct[: -len(suffix)]
        + "\n\nCoverage checklist (use this question only to decide which "
        "interpretation to cover; do not mention the checklist):\n"
        + render_checklist((question,))
        + suffix
    )


def score_generation(
    case: Case,
    answer: str,
    arm: str,
    candidate_index: int | None = None,
    decoy: Decoy | None = None,
) -> dict[str, Any]:
    coverage, strict, present = facet_score(list(case.alias_groups), answer)
    return {
        "id": case.id,
        "arm": arm,
        "candidate_index": candidate_index,
        "decoy_case_id": decoy.case_id if decoy else None,
        "decoy_facet_index": decoy.facet_index if decoy else None,
        "answer": answer,
        "str_em": coverage,
        "str_hit": strict,
        "present_vector": [bool(value) for value in present],
        "present_facets": sum(present),
        "facet_count": len(present),
        "word_count": len(answer.split()),
    }


def metric_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "str_em": mean(row["str_em"] for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
        "median_words": median(row["word_count"] for row in rows),
        "mean_words": mean(row["word_count"] for row in rows),
    }


def policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "str_em": mean(row["str_em"] for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
    }


def validate_rows(cases: list[Case], rows: list[dict[str, Any]]) -> bool:
    by_key = Counter(
        (row["id"], row["arm"], row.get("candidate_index")) for row in rows
    )
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
    return len(rows) == len(expected) and all(by_key[key] == 1 for key in expected)


def summarize(
    eligible: list[Case],
    old_cases: list[Case],
    fresh_pool: list[Case],
    cases: list[Case],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_by_id = {case.id: case for case in cases}
    direct = {row["id"]: row for row in rows if row["arm"] == "fixed_direct"}
    all_true = {row["id"]: row for row in rows if row["arm"] == "all_true"}
    decoy = {row["id"]: row for row in rows if row["arm"] == "single_decoy"}
    singles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] == "single_true":
            singles[row["id"]].append(row)
    for problem_rows in singles.values():
        problem_rows.sort(key=lambda row: row["candidate_index"])

    direct_metrics = metric_summary(direct.values())
    all_metrics = metric_summary(all_true.values())
    decoy_metrics = metric_summary(decoy.values())

    uniform_problem_rows = []
    for case in cases:
        problem_rows = singles[case.id]
        uniform_problem_rows.append(
            {
                "str_em": mean(row["str_em"] for row in problem_rows),
                "str_hit": mean(float(row["str_hit"]) for row in problem_rows),
                "word_count": mean(row["word_count"] for row in problem_rows),
            }
        )
    uniform_metrics = metric_summary(uniform_problem_rows)

    fixed_position: dict[str, dict[str, Any]] = {}
    fixed_position_rows: dict[int, list[dict[str, Any]]] = {}
    for position in range(1, 7):
        chosen = []
        for case in cases:
            candidates = [
                row for row in singles[case.id] if row["candidate_index"] == position
            ]
            chosen.append(candidates[0] if candidates else direct[case.id])
        fixed_position_rows[position] = chosen
        fixed_position[str(position)] = policy_summary(chosen)
    best_position = max(
        range(1, 7),
        key=lambda position: (
            fixed_position[str(position)]["str_hit"],
            fixed_position[str(position)]["str_em"],
            -position,
        ),
    )
    best_fixed = {
        "position": best_position,
        **fixed_position[str(best_position)],
    }

    keep_all_hit: list[bool] = []
    keep_single_hit: list[bool] = []
    keep_all_em: list[float] = []
    keep_single_em: list[float] = []
    mixed_ids: list[str] = []
    direct_failure_ids: list[str] = []
    repair_rows = 0
    direct_failure_single_rows = 0
    repair_by_index: Counter[int] = Counter()
    single_only_ids: list[str] = []
    all_only_ids: list[str] = []

    injected_deltas: list[float] = []
    non_injected_deltas: list[float] = []
    for case in cases:
        source = direct[case.id]
        all_row = all_true[case.id]
        candidate_rows = singles[case.id]
        d_hit = bool(source["str_hit"])
        a_hit = bool(all_row["str_hit"])
        s_hits = [bool(row["str_hit"]) for row in candidate_rows]
        all_oracle_hit = d_hit or a_hit
        single_oracle_hit = d_hit or any(s_hits)
        keep_all_hit.append(all_oracle_hit)
        keep_single_hit.append(single_oracle_hit)
        keep_all_em.append(max(source["str_em"], all_row["str_em"]))
        keep_single_em.append(
            max([source["str_em"]] + [row["str_em"] for row in candidate_rows])
        )
        if single_oracle_hit and not all_oracle_hit:
            single_only_ids.append(case.id)
        if all_oracle_hit and not single_oracle_hit:
            all_only_ids.append(case.id)

        if not d_hit:
            direct_failure_ids.append(case.id)
            direct_failure_single_rows += len(candidate_rows)
            repair_rows += sum(s_hits)
            if any(s_hits) and not all(s_hits):
                mixed_ids.append(case.id)
            for row in candidate_rows:
                if row["str_hit"]:
                    repair_by_index[int(row["candidate_index"])] += 1

        for row in candidate_rows:
            index = int(row["candidate_index"]) - 1
            direct_vector = source["present_vector"]
            single_vector = row["present_vector"]
            injected_deltas.append(
                float(single_vector[index]) - float(direct_vector[index])
            )
            other_deltas = [
                float(single_vector[other]) - float(direct_vector[other])
                for other in range(len(direct_vector))
                if other != index
            ]
            non_injected_deltas.append(mean(other_deltas))

    mcnemar_p, single_only, all_only = exact_mcnemar_p(
        keep_single_hit, keep_all_hit
    )
    keep_all_metrics = {
        "n": len(cases),
        "str_hit": mean(float(value) for value in keep_all_hit),
        "str_em": mean(keep_all_em),
    }
    keep_single_metrics = {
        "n": len(cases),
        "str_hit": mean(float(value) for value in keep_single_hit),
        "str_em": mean(keep_single_em),
    }

    half_results: dict[str, Any] = {}
    for half in (0, 1):
        indices = [index for index, case in enumerate(cases) if fixed_half(case.id) == half]
        half_all = mean(float(keep_all_hit[index]) for index in indices)
        half_single = mean(float(keep_single_hit[index]) for index in indices)
        half_results[str(half)] = {
            "n": len(indices),
            "keep_or_all_str_hit": half_all,
            "keep_or_single_str_hit": half_single,
            "single_minus_all": half_single - half_all,
        }

    apparatus_gates = {
        "g1_exact_fresh_counts_and_zero_overlap": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(old_cases) == EXPECTED_OLD
            and len(fresh_pool) == EXPECTED_FRESH_POOL
            and len(cases) == 192
            and not ({case.id for case in old_cases} & {case.id for case in cases})
        ),
        "g2_nodes_and_generation_rows_complete": (
            all(2 <= len(case.facet_questions) <= 6 for case in cases)
            and validate_rows(cases, rows)
        ),
        "g3_direct_operating_point_and_failures": (
            0.35 <= direct_metrics["str_hit"] <= 0.75
            and 30 <= direct_metrics["median_words"] <= 160
            and len(direct_failure_ids) >= 60
        ),
        "g4_all_true_beats_direct_hit_by_5_points": (
            all_metrics["str_hit"] - direct_metrics["str_hit"] >= 0.05
        ),
    }
    actionability_gates = {
        "g5_single_oracle_beats_all_oracle_hit_by_5_points": (
            keep_single_metrics["str_hit"] - keep_all_metrics["str_hit"] >= 0.05
        ),
        "g6_oracle_mcnemar_below_0_05_and_single_only_wins": (
            mcnemar_p < 0.05 and single_only > all_only
        ),
        "g7_at_least_24_mixed_intervention_problems": len(mixed_ids) >= 24,
        "g8_repair_prevalence_between_5_and_50pct": (
            direct_failure_single_rows > 0
            and 0.05 <= repair_rows / direct_failure_single_rows <= 0.50
        ),
        "g9_single_oracle_beats_best_fixed_hit_by_10_points": (
            keep_single_metrics["str_hit"] - best_fixed["str_hit"] >= 0.10
        ),
        "g10_single_oracle_beats_all_oracle_in_both_halves_by_2_points": all(
            result["single_minus_all"] >= 0.02 for result in half_results.values()
        ),
        "g11_uniform_single_beats_single_decoy_em_by_2_points": (
            uniform_metrics["str_em"] - decoy_metrics["str_em"] >= 0.02
        ),
        "g12_injected_delta_exceeds_non_injected_by_3_points": (
            mean(injected_deltas) - mean(non_injected_deltas) >= 0.03
        ),
    }
    if all(apparatus_gates.values()) and all(actionability_gates.values()):
        outcome = "SINGLE_NODE_ACTIONABILITY_PASS"
    elif all(apparatus_gates.values()):
        outcome = "STATIC_REPLICATES_SELECTION_FAIL"
    else:
        outcome = "FRESH_REPLICATION_FAIL"

    return {
        "outcome": outcome,
        "counts": {
            "eligible": len(eligible),
            "old_p1x": len(old_cases),
            "fresh_pool": len(fresh_pool),
            "selected": len(cases),
            "old_new_overlap": len(
                {case.id for case in old_cases} & {case.id for case in cases}
            ),
            "generation_rows": len(rows),
            "expected_generation_rows": 3 * len(cases)
            + sum(len(case.facet_questions) for case in cases),
            "direct_failures": len(direct_failure_ids),
            "mixed_intervention_problems": len(mixed_ids),
            "direct_failure_single_rows": direct_failure_single_rows,
            "strict_repair_rows": repair_rows,
        },
        "absolute": {
            "fixed_direct": direct_metrics,
            "all_true": all_metrics,
            "uniform_single": uniform_metrics,
            "single_decoy": decoy_metrics,
            "keep_or_all_oracle": keep_all_metrics,
            "keep_or_single_oracle": keep_single_metrics,
            "best_fixed_position": best_fixed,
        },
        "fixed_position_policies": fixed_position,
        "paired_changes": {
            "all_true_minus_direct_str_hit": all_metrics["str_hit"]
            - direct_metrics["str_hit"],
            "all_true_minus_direct_str_em": all_metrics["str_em"]
            - direct_metrics["str_em"],
            "single_oracle_minus_all_oracle_str_hit": keep_single_metrics["str_hit"]
            - keep_all_metrics["str_hit"],
            "single_oracle_minus_best_fixed_str_hit": keep_single_metrics["str_hit"]
            - best_fixed["str_hit"],
            "uniform_single_minus_decoy_str_em": uniform_metrics["str_em"]
            - decoy_metrics["str_em"],
        },
        "oracle_mcnemar": {
            "p_exact_two_sided": mcnemar_p,
            "single_only_successes": single_only,
            "all_only_successes": all_only,
            "single_only_ids": single_only_ids,
            "all_only_ids": all_only_ids,
        },
        "node_specificity": {
            "mean_injected_facet_coverage_delta": mean(injected_deltas),
            "mean_non_injected_facet_coverage_delta": mean(non_injected_deltas),
            "injected_minus_non_injected_delta": mean(injected_deltas)
            - mean(non_injected_deltas),
        },
        "repair_prevalence": (
            repair_rows / direct_failure_single_rows
            if direct_failure_single_rows
            else 0.0
        ),
        "repair_rows_by_released_index": dict(sorted(repair_by_index.items())),
        "mixed_intervention_ids": mixed_ids,
        "fixed_id_hash_halves": half_results,
        "apparatus_gates": apparatus_gates,
        "actionability_gates": actionability_gates,
        "selected_facet_histogram": dict(
            sorted(Counter(len(case.facet_questions) for case in cases).items())
        ),
        "selected_ids_by_facet_count": {
            str(count): [
                case.id for case in cases if len(case.facet_questions) == count
            ]
            for count in sorted({len(case.facet_questions) for case in cases})
        },
        "case_count_check": len(case_by_id) == len(cases),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eligible = aligned_clean_cases(args.alce, args.original)
    old_cases, fresh_pool, cases = select_fresh_cases(eligible, args.n)
    decoys = build_single_decoys(cases)
    print(
        f"[apparatus] eligible={len(eligible)} old={len(old_cases)} "
        f"fresh_pool={len(fresh_pool)} selected={len(cases)} "
        f"overlap={len({case.id for case in old_cases} & {case.id for case in cases})} "
        f"facets={dict(sorted(Counter(len(case.facet_questions) for case in cases).items()))}",
        flush=True,
    )

    prompts: list[str] = []
    keys: list[tuple[Case, str, int | None, Decoy | None]] = []
    for case in cases:
        prompts.append(render_user(case, "fixed_direct"))
        keys.append((case, "fixed_direct", None, None))
        prompts.append(render_user(case, "true_facets"))
        keys.append((case, "all_true", None, None))
        for index, question in enumerate(case.facet_questions, 1):
            prompts.append(render_single_user(case, question))
            keys.append((case, "single_true", index, None))
        decoy = decoys[case.id]
        prompts.append(render_single_user(case, decoy.question))
        keys.append((case, "single_decoy", None, decoy))

    print(f"[apparatus] generation_prompts={len(prompts)}", flush=True)
    runner = ModelRunner(args.model)
    answers = runner.generate(prompts, args.batch_size, args.max_new_tokens)
    if len(answers) != len(keys):
        raise RuntimeError(f"generated {len(answers)} answers for {len(keys)} prompts")
    rows = [
        score_generation(case, answer, arm, candidate_index, decoy)
        for (case, arm, candidate_index, decoy), answer in zip(keys, answers)
    ]

    metrics = summarize(eligible, old_cases, fresh_pool, cases, rows)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_ASQA_SINGLE_NODE_INTERVENTION_P3X.md",
        "outcome": metrics["outcome"],
        "model": args.model,
        "selection_salt": SELECTION_SALT,
        "selected_id_sha256": hash_key("\n".join(case.id for case in cases) + "\n"),
        "old_p1x_id_sha256": hash_key(
            "\n".join(case.id for case in old_cases) + "\n"
        ),
        "generation": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        **metrics,
        "protocol_match": {
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            "n": args.n == 192,
            "max_new_tokens": args.max_new_tokens == 192,
            "selection_salt": SELECTION_SALT
            == "20260815-asqa-single-node-p3x",
        },
        "interpretation_guard": (
            "P3x is a fresh textual intervention Oracle and target audit. It is not "
            "HSGR, a hidden-state result, or a deployable selector. A pass licenses "
            "only a separately frozen hidden marginal-utility screen."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_single_node_intervention_p3x_generations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_single_node_intervention_p3x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "selected_ids.txt").write_text(
        "\n".join(case.id for case in cases) + "\n", encoding="utf-8"
    )
    (args.out_dir / "decoy_mapping.json").write_text(
        json.dumps(
            {case_id: asdict(decoy) for case_id, decoy in decoys.items()},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
