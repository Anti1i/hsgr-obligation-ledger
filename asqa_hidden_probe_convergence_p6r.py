"""Frozen convergence audit for the ASQA P6x hidden probe."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

from asqa_clean_fixed_support_p1x import aligned_clean_cases, mean, select_cases
from asqa_missing_selector_p6x import (
    CS,
    EXPECTED_CASES,
    EXPECTED_ELIGIBLE,
    EXPECTED_P1X_ROWS,
    EXPECTED_P3X_ROWS,
    LAYERS,
    ModelRunner,
    build_selector_cases,
    candidate_auroc,
    fold_id,
    make_candidates,
    paired_boolean,
    target_selection_accuracy,
)
from asqa_obligation_repair_p5x import load_p3x_rows
from asqa_set_guide_patch_p4x import load_p1x_rows
from asqa_single_node_intervention_p3x import select_fresh_cases


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_HIDDEN_CONVERGENCE_P6R.md"
MAX_ITER = 5000
TOL = 1e-6


def load_p6x_scores(path: Path, candidates: list[Any]) -> dict[str, Any]:
    import numpy as np

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {(str(row["id"]), int(row["facet_index"]) - 1): row for row in rows}
    if len(rows) != len(candidates) or len(by_key) != len(candidates):
        raise RuntimeError(f"P6x candidate count mismatch: rows={len(rows)} expected={len(candidates)}")
    aligned = []
    for candidate in candidates:
        key = (candidate.case_id, candidate.facet_index)
        if key not in by_key:
            raise RuntimeError(f"P6x candidate missing: {key}")
        row = by_key[key]
        if bool(row["missing_label"]) != bool(candidate.missing):
            raise RuntimeError(f"P6x label mismatch: {key}")
        aligned.append(row)
    return {
        "hidden_probe": np.asarray([float(row["hidden_probe_score"]) for row in aligned]),
        "logit": np.asarray([float(row["logit_score"]) for row in aligned]),
        "random": np.asarray([float(row["random_score"]) for row in aligned]),
    }


def fit_one(x: Any, labels: Any, c_value: float) -> tuple[Any, dict[str, Any]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            dual=False,
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=0,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, labels)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    n_iter = int(model.named_steps["logisticregression"].n_iter_[0])
    return model, {"n_iter": n_iter, "convergence_warning": warned, "converged": not warned and n_iter < MAX_ITER}


def fit_stable_probe(candidates: list[Any], features: dict[int, Any], exact_ids: set[str]):
    import numpy as np

    labels = np.asarray([int(candidate.missing) for candidate in candidates], dtype=np.int64)
    folds = np.asarray([fold_id(candidate.case_id) for candidate in candidates], dtype=np.int64)
    cells: list[dict[str, Any]] = []
    for layer in LAYERS:
        for c_value in CS:
            oof = np.full(len(candidates), np.nan, dtype=np.float64)
            fit_info = []
            for fold in range(5):
                train = folds != fold
                valid = folds == fold
                model, info = fit_one(features[layer][train], labels[train], c_value)
                oof[valid] = model.predict_proba(features[layer][valid])[:, 1]
                fit_info.append({"fold": fold, **info})
            if not np.isfinite(oof).all():
                raise RuntimeError("non-finite stable OOF scores")
            cells.append(
                {
                    "layer": layer,
                    "C": c_value,
                    "oof_candidate_auroc": candidate_auroc(candidates, oof),
                    "oof_exact_one_selection_accuracy": target_selection_accuracy(candidates, oof, exact_ids)[0],
                    "fits": fit_info,
                    "all_folds_converged": all(info["converged"] for info in fit_info),
                }
            )
            print(f"[probe] layer={layer} C={c_value} complete", flush=True)
    selected = min(
        cells,
        key=lambda cell: (
            -cell["oof_exact_one_selection_accuracy"],
            -cell["oof_candidate_auroc"],
            cell["C"],
            cell["layer"],
        ),
    )
    final_model, final_info = fit_one(features[int(selected["layer"])], labels, float(selected["C"]))
    return {"cells": cells, "selected": selected, "final_fit": final_info}, final_model


def outcome_for(apparatus_pass: bool, converged: bool, recovery: bool) -> str:
    if not apparatus_pass:
        return "APPARATUS_FAIL"
    if not converged:
        return "SOLVER_STILL_FAIL"
    if recovery:
        return "CONVERGED_HIDDEN_RECOVERY"
    return "CONVERGED_HIDDEN_NO_ADVANTAGE"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    eligible = aligned_clean_cases(args.alce, args.original)
    eval_cases = select_cases(eligible, EXPECTED_CASES)
    old_cases, fresh_pool, train_cases = select_fresh_cases(eligible, EXPECTED_CASES)
    if [case.id for case in old_cases] != [case.id for case in eval_cases]:
        raise RuntimeError("P1x reconstruction mismatch")
    p1x = load_p1x_rows(args.p1x_generations, eval_cases)
    p3x_rows = load_p3x_rows(args.p3x_generations, train_cases)
    train_direct = {row["id"]: row for row in p3x_rows if row["arm"] == "fixed_direct"}
    eval_direct = {case.id: p1x[(case.id, "fixed_direct")] for case in eval_cases}
    train_items, train_rescore = build_selector_cases(train_cases, train_direct)
    eval_items, eval_rescore = build_selector_cases(eval_cases, eval_direct)
    train_mixed = [item for item in train_items if item.mixed]
    eval_mixed = [item for item in eval_items if item.mixed]
    train_repairs = [item for item in train_items if item.exactly_one_missing]
    eval_repairs = [item for item in eval_items if item.exactly_one_missing]
    train_candidates = make_candidates(train_mixed, "train_p3x")
    eval_candidates = make_candidates(eval_mixed, "eval_p1x")
    frozen = load_p6x_scores(args.p6x_candidates, eval_candidates)

    runner = ModelRunner(args.model)
    train_features, train_logits = runner.extract_selector(
        [candidate.prompt for candidate in train_candidates], LAYERS, args.batch_size
    )
    selection, probe = fit_stable_probe(
        train_candidates, train_features, {item.case.id for item in train_repairs}
    )
    layer = int(selection["selected"]["layer"])
    eval_features, eval_logits = runner.extract_selector(
        [candidate.prompt for candidate in eval_candidates], LAYERS, args.batch_size
    )
    stable_scores = probe.predict_proba(eval_features[layer])[:, 1]
    arrays = [train_logits, eval_logits, stable_scores, frozen["hidden_probe"], frozen["logit"]]
    finite = all(np.isfinite(array).all() for array in arrays)
    logit_max_abs_error = float(np.max(np.abs(eval_logits - frozen["logit"])))

    exact_ids = {item.case.id for item in eval_repairs}
    ordered_ids = [item.case.id for item in eval_repairs]
    targets = {item.case.id: item.missing_indices[0] for item in eval_repairs}
    score_sets = {
        "stable_hidden": stable_scores,
        "p6x_hidden": frozen["hidden_probe"],
        "logit": frozen["logit"],
        "random": frozen["random"],
    }
    absolute = {}
    correctness = {}
    for name, scores in score_sets.items():
        accuracy, mapping = target_selection_accuracy(eval_candidates, scores, exact_ids)
        correctness[name] = [mapping[case_id] == targets[case_id] for case_id in ordered_ids]
        absolute[name] = {
            "candidate_auroc": candidate_auroc(eval_candidates, scores),
            "exact_one_target_selection_accuracy": accuracy,
        }
    paired = {
        "stable_vs_p6x_hidden": paired_boolean(
            "stable_hidden", correctness["stable_hidden"], "p6x_hidden", correctness["p6x_hidden"]
        ),
        "stable_vs_logit": paired_boolean(
            "stable_hidden", correctness["stable_hidden"], "logit", correctness["logit"]
        ),
        "stable_vs_random": paired_boolean(
            "stable_hidden", correctness["stable_hidden"], "random", correctness["random"]
        ),
    }

    apparatus_gates = {
        "exact_counts_and_zero_overlap": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(train_cases) == EXPECTED_CASES
            and len(eval_cases) == EXPECTED_CASES
            and len(fresh_pool) == 235
            and len(p1x) == EXPECTED_P1X_ROWS
            and len(p3x_rows) == EXPECTED_P3X_ROWS
            and len(train_candidates) == 324
            and len(eval_candidates) == 288
            and len(eval_repairs) == 73
            and not ({case.id for case in train_cases} & {case.id for case in eval_cases})
        ),
        "exact_rescore_and_finite": train_rescore and eval_rescore and finite,
        "p6x_alignment_and_logit_replay": logit_max_abs_error <= 1e-3,
    }
    every_cv_fit_converged = all(
        cell["all_folds_converged"] for cell in selection["cells"]
    )
    convergence_gates = {
        "all_45_cv_fits_converged": every_cv_fit_converged,
        "final_refit_converged": bool(selection["final_fit"]["converged"]),
    }
    stable_vs_logit = paired["stable_vs_logit"]
    recovery_gates = {
        "stable_auc_and_selection": (
            absolute["stable_hidden"]["candidate_auroc"] >= 0.70
            and absolute["stable_hidden"]["exact_one_target_selection_accuracy"] >= 0.50
        ),
        "stable_beats_logit_by_5_points": stable_vs_logit["delta"] >= 0.05,
    }
    apparatus_pass = all(apparatus_gates.values())
    convergence_pass = all(convergence_gates.values())
    recovery_pass = all(recovery_gates.values())
    outcome = outcome_for(apparatus_pass, convergence_pass, recovery_pass)
    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "eligible": len(eligible),
            "train_cases": len(train_cases),
            "eval_cases": len(eval_cases),
            "train_candidates": len(train_candidates),
            "eval_candidates": len(eval_candidates),
            "eval_exact_one_missing": len(eval_repairs),
        },
        "optimizer": {
            "objective": "balanced L2 logistic",
            "scaler_with_mean": False,
            "solver": "liblinear",
            "dual": False,
            "max_iter": MAX_ITER,
            "tol": TOL,
        },
        "probe_selection": selection,
        "eval_selector_absolute": absolute,
        "eval_selector_paired": paired,
        "logit_replay_max_abs_error": logit_max_abs_error,
        "apparatus_gates": apparatus_gates,
        "convergence_gates": convergence_gates,
        "hidden_recovery_gates": recovery_gates,
        "interpretation_guard": (
            "P6r changes only the optimizer used to solve the same frozen hidden-probe objective. "
            "It does not test facet induction, hierarchy, generation, or end-to-end training."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "asqa_hidden_probe_convergence_p6r_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "asqa_hidden_probe_convergence_p6r_scores.jsonl").open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(eval_candidates):
            handle.write(
                json.dumps(
                    {
                        "id": candidate.case_id,
                        "facet_index": candidate.facet_index + 1,
                        "missing_label": candidate.missing,
                        "stable_hidden_score": float(stable_scores[index]),
                        "p6x_hidden_score": float(frozen["hidden_probe"][index]),
                        "logit_score": float(frozen["logit"][index]),
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--p3x-generations", type=Path, required=True)
    parser.add_argument("--p6x-candidates", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
