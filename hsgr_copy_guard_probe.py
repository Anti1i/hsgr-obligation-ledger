"""OOF structural probe for the HSGR predecessor-copy Guide guard.

This CPU-only diagnostic reuses frozen features from the structured verifier.
Unlike the answer-correctness probe, supervision is derived solely from the
known hierarchy state: whether a candidate exactly repeats a verified
predecessor value.  It tests whether a hidden reader can implement the useful
copy guard discovered by the lexical action ceiling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

from hsgr_focus_route_ceiling import exact_mcnemar
from hsgr_structured_hidden_verifier import LAYERS, ROUTES
from mh_ceiling import answers_match, normalize
from mh_e0 import hop_deps, load_rows
from mh_latent_rerank import auroc, fit_probe, within_problem_auroc


PENALTIES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
SUPPORT_WEIGHT = 0.75
EXACT_GUARD_WEIGHT = 0.75


def dependency_state(rows: list[dict]) -> dict:
    state = {}
    for row in rows:
        decomp = row["question_decomposition"]
        deps = set(hop_deps(decomp)[-1])
        state[row["_uid"]] = {
            "values": {
                normalize(str(decomp[index]["answer"])) for index in deps
            },
            "n_hops": len(decomp),
        }
    return state


def structural_oof(payload, route, copy_labels, folds, seed):
    import torch

    metas = payload["metas"]
    pids = sorted({meta["id"] for meta in metas})
    labels = torch.tensor(copy_labels, dtype=torch.float32)
    per_layer = {layer: [None] * len(metas) for layer in LAYERS}
    fold_internal = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        X = payload["features"][route][layer].float()
        for fold_index, hold in enumerate(folds):
            hold_set = set(hold)
            test_idx = [
                i for i, meta in enumerate(metas) if meta["id"] in hold_set
            ]
            train_pids = [pid for pid in pids if pid not in hold_set]
            rng = random.Random(seed + 31 * fold_index + layer)
            rng.shuffle(train_pids)
            n_val = max(1, len(train_pids) // 7)
            val_set = set(train_pids[:n_val])
            train_idx = [
                i for i, meta in enumerate(metas)
                if meta["id"] not in hold_set and meta["id"] not in val_set
            ]
            val_idx = [
                i for i, meta in enumerate(metas) if meta["id"] in val_set
            ]
            scorer, criterion = fit_probe(
                X[train_idx],
                labels[train_idx],
                [metas[i]["id"] for i in train_idx],
                X[val_idx],
                labels[val_idx],
                [metas[i]["id"] for i in val_idx],
                use_rank=True,
                epochs=250,
                rank_w=1.0,
                seed=seed + fold_index,
            )
            fold_internal[layer][fold_index] = criterion
            for index, score in zip(test_idx, scorer(X[test_idx])):
                per_layer[layer][index] = float(score)
    scores = [None] * len(metas)
    selected_layers = {}
    for fold_index, hold in enumerate(folds):
        layer = max(
            LAYERS,
            key=lambda value: fold_internal[value].get(
                fold_index, float("-inf")
            ),
        )
        selected_layers[fold_index] = layer
        hold_set = set(hold)
        for index, meta in enumerate(metas):
            if meta["id"] in hold_set:
                scores[index] = per_layer[layer][index]
    if any(score is None for score in scores):
        raise RuntimeError(f"incomplete structural OOF scores for {route}")
    pooled = auroc(scores, copy_labels)
    within, within_n = within_problem_auroc(
        scores, copy_labels, [meta["id"] for meta in metas]
    )
    return scores, {
        "selected_layer_per_fold": selected_layers,
        "internal_wp_per_fold": fold_internal,
        "oof_pooled_auroc": pooled,
        "oof_within_problem_auroc": within,
        "oof_within_problem_n": within_n,
    }


def problem_scores(metas, hidden_scores, state, explicit, hidden_penalty):
    by_problem = defaultdict(list)
    for index, meta in enumerate(metas):
        by_problem[meta["id"]].append((meta, float(hidden_scores[index])))
    outcomes = {}
    selected_norms = {}
    for pid, items in by_problem.items():
        raw = [score for _, score in items]
        score_mean = sum(raw) / len(raw)
        variance = sum((score - score_mean) ** 2 for score in raw) / max(
            1, len(raw) - 1
        )
        score_sd = math.sqrt(variance) if variance > 0 else 1.0
        grouped = defaultdict(
            lambda: {"count": 0, "mention": 0.0, "copy_z": 0.0}
        )
        order = []
        for meta, score in items:
            key = meta["norm"]
            if not key:
                continue
            if key not in grouped:
                order.append(key)
            grouped[key]["count"] += 1
            grouped[key]["mention"] += float(meta["mentions"]["correct"])
            grouped[key]["copy_z"] += (score - score_mean) / score_sd
        best = max(
            order,
            key=lambda key: (
                grouped[key]["count"] / len(items)
                + SUPPORT_WEIGHT
                * grouped[key]["mention"]
                / grouped[key]["count"]
                - (EXACT_GUARD_WEIGHT if explicit else 0.0)
                * float(key in state[pid]["values"])
                - hidden_penalty
                * grouped[key]["copy_z"]
                / grouped[key]["count"]
            ),
            default=None,
        )
        first = items[0][0]
        outcomes[pid] = bool(
            best and answers_match(best, first["gold"], first["aliases"])
        )
        selected_norms[pid] = best
    return outcomes, selected_norms


def nested_policy(metas, scores, state, folds, explicit):
    all_outcomes = {
        penalty: problem_scores(metas, scores, state, explicit, penalty)[0]
        for penalty in PENALTIES
    }
    selected = {}
    picked = {}
    for fold_index, hold in enumerate(folds):
        hold_set = set(hold)
        train = [pid for pid in all_outcomes[0.0] if pid not in hold_set]
        candidates = []
        for penalty in PENALTIES:
            accuracy = sum(
                all_outcomes[penalty][pid] for pid in train
            ) / len(train)
            candidates.append((accuracy, -penalty, penalty))
        _, _, best_penalty = max(candidates)
        picked[fold_index] = best_penalty
        for pid in hold:
            selected[pid] = all_outcomes[best_penalty][pid]
    curve = {
        str(penalty): sum(outcomes.values()) / len(outcomes)
        for penalty, outcomes in all_outcomes.items()
    }
    return selected, picked, curve


def metric(hits, baseline):
    ids = sorted(baseline)
    values = [hits[pid] for pid in ids]
    base = [baseline[pid] for pid in ids]
    return {
        "accuracy": sum(values) / len(values),
        "delta": sum(values) / len(values) - sum(base) / len(base),
        "paired": exact_mcnemar(values, base),
    }


def main(args):
    import torch

    payload = torch.load(args.features, map_location="cpu")
    metas = payload["metas"]
    rows = load_rows(args.data, args.limit, seed=0)
    state = dependency_state(rows)
    if {meta["id"] for meta in metas} != set(state):
        raise RuntimeError("feature IDs do not match the expected development slice")
    copy_labels = [
        int(bool(meta["norm"]) and meta["norm"] in state[meta["id"]]["values"])
        for meta in metas
    ]
    pids = sorted(state)
    rng = random.Random(args.seed)
    rng.shuffle(pids)
    folds = [pids[index::args.folds] for index in range(args.folds)]

    zero_scores = [0.0] * len(metas)
    lexical, _ = problem_scores(metas, zero_scores, state, False, 0.0)
    exact, _ = problem_scores(metas, zero_scores, state, True, 0.0)
    sc_metas = [
        dict(meta, mentions={"correct": 0, "wrong": 0}) for meta in metas
    ]
    sc8, _ = problem_scores(sc_metas, zero_scores, state, False, 0.0)

    readers = {}
    policies = {}
    policy_hits = {}
    for route in ROUTES:
        scores, readers[route] = structural_oof(
            payload,
            route,
            copy_labels,
            folds,
            args.seed + (0 if route == "correct" else 5000),
        )
        replace, replace_picked, replace_curve = nested_policy(
            metas, scores, state, folds, explicit=False
        )
        hybrid, hybrid_picked, hybrid_curve = nested_policy(
            metas, scores, state, folds, explicit=True
        )
        policy_hits[(route, "replace")] = replace
        policy_hits[(route, "hybrid")] = hybrid
        policies[route] = {
            "replace_exact_guard": {
                **metric(replace, sc8),
                "vs_exact_guard": metric(replace, exact),
                "penalty_per_fold": replace_picked,
                "curve": replace_curve,
            },
            "hybrid_with_exact_guard": {
                **metric(hybrid, sc8),
                "vs_exact_guard": metric(hybrid, exact),
                "penalty_per_fold": hybrid_picked,
                "curve": hybrid_curve,
            },
        }

    descriptive_best_route = max(
        ROUTES,
        key=lambda route: policies[route]["replace_exact_guard"]["accuracy"],
    )
    # Predecessor-context (the prior experiment's "wrong" route) is the
    # pre-registered primary arm.  The correct final-hop block is a control;
    # do not choose between routes using outcome accuracy.
    primary_route = "wrong"
    primary = policies[primary_route]["replace_exact_guard"]
    hybrid = policies[primary_route]["hybrid_with_exact_guard"]
    report = {
        "experiment": "HSGR structural predecessor-copy hidden reader",
        "claim_boundary": (
            "Development OOF diagnostic. Probe supervision is structural copy "
            "state, while policy penalty is selected on other OOF problems."
        ),
        "n_problems": len(pids),
        "n_candidates": len(metas),
        "copy_positive": sum(copy_labels),
        "sc8": sum(sc8.values()) / len(sc8),
        "lexical_guide": metric(lexical, sc8),
        "exact_copy_guard": metric(exact, sc8),
        "readers": readers,
        "policies": policies,
        "primary_route": primary_route,
        "descriptive_best_route": descriptive_best_route,
        "gates": {
            "structural_reader": (
                readers[primary_route]["oof_pooled_auroc"] >= 0.90
                and readers[primary_route]["oof_within_problem_auroc"] >= 0.80
            ),
            "hidden_replaces_guard": (
                primary["accuracy"] >= 0.565
                and primary["paired"]["p"] < 0.05
                and primary["vs_exact_guard"]["delta"] >= -0.01
            ),
            "hybrid_nonnegative": hybrid["vs_exact_guard"]["delta"] >= 0.0,
        },
    }
    report["decision"] = (
        "PASS" if all(report["gates"].values()) else "GATE FAIL"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "hsgr_copy_guard_probe_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="hsgr_copy_guard_probe")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    main(parser.parse_args())
