"""Paired hidden-state diagnostic for a directed HSGR Guide transition.

For every SC candidate, the HSGR Guide supplies a destination-support context
and a length-matched predecessor context.  A single structural reader is fit
on their hidden-state difference, rather than subtracting independently fit
support and copy readers.  Supervision uses only strict route preference:
destination mention without predecessor mention versus the reverse.

The second fresh set has already been observed.  It is used here only for
mechanism development.  A positive decision means the method is worth freezing
for the last untouched split; it is not a confirmatory claim.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from hsgr_copy_guard_probe import dependency_state, metric
from hsgr_dual_route_guide_diagnostic import (
    WEIGHTS,
    normalize_within,
    select_policy,
)
from hsgr_structured_hidden_verifier import LAYERS
from mh_e0 import load_rows
from mh_latent_rerank import auroc, fit_probe, within_problem_auroc


SEED = 20260820
FOLDS = 5
FROZEN_TRANSITION_REFERENCE = 0.509375


def strict_direction_labels(metas):
    """Return 1 for destination-only, 0 for predecessor-only, else None."""
    labels = []
    for meta in metas:
        correct = int(meta["mentions"]["correct"])
        wrong = int(meta["mentions"]["wrong"])
        labels.append(1 if correct > wrong else 0 if wrong > correct else None)
    return labels


def rotation_indices(metas):
    """Rotate predecessor states within each problem, preserving marginals."""
    grouped = defaultdict(list)
    for index, meta in enumerate(metas):
        grouped[meta["id"]].append(index)
    rotated = list(range(len(metas)))
    for indices in grouped.values():
        shifted = indices[1:] + indices[:1]
        for target, source in zip(indices, shifted):
            rotated[target] = source
    return rotated


def paired_matrix(payload, layer, wrong_indices=None, sign=1.0):
    import torch

    correct = payload["features"]["correct"][layer].float()
    wrong = payload["features"]["wrong"][layer].float()
    if wrong_indices is not None:
        wrong = wrong[torch.tensor(wrong_indices, dtype=torch.long)]
    matrix = (correct - wrong) * float(sign)
    return matrix / (matrix.norm(dim=1, keepdim=True) + 1e-8)


def eligible_diagnostics(scores, labels, metas):
    indices = [index for index, label in enumerate(labels) if label is not None]
    values = [scores[index] for index in indices]
    targets = [labels[index] for index in indices]
    pids = [metas[index]["id"] for index in indices]
    within, within_n = within_problem_auroc(values, targets, pids)
    return {
        "eligible": len(indices),
        "positives": sum(targets),
        "negatives": len(targets) - sum(targets),
        "pooled_auroc": auroc(values, targets),
        "within_auroc": within,
        "within_n": within_n,
    }


def paired_oof_and_transfer(dev_payload, held_payload, seed):
    import torch

    dev_metas = dev_payload["metas"]
    held_metas = held_payload["metas"]
    dev_labels = strict_direction_labels(dev_metas)
    held_labels = strict_direction_labels(held_metas)
    pids = sorted({meta["id"] for meta in dev_metas})
    random.Random(seed).shuffle(pids)
    folds = [pids[index::FOLDS] for index in range(FOLDS)]
    held_rotated = rotation_indices(held_metas)

    dev_matrices = {
        layer: paired_matrix(dev_payload, layer) for layer in LAYERS
    }
    held_matrices = {
        layer: paired_matrix(held_payload, layer) for layer in LAYERS
    }
    held_swapped = {
        layer: paired_matrix(held_payload, layer, sign=-1.0)
        for layer in LAYERS
    }
    held_mismatched = {
        layer: paired_matrix(
            held_payload, layer, wrong_indices=held_rotated
        )
        for layer in LAYERS
    }

    oof = [None] * len(dev_metas)
    held_members = {name: [] for name in ("paired", "sign_swap", "mismatch")}
    selected = {}

    for fold_index, hold in enumerate(folds):
        hold_set = set(hold)
        train_pids = [pid for pid in pids if pid not in hold_set]
        shuffled = list(train_pids)
        random.Random(seed + 101 * fold_index).shuffle(shuffled)
        n_val = max(1, len(shuffled) // 7)
        val_set = set(shuffled[:n_val])
        train_idx = [
            index
            for index, meta in enumerate(dev_metas)
            if meta["id"] not in hold_set
            and meta["id"] not in val_set
            and dev_labels[index] is not None
        ]
        val_idx = [
            index
            for index, meta in enumerate(dev_metas)
            if meta["id"] in val_set and dev_labels[index] is not None
        ]
        test_idx = [
            index
            for index, meta in enumerate(dev_metas)
            if meta["id"] in hold_set
        ]
        candidates = []
        for layer in LAYERS:
            labels = torch.tensor(
                [dev_labels[index] for index in train_idx],
                dtype=torch.float32,
            )
            val_labels = torch.tensor(
                [dev_labels[index] for index in val_idx],
                dtype=torch.float32,
            )
            scorer, criterion = fit_probe(
                dev_matrices[layer][train_idx],
                labels,
                [dev_metas[index]["id"] for index in train_idx],
                dev_matrices[layer][val_idx],
                val_labels,
                [dev_metas[index]["id"] for index in val_idx],
                use_rank=True,
                epochs=250,
                rank_w=1.0,
                seed=seed + 1000 * fold_index + layer,
            )
            candidates.append(
                (
                    criterion,
                    layer,
                    scorer,
                    scorer(dev_matrices[layer][test_idx]),
                )
            )
        criterion, layer, scorer, test_scores = max(
            candidates, key=lambda item: item[0]
        )
        selected[str(fold_index)] = {
            "layer": layer,
            "internal_criterion": criterion,
            "train_eligible": len(train_idx),
            "validation_eligible": len(val_idx),
        }
        test_metas = [dev_metas[index] for index in test_idx]
        for index, score in zip(
            test_idx, normalize_within(test_scores, test_metas)
        ):
            oof[index] = score
        for name, matrix in (
            ("paired", held_matrices[layer]),
            ("sign_swap", held_swapped[layer]),
            ("mismatch", held_mismatched[layer]),
        ):
            held_members[name].append(
                normalize_within(scorer(matrix), held_metas)
            )

    if any(score is None for score in oof):
        raise RuntimeError("incomplete paired OOF predictions")
    transferred = {
        name: [
            sum(member[index] for member in members) / len(members)
            for index in range(len(held_metas))
        ]
        for name, members in held_members.items()
    }
    diagnostics = {
        "selected_layers": selected,
        "dev": eligible_diagnostics(oof, dev_labels, dev_metas),
        "heldout": eligible_diagnostics(
            transferred["paired"], held_labels, held_metas
        ),
    }
    return oof, transferred, diagnostics


def tune_weight(metas, state, signal):
    curve = {}
    candidates = []
    for weight in WEIGHTS:
        outcomes, _ = select_policy(
            metas, state, signal, weight, routed=True, explicit=True
        )
        accuracy = sum(outcomes.values()) / len(outcomes)
        curve[str(weight)] = accuracy
        candidates.append((accuracy, -weight, weight))
    _, _, selected = max(candidates)
    return selected, curve


def paired_counts(metric_value):
    paired = metric_value["paired"]
    return paired["a_only"], paired["b_only"]


def main(args):
    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    dev_payload = torch.load(args.dev_features, map_location="cpu")
    held_payload = torch.load(args.heldout_features, map_location="cpu")
    dev_metas = dev_payload["metas"]
    held_metas = held_payload["metas"]
    if len(dev_metas) != 1600:
        raise RuntimeError(f"expected 1600 development units, got {len(dev_metas)}")
    if len(held_metas) != 2560:
        raise RuntimeError(f"expected 2560 held-out units, got {len(held_metas)}")

    dev_rows = load_rows(args.data, 200, seed=0)
    dev_state = dependency_state(dev_rows)
    held_ids = {meta["id"] for meta in held_metas}
    all_rows = load_rows(args.data, 0, seed=0)
    held_rows = [row for row in all_rows if row["_uid"] in held_ids]
    held_state = dependency_state(held_rows)
    if set(held_state) != held_ids:
        raise RuntimeError("held-out feature IDs do not match source rows")

    dev_scores, held_scores, reader_report = paired_oof_and_transfer(
        dev_payload, held_payload, args.seed
    )
    pair_weight, dev_curve = tune_weight(dev_metas, dev_state, dev_scores)
    length_weight, length_curve = tune_weight(
        dev_metas,
        dev_state,
        [len(meta["norm"] or "") for meta in dev_metas],
    )

    sc8, _ = select_policy(
        held_metas, held_state, None, 0.0, routed=False, explicit=False
    )
    explicit, explicit_answers = select_policy(
        held_metas, held_state, None, 0.0, routed=True, explicit=True
    )
    policies = {}
    answers = {}
    signals = {
        "paired_transition": held_scores["paired"],
        "sign_swap_control": held_scores["sign_swap"],
        "mismatch_control": held_scores["mismatch"],
        "length_control": [len(meta["norm"] or "") for meta in held_metas],
    }
    weights = {
        "paired_transition": pair_weight,
        "sign_swap_control": pair_weight,
        "mismatch_control": pair_weight,
        "length_control": length_weight,
    }
    for name, signal in signals.items():
        policies[name], answers[name] = select_policy(
            held_metas,
            held_state,
            signal,
            weights[name],
            routed=True,
            explicit=True,
        )

    report = {
        "experiment": "HSGR paired directed-transition hidden diagnostic",
        "claim_boundary": (
            "Mechanism development on an already observed second fresh set; "
            "not a confirmatory held-out claim. Uses oracle decomposition, "
            "verified predecessor values, and gold support routing."
        ),
        "n_problems": len(held_state),
        "n_candidates": len(held_metas),
        "reader": reader_report,
        "dev_selection": {
            "paired_weight": pair_weight,
            "paired_curve": dev_curve,
            "length_weight": length_weight,
            "length_curve": length_curve,
        },
        "baseline": {
            "sc8": sum(sc8.values()) / len(sc8),
            "explicit": metric(explicit, sc8),
            "frozen_transition_reference": FROZEN_TRANSITION_REFERENCE,
        },
        "policy": {},
    }
    for name, outcomes in policies.items():
        report["policy"][name] = {
            **metric(outcomes, sc8),
            "vs_explicit": metric(outcomes, explicit),
        }

    primary = report["policy"]["paired_transition"]
    vs_explicit = primary["vs_explicit"]
    fixes, breaks = paired_counts(vs_explicit)
    report["diagnostic_gates"] = {
        "reader_transfer": (
            reader_report["dev"]["within_auroc"] >= 0.75
            and reader_report["heldout"]["within_auroc"] >= 0.75
        ),
        "nonzero_dev_weight": pair_weight > 0.0,
        "headroom": primary["delta"] >= 0.06 and primary["paired"]["p"] < 0.05,
        "safe_net_vs_explicit": (
            vs_explicit["delta"] >= 0.01
            and breaks <= 8
            and fixes >= breaks + 4
        ),
        "beyond_length": (
            primary["accuracy"]
            >= report["policy"]["length_control"]["accuracy"] + 0.01
        ),
        "route_coupling": (
            primary["accuracy"]
            >= report["policy"]["mismatch_control"]["accuracy"] + 0.01
        ),
        "directionality": (
            primary["accuracy"]
            >= report["policy"]["sign_swap_control"]["accuracy"] + 0.02
        ),
        "matches_frozen_transition": (
            primary["accuracy"] >= FROZEN_TRANSITION_REFERENCE
        ),
    }
    report["decision"] = (
        "WORTH FINAL HOLDOUT"
        if all(report["diagnostic_gates"].values())
        else "DO NOT CONSUME FINAL HOLDOUT"
    )

    rows_by_id = {row["_uid"]: row for row in held_rows}
    cases_path = os.path.join(args.out_dir, "paired_transition_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in sorted(held_state):
            primary_hit = policies["paired_transition"][pid]
            explicit_hit = explicit[pid]
            if primary_hit == explicit_hit:
                continue
            row = rows_by_id[pid]
            handle.write(
                json.dumps(
                    {
                        "id": pid,
                        "hops": len(row["question_decomposition"]),
                        "question": row["question"],
                        "gold": row["answer"],
                        "explicit": explicit_answers[pid],
                        "paired_transition": answers["paired_transition"][pid],
                        "paired_fixes_explicit": bool(
                            primary_hit and not explicit_hit
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report_path = os.path.join(
        args.out_dir, "paired_transition_diagnostic_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--heldout-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_paired_transition_diagnostic")
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args())

