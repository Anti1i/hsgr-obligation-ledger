"""Skew-bilinear hidden interaction diagnostic for a directed HSGR edge.

The feature is an antisymmetric outer product of random-projected destination
and predecessor hidden states.  It changes sign when the Guide edge direction
is reversed and contains no destination-only or predecessor-only term.  The
same fitted reader is evaluated under sign reversal and within-problem route
mismatch controls.

This is mechanism development on an already observed held-out set.  It must
pass the frozen diagnostic gates before the final untouched split is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

from hsgr_copy_guard_probe import dependency_state, metric
from hsgr_dual_route_guide_diagnostic import normalize_within, select_policy
from hsgr_paired_transition_diagnostic import (
    FROZEN_TRANSITION_REFERENCE,
    eligible_diagnostics,
    paired_counts,
    rotation_indices,
    strict_direction_labels,
    tune_weight,
)
from hsgr_structured_hidden_verifier import LAYERS
from mh_e0 import load_rows
from mh_latent_rerank import fit_probe


SEED = 20260821
FOLDS = 5
PROJECTED_DIM = 32


def problem_id_sha(metas):
    ids = list(dict.fromkeys(meta["id"] for meta in metas))
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def projection(torch, input_dim, output_dim, seed):
    if output_dim > input_dim:
        raise ValueError("output_dim cannot exceed hidden feature dimension")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def skew_features(payload, layer, projected_dim, seed, wrong_indices=None):
    import torch

    correct = payload["features"]["correct"][layer].float()
    wrong = payload["features"]["wrong"][layer].float()
    if wrong_indices is not None:
        wrong = wrong[torch.tensor(wrong_indices, dtype=torch.long)]
    matrix = projection(
        torch, correct.shape[1], projected_dim, seed + 1009 * layer
    )
    correct = correct @ matrix
    wrong = wrong @ matrix
    correct = correct / (correct.norm(dim=1, keepdim=True) + 1e-8)
    wrong = wrong / (wrong.norm(dim=1, keepdim=True) + 1e-8)
    upper = torch.triu_indices(projected_dim, projected_dim, offset=1)
    skew = (
        correct.unsqueeze(2) * wrong.unsqueeze(1)
        - wrong.unsqueeze(2) * correct.unsqueeze(1)
    )
    features = skew[:, upper[0], upper[1]]
    return features / (features.norm(dim=1, keepdim=True) + 1e-8)


def augmented_fit(
    X,
    labels,
    metas,
    train_idx,
    val_idx,
    seed,
):
    import torch

    train_y = torch.tensor(
        [labels[index] for index in train_idx], dtype=torch.float32
    )
    val_y = torch.tensor(
        [labels[index] for index in val_idx], dtype=torch.float32
    )
    train_x = torch.cat([X[train_idx], -X[train_idx]], dim=0)
    val_x = torch.cat([X[val_idx], -X[val_idx]], dim=0)
    train_targets = torch.cat([train_y, 1.0 - train_y], dim=0)
    val_targets = torch.cat([val_y, 1.0 - val_y], dim=0)
    train_ids = [metas[index]["id"] for index in train_idx] * 2
    val_ids = [metas[index]["id"] for index in val_idx] * 2
    return fit_probe(
        train_x,
        train_targets,
        train_ids,
        val_x,
        val_targets,
        val_ids,
        use_rank=True,
        epochs=250,
        rank_w=1.0,
        seed=seed,
    )


def bilinear_oof_and_transfer(dev_payload, held_payload, seed, projected_dim):
    dev_metas = dev_payload["metas"]
    held_metas = held_payload["metas"]
    dev_labels = strict_direction_labels(dev_metas)
    held_labels = strict_direction_labels(held_metas)
    pids = sorted({meta["id"] for meta in dev_metas})
    random.Random(seed).shuffle(pids)
    folds = [pids[index::FOLDS] for index in range(FOLDS)]
    held_rotated = rotation_indices(held_metas)

    dev_features = {
        layer: skew_features(
            dev_payload, layer, projected_dim, seed
        )
        for layer in LAYERS
    }
    held_features = {
        layer: skew_features(
            held_payload, layer, projected_dim, seed
        )
        for layer in LAYERS
    }
    held_mismatch = {
        layer: skew_features(
            held_payload,
            layer,
            projected_dim,
            seed,
            wrong_indices=held_rotated,
        )
        for layer in LAYERS
    }

    oof = [None] * len(dev_metas)
    members = {name: [] for name in ("bilinear", "sign_swap", "mismatch")}
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
            scorer, criterion = augmented_fit(
                dev_features[layer],
                dev_labels,
                dev_metas,
                train_idx,
                val_idx,
                seed + 1000 * fold_index + layer,
            )
            candidates.append(
                (
                    criterion,
                    layer,
                    scorer,
                    scorer(dev_features[layer][test_idx]),
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
            ("bilinear", held_features[layer]),
            ("sign_swap", -held_features[layer]),
            ("mismatch", held_mismatch[layer]),
        ):
            members[name].append(
                normalize_within(scorer(matrix), held_metas)
            )

    if any(score is None for score in oof):
        raise RuntimeError("incomplete bilinear OOF predictions")
    transferred = {
        name: [
            sum(member[index] for member in group) / len(group)
            for index in range(len(held_metas))
        ]
        for name, group in members.items()
    }
    diagnostics = {
        "feature": {
            "type": "skew_symmetric_outer_product",
            "projected_dim": projected_dim,
            "feature_dim": projected_dim * (projected_dim - 1) // 2,
            "projection_seed": seed,
        },
        "selected_layers": selected,
        "dev": eligible_diagnostics(oof, dev_labels, dev_metas),
        "heldout": eligible_diagnostics(
            transferred["bilinear"], held_labels, held_metas
        ),
    }
    return oof, transferred, diagnostics


def main(args):
    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(args.threads)
    dev_payload = torch.load(args.dev_features, map_location="cpu")
    held_payload = torch.load(args.heldout_features, map_location="cpu")
    dev_metas = dev_payload["metas"]
    held_metas = held_payload["metas"]
    if len(dev_metas) != 1600 or len(held_metas) != 2560:
        raise RuntimeError("unexpected feature unit count")

    dev_rows = load_rows(args.data, 200, seed=0)
    dev_state = dependency_state(dev_rows)
    held_ids = {meta["id"] for meta in held_metas}
    all_rows = load_rows(args.data, 0, seed=0)
    held_rows = [row for row in all_rows if row["_uid"] in held_ids]
    held_state = dependency_state(held_rows)
    if set(held_state) != held_ids:
        raise RuntimeError("held-out feature IDs do not match source rows")

    dev_scores, held_scores, reader_report = bilinear_oof_and_transfer(
        dev_payload, held_payload, args.seed, args.projected_dim
    )
    weight, dev_curve = tune_weight(dev_metas, dev_state, dev_scores)
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
    signals = {
        "bilinear_transition": held_scores["bilinear"],
        "sign_swap_control": held_scores["sign_swap"],
        "mismatch_control": held_scores["mismatch"],
        "length_control": [len(meta["norm"] or "") for meta in held_metas],
    }
    weights = {
        "bilinear_transition": weight,
        "sign_swap_control": weight,
        "mismatch_control": weight,
        "length_control": length_weight,
    }
    outcomes = {}
    answers = {}
    for name, signal in signals.items():
        outcomes[name], answers[name] = select_policy(
            held_metas,
            held_state,
            signal,
            weights[name],
            routed=True,
            explicit=True,
        )

    report = {
        "experiment": "HSGR skew-bilinear directed-transition diagnostic",
        "claim_boundary": (
            "Mechanism development on an already observed second fresh set; "
            "not a confirmatory claim. Uses oracle decomposition, verified "
            "predecessor values, and gold support routing."
        ),
        "heldout_problem_id_sha256": problem_id_sha(held_metas),
        "n_problems": len(held_state),
        "n_candidates": len(held_metas),
        "reader": reader_report,
        "dev_selection": {
            "bilinear_weight": weight,
            "bilinear_curve": dev_curve,
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
    for name, values in outcomes.items():
        report["policy"][name] = {
            **metric(values, sc8),
            "vs_explicit": metric(values, explicit),
        }

    primary = report["policy"]["bilinear_transition"]
    vs_explicit = primary["vs_explicit"]
    fixes, breaks = paired_counts(vs_explicit)
    report["diagnostic_gates"] = {
        "reader_transfer": (
            reader_report["dev"]["pooled_auroc"] >= 0.70
            and reader_report["heldout"]["pooled_auroc"] >= 0.70
            and reader_report["dev"]["within_auroc"] >= 0.75
            and reader_report["heldout"]["within_auroc"] >= 0.75
        ),
        "nonzero_dev_weight": weight > 0.0,
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
    cases_path = os.path.join(args.out_dir, "bilinear_transition_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in sorted(held_state):
            primary_hit = outcomes["bilinear_transition"][pid]
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
                        "bilinear_transition": answers[
                            "bilinear_transition"
                        ][pid],
                        "bilinear_fixes_explicit": bool(
                            primary_hit and not explicit_hit
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report_path = os.path.join(
        args.out_dir, "bilinear_transition_diagnostic_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--heldout-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_bilinear_transition_diagnostic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--projected-dim", type=int, default=PROJECTED_DIM)
    parser.add_argument("--threads", type=int, default=16)
    main(parser.parse_args())

