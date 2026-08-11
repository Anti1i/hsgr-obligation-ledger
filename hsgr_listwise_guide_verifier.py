"""Guide-conditioned listwise hidden verifier on consumed feature sets.

Training uses the original n=200 development set plus the already consumed
first n=320 fresh set.  Problem-disjoint OOF predictions choose the safe hybrid
weight.  The already observed second n=320 fresh set is a development-stage
validation target, not a confirmatory claim.  The last untouched split remains
sealed unless every frozen gate passes.

The hidden model encodes, per layer, both the directed route margin and a
route-interaction term.  It is compared with an equal-training-procedure
non-hidden model, a hidden route swap, and a within-problem predecessor-state
mismatch.  Unlike prior structural probes, supervision is candidate outcome
correctness and the loss is explicitly listwise within each problem.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import Counter, defaultdict

from hsgr_copy_guard_probe import dependency_state, metric
from hsgr_dual_route_guide_diagnostic import (
    WEIGHTS,
    normalize_within,
    select_policy,
)
from hsgr_paired_transition_diagnostic import paired_counts, rotation_indices
from hsgr_structured_hidden_verifier import LAYERS
from mh_e0 import load_rows
from mh_latent_rerank import auroc, within_problem_auroc


SEED = 20260822
FOLDS = 5
EPOCHS = 100
ENCODER_WIDTH = 16
HEAD_WIDTH = 32
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
RANK_WEIGHT = 1.0


def load_feature_payload(torch, path):
    payload = torch.load(path, map_location="cpu")
    metas = payload.get("metas")
    features = payload.get("features")
    if not isinstance(metas, list) or not isinstance(features, dict):
        raise RuntimeError(f"invalid feature payload: {path}")
    if sorted(features) != ["correct", "wrong"]:
        raise RuntimeError(f"expected correct/wrong routes: {path}")
    for route in ("correct", "wrong"):
        if set(features[route]) != set(LAYERS):
            raise RuntimeError(f"unexpected layers for {route}: {path}")
        for layer in LAYERS:
            tensor = features[route][layer]
            if tensor.shape[0] != len(metas) or tensor.shape[1] != 256:
                raise RuntimeError(
                    f"unexpected {route}/L{layer} shape {tuple(tensor.shape)}: {path}"
                )
    return payload


def concat_payloads(torch, paths):
    payloads = [load_feature_payload(torch, path) for path in paths]
    seen = set()
    metas = []
    for path, payload in zip(paths, payloads):
        ids = {meta["id"] for meta in payload["metas"]}
        overlap = seen & ids
        if overlap:
            raise RuntimeError(
                f"feature training sets overlap at {len(overlap)} problems: {path}"
            )
        seen.update(ids)
        metas.extend(payload["metas"])
    features = {
        route: {
            layer: torch.cat(
                [payload["features"][route][layer] for payload in payloads],
                dim=0,
            )
            for layer in LAYERS
        }
        for route in ("correct", "wrong")
    }
    return {"metas": metas, "features": features}


def within_z(values, metas):
    grouped = defaultdict(list)
    for index, meta in enumerate(metas):
        grouped[meta["id"]].append(index)
    result = [0.0] * len(values)
    for indices in grouped.values():
        local = [float(values[index]) for index in indices]
        center = sum(local) / len(local)
        variance = sum((value - center) ** 2 for value in local) / max(
            1, len(local) - 1
        )
        scale = math.sqrt(variance) if variance > 0 else 1.0
        for index in indices:
            result[index] = (float(values[index]) - center) / scale
    return result


def make_data(torch, payload, state, hop_by_pid):
    metas = payload["metas"]
    counts = Counter((meta["id"], meta["norm"]) for meta in metas if meta["norm"])
    lengths = [len(meta["norm"] or "") for meta in metas]
    length_z = within_z(lengths, metas)
    scalar = []
    labels = []
    for index, meta in enumerate(metas):
        norm = meta["norm"]
        scalar.append(
            [
                counts[(meta["id"], norm)] / 8.0 if norm else 0.0,
                float(meta["mentions"]["correct"]),
                float(meta["mentions"]["wrong"]),
                float(bool(norm) and norm in state[meta["id"]]["values"]),
                float(length_z[index]),
            ]
        )
        labels.append(float(meta["label"]))
    return {
        "metas": metas,
        "pids": [meta["id"] for meta in metas],
        "hops": torch.tensor(
            [hop_by_pid[meta["id"]] for meta in metas], dtype=torch.long
        ),
        "scalar": torch.tensor(scalar, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "features": {
            route: {
                layer: payload["features"][route][layer].float()
                for layer in LAYERS
            }
            for route in ("correct", "wrong")
        },
        "rotation": torch.tensor(rotation_indices(metas), dtype=torch.long),
    }


def make_model(torch, hidden):
    nn = torch.nn

    class CandidateVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = bool(hidden)
            if self.hidden:
                self.encoders = nn.ModuleDict(
                    {
                        str(layer): nn.Sequential(
                            nn.Linear(512, ENCODER_WIDTH),
                            nn.GELU(),
                            nn.LayerNorm(ENCODER_WIDTH),
                        )
                        for layer in LAYERS
                    }
                )
                input_dim = len(LAYERS) * ENCODER_WIDTH + 5
            else:
                input_dim = 5
            self.head = nn.Sequential(
                nn.Linear(input_dim, HEAD_WIDTH),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(HEAD_WIDTH, 1),
            )

        def forward(self, data, route_mode="normal"):
            parts = [data["scalar"]]
            if self.hidden:
                for layer in LAYERS:
                    correct = data["features"]["correct"][layer]
                    wrong = data["features"]["wrong"][layer]
                    if route_mode == "swap":
                        correct, wrong = wrong, correct
                    elif route_mode == "mismatch":
                        wrong = wrong[data["rotation"]]
                    elif route_mode != "normal":
                        raise ValueError(route_mode)
                    pair = torch.cat(
                        [correct - wrong, correct * wrong], dim=1
                    )
                    parts.append(self.encoders[str(layer)](pair))
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    return CandidateVerifier()


def indices_for_pids(data, allowed):
    allowed = set(allowed)
    return [index for index, pid in enumerate(data["pids"]) if pid in allowed]


def rank_pairs(data, indices):
    grouped = defaultdict(lambda: {"pos": [], "neg": []})
    for index in indices:
        bucket = "pos" if data["labels"][index].item() > 0.5 else "neg"
        grouped[data["pids"][index]][bucket].append(index)
    positive, negative = [], []
    for values in grouped.values():
        for pos in values["pos"]:
            for neg in values["neg"]:
                positive.append(pos)
                negative.append(neg)
    return positive, negative


def pair_accuracy(scores, data, indices, hop_balanced=False):
    positive, negative = rank_pairs(data, indices)
    if not positive:
        return 0.5
    correct = [
        float(float(scores[pos]) > float(scores[neg]))
        for pos, neg in zip(positive, negative)
    ]
    if not hop_balanced:
        return sum(correct) / len(correct)
    grouped = defaultdict(list)
    for value, pos in zip(correct, positive):
        grouped[int(data["hops"][pos].item())].append(value)
    return sum(sum(values) / len(values) for values in grouped.values()) / len(
        grouped
    )


def mean_loss_by_hop(torch, losses, hops):
    """Give every observed hop stratum equal mass in an already scalar loss."""
    return torch.stack(
        [losses[hops == hop].mean() for hop in torch.unique(hops)]
    ).mean()


def hop_class_balanced_bce(torch, logits, labels, hops):
    """Give every (hop, correctness) stratum equal mass when it is present."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    strata = []
    for hop in torch.unique(hops):
        for label in (0.0, 1.0):
            mask = (hops == hop) & (labels == label)
            if bool(mask.any()):
                strata.append(losses[mask].mean())
    return torch.stack(strata).mean()


def fit_model(
    torch, data, train_idx, val_idx, hidden, seed, hop_balanced=False
):
    torch.manual_seed(seed)
    model = make_model(torch, hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    train_pos, train_neg = rank_pairs(data, train_idx)
    pos_tensor = torch.tensor(train_pos, dtype=torch.long)
    neg_tensor = torch.tensor(train_neg, dtype=torch.long)
    y_train = data["labels"][train_tensor]
    train_hops = data["hops"][train_tensor]
    positives = float(y_train.sum().item())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor(
        negatives / max(1.0, positives), dtype=torch.float32
    )
    best = None
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        if hop_balanced:
            bce = hop_class_balanced_bce(
                torch, logits[train_tensor], y_train, train_hops
            )
        else:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[train_tensor], y_train, pos_weight=pos_weight
            )
        rank_losses = torch.nn.functional.softplus(
            -(logits[pos_tensor] - logits[neg_tensor])
        )
        rank_loss = (
            mean_loss_by_hop(torch, rank_losses, data["hops"][pos_tensor])
            if hop_balanced
            else rank_losses.mean()
        )
        loss = bce + RANK_WEIGHT * rank_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                values = model(data).tolist()
            val_pair = pair_accuracy(
                values, data, val_idx, hop_balanced=hop_balanced
            )
            val_pos, val_neg = rank_pairs(data, val_idx)
            if val_pos:
                vp = torch.tensor(val_pos, dtype=torch.long)
                vn = torch.tensor(val_neg, dtype=torch.long)
                with torch.no_grad():
                    val_rank_losses = torch.nn.functional.softplus(
                        -(model(data)[vp] - model(data)[vn])
                    )
                    if hop_balanced:
                        val_rank_losses = mean_loss_by_hop(
                            torch, val_rank_losses, data["hops"][vp]
                        )
                    else:
                        val_rank_losses = val_rank_losses.mean()
                    val_rank = float(val_rank_losses.item())
            else:
                val_rank = float("inf")
            row = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "validation_pair_accuracy": val_pair,
                "validation_rank_loss": val_rank,
            }
            history.append(row)
            key = (val_pair, -val_rank, -epoch)
            if best is None or key > best:
                best = key
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("model training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selected = max(
        history,
        key=lambda row: (
            row["validation_pair_accuracy"],
            -row["validation_rank_loss"],
            -row["epoch"],
        ),
    )
    return model, {"selected": selected, "history": history}


def cv_scores(
    torch, train_data, held_data, hidden, seed, hop_balanced=False
):
    pids = sorted(set(train_data["pids"]))
    random.Random(seed).shuffle(pids)
    folds = [pids[index::FOLDS] for index in range(FOLDS)]
    oof = [None] * len(train_data["metas"])
    held_members = defaultdict(list)
    fold_report = []
    for fold_index, hold in enumerate(folds):
        hold_set = set(hold)
        available = [pid for pid in pids if pid not in hold_set]
        random.Random(seed + 101 * fold_index).shuffle(available)
        n_val = max(1, len(available) // 7)
        val_pids = set(available[:n_val])
        fit_pids = set(available[n_val:])
        train_idx = indices_for_pids(train_data, fit_pids)
        val_idx = indices_for_pids(train_data, val_pids)
        test_idx = indices_for_pids(train_data, hold_set)
        model, diagnostics = fit_model(
            torch,
            train_data,
            train_idx,
            val_idx,
            hidden,
            seed + 1000 * fold_index,
            hop_balanced=hop_balanced,
        )
        with torch.no_grad():
            train_scores = model(train_data).tolist()
        local = normalize_within(
            [train_scores[index] for index in test_idx],
            [train_data["metas"][index] for index in test_idx],
        )
        for index, score in zip(test_idx, local):
            oof[index] = score
        modes = ("normal", "swap", "mismatch") if hidden else ("normal",)
        for mode in modes:
            with torch.no_grad():
                scores = model(held_data, route_mode=mode).tolist()
            held_members[mode].append(
                normalize_within(scores, held_data["metas"])
            )
        fold_report.append(
            {
                "fold": fold_index,
                "outer_problems": len(hold_set),
                "train_problems": len(fit_pids),
                "validation_problems": len(val_pids),
                **diagnostics["selected"],
            }
        )
        print(
            f"[cv {'hidden' if hidden else 'nonhidden'}] "
            + json.dumps(fold_report[-1], sort_keys=True),
            flush=True,
        )
    if any(score is None for score in oof):
        raise RuntimeError("incomplete OOF predictions")
    held = {
        mode: [
            sum(member[index] for member in members) / len(members)
            for index in range(len(held_data["metas"]))
        ]
        for mode, members in held_members.items()
    }
    return oof, held, fold_report


def candidate_diagnostics(scores, data):
    values = [float(value) for value in scores]
    labels = [int(value.item()) for value in data["labels"]]
    within, within_n = within_problem_auroc(values, labels, data["pids"])
    return {
        "pooled_auroc": auroc(values, labels),
        "within_auroc": within,
        "within_n": within_n,
        "positives": sum(labels),
        "n": len(labels),
    }


def tune_policy_weight(metas, state, signal):
    candidates = []
    curve = {}
    for weight in WEIGHTS:
        outcomes, _ = select_policy(
            metas, state, signal, weight, routed=True, explicit=True
        )
        accuracy = sum(outcomes.values()) / len(outcomes)
        curve[str(weight)] = accuracy
        candidates.append((accuracy, -weight, weight))
    _, _, weight = max(candidates)
    return weight, curve


def tune_policy_weight_by_hop(metas, state, signal, hop_by_pid):
    """Choose Guide strength per depth using training OOF predictions only."""
    weights = {}
    curves = {}
    for hop in sorted(set(hop_by_pid[meta["id"]] for meta in metas)):
        indices = [
            index
            for index, meta in enumerate(metas)
            if hop_by_pid[meta["id"]] == hop
        ]
        local_metas = [metas[index] for index in indices]
        local_signal = [signal[index] for index in indices]
        weights[hop], curves[str(hop)] = tune_policy_weight(
            local_metas, state, local_signal
        )
    return weights, curves


def select_policy_with_weights(
    metas,
    state,
    signal,
    weights,
    hop_by_pid,
    routed=True,
    explicit=True,
):
    if not isinstance(weights, dict):
        return select_policy(
            metas,
            state,
            signal,
            weights,
            routed=routed,
            explicit=explicit,
        )
    outcomes = {}
    answers = {}
    for hop, weight in sorted(weights.items()):
        indices = [
            index
            for index, meta in enumerate(metas)
            if hop_by_pid[meta["id"]] == hop
        ]
        local_metas = [metas[index] for index in indices]
        local_signal = (
            None if signal is None else [signal[index] for index in indices]
        )
        local_outcomes, local_answers = select_policy(
            local_metas,
            state,
            local_signal,
            weight,
            routed=routed,
            explicit=explicit,
        )
        outcomes.update(local_outcomes)
        answers.update(local_answers)
    expected = {meta["id"] for meta in metas}
    if set(outcomes) != expected:
        raise RuntimeError("hop-conditioned policy did not cover every problem")
    return outcomes, answers


def by_hop(outcomes, baselines, rows_by_id):
    grouped = defaultdict(list)
    for pid in baselines:
        grouped[str(len(rows_by_id[pid]["question_decomposition"]))].append(pid)
    result = {}
    for hop, ids in grouped.items():
        result[hop] = {
            "n": len(ids),
            "sc8": sum(baselines[pid] for pid in ids) / len(ids),
            **{
                name: sum(values[pid] for pid in ids) / len(ids)
                for name, values in outcomes.items()
            },
        }
    return result


def main(args):
    import torch

    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    train_payload = concat_payloads(
        torch, [args.dev_features, args.first_features]
    )
    held_payload = load_feature_payload(torch, args.second_features)
    if len(train_payload["metas"]) != 4160:
        raise RuntimeError(
            f"expected 4160 training candidates, got {len(train_payload['metas'])}"
        )
    if len(held_payload["metas"]) != 2560:
        raise RuntimeError("expected 2560 second-set candidates")

    all_rows = load_rows(args.data, 0, seed=0)
    rows_by_id = {row["_uid"]: row for row in all_rows}
    train_ids = {meta["id"] for meta in train_payload["metas"]}
    held_ids = {meta["id"] for meta in held_payload["metas"]}
    if train_ids & held_ids:
        raise RuntimeError("training and second validation IDs overlap")
    train_rows = [rows_by_id[pid] for pid in train_ids]
    held_rows = [rows_by_id[pid] for pid in held_ids]
    train_state = dependency_state(train_rows)
    held_state = dependency_state(held_rows)
    hop_by_pid = {
        pid: len(rows_by_id[pid]["question_decomposition"])
        for pid in train_ids | held_ids
    }
    train_data = make_data(torch, train_payload, train_state, hop_by_pid)
    held_data = make_data(torch, held_payload, held_state, hop_by_pid)

    hidden_oof, hidden_held, hidden_folds = cv_scores(
        torch,
        train_data,
        held_data,
        True,
        args.seed,
        hop_balanced=args.hop_balanced,
    )
    nonhidden_oof, nonhidden_held, nonhidden_folds = cv_scores(
        torch,
        train_data,
        held_data,
        False,
        args.seed + 10000,
        hop_balanced=args.hop_balanced,
    )
    if args.hop_policy:
        hidden_weight, hidden_curve = tune_policy_weight_by_hop(
            train_data["metas"], train_state, hidden_oof, hop_by_pid
        )
        nonhidden_weight, nonhidden_curve = tune_policy_weight_by_hop(
            train_data["metas"], train_state, nonhidden_oof, hop_by_pid
        )
        length_weight, length_curve = tune_policy_weight_by_hop(
            train_data["metas"],
            train_state,
            [len(meta["norm"] or "") for meta in train_data["metas"]],
            hop_by_pid,
        )
    else:
        hidden_weight, hidden_curve = tune_policy_weight(
            train_data["metas"], train_state, hidden_oof
        )
        nonhidden_weight, nonhidden_curve = tune_policy_weight(
            train_data["metas"], train_state, nonhidden_oof
        )
        length_weight, length_curve = tune_policy_weight(
            train_data["metas"],
            train_state,
            [len(meta["norm"] or "") for meta in train_data["metas"]],
        )

    train_sc, _ = select_policy(
        train_data["metas"], train_state, None, 0.0, routed=False, explicit=False
    )
    train_explicit, _ = select_policy(
        train_data["metas"], train_state, None, 0.0, routed=True, explicit=True
    )
    train_hidden, _ = select_policy_with_weights(
        train_data["metas"],
        train_state,
        hidden_oof,
        hidden_weight,
        hop_by_pid,
        routed=True,
        explicit=True,
    )
    train_nonhidden, _ = select_policy_with_weights(
        train_data["metas"],
        train_state,
        nonhidden_oof,
        nonhidden_weight,
        hop_by_pid,
        routed=True,
        explicit=True,
    )

    sc8, _ = select_policy(
        held_data["metas"], held_state, None, 0.0, routed=False, explicit=False
    )
    explicit, explicit_answers = select_policy(
        held_data["metas"], held_state, None, 0.0, routed=True, explicit=True
    )
    signals = {
        "hidden_listwise": hidden_held["normal"],
        "nonhidden_control": nonhidden_held["normal"],
        "route_swap_control": hidden_held["swap"],
        "route_mismatch_control": hidden_held["mismatch"],
        "length_control": [
            len(meta["norm"] or "") for meta in held_data["metas"]
        ],
    }
    weights = {
        "hidden_listwise": hidden_weight,
        "nonhidden_control": nonhidden_weight,
        "route_swap_control": hidden_weight,
        "route_mismatch_control": hidden_weight,
        "length_control": length_weight,
    }
    outcomes = {}
    answers = {}
    for name, signal in signals.items():
        outcomes[name], answers[name] = select_policy_with_weights(
            held_data["metas"],
            held_state,
            signal,
            weights[name],
            hop_by_pid,
            routed=True,
            explicit=True,
        )

    report = {
        "experiment": "HSGR Guide-conditioned listwise hidden verifier"
        + (" with hop-balanced loss" if args.hop_balanced else "")
        + (" and OOF depth-calibrated Guide policy" if args.hop_policy else ""),
        "claim_boundary": (
            "Outcome-supervised verifier trained on consumed n=200+n=320 "
            "problems and evaluated for development on an already observed "
            "second n=320 set. Uses oracle decomposition, verified predecessor "
            "values, and gold support routing; not an end-to-end or fresh claim."
        ),
        "data": {
            "train_problems": len(train_ids),
            "train_candidates": len(train_data["metas"]),
            "second_problems": len(held_ids),
            "second_candidates": len(held_data["metas"]),
            "overlap": len(train_ids & held_ids),
        },
        "architecture": {
            "layers": list(LAYERS),
            "per_layer_input": "[correct-wrong, correct*wrong]",
            "encoder_width": ENCODER_WIDTH,
            "head_width": HEAD_WIDTH,
            "epochs": EPOCHS,
            "loss": (
                "equal (hop, class) BCE strata + equal-hop within-problem "
                "pairwise logistic"
                if args.hop_balanced
                else "class-balanced BCE + within-problem pairwise logistic"
            ),
            "hop_balanced": bool(args.hop_balanced),
            "hop_policy": bool(args.hop_policy),
            "training_problem_hops": dict(
                sorted(Counter(hop_by_pid[pid] for pid in train_ids).items())
            ),
        },
        "training_oof": {
            "hidden_reader": candidate_diagnostics(hidden_oof, train_data),
            "nonhidden_reader": candidate_diagnostics(
                nonhidden_oof, train_data
            ),
            "hidden_folds": hidden_folds,
            "nonhidden_folds": nonhidden_folds,
            "weights": {
                "hidden": hidden_weight,
                "nonhidden": nonhidden_weight,
                "length": length_weight,
            },
            "curves": {
                "hidden": hidden_curve,
                "nonhidden": nonhidden_curve,
                "length": length_curve,
            },
            "baseline_sc8": sum(train_sc.values()) / len(train_sc),
            "explicit": metric(train_explicit, train_sc),
            "hidden_policy": {
                **metric(train_hidden, train_sc),
                "vs_explicit": metric(train_hidden, train_explicit),
            },
            "nonhidden_policy": {
                **metric(train_nonhidden, train_sc),
                "vs_explicit": metric(train_nonhidden, train_explicit),
            },
        },
        "second_validation": {
            "hidden_reader": candidate_diagnostics(
                hidden_held["normal"], held_data
            ),
            "nonhidden_reader": candidate_diagnostics(
                nonhidden_held["normal"], held_data
            ),
            "baseline": {
                "sc8": sum(sc8.values()) / len(sc8),
                "explicit": metric(explicit, sc8),
            },
            "policy": {},
        },
    }
    for name, values in outcomes.items():
        report["second_validation"]["policy"][name] = {
            **metric(values, sc8),
            "vs_explicit": metric(values, explicit),
        }

    hop_report = by_hop(outcomes, sc8, rows_by_id)
    report["second_validation"]["by_hop"] = hop_report
    primary = report["second_validation"]["policy"]["hidden_listwise"]
    vs_explicit = primary["vs_explicit"]
    fixes, breaks = paired_counts(vs_explicit)
    oof_hidden = report["training_oof"]["hidden_policy"]
    oof_nonhidden = report["training_oof"]["nonhidden_policy"]
    depth_delta = {
        hop: values["hidden_listwise"] - values["sc8"]
        for hop, values in hop_report.items()
    }
    report["second_validation"]["depth_delta"] = depth_delta
    report["gates"] = {
        "oof_value": (
            oof_hidden["vs_explicit"]["delta"] >= 0.01
            and oof_hidden["accuracy"] >= oof_nonhidden["accuracy"] + 0.005
        ),
        "reader_transfer": (
            report["training_oof"]["hidden_reader"]["within_auroc"] >= 0.75
            and report["second_validation"]["hidden_reader"]["within_auroc"]
            >= 0.75
        ),
        "headroom": primary["delta"] >= 0.06 and primary["paired"]["p"] < 0.05,
        "safe_net_vs_explicit": (
            vs_explicit["delta"] >= 0.02
            and breaks <= 8
            and fixes >= breaks + 4
        ),
        "hidden_contribution": (
            primary["accuracy"]
            >= report["second_validation"]["policy"]["nonhidden_control"][
                "accuracy"
            ]
            + 0.01
        ),
        "route_directionality": (
            primary["accuracy"]
            >= report["second_validation"]["policy"]["route_swap_control"][
                "accuracy"
            ]
            + 0.02
        ),
        "route_coupling": (
            primary["accuracy"]
            >= report["second_validation"]["policy"][
                "route_mismatch_control"
            ]["accuracy"]
            + 0.01
        ),
        "depth_signature": (
            min(depth_delta.values()) >= 0.0
            and depth_delta["4"] >= depth_delta["2"] - 0.01
        ),
    }
    report["decision"] = (
        "WORTH FINAL HOLDOUT"
        if all(report["gates"].values())
        else "DO NOT CONSUME FINAL HOLDOUT"
    )

    cases_path = os.path.join(args.out_dir, "listwise_guide_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in sorted(held_ids):
            primary_hit = outcomes["hidden_listwise"][pid]
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
                        "hidden_listwise": answers["hidden_listwise"][pid],
                        "hidden_fixes_explicit": bool(
                            primary_hit and not explicit_hit
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report_path = os.path.join(args.out_dir, "listwise_guide_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--first-features", required=True)
    parser.add_argument("--second-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_listwise_guide_verifier")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hop-balanced", action="store_true")
    parser.add_argument("--hop-policy", action="store_true")
    main(parser.parse_args())
