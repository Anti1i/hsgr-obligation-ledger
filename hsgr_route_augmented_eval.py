"""Nested-OOF test of an absolute-hidden verifier plus route residual."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import defaultdict

from hsgr_route_counterfactual_eval import (
    EPOCHS,
    HEAD_WIDTH,
    LEARNING_RATE,
    RANK_WEIGHT,
    SEED,
    WEIGHT_DECAY,
    by_hop,
    hash_half,
    holm_adjust,
    indices_for_pids,
    make_data,
    metric,
    outer_splits,
    pair_accuracy,
    rank_pairs,
    select_policy,
    tune_explicit,
    tune_weight,
    validate_payload,
)
from hsgr_structured_hidden_verifier import LAYERS


KINDS = ("route_augmented", "ordinary_wide", "activation_wide")
ABS_WIDTH = 16
ROUTE_WIDTH = 16
ORDINARY_WIDTH = 48
ACTIVATION_WIDTH = 24


def make_model(torch, kind):
    nn = torch.nn
    if kind not in KINDS:
        raise ValueError(kind)

    class AugmentedVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "route_augmented":
                self.absolute_encoders = nn.ModuleDict(
                    {
                        str(layer): nn.Sequential(
                            nn.Linear(256, ABS_WIDTH),
                            nn.GELU(),
                            nn.LayerNorm(ABS_WIDTH),
                        )
                        for layer in LAYERS
                    }
                )
                self.route_encoders = nn.ModuleDict(
                    {
                        str(layer): nn.Sequential(
                            nn.Linear(512, ROUTE_WIDTH),
                            nn.GELU(),
                            nn.LayerNorm(ROUTE_WIDTH),
                        )
                        for layer in LAYERS
                    }
                )
                encoded_width = ABS_WIDTH + ROUTE_WIDTH
            else:
                input_width = 256 if kind == "ordinary_wide" else 512
                output_width = (
                    ORDINARY_WIDTH if kind == "ordinary_wide" else ACTIVATION_WIDTH
                )
                self.encoders = nn.ModuleDict(
                    {
                        str(layer): nn.Sequential(
                            nn.Linear(input_width, output_width),
                            nn.GELU(),
                            nn.LayerNorm(output_width),
                        )
                        for layer in LAYERS
                    }
                )
                encoded_width = output_width
            self.head = nn.Sequential(
                nn.Linear(7 + len(LAYERS) * encoded_width, HEAD_WIDTH),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(HEAD_WIDTH, 1),
            )

        def forward(self, data, mode="normal"):
            if mode not in ("normal", "swap", "mismatch"):
                raise ValueError(mode)
            if self.kind != "route_augmented" and mode != "normal":
                raise ValueError(f"{mode} is only defined for route_augmented")
            parts = [data["scalar"]]
            if self.kind == "route_augmented":
                for layer in LAYERS:
                    matched = data["features"]["matched"][layer]
                    other = data["features"]["counterfactual"][layer]
                    if mode == "swap":
                        matched, other = other, matched
                    elif mode == "mismatch":
                        other = data["features"]["mismatch"][layer]
                    route = torch.cat(
                        [matched - other, matched * other], dim=1
                    )
                    parts.append(self.absolute_encoders[str(layer)](matched))
                    parts.append(self.route_encoders[str(layer)](route))
            elif self.kind == "ordinary_wide":
                for layer in LAYERS:
                    parts.append(
                        self.encoders[str(layer)](
                            data["features"]["matched"][layer]
                        )
                    )
            else:
                for layer in LAYERS:
                    end = data["features"]["matched"][layer]
                    start = data["features"]["matched_start"][layer]
                    delta = torch.cat([end - start, end * start], dim=1)
                    parts.append(self.encoders[str(layer)](delta))
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    return AugmentedVerifier()


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def fit_model(torch, data, kind, fit_idx, checkpoint_idx, seed):
    torch.manual_seed(seed)
    model = make_model(torch, kind)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fit_tensor = torch.tensor(fit_idx, dtype=torch.long)
    positive, negative = rank_pairs(data, fit_idx)
    pos_tensor = torch.tensor(positive, dtype=torch.long)
    neg_tensor = torch.tensor(negative, dtype=torch.long)
    labels = data["labels"][fit_tensor]
    n_pos = float(labels.sum().item())
    n_neg = float(len(labels) - n_pos)
    pos_weight = torch.tensor(n_neg / max(1.0, n_pos), dtype=torch.float32)
    best_key = None
    best_state = None
    selected = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[fit_tensor], labels, pos_weight=pos_weight
        )
        rank = torch.nn.functional.softplus(
            -(logits[pos_tensor] - logits[neg_tensor])
        ).mean()
        loss = bce + RANK_WEIGHT * rank
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                scores = model(data).tolist()
            accuracy = pair_accuracy(scores, data, checkpoint_idx)
            cp_pos, cp_neg = rank_pairs(data, checkpoint_idx)
            with torch.no_grad():
                cp_logits = model(data)
                rank_loss = float(
                    torch.nn.functional.softplus(
                        -(
                            cp_logits[torch.tensor(cp_pos)]
                            - cp_logits[torch.tensor(cp_neg)]
                        )
                    )
                    .mean()
                    .item()
                )
            key = (accuracy, -rank_loss, -epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                selected = {
                    "epoch": epoch,
                    "checkpoint_pair_accuracy": accuracy,
                    "checkpoint_rank_loss": rank_loss,
                }
    model.load_state_dict(best_state)
    model.eval()
    return model, selected


def run_nested(torch, data, seed):
    pids = sorted(set(data["pids"]))
    folds = outer_splits(pids, seed)
    outcomes = defaultdict(dict)
    fold_report = []
    parameters = {kind: parameter_count(make_model(torch, kind)) for kind in KINDS}
    for fold_index, test_pids in enumerate(folds):
        test_set = set(test_pids)
        available = [pid for pid in pids if pid not in test_set]
        random.Random(seed + 1009 * fold_index).shuffle(available)
        n_checkpoint = max(1, len(available) // 8)
        n_policy = max(1, len(available) // 8)
        checkpoint_pids = available[:n_checkpoint]
        policy_pids = available[n_checkpoint : n_checkpoint + n_policy]
        fit_pids = available[n_checkpoint + n_policy :]
        fit_idx = indices_for_pids(data, fit_pids)
        checkpoint_idx = indices_for_pids(data, checkpoint_pids)
        evidence_weight, parent_penalty, explicit_curve = tune_explicit(
            data, policy_pids
        )
        sc, _ = select_policy(data, test_pids, explicit=False)
        explicit, _ = select_policy(
            data,
            test_pids,
            explicit=True,
            evidence_weight=evidence_weight,
            parent_penalty=parent_penalty,
        )
        for pid in test_pids:
            outcomes["sc8"][pid] = sc[pid]
            outcomes["explicit_predicted_state"][pid] = explicit[pid]
        models = {}
        weights = {}
        checkpoints = {}
        curves = {}
        for kind_index, kind in enumerate(KINDS):
            model, checkpoint = fit_model(
                torch,
                data,
                kind,
                fit_idx,
                checkpoint_idx,
                seed + 10000 * fold_index + 1000 * kind_index,
            )
            with torch.no_grad():
                scores = model(data).tolist()
            weight, curve = tune_weight(
                data,
                policy_pids,
                scores,
                evidence_weight,
                parent_penalty,
            )
            held, _ = select_policy(
                data,
                test_pids,
                scores,
                weight,
                explicit=True,
                evidence_weight=evidence_weight,
                parent_penalty=parent_penalty,
            )
            models[kind] = model
            weights[kind] = weight
            checkpoints[kind] = checkpoint
            curves[kind] = curve
            for pid in test_pids:
                outcomes[kind][pid] = held[pid]
        primary_model = models["route_augmented"]
        primary_weight = weights["route_augmented"]
        for name, mode in (("route_swap", "swap"), ("route_mismatch", "mismatch")):
            with torch.no_grad():
                signal = primary_model(data, mode=mode).tolist()
            held, _ = select_policy(
                data,
                test_pids,
                signal,
                primary_weight,
                explicit=True,
                evidence_weight=evidence_weight,
                parent_penalty=parent_penalty,
            )
            for pid in test_pids:
                outcomes[name][pid] = held[pid]
        local_primary = sum(
            outcomes["route_augmented"][pid] for pid in test_pids
        ) / len(test_pids)
        local_ordinary = sum(
            outcomes["ordinary_wide"][pid] for pid in test_pids
        ) / len(test_pids)
        fold_report.append(
            {
                "fold": fold_index,
                "fit_problems": len(fit_pids),
                "checkpoint_validation_problems": len(checkpoint_pids),
                "policy_validation_problems": len(policy_pids),
                "test_problems": len(test_pids),
                "parameters": parameters,
                "checkpoints": checkpoints,
                "weights": {
                    "explicit_evidence": evidence_weight,
                    "explicit_parent_penalty": parent_penalty,
                    **weights,
                },
                "curves": {"explicit": explicit_curve, **curves},
                "test_route_augmented": local_primary,
                "test_ordinary_wide": local_ordinary,
                "test_delta_vs_ordinary": local_primary - local_ordinary,
            }
        )
        print("[outer-fold] " + json.dumps(fold_report[-1], sort_keys=True), flush=True)
    return outcomes, fold_report, parameters


def main(args):
    import torch

    torch.set_num_threads(args.threads)
    payload = torch.load(args.features, map_location="cpu")
    validate_payload(torch, payload)
    data = make_data(torch, payload)
    outcomes, folds, parameters = run_nested(torch, data, args.seed)
    primary = outcomes["route_augmented"]
    names = (
        "sc8",
        "explicit_predicted_state",
        "ordinary_wide",
        "activation_wide",
        "route_swap",
        "route_mismatch",
    )
    comparisons = {
        name: metric(primary, outcomes[name], args.seed + index)
        for index, name in enumerate(names)
    }
    adjusted = holm_adjust(
        {
            name: comparisons[name]["paired"]["p"]
            for name in ("sc8", "ordinary_wide", "activation_wide")
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    hop_report = by_hop(outcomes, data["gold_hop"])
    depth_delta = {
        hop: values["route_augmented"] - values["sc8"]
        for hop, values in hop_report.items()
        if hop in ("2", "3", "4")
    }
    half_delta = {}
    for half in (0, 1):
        ids = [pid for pid in primary if hash_half(pid) == half]
        half_delta[str(half)] = (
            sum(primary[pid] for pid in ids)
            - sum(outcomes["ordinary_wide"][pid] for pid in ids)
        ) / len(ids)
    gates = {
        "beyond_parameter_matched_ordinary": (
            comparisons["ordinary_wide"]["delta"] >= 0.01
            and comparisons["ordinary_wide"]["holm_adjusted_p"] < 0.05
        ),
        "beyond_parameter_matched_activation_delta": (
            comparisons["activation_wide"]["delta"] >= 0.01
            and comparisons["activation_wide"]["holm_adjusted_p"] < 0.05
        ),
        "selection_value": (
            comparisons["sc8"]["delta"] >= 0.02
            and comparisons["sc8"]["holm_adjusted_p"] < 0.05
        ),
        "route_controls": (
            comparisons["route_swap"]["delta"] >= 0.02
            and comparisons["route_mismatch"]["delta"] >= 0.01
        ),
        "depth_signature": (
            set(depth_delta) == {"2", "3", "4"}
            and min(depth_delta.values()) >= 0.0
            and depth_delta["4"] >= depth_delta["2"] - 0.01
        ),
        "oof_stability_vs_ordinary": (
            sum(row["test_delta_vs_ordinary"] > 0 for row in folds) >= 4
            and all(value > 0 for value in half_delta.values())
        ),
        "feature_validity": (
            payload["accounting"]["matched_counterfactual_length_mismatches"] == 0
            and payload["accounting"]["hierarchy_and_predecessor_generation"].get(
                "matched_truncated_candidates"
            )
            == 0
            and parameters["ordinary_wide"] >= parameters["route_augmented"]
        ),
    }
    report = {
        "experiment": "Absolute-hidden verifier with route-counterfactual residual",
        "protocol": "EXPERIMENT_PROTOCOL_ROUTE_AUGMENTED_GUIDE_V2.md",
        "source_feature_protocol": payload["protocol"],
        "data": payload["data"],
        "parameter_counts": parameters,
        "nested_oof": {
            "accuracy": {
                name: sum(values.values()) / len(values)
                for name, values in outcomes.items()
            },
            "comparisons": comparisons,
            "folds": folds,
            "gold_hop_stratification_for_evaluation_only": hop_report,
            "depth_delta_vs_sc8": depth_delta,
            "id_hash_half_delta_vs_ordinary": half_delta,
        },
        "gates": gates,
        "decision": (
            "ROUTE RESIDUAL SURVIVES DEVELOPMENT"
            if all(gates.values())
            else "STOP HIDDEN GUIDE SELECTION ROUTE"
        ),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "route_augmented_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="hsgr_route_augmented_eval")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args())
