"""Nested-OOF evaluation for the structure-de-oracled counterfactual Guide.

All 840 previously consumed problems are development data.  Each outer fold
has disjoint fit, checkpoint-validation, policy-validation, and test problems.
The remaining 377 problems are not selected or read by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

from hsgr_focus_route_ceiling import exact_mcnemar
from hsgr_structured_hidden_verifier import LAYERS


SEED = 20260831
FOLDS = 5
EPOCHS = 100
ENCODER_WIDTH = 16
HEAD_WIDTH = 32
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
RANK_WEIGHT = 1.0
SIGNAL_WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
EXPLICIT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
KINDS = ("guide", "ordinary", "activation_delta", "nonhidden")


def validate_payload(torch, payload):
    metas = payload.get("metas")
    scalars = payload.get("scalars")
    features = payload.get("features")
    if not isinstance(metas, list) or len({meta["id"] for meta in metas}) != 840:
        raise RuntimeError("expected 840-problem feature payload")
    if not torch.is_tensor(scalars) or scalars.shape != (len(metas), 7):
        raise RuntimeError(f"unexpected scalar shape {getattr(scalars, 'shape', None)}")
    expected = {"matched", "counterfactual", "mismatch", "matched_start"}
    if set(features or {}) != expected:
        raise RuntimeError(f"unexpected feature views: {sorted(features or {})}")
    for view in expected:
        if set(features[view]) != set(LAYERS):
            raise RuntimeError(f"unexpected layers for {view}")
        for layer in LAYERS:
            if tuple(features[view][layer].shape) != (len(metas), 256):
                raise RuntimeError(f"unexpected {view}/L{layer} shape")
    if payload.get("accounting", {}).get("matched_counterfactual_length_mismatches") != 0:
        raise RuntimeError("primary prompt lengths are not matched")


def grouped_indices(metas):
    grouped = defaultdict(list)
    for index, meta in enumerate(metas):
        grouped[meta["id"]].append(index)
    return grouped


def normalize_within(values, metas, allowed=None):
    allowed = set(allowed) if allowed is not None else None
    grouped = defaultdict(list)
    for index, meta in enumerate(metas):
        if allowed is None or meta["id"] in allowed:
            grouped[meta["id"]].append(index)
    result = {}
    for pid, indices in grouped.items():
        local = [float(values[index]) for index in indices]
        center = sum(local) / len(local)
        variance = sum((value - center) ** 2 for value in local) / max(1, len(local) - 1)
        scale = math.sqrt(variance) if variance else 1.0
        for index, value in zip(indices, local):
            result[index] = (value - center) / scale
    return result


def make_data(torch, payload):
    metas = payload["metas"]
    state = payload["state_summary"]
    pids = sorted(state)
    shuffled = list(pids)
    random.Random(SEED + 97).shuffle(shuffled)
    if any(pid == source for pid, source in zip(pids, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]
    permuted = {pid: state[source] for pid, source in zip(pids, shuffled)}
    scalar_permuted = payload["scalars"].float().clone()
    for index, meta in enumerate(metas):
        source = permuted[meta["id"]]
        scalar_permuted[index, 5] = min(4, source["predicted_depth"]) / 4.0
        scalar_permuted[index, 6] = min(3, source["n_parents"]) / 3.0
    return {
        "metas": metas,
        "pids": [meta["id"] for meta in metas],
        "labels": torch.tensor([float(meta["label"]) for meta in metas], dtype=torch.float32),
        "scalar": payload["scalars"].float(),
        "scalar_permuted": scalar_permuted,
        "features": {
            view: {layer: payload["features"][view][layer].float() for layer in LAYERS}
            for view in payload["features"]
        },
        "gold_hop": {
            pid: int(next(meta.get("n_hops") for meta in metas if meta["id"] == pid))
            for pid in pids
        },
    }


def make_model(torch, kind):
    nn = torch.nn
    if kind not in KINDS:
        raise ValueError(kind)

    class CandidateVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            widths = {
                "guide": 512,
                "ordinary": 256,
                "activation_delta": 512,
                "nonhidden": 0,
            }
            hidden_width = widths[kind]
            if hidden_width:
                self.encoders = nn.ModuleDict(
                    {
                        str(layer): nn.Sequential(
                            nn.Linear(hidden_width, ENCODER_WIDTH),
                            nn.GELU(),
                            nn.LayerNorm(ENCODER_WIDTH),
                        )
                        for layer in LAYERS
                    }
                )
            input_dim = 7 + (len(LAYERS) * ENCODER_WIDTH if hidden_width else 0)
            self.head = nn.Sequential(
                nn.Linear(input_dim, HEAD_WIDTH),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(HEAD_WIDTH, 1),
            )

        def forward(self, data, mode="normal"):
            if mode not in ("normal", "swap", "mismatch", "state_permute"):
                raise ValueError(mode)
            scalar = data["scalar_permuted"] if mode == "state_permute" else data["scalar"]
            parts = [scalar]
            if self.kind == "guide":
                for layer in LAYERS:
                    matched = data["features"]["matched"][layer]
                    other = data["features"]["counterfactual"][layer]
                    if mode == "swap":
                        matched, other = other, matched
                    elif mode == "mismatch":
                        other = data["features"]["mismatch"][layer]
                    pair = torch.cat([matched - other, matched * other], dim=1)
                    parts.append(self.encoders[str(layer)](pair))
            elif self.kind == "ordinary":
                for layer in LAYERS:
                    parts.append(self.encoders[str(layer)](data["features"]["matched"][layer]))
            elif self.kind == "activation_delta":
                for layer in LAYERS:
                    end = data["features"]["matched"][layer]
                    start = data["features"]["matched_start"][layer]
                    pair = torch.cat([end - start, end * start], dim=1)
                    parts.append(self.encoders[str(layer)](pair))
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    return CandidateVerifier()


def indices_for_pids(data, allowed):
    allowed = set(allowed)
    return [index for index, pid in enumerate(data["pids"]) if pid in allowed]


def rank_pairs(data, indices):
    buckets = defaultdict(lambda: {"pos": [], "neg": []})
    for index in indices:
        key = "pos" if data["labels"][index].item() > 0.5 else "neg"
        buckets[data["pids"][index]][key].append(index)
    positive, negative = [], []
    for value in buckets.values():
        for pos in value["pos"]:
            for neg in value["neg"]:
                positive.append(pos)
                negative.append(neg)
    return positive, negative


def pair_accuracy(scores, data, indices):
    positive, negative = rank_pairs(data, indices)
    if not positive:
        return 0.5
    return sum(float(float(scores[pos]) > float(scores[neg])) for pos, neg in zip(positive, negative)) / len(positive)


def fit_model(torch, data, kind, fit_idx, checkpoint_idx, seed):
    torch.manual_seed(seed)
    model = make_model(torch, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        rank = torch.nn.functional.softplus(-(logits[pos_tensor] - logits[neg_tensor])).mean()
        loss = bce + RANK_WEIGHT * rank
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                values = model(data).tolist()
            accuracy = pair_accuracy(values, data, checkpoint_idx)
            cp_pos, cp_neg = rank_pairs(data, checkpoint_idx)
            if cp_pos:
                with torch.no_grad():
                    cp_logits = model(data)
                    rank_loss = float(
                        torch.nn.functional.softplus(
                            -(cp_logits[torch.tensor(cp_pos)] - cp_logits[torch.tensor(cp_neg)])
                        ).mean().item()
                    )
            else:
                rank_loss = float("inf")
            key = (accuracy, -rank_loss, -epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                selected = {
                    "epoch": epoch,
                    "checkpoint_pair_accuracy": accuracy,
                    "checkpoint_rank_loss": rank_loss,
                }
    if best_state is None:
        raise RuntimeError("no model checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, selected


def select_policy(
    data,
    allowed_pids,
    signal=None,
    weight=0.0,
    explicit=True,
    evidence_weight=0.0,
    parent_penalty=0.0,
):
    allowed = set(allowed_pids)
    metas = data["metas"]
    if signal is None:
        signal = [0.0] * len(metas)
    normalized = normalize_within(signal, metas, allowed)
    by_problem = defaultdict(list)
    for index, meta in enumerate(metas):
        if meta["id"] in allowed:
            by_problem[meta["id"]].append((index, meta))
    outcomes = {}
    answers = {}
    for pid, items in by_problem.items():
        grouped = defaultdict(lambda: {"count": 0, "evidence": 0.0, "parent": 0.0, "signal": 0.0, "correct": 0})
        order = []
        for index, meta in items:
            key = meta.get("norm")
            if not key:
                continue
            if key not in grouped:
                order.append(key)
            scalar = data["scalar"][index]
            grouped[key]["count"] += 1
            grouped[key]["evidence"] += float(scalar[1].item())
            grouped[key]["parent"] += float(scalar[2].item())
            grouped[key]["signal"] += float(normalized[index])
            grouped[key]["correct"] = max(grouped[key]["correct"], int(meta["label"]))
        def score(key):
            value = grouped[key]
            count = value["count"]
            base = count / len(items)
            if explicit:
                base += evidence_weight * value["evidence"] / count
                base -= parent_penalty * value["parent"] / count
            return base + weight * value["signal"] / count
        best = max(order, key=score, default=None)
        outcomes[pid] = bool(best and grouped[best]["correct"])
        answers[pid] = best
    return outcomes, answers


def tune_explicit(data, policy_pids):
    candidates = []
    curve = {}
    for evidence_weight in EXPLICIT_WEIGHTS:
        for parent_penalty in EXPLICIT_WEIGHTS:
            outcomes, _ = select_policy(
                data,
                policy_pids,
                explicit=True,
                evidence_weight=evidence_weight,
                parent_penalty=parent_penalty,
            )
            accuracy = sum(outcomes.values()) / len(outcomes)
            key = f"e{evidence_weight}_p{parent_penalty}"
            curve[key] = accuracy
            candidates.append(
                (
                    accuracy,
                    -(evidence_weight + parent_penalty),
                    -evidence_weight,
                    evidence_weight,
                    parent_penalty,
                )
            )
    _, _, _, evidence_weight, parent_penalty = max(candidates)
    return evidence_weight, parent_penalty, curve


def tune_weight(data, policy_pids, signal, evidence_weight, parent_penalty):
    candidates = []
    curve = {}
    for weight in SIGNAL_WEIGHTS:
        outcomes, _ = select_policy(
            data,
            policy_pids,
            signal,
            weight,
            explicit=True,
            evidence_weight=evidence_weight,
            parent_penalty=parent_penalty,
        )
        accuracy = sum(outcomes.values()) / len(outcomes)
        curve[str(weight)] = accuracy
        candidates.append((accuracy, -weight, weight))
    _, _, selected = max(candidates)
    return selected, curve


def metric(hits, baseline, bootstrap_seed=0):
    ids = sorted(set(hits) & set(baseline))
    values = [int(hits[pid]) for pid in ids]
    base = [int(baseline[pid]) for pid in ids]
    deltas = [left - right for left, right in zip(values, base)]
    rng = random.Random(bootstrap_seed)
    samples = []
    for _ in range(10000):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    return {
        "accuracy": sum(values) / len(values),
        "baseline_accuracy": sum(base) / len(base),
        "delta": sum(deltas) / len(deltas),
        "paired": exact_mcnemar(values, base),
        "paired_bootstrap_95ci": [samples[249], samples[9749]],
        "n": len(ids),
    }


def holm_adjust(p_values):
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * float(p_values[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def hash_half(pid):
    return int(hashlib.sha256(pid.encode("utf-8")).hexdigest()[0], 16) % 2


def outer_splits(pids, seed):
    shuffled = list(pids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::FOLDS] for index in range(FOLDS)]


def run_nested_oof(torch, data, seed):
    pids = sorted(set(data["pids"]))
    folds = outer_splits(pids, seed)
    outcomes = defaultdict(dict)
    answers = defaultdict(dict)
    fold_report = []
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
        sc, sc_answers = select_policy(data, test_pids, explicit=False)
        explicit, explicit_answers = select_policy(
            data,
            test_pids,
            explicit=True,
            evidence_weight=evidence_weight,
            parent_penalty=parent_penalty,
        )
        for pid in test_pids:
            outcomes["sc8"][pid] = sc[pid]
            outcomes["explicit_predicted_state"][pid] = explicit[pid]
            answers["sc8"][pid] = sc_answers[pid]
            answers["explicit_predicted_state"][pid] = explicit_answers[pid]

        models = {}
        diagnostics = {}
        full_scores = {}
        selected_weights = {}
        curves = {}
        for kind_index, kind in enumerate(KINDS):
            model, selected = fit_model(
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
            models[kind] = model
            diagnostics[kind] = selected
            full_scores[kind] = scores
            selected_weights[kind] = weight
            curves[kind] = curve
            name = {
                "guide": "matched_guide",
                "ordinary": "ordinary_hidden_verifier",
                "activation_delta": "activation_delta_verifier",
                "nonhidden": "nonhidden_listwise",
            }[kind]
            held, held_answers = select_policy(
                data,
                test_pids,
                scores,
                weight,
                explicit=True,
                evidence_weight=evidence_weight,
                parent_penalty=parent_penalty,
            )
            for pid in test_pids:
                outcomes[name][pid] = held[pid]
                answers[name][pid] = held_answers[pid]

        guide_model = models["guide"]
        guide_weight = selected_weights["guide"]
        controls = {
            "route_swap": "swap",
            "route_mismatch": "mismatch",
            "state_label_permutation": "state_permute",
        }
        for name, mode in controls.items():
            with torch.no_grad():
                signal = guide_model(data, mode=mode).tolist()
            held, held_answers = select_policy(
                data,
                test_pids,
                signal,
                guide_weight,
                explicit=True,
                evidence_weight=evidence_weight,
                parent_penalty=parent_penalty,
            )
            for pid in test_pids:
                outcomes[name][pid] = held[pid]
                answers[name][pid] = held_answers[pid]

        length_signal = [float(value) for value in data["scalar"][:, 4].tolist()]
        length_weight, length_curve = tune_weight(
            data,
            policy_pids,
            length_signal,
            evidence_weight,
            parent_penalty,
        )
        length_hits, length_answers = select_policy(
            data,
            test_pids,
            length_signal,
            length_weight,
            explicit=True,
            evidence_weight=evidence_weight,
            parent_penalty=parent_penalty,
        )
        for pid in test_pids:
            outcomes["length_control"][pid] = length_hits[pid]
            answers["length_control"][pid] = length_answers[pid]

        local_primary = sum(outcomes["matched_guide"][pid] for pid in test_pids) / len(test_pids)
        local_sc = sum(outcomes["sc8"][pid] for pid in test_pids) / len(test_pids)
        fold_report.append(
            {
                "fold": fold_index,
                "fit_problems": len(fit_pids),
                "checkpoint_validation_problems": len(checkpoint_pids),
                "policy_validation_problems": len(policy_pids),
                "test_problems": len(test_pids),
                "selected_checkpoints": diagnostics,
                "selected_weights": {
                    "explicit_evidence": evidence_weight,
                    "explicit_parent_penalty": parent_penalty,
                    **selected_weights,
                    "length": length_weight,
                },
                "weight_curves": {
                    "explicit": explicit_curve,
                    **curves,
                    "length": length_curve,
                },
                "test_sc8": local_sc,
                "test_matched_guide": local_primary,
                "test_delta": local_primary - local_sc,
            }
        )
        print("[outer-fold] " + json.dumps(fold_report[-1], sort_keys=True), flush=True)
    return outcomes, answers, fold_report


def by_hop(outcomes, gold_hop):
    report = {}
    for hop in sorted(set(gold_hop.values())):
        ids = [pid for pid, value in gold_hop.items() if value == hop]
        report[str(hop)] = {
            "n": len(ids),
            **{
                name: sum(values[pid] for pid in ids) / len(ids)
                for name, values in outcomes.items()
            },
        }
    return report


def main(args):
    import torch

    torch.set_num_threads(args.threads)
    payload = torch.load(args.features, map_location="cpu")
    validate_payload(torch, payload)
    data = make_data(torch, payload)
    outcomes, answers, folds = run_nested_oof(torch, data, args.seed)
    primary = outcomes["matched_guide"]
    comparisons = {
        name: metric(primary, outcomes[name], args.seed + index)
        for index, name in enumerate(
            (
                "sc8",
                "explicit_predicted_state",
                "nonhidden_listwise",
                "ordinary_hidden_verifier",
                "activation_delta_verifier",
                "route_swap",
                "route_mismatch",
                "state_label_permutation",
                "length_control",
            )
        )
    }
    raw_primary_p = {
        name: comparisons[name]["paired"]["p"]
        for name in ("sc8", "ordinary_hidden_verifier", "activation_delta_verifier")
    }
    adjusted = holm_adjust(raw_primary_p)
    for name, p_value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = p_value
    hop_report = by_hop(outcomes, data["gold_hop"])
    depth_delta = {
        hop: values["matched_guide"] - values["sc8"]
        for hop, values in hop_report.items()
        if hop in ("2", "3", "4")
    }
    half_delta = {}
    for half in (0, 1):
        ids = [pid for pid in primary if hash_half(pid) == half]
        half_delta[str(half)] = (
            sum(primary[pid] for pid in ids) - sum(outcomes["sc8"][pid] for pid in ids)
        ) / len(ids)
    guide_accounting = payload["accounting"]["hierarchy_and_predecessor_generation"]
    candidate_accounting = payload["accounting"]["candidate_generation"]
    accounting_complete = (
        payload["accounting"]["matched_counterfactual_length_mismatches"] == 0
        and all(
            key in guide_accounting
            for key in (
                "planner_prompt_tokens",
                "planner_generated_tokens",
                "executor_prompt_tokens",
                "executor_generated_tokens",
                "matched_attended_prompt_tokens",
                "counterfactual_attended_prompt_tokens",
            )
        )
        and candidate_accounting.get("completion_accounting_complete") is True
    )
    gates = {
        "selection_value": (
            comparisons["sc8"]["delta"] >= 0.02
            and comparisons["sc8"]["holm_adjusted_p"] < 0.05
        ),
        "beyond_explicit_nonhidden": (
            comparisons["explicit_predicted_state"]["delta"] >= 0.01
            and comparisons["nonhidden_listwise"]["delta"] >= 0.01
        ),
        "beyond_ordinary_hidden": (
            comparisons["ordinary_hidden_verifier"]["delta"] >= 0.01
            and comparisons["ordinary_hidden_verifier"]["holm_adjusted_p"] < 0.05
        ),
        "beyond_activation_delta": (
            comparisons["activation_delta_verifier"]["delta"] >= 0.01
            and comparisons["activation_delta_verifier"]["holm_adjusted_p"] < 0.05
        ),
        "route_directionality": comparisons["route_swap"]["delta"] >= 0.02,
        "route_coupling": comparisons["route_mismatch"]["delta"] >= 0.01,
        "hierarchy_dependence": comparisons["state_label_permutation"]["delta"] >= 0.01,
        "depth_signature": (
            set(depth_delta) == {"2", "3", "4"}
            and min(depth_delta.values()) >= 0.0
            and depth_delta["4"] >= depth_delta["2"] - 0.01
        ),
        "oof_stability": (
            sum(row["test_delta"] > 0 for row in folds) >= 4
            and all(value > 0 for value in half_delta.values())
        ),
        "no_structure_oracle_and_accounting_audit": accounting_complete,
    }
    report = {
        "experiment": "Structure-de-oracled route-counterfactual hidden Guide",
        "protocol": payload["protocol"],
        "claim_boundary": payload["claim_boundary"],
        "data": payload["data"],
        "nested_oof": {
            "folds": folds,
            "accuracy": {
                name: sum(values.values()) / len(values) for name, values in outcomes.items()
            },
            "matched_guide_comparisons": comparisons,
            "gold_hop_stratification_for_evaluation_only": hop_report,
            "depth_delta_vs_sc8": depth_delta,
            "id_hash_half_delta_vs_sc8": half_delta,
        },
        "accounting": payload["accounting"],
        "gates": gates,
        "decision": (
            "ELIGIBLE FOR ONE FINAL377 RUN" if all(gates.values()) else "DO NOT CONSUME FINAL377"
        ),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "route_counterfactual_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    cases_path = os.path.join(args.out_dir, "route_counterfactual_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in sorted(primary):
            if primary[pid] == outcomes["ordinary_hidden_verifier"][pid]:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": pid,
                        "gold_hops": data["gold_hop"][pid],
                        "matched_guide_answer": answers["matched_guide"][pid],
                        "ordinary_hidden_answer": answers["ordinary_hidden_verifier"][pid],
                        "guide_fixes_ordinary": bool(
                            primary[pid] and not outcomes["ordinary_hidden_verifier"][pid]
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="hsgr_route_counterfactual_eval")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args())
