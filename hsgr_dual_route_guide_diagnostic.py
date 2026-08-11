"""Post-hoc diagnostic for a dual-route hidden HSGR Guide.

The Guide compares hidden evidence-alignment signals under the routed final-hop
support block and a length-matched predecessor block.  Reader supervision is
structural only: exact candidate mentions in the corresponding route.  Policy
weights are selected on problem-disjoint development OOF predictions before
held-out outcomes are scored.

The n=320 set was already inspected when this diagnostic was designed, so this
script is mechanism development, not a second fresh held-out claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

from hsgr_copy_guard_probe import (
    EXACT_GUARD_WEIGHT,
    SUPPORT_WEIGHT,
    dependency_state,
    metric,
)
from hsgr_structured_hidden_verifier import (
    LAYERS,
    SYSTEM as VERIFY_SYSTEM,
    projectors,
)
from mh_ceiling import answers_match
from mh_e0 import load_rows
from mh_latent_rerank import auroc, fit_probe, within_problem_auroc
from pilot import Runner


WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
SEED = 20260818


def normalize_within(scores, metas):
    grouped = defaultdict(list)
    for index, meta in enumerate(metas):
        grouped[meta["id"]].append(index)
    normalized = [0.0] * len(scores)
    for indices in grouped.values():
        values = [float(scores[index]) for index in indices]
        center = sum(values) / len(values)
        variance = sum((value - center) ** 2 for value in values) / max(
            1, len(values) - 1
        )
        scale = math.sqrt(variance) if variance > 0 else 1.0
        for index in indices:
            normalized[index] = (float(scores[index]) - center) / scale
    return normalized


def extract_correct_features(runner, metas, batch_size, max_context):
    torch = runner.torch
    tokenizer = runner.tok
    matrices = projectors(torch, runner.model.config.hidden_size, "cuda")
    features = {layer: [] for layer in LAYERS}
    for start in range(0, len(metas), batch_size):
        batch = metas[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": meta["prompts"]["correct"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for meta in batch
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context,
        ).to("cuda")
        with torch.no_grad():
            result = runner.model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer in LAYERS:
            hidden = result.hidden_states[layer][:, -1, :].float()
            projected = hidden @ matrices[layer]
            projected = projected / (
                projected.norm(dim=1, keepdim=True) + 1e-8
            )
            features[layer].append(projected.half().cpu())
        del enc, result
        torch.cuda.empty_cache()
        if start == 0 or (start + batch_size) % 160 == 0:
            print(
                f"[correct-route] {min(start + batch_size, len(metas))}/{len(metas)}",
                flush=True,
            )
    return {layer: torch.cat(parts) for layer, parts in features.items()}


def reader_oof_and_transfer(
    dev_payload,
    route,
    dev_labels,
    held_features,
    held_metas,
    held_labels,
    seed,
):
    import torch

    dev_metas = dev_payload["metas"]
    pids = sorted({meta["id"] for meta in dev_metas})
    rng = random.Random(seed)
    rng.shuffle(pids)
    folds = [pids[index::5] for index in range(5)]
    labels = torch.tensor(dev_labels, dtype=torch.float32)
    oof = [None] * len(dev_metas)
    held_members = []
    selected = {}

    for fold_index, hold in enumerate(folds):
        hold_set = set(hold)
        train_pids = [pid for pid in pids if pid not in hold_set]
        candidates = []
        for layer in LAYERS:
            shuffled = list(train_pids)
            random.Random(seed + 101 * fold_index + layer).shuffle(shuffled)
            n_val = max(1, len(shuffled) // 7)
            val_set = set(shuffled[:n_val])
            train_idx = [
                index
                for index, meta in enumerate(dev_metas)
                if meta["id"] not in hold_set and meta["id"] not in val_set
            ]
            val_idx = [
                index
                for index, meta in enumerate(dev_metas)
                if meta["id"] in val_set
            ]
            X = dev_payload["features"][route][layer].float()
            scorer, criterion = fit_probe(
                X[train_idx],
                labels[train_idx],
                [dev_metas[index]["id"] for index in train_idx],
                X[val_idx],
                labels[val_idx],
                [dev_metas[index]["id"] for index in val_idx],
                use_rank=True,
                epochs=250,
                rank_w=1.0,
                seed=seed + 1000 * fold_index + layer,
            )
            test_idx = [
                index
                for index, meta in enumerate(dev_metas)
                if meta["id"] in hold_set
            ]
            candidates.append(
                (
                    criterion,
                    layer,
                    test_idx,
                    scorer(X[test_idx]),
                    scorer(held_features[layer].float()),
                )
            )
        criterion, layer, test_idx, test_scores, held_scores = max(
            candidates, key=lambda item: item[0]
        )
        selected[str(fold_index)] = {
            "layer": layer,
            "internal_criterion": criterion,
        }
        test_metas = [dev_metas[index] for index in test_idx]
        test_scores = normalize_within(test_scores, test_metas)
        for index, score in zip(test_idx, test_scores):
            oof[index] = score
        held_members.append(normalize_within(held_scores, held_metas))

    if any(score is None for score in oof):
        raise RuntimeError(f"incomplete OOF predictions for route={route}")
    transferred = [
        sum(member[index] for member in held_members) / len(held_members)
        for index in range(len(held_metas))
    ]
    dev_wp, dev_wp_n = within_problem_auroc(
        oof, dev_labels, [meta["id"] for meta in dev_metas]
    )
    held_wp, held_wp_n = within_problem_auroc(
        transferred, held_labels, [meta["id"] for meta in held_metas]
    )
    diagnostics = {
        "selected_layers": selected,
        "dev_oof_pooled_auroc": auroc(oof, dev_labels),
        "dev_oof_within_auroc": dev_wp,
        "dev_oof_within_n": dev_wp_n,
        "heldout_pooled_auroc": auroc(transferred, held_labels),
        "heldout_within_auroc": held_wp,
        "heldout_within_n": held_wp_n,
        "dev_positives": sum(dev_labels),
        "heldout_positives": sum(held_labels),
    }
    return oof, transferred, diagnostics


def select_policy(metas, state, signal, weight, routed=True, explicit=True):
    by_problem = defaultdict(list)
    if signal is None:
        signal = [0.0] * len(metas)
    signal = normalize_within(signal, metas)
    for index, meta in enumerate(metas):
        by_problem[meta["id"]].append((meta, signal[index]))

    outcomes = {}
    answers = {}
    for pid, items in by_problem.items():
        grouped = defaultdict(
            lambda: {"count": 0, "mention": 0.0, "signal": 0.0}
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
            grouped[key]["signal"] += float(score)
        best = max(
            order,
            key=lambda key: (
                grouped[key]["count"] / len(items)
                + (SUPPORT_WEIGHT if routed else 0.0)
                * grouped[key]["mention"]
                / grouped[key]["count"]
                - (EXACT_GUARD_WEIGHT if explicit else 0.0)
                * float(key in state[pid]["values"])
                + weight
                * grouped[key]["signal"]
                / grouped[key]["count"]
            ),
            default=None,
        )
        first = items[0][0]
        outcomes[pid] = bool(
            best and answers_match(best, first["gold"], first["aliases"])
        )
        answers[pid] = best
    return outcomes, answers


def tune_on_dev(metas, state, signal):
    candidates = []
    curves = {}
    for weight in WEIGHTS:
        outcomes, _ = select_policy(metas, state, signal, weight)
        accuracy = sum(outcomes.values()) / len(outcomes)
        curves[str(weight)] = accuracy
        candidates.append((accuracy, -weight, weight))
    _, _, selected = max(candidates)
    return selected, curves


def combine(left, right, right_scale=-1.0):
    return [float(a) + right_scale * float(b) for a, b in zip(left, right)]


def main(args):
    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    dev_payload = torch.load(args.dev_features, map_location="cpu")
    held_payload = torch.load(args.heldout_features, map_location="cpu")
    held_metas = held_payload["metas"]
    if len(held_metas) != 2560:
        raise RuntimeError(f"expected 2560 held-out units, got {len(held_metas)}")

    combined_path = os.path.join(args.out_dir, "dual_route_hidden_features.pt")
    reusable_combined = args.combined_features or combined_path
    if os.path.isfile(reusable_combined):
        combined_payload = torch.load(reusable_combined, map_location="cpu")
        combined_metas = combined_payload["metas"]
        if [
            (meta["id"], meta["cand"]) for meta in combined_metas
        ] != [
            (meta["id"], meta["cand"]) for meta in held_metas
        ]:
            raise RuntimeError("combined feature metadata/order mismatch")
        correct_features = combined_payload["features"]["correct"]
        print(f"[correct-route] reuse {reusable_combined}", flush=True)
    else:
        runner = Runner(args.model)
        correct_features = extract_correct_features(
            runner, held_metas, args.bs, args.max_context
        )
        del runner.model
        torch.cuda.empty_cache()
        combined_payload = {
            "features": {
                "correct": correct_features,
                "wrong": held_payload["features"],
            },
            "metas": held_metas,
        }
        torch.save(combined_payload, combined_path)
        print(f"[correct-route] saved {combined_path}", flush=True)

    dev_metas = dev_payload["metas"]
    dev_rows = load_rows(args.data, 200, seed=0)
    dev_state = dependency_state(dev_rows)
    all_rows = load_rows(args.data, 0, seed=0)
    held_ids = {meta["id"] for meta in held_metas}
    held_rows = [row for row in all_rows if row["_uid"] in held_ids]
    held_state = dependency_state(held_rows)
    if len(held_state) != 320:
        raise RuntimeError(f"expected 320 held-out states, got {len(held_state)}")

    dev_labels = {
        "support_correct": [meta["mentions"]["correct"] for meta in dev_metas],
        "support_wrong": [meta["mentions"]["wrong"] for meta in dev_metas],
        "copy": [
            int(bool(meta["norm"]) and meta["norm"] in dev_state[meta["id"]]["values"])
            for meta in dev_metas
        ],
    }
    held_labels = {
        "support_correct": [meta["mentions"]["correct"] for meta in held_metas],
        "support_wrong": [meta["mentions"]["wrong"] for meta in held_metas],
        "copy": [
            int(bool(meta["norm"]) and meta["norm"] in held_state[meta["id"]]["values"])
            for meta in held_metas
        ],
    }
    dev_labels["copy_correct_control"] = list(dev_labels["copy"])
    held_labels["copy_correct_control"] = list(held_labels["copy"])

    reader_specs = {
        "support_correct": ("correct", SEED + 100),
        "support_wrong": ("wrong", SEED + 200),
        "copy": ("wrong", SEED + 300),
        "copy_correct_control": ("correct", SEED + 400),
    }
    dev_scores = {}
    held_scores = {}
    reader_report = {}
    held_feature_routes = {
        "correct": correct_features,
        "wrong": held_payload["features"],
    }
    for name, (route, seed) in reader_specs.items():
        print(f"[reader] {name} route={route}", flush=True)
        dev_scores[name], held_scores[name], reader_report[name] = (
            reader_oof_and_transfer(
                dev_payload,
                route,
                dev_labels[name],
                held_feature_routes[route],
                held_metas,
                held_labels[name],
                seed,
            )
        )

    dev_signals = {
        "route_delta": combine(
            dev_scores["support_correct"], dev_scores["support_wrong"]
        ),
        "route_swap": combine(
            dev_scores["support_wrong"], dev_scores["support_correct"]
        ),
        "correct_only": dev_scores["support_correct"],
        "anti_wrong": [-score for score in dev_scores["support_wrong"]],
        "copy_hidden": [-score for score in dev_scores["copy"]],
        "copy_correct_control": [
            -score for score in dev_scores["copy_correct_control"]
        ],
        "transition_guard": [
            correct - wrong - copy
            for correct, wrong, copy in zip(
                dev_scores["support_correct"],
                dev_scores["support_wrong"],
                dev_scores["copy"],
            )
        ],
        "length_only": [len(meta["norm"] or "") for meta in dev_metas],
    }
    held_signals = {
        "route_delta": combine(
            held_scores["support_correct"], held_scores["support_wrong"]
        ),
        "route_swap": combine(
            held_scores["support_wrong"], held_scores["support_correct"]
        ),
        "correct_only": held_scores["support_correct"],
        "anti_wrong": [-score for score in held_scores["support_wrong"]],
        "copy_hidden": [-score for score in held_scores["copy"]],
        "copy_correct_control": [
            -score for score in held_scores["copy_correct_control"]
        ],
        "transition_guard": [
            correct - wrong - copy
            for correct, wrong, copy in zip(
                held_scores["support_correct"],
                held_scores["support_wrong"],
                held_scores["copy"],
            )
        ],
        "length_only": [len(meta["norm"] or "") for meta in held_metas],
    }

    sc8, _ = select_policy(
        held_metas, held_state, None, 0.0, routed=False, explicit=False
    )
    lexical, _ = select_policy(
        held_metas, held_state, None, 0.0, routed=True, explicit=False
    )
    explicit, explicit_answers = select_policy(
        held_metas, held_state, None, 0.0, routed=True, explicit=True
    )
    selected_weights = {}
    dev_curves = {}
    policies = {}
    policy_answers = {}
    for name, dev_signal in dev_signals.items():
        selected_weights[name], dev_curves[name] = tune_on_dev(
            dev_metas, dev_state, dev_signal
        )
        policies[name], policy_answers[name] = select_policy(
            held_metas,
            held_state,
            held_signals[name],
            selected_weights[name],
        )

    report = {
        "experiment": "HSGR dual-route hidden Guide diagnostic",
        "claim_boundary": (
            "Post-hoc mechanism development on the previously inspected n=320; "
            "reader labels are structural and weights use only dev OOF outcomes, "
            "but a new untouched split is required for a confirmatory claim."
        ),
        "data": {"dev_problems": 200, "heldout_problems": 320},
        "reader": reader_report,
        "selected_weights_from_dev": selected_weights,
        "dev_oof_weight_curves": dev_curves,
        "baseline": {
            "sc8": sum(sc8.values()) / len(sc8),
            "lexical": metric(lexical, sc8),
            "explicit": metric(explicit, sc8),
        },
        "policy": {
            name: {
                **metric(outcomes, sc8),
                "vs_explicit": metric(outcomes, explicit),
            }
            for name, outcomes in policies.items()
        },
        "by_hop": {},
    }
    meta_by_id = {meta["id"]: meta for meta in held_metas}
    for hop in (2, 3, 4):
        ids = [pid for pid, meta in meta_by_id.items() if meta["n_hops"] == hop]
        report["by_hop"][str(hop)] = {
            "n": len(ids),
            "sc8": sum(sc8[pid] for pid in ids) / len(ids),
            "explicit": sum(explicit[pid] for pid in ids) / len(ids),
            **{
                name: sum(outcomes[pid] for pid in ids) / len(ids)
                for name, outcomes in policies.items()
            },
        }

    route = report["policy"]["route_delta"]
    route_accuracy = route["accuracy"]
    transition = report["policy"]["transition_guard"]
    transition_accuracy = transition["accuracy"]
    report["gates"] = {
        "readers_transfer": (
            reader_report["support_correct"]["heldout_within_auroc"] >= 0.75
            and reader_report["support_wrong"]["heldout_within_auroc"] >= 0.75
        ),
        "route_delta_headroom": (
            route["delta"] >= 0.06 and route["paired"]["p"] < 0.05
        ),
        "nonnegative_vs_explicit": route["vs_explicit"]["delta"] >= 0.0,
        "route_specificity": (
            route_accuracy
            >= report["policy"]["route_swap"]["accuracy"] + 0.02
        ),
        "beyond_length": (
            route_accuracy
            >= report["policy"]["length_only"]["accuracy"] + 0.01
        ),
        "safe_vs_explicit": route["vs_explicit"]["paired"]["b_only"] <= 3,
        "transition_guard_headroom": (
            transition["delta"] >= 0.06
            and transition["paired"]["p"] < 0.05
        ),
        "transition_guard_beyond_length": (
            transition_accuracy
            >= report["policy"]["length_only"]["accuracy"] + 0.01
        ),
        "transition_guard_safe": (
            transition["vs_explicit"]["delta"] >= 0.0
            and transition["vs_explicit"]["paired"]["b_only"] <= 3
        ),
        "predecessor_route_copy_specificity": (
            report["policy"]["copy_hidden"]["accuracy"]
            >= report["policy"]["copy_correct_control"]["accuracy"] + 0.01
        ),
    }
    report["decision"] = (
        "DIAGNOSTIC PASS" if all(report["gates"].values()) else "DIAGNOSTIC FAIL"
    )

    with open(
        os.path.join(args.out_dir, "dual_route_guide_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    with open(
        os.path.join(args.out_dir, "dual_route_guide_cases.jsonl"),
        "w",
        encoding="utf-8",
    ) as handle:
        for pid in sorted(sc8):
            handle.write(
                json.dumps(
                    {
                        "id": pid,
                        "n_hops": meta_by_id[pid]["n_hops"],
                        "explicit": {
                            "answer": explicit_answers[pid],
                            "hit": explicit[pid],
                        },
                        **{
                            name: {
                                "answer": policy_answers[name][pid],
                                "hit": policies[name][pid],
                            }
                            for name in policies
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--heldout-features", required=True)
    parser.add_argument("--combined-features")
    parser.add_argument("--out-dir", default="hsgr_dual_route_diagnostic")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=3072)
    main(parser.parse_args())
