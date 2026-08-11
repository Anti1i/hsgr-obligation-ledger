"""Two-channel hidden HSGR transition consensus diagnostic.

The stable channel is a linear directed margin over destination and predecessor
hidden states.  The relation channel is a skew-bilinear interaction with no
single-route term.  Their relative weight and the final policy weight are
selected only on problem-disjoint development OOF predictions.

The already observed second fresh set is used for mechanism development only.
The final untouched split remains sealed unless every causal diagnostic passes.
"""
from __future__ import annotations

import argparse
import json
import os

from hsgr_bilinear_transition_diagnostic import (
    PROJECTED_DIM,
    SEED as BILINEAR_SEED,
    bilinear_oof_and_transfer,
)
from hsgr_copy_guard_probe import dependency_state, metric
from hsgr_dual_route_guide_diagnostic import WEIGHTS, select_policy
from hsgr_paired_transition_diagnostic import (
    FROZEN_TRANSITION_REFERENCE,
    SEED as PAIRED_SEED,
    paired_counts,
    paired_oof_and_transfer,
    tune_weight,
)
from mh_e0 import load_rows


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def combine(left, right, alpha):
    return [
        float(a) + float(alpha) * float(b) for a, b in zip(left, right)
    ]


def tune_consensus(metas, state, paired_scores, bilinear_scores):
    candidates = []
    curves = {}
    for alpha in ALPHAS:
        signal = combine(paired_scores, bilinear_scores, alpha)
        curves[str(alpha)] = {}
        for weight in WEIGHTS:
            outcomes, _ = select_policy(
                metas,
                state,
                signal,
                weight,
                routed=True,
                explicit=True,
            )
            accuracy = sum(outcomes.values()) / len(outcomes)
            curves[str(alpha)][str(weight)] = accuracy
            candidates.append((accuracy, -alpha, -weight, alpha, weight))
    _, _, _, alpha, weight = max(candidates)
    return alpha, weight, curves


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

    pair_dev, pair_held, pair_reader = paired_oof_and_transfer(
        dev_payload, held_payload, PAIRED_SEED
    )
    bil_dev, bil_held, bil_reader = bilinear_oof_and_transfer(
        dev_payload,
        held_payload,
        BILINEAR_SEED,
        args.projected_dim,
    )
    alpha, consensus_weight, consensus_curves = tune_consensus(
        dev_metas, dev_state, pair_dev, bil_dev
    )
    paired_weight, paired_curve = tune_weight(
        dev_metas, dev_state, pair_dev
    )
    bilinear_weight, bilinear_curve = tune_weight(
        dev_metas, dev_state, bil_dev
    )
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
        "transition_consensus": combine(
            pair_held["paired"], bil_held["bilinear"], alpha
        ),
        "sign_swap_control": combine(
            pair_held["sign_swap"], bil_held["sign_swap"], alpha
        ),
        "mismatch_control": combine(
            pair_held["mismatch"], bil_held["mismatch"], alpha
        ),
        "paired_only": pair_held["paired"],
        "bilinear_only": bil_held["bilinear"],
        "length_control": [len(meta["norm"] or "") for meta in held_metas],
    }
    weights = {
        "transition_consensus": consensus_weight,
        "sign_swap_control": consensus_weight,
        "mismatch_control": consensus_weight,
        "paired_only": paired_weight,
        "bilinear_only": bilinear_weight,
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
        "experiment": "HSGR two-channel hidden transition consensus",
        "claim_boundary": (
            "Mechanism development on an already observed second fresh set; "
            "not a confirmatory claim. Uses oracle decomposition, verified "
            "predecessor values, and gold support routing."
        ),
        "n_problems": len(held_state),
        "n_candidates": len(held_metas),
        "reader": {
            "paired": pair_reader,
            "bilinear": bil_reader,
        },
        "dev_selection": {
            "alpha_candidates": list(ALPHAS),
            "selected_alpha": alpha,
            "selected_consensus_weight": consensus_weight,
            "consensus_curves": consensus_curves,
            "paired_weight": paired_weight,
            "paired_curve": paired_curve,
            "bilinear_weight": bilinear_weight,
            "bilinear_curve": bilinear_curve,
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

    primary = report["policy"]["transition_consensus"]
    vs_explicit = primary["vs_explicit"]
    fixes, breaks = paired_counts(vs_explicit)
    report["diagnostic_gates"] = {
        "uses_relation_channel": alpha > 0.0,
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
        "interaction_gain": (
            primary["accuracy"]
            >= report["policy"]["paired_only"]["accuracy"] + 0.005
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
    cases_path = os.path.join(args.out_dir, "transition_consensus_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in sorted(held_state):
            primary_hit = outcomes["transition_consensus"][pid]
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
                        "transition_consensus": answers[
                            "transition_consensus"
                        ][pid],
                        "consensus_fixes_explicit": bool(
                            primary_hit and not explicit_hit
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report_path = os.path.join(
        args.out_dir, "transition_consensus_diagnostic_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--heldout-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_transition_consensus")
    parser.add_argument("--projected-dim", type=int, default=PROJECTED_DIM)
    parser.add_argument("--threads", type=int, default=16)
    main(parser.parse_args())

