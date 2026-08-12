"""Oracle-structure root-cause audit for the route residual.

This diagnostic adapts the three already consumed oracle-route feature caches
to the parameter-matched nested-OOF evaluator.  It is deliberately not an
end-to-end or confirmatory experiment: gold decomposition, verified
predecessors, and gold support routing were used when the caches were made.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from hsgr_copy_guard_probe import dependency_state
from hsgr_listwise_guide_verifier import (
    concat_payloads,
    make_data as make_oracle_data,
)
from hsgr_route_augmented_eval import run_nested
from hsgr_route_counterfactual_eval import (
    SEED,
    by_hop,
    hash_half,
    holm_adjust,
    metric,
)
from hsgr_structured_hidden_verifier import LAYERS
from mh_e0 import load_rows


EXPECTED_PROBLEMS = 840
EXPECTED_CANDIDATES = 6720


def adapt_data(torch, payload, state, hop_by_pid):
    """Map oracle correct/wrong route caches to the v2 evaluator schema."""
    source = make_oracle_data(torch, payload, state, hop_by_pid)
    hop_column = torch.tensor(
        [[min(4, hop_by_pid[pid]) / 4.0] for pid in source["pids"]],
        dtype=torch.float32,
    )
    # The old cache has no stable graph-degree field.  A zero column preserves
    # the frozen seven-scalar architecture without giving either model extra
    # information.  Explicit selection uses columns 1 and 2, as before.
    degree_column = torch.zeros((len(source["pids"]), 1), dtype=torch.float32)
    scalar = torch.cat([source["scalar"], hop_column, degree_column], dim=1)
    rotation = source["rotation"]
    return {
        "metas": source["metas"],
        "pids": source["pids"],
        "labels": source["labels"],
        "scalar": scalar,
        "features": {
            "matched": {
                layer: source["features"]["correct"][layer]
                for layer in LAYERS
            },
            "counterfactual": {
                layer: source["features"]["wrong"][layer]
                for layer in LAYERS
            },
            "mismatch": {
                layer: source["features"]["wrong"][layer][rotation]
                for layer in LAYERS
            },
            # Required only because the shared evaluator also instantiates its
            # old activation control.  That output is invalid for this cache
            # and is excluded from every comparison and gate.
            "matched_start": {
                layer: source["features"]["correct"][layer]
                for layer in LAYERS
            },
        },
        "gold_hop": dict(hop_by_pid),
    }


def validate_adapted(data):
    pids = set(data["pids"])
    if len(pids) != EXPECTED_PROBLEMS:
        raise RuntimeError(f"expected {EXPECTED_PROBLEMS} problems, got {len(pids)}")
    if len(data["metas"]) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"expected {EXPECTED_CANDIDATES} candidates, got {len(data['metas'])}"
        )
    if tuple(data["scalar"].shape) != (EXPECTED_CANDIDATES, 7):
        raise RuntimeError(f"invalid scalar shape {tuple(data['scalar'].shape)}")
    if set(Counter(data["pids"]).values()) != {8}:
        counts = Counter(data["pids"])
        bad = {pid: count for pid, count in counts.items() if count != 8}
        raise RuntimeError(f"expected SC@8 candidates per problem; bad={bad}")
    for view in ("matched", "counterfactual", "mismatch"):
        for layer in LAYERS:
            tensor = data["features"][view][layer]
            if tuple(tensor.shape) != (EXPECTED_CANDIDATES, 256):
                raise RuntimeError(
                    f"invalid {view}/L{layer} shape {tuple(tensor.shape)}"
                )


def main(args):
    import torch

    torch.set_num_threads(args.threads)
    payload = concat_payloads(torch, args.features)
    all_rows = load_rows(args.data, 0, seed=0)
    rows_by_id = {row["_uid"]: row for row in all_rows}
    ids = {meta["id"] for meta in payload["metas"]}
    missing = ids - set(rows_by_id)
    if missing:
        raise RuntimeError(f"missing {len(missing)} feature IDs from data")
    rows = [rows_by_id[pid] for pid in ids]
    state = dependency_state(rows)
    hop_by_pid = {
        pid: len(rows_by_id[pid]["question_decomposition"])
        for pid in ids
    }
    data = adapt_data(torch, payload, state, hop_by_pid)
    validate_adapted(data)

    outcomes, folds, parameters = run_nested(torch, data, args.seed)
    primary = outcomes["route_augmented"]
    comparison_names = (
        "sc8",
        "explicit_predicted_state",
        "ordinary_wide",
        "route_swap",
        "route_mismatch",
    )
    comparisons = {
        name: metric(primary, outcomes[name], args.seed + index)
        for index, name in enumerate(comparison_names)
    }
    adjusted = holm_adjust(
        {
            name: comparisons[name]["paired"]["p"]
            for name in (
                "sc8",
                "explicit_predicted_state",
                "ordinary_wide",
            )
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    hop_report = by_hop(
        {
            name: outcomes[name]
            for name in (
                "sc8",
                "explicit_predicted_state",
                "route_augmented",
                "ordinary_wide",
                "route_swap",
                "route_mismatch",
            )
        },
        hop_by_pid,
    )
    depth_delta = {
        hop: values["route_augmented"] - values["ordinary_wide"]
        for hop, values in hop_report.items()
        if hop in ("2", "3", "4")
    }
    half_delta = {}
    for half in (0, 1):
        half_ids = [pid for pid in primary if hash_half(pid) == half]
        half_delta[str(half)] = (
            sum(primary[pid] for pid in half_ids)
            - sum(outcomes["ordinary_wide"][pid] for pid in half_ids)
        ) / len(half_ids)

    gates = {
        "beyond_parameter_matched_ordinary": (
            comparisons["ordinary_wide"]["delta"] >= 0.01
            and comparisons["ordinary_wide"]["holm_adjusted_p"] < 0.05
        ),
        "selection_value": (
            comparisons["sc8"]["delta"] >= 0.02
            and comparisons["sc8"]["holm_adjusted_p"] < 0.05
        ),
        "beyond_explicit_oracle_route": (
            comparisons["explicit_predicted_state"]["delta"] >= 0.01
            and comparisons["explicit_predicted_state"]["holm_adjusted_p"]
            < 0.05
        ),
        "route_controls": (
            comparisons["route_swap"]["delta"] >= 0.02
            and comparisons["route_swap"]["paired"]["p"] < 0.05
            and comparisons["route_mismatch"]["delta"] >= 0.01
            and comparisons["route_mismatch"]["paired"]["p"] < 0.05
        ),
        "oof_stability_vs_ordinary": (
            sum(row["test_delta_vs_ordinary"] > 0 for row in folds) >= 4
            and all(value > 0 for value in half_delta.values())
        ),
        "depth_signature_vs_ordinary": (
            set(depth_delta) == {"2", "3", "4"}
            and min(depth_delta.values()) >= 0.0
            and depth_delta["4"] >= depth_delta["2"] - 0.01
        ),
        "feature_and_capacity_validity": (
            len(ids) == EXPECTED_PROBLEMS
            and len(data["metas"]) == EXPECTED_CANDIDATES
            and parameters["ordinary_wide"] >= parameters["route_augmented"]
        ),
    }
    if not gates["beyond_parameter_matched_ordinary"]:
        decision = (
            "ORACLE ROUTE RESIDUAL FAILS: DO NOT INVEST IN A STRUCTURE "
            "PREDICTOR FOR THIS SELECTION MECHANISM"
        )
    elif all(gates.values()):
        decision = (
            "ORACLE ROUTE RESIDUAL HAS MECHANISM HEADROOM; DEVELOPMENT "
            "DIAGNOSTIC ONLY; FINAL377 REMAINS SEALED"
        )
    else:
        decision = "ORACLE ROUTE RESIDUAL INCONCLUSIVE; FINAL377 REMAINS SEALED"

    selected_outcomes = {
        name: outcomes[name]
        for name in (
            "sc8",
            "explicit_predicted_state",
            "route_augmented",
            "ordinary_wide",
            "route_swap",
            "route_mismatch",
        )
    }
    report = {
        "experiment": "Oracle-structure absolute hidden plus route residual",
        "protocol": "EXPERIMENT_PROTOCOL_ORACLE_ROUTE_RESIDUAL_DIAGNOSTIC.md",
        "claim_boundary": (
            "Root-cause diagnostic using oracle decomposition, verified "
            "predecessors, and gold support routing on consumed data only."
        ),
        "data": {
            "problems": len(ids),
            "candidates": len(data["metas"]),
            "feature_paths": list(args.features),
            "hop_counts": dict(sorted(Counter(hop_by_pid.values()).items())),
        },
        "parameter_counts": {
            "route_augmented": parameters["route_augmented"],
            "ordinary_wide": parameters["ordinary_wide"],
        },
        "activation_delta_control": (
            "unavailable: oracle caches do not contain same-prompt start states"
        ),
        "nested_oof": {
            "accuracy": {
                name: sum(values.values()) / len(values)
                for name, values in selected_outcomes.items()
            },
            "comparisons": comparisons,
            "folds": folds,
            "gold_hop_stratification": hop_report,
            "depth_delta_vs_ordinary": depth_delta,
            "id_hash_half_delta_vs_ordinary": half_delta,
        },
        "gates": gates,
        "decision": decision,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "oracle_route_augmented_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", nargs=3, required=True)
    parser.add_argument("--out-dir", default="hsgr_oracle_route_augmented_eval")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED + 404)
    main(parser.parse_args())
