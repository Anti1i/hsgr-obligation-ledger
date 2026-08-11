"""Second fresh confirmatory test for the HSGR hidden TransitionGuard.

Frozen before candidate generation:

* SC@8 uses the original full-evidence MuSiQue prompt;
* routed support and exact predecessor guard weights are both 0.75;
* TransitionGuard = correct-support - predecessor-support - copy-risk;
* TransitionGuard weight is 0.30;
* all hidden readers use structural labels and the original n=200 dev split.

The experiment still uses oracle decomposition, verified predecessor values,
and gold support routing.  It is a mechanism test, not an end-to-end method.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter

from hsgr_contrastive_guide_heldout import generate_sc8, sha_ids
from hsgr_copy_guard_probe import dependency_state, metric
from hsgr_dual_route_guide_diagnostic import (
    SEED as READER_SEED,
    reader_oof_and_transfer,
    select_policy,
)
from hsgr_dynamic_route_read import select_new_units
from hsgr_focus_route_ceiling import unit_id
from hsgr_structured_hidden_verifier import (
    LAYERS,
    SYSTEM as VERIFY_SYSTEM,
    build_units,
    projectors,
)
from mh_e0 import load_rows
from pilot import Runner


FIRST_CALIB_SHA = "47c7da894480c9405eec1b483dfec6d84d7425a0685cd13e5ae447d77ebc6c5e"
FIRST_TEST_SHA = "3262ab2b75860743be0039aef399aad029c82b79d8f7998b154880934c3a1921"
EXPECTED_SECOND_SHA = None  # Frozen after the selection-only dry run.
SELECTION_SEED = 20260819
TRANSITION_WEIGHT = 0.30
ROUTE_DELTA_WEIGHT = 0.20
COPY_WEIGHT = 0.15
LENGTH_WEIGHT = 0.20


def select_second_rows(args):
    first_calib, first_test, first_meta = select_new_units(
        data=args.data,
        exclude_cases=[args.edge_cases, args.focus_cases],
        original_limit=200,
        calib_n=80,
        test_n=320,
        seed=20260814,
    )
    if sha_ids([unit["id"] for unit in first_calib]) != FIRST_CALIB_SHA:
        raise RuntimeError("first held-out calibration reconstruction mismatch")
    if sha_ids([unit["id"] for unit in first_test]) != FIRST_TEST_SHA:
        raise RuntimeError("first held-out test reconstruction mismatch")
    _, full_pool, pool_meta = select_new_units(
        data=args.data,
        exclude_cases=[args.edge_cases, args.focus_cases],
        original_limit=200,
        calib_n=0,
        test_n=first_meta["untouched_pool"],
        seed=0,
    )
    consumed = {unit["id"] for unit in first_calib + first_test}
    remaining = sorted(
        (unit for unit in full_pool if unit["id"] not in consumed),
        key=lambda unit: unit["id"],
    )
    if len(remaining) != 697:
        raise RuntimeError(f"expected 697 second-stage untouched units, got {len(remaining)}")
    chosen = random.Random(SELECTION_SEED).sample(remaining, 320)
    ids = [unit["id"] for unit in chosen]
    digest = sha_ids(ids)
    if EXPECTED_SECOND_SHA is None and not args.dry_run:
        raise RuntimeError("second held-out hash has not been frozen")
    if EXPECTED_SECOND_SHA is not None and digest != EXPECTED_SECOND_SHA:
        raise RuntimeError(f"second held-out hash mismatch: {digest}")

    all_rows = load_rows(args.data, 0, seed=0)
    by_id = {unit_id(row): row for row in all_rows}
    rows = [by_id[uid] for uid in ids]
    selection = {
        "base_untouched_pool": first_meta["untouched_pool"],
        "first_consumed": len(consumed),
        "first_calib_sha256": FIRST_CALIB_SHA,
        "first_test_sha256": FIRST_TEST_SHA,
        "remaining_before_second": len(remaining),
        "remaining_after_second": len(remaining) - len(chosen),
        "selection_seed": SELECTION_SEED,
        "n": len(chosen),
        "id_sha256": digest,
        "hop_counts": dict(Counter(unit["n_hops"] for unit in chosen)),
        "base_pool_reconstruction": pool_meta,
    }
    return rows, selection


def extract_route_features(runner, metas, route, batch_size, max_context):
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
                    {"role": "user", "content": meta["prompts"][route]},
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
                f"[hidden-{route}] {min(start + batch_size, len(metas))}/{len(metas)}",
                flush=True,
            )
    return {layer: torch.cat(parts) for layer, parts in features.items()}


def difference(left, right, third=None):
    if third is None:
        return [float(a) - float(b) for a, b in zip(left, right)]
    return [
        float(a) - float(b) - float(c)
        for a, b, c in zip(left, right, third)
    ]


def reader_labels(metas, state):
    labels = {
        "support_correct": [meta["mentions"]["correct"] for meta in metas],
        "support_wrong": [meta["mentions"]["wrong"] for meta in metas],
        "copy": [
            int(bool(meta["norm"]) and meta["norm"] in state[meta["id"]]["values"])
            for meta in metas
        ],
    }
    labels["copy_correct_control"] = list(labels["copy"])
    return labels


def exact_paired_fields(metric_value):
    paired = metric_value["paired"]
    return paired["a_only"], paired["b_only"]


def main(args):
    rows, selection = select_second_rows(args)
    ids = [row["_uid"] for row in rows]
    print("[selection] " + json.dumps(selection, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "first_id": ids[0],
                    "first_question": rows[0]["question"],
                    "expected_hash_frozen": EXPECTED_SECOND_SHA,
                },
                ensure_ascii=False,
            )
        )
        return

    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = Runner(args.model)
    sc_path = os.path.join(args.out_dir, "transition_fresh_sc8.jsonl")
    sc_rows = generate_sc8(
        runner, rows, sc_path, args.batch_problems, args.max_new
    )
    sc_by_id = {row["id"]: row for row in sc_rows}
    metas = build_units(rows, sc_by_id)
    if len(metas) != 2560:
        raise RuntimeError(f"expected 2560 candidate units, got {len(metas)}")

    held_features = {
        route: extract_route_features(
            runner, metas, route, args.bs_hidden, args.max_context
        )
        for route in ("correct", "wrong")
    }
    del runner.model
    torch.cuda.empty_cache()
    feature_path = os.path.join(args.out_dir, "transition_fresh_features.pt")
    torch.save({"features": held_features, "metas": metas}, feature_path)

    dev_payload = torch.load(args.dev_features, map_location="cpu")
    dev_metas = dev_payload["metas"]
    dev_rows = load_rows(args.data, 200, seed=0)
    dev_state = dependency_state(dev_rows)
    held_state = dependency_state(rows)
    dev_labels = reader_labels(dev_metas, dev_state)
    held_labels = reader_labels(metas, held_state)

    reader_specs = {
        "support_correct": ("correct", READER_SEED + 100),
        "support_wrong": ("wrong", READER_SEED + 200),
        "copy": ("wrong", READER_SEED + 300),
        "copy_correct_control": ("correct", READER_SEED + 400),
    }
    dev_scores = {}
    held_scores = {}
    reader_report = {}
    for name, (route, seed) in reader_specs.items():
        print(f"[reader] {name} route={route}", flush=True)
        dev_scores[name], held_scores[name], reader_report[name] = (
            reader_oof_and_transfer(
                dev_payload,
                route,
                dev_labels[name],
                held_features[route],
                metas,
                held_labels[name],
                seed,
            )
        )

    signals = {
        "transition_guard": difference(
            held_scores["support_correct"],
            held_scores["support_wrong"],
            held_scores["copy"],
        ),
        "transition_swap": difference(
            held_scores["support_wrong"],
            held_scores["support_correct"],
            held_scores["copy"],
        ),
        "route_delta": difference(
            held_scores["support_correct"], held_scores["support_wrong"]
        ),
        "copy_hidden": [-score for score in held_scores["copy"]],
        "copy_correct_control": [
            -score for score in held_scores["copy_correct_control"]
        ],
        "length_only": [len(meta["norm"] or "") for meta in metas],
    }
    weights = {
        "transition_guard": TRANSITION_WEIGHT,
        "transition_swap": TRANSITION_WEIGHT,
        "route_delta": ROUTE_DELTA_WEIGHT,
        "copy_hidden": COPY_WEIGHT,
        "copy_correct_control": 0.20,
        "length_only": LENGTH_WEIGHT,
    }

    sc8, sc_answers = select_policy(
        metas, held_state, None, 0.0, routed=False, explicit=False
    )
    lexical, lexical_answers = select_policy(
        metas, held_state, None, 0.0, routed=True, explicit=False
    )
    explicit, explicit_answers = select_policy(
        metas, held_state, None, 0.0, routed=True, explicit=True
    )
    policies = {}
    answers = {}
    for name, signal in signals.items():
        policies[name], answers[name] = select_policy(
            metas, held_state, signal, weights[name], routed=True, explicit=True
        )

    report = {
        "experiment": "HSGR TransitionGuard second fresh held-out",
        "claim_boundary": (
            "Uses oracle decomposition, verified predecessor values, and gold "
            "support routing; confirmatory mechanism test, not end-to-end use."
        ),
        "selection": selection,
        "fixed_weights": {
            "support": 0.75,
            "exact_guard": 0.75,
            **weights,
        },
        "reader": reader_report,
        "baseline": {
            "sc8": sum(sc8.values()) / len(sc8),
            "oracle8": sum(row["oracle"] for row in sc_rows) / len(sc_rows),
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
    row_by_id = {row["_uid"]: row for row in rows}
    for hop in (2, 3, 4):
        hop_ids = [
            pid
            for pid in ids
            if len(row_by_id[pid]["question_decomposition"]) == hop
        ]
        report["by_hop"][str(hop)] = {
            "n": len(hop_ids),
            "sc8": sum(sc8[pid] for pid in hop_ids) / len(hop_ids),
            "explicit": sum(explicit[pid] for pid in hop_ids) / len(hop_ids),
            **{
                name: sum(outcomes[pid] for pid in hop_ids) / len(hop_ids)
                for name, outcomes in policies.items()
            },
        }

    transition = report["policy"]["transition_guard"]
    vs_explicit = transition["vs_explicit"]
    fixes, breaks = exact_paired_fields(vs_explicit)
    depth_delta = {
        hop: values["transition_guard"] - values["sc8"]
        for hop, values in report["by_hop"].items()
    }
    report["transition_depth_delta"] = depth_delta
    copy_wrong = report["policy"]["copy_hidden"]["accuracy"]
    copy_correct = report["policy"]["copy_correct_control"]["accuracy"]
    report["gates"] = {
        "reader_transfer": (
            reader_report["support_correct"]["heldout_within_auroc"] >= 0.75
            and reader_report["support_wrong"]["heldout_within_auroc"] >= 0.75
            and reader_report["copy"]["heldout_within_auroc"] >= 0.80
        ),
        "transition_headroom": (
            transition["delta"] >= 0.06
            and transition["paired"]["p"] < 0.05
        ),
        "safe_net_vs_explicit": (
            vs_explicit["delta"] >= 0.0
            and breaks <= 8
            and fixes >= breaks + 4
        ),
        "beyond_length": (
            transition["accuracy"]
            >= report["policy"]["length_only"]["accuracy"] + 0.01
        ),
        "predecessor_route_specificity": copy_wrong >= copy_correct + 0.01,
        "transition_directionality": (
            transition["accuracy"]
            >= report["policy"]["transition_swap"]["accuracy"] + 0.02
        ),
        "depth_signature": (
            min(depth_delta.values()) >= 0.0
            and depth_delta["4"] >= depth_delta["2"] - 0.01
        ),
    }
    report["decision"] = "PASS" if all(report["gates"].values()) else "GATE FAIL"

    report_path = os.path.join(args.out_dir, "transition_fresh_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    cases_path = os.path.join(args.out_dir, "transition_fresh_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in ids:
            handle.write(
                json.dumps(
                    {
                        "id": pid,
                        "n_hops": len(row_by_id[pid]["question_decomposition"]),
                        "gold": row_by_id[pid]["answer"],
                        "selected": {
                            "sc8": {"answer": sc_answers[pid], "hit": sc8[pid]},
                            "lexical": {
                                "answer": lexical_answers[pid],
                                "hit": lexical[pid],
                            },
                            "explicit": {
                                "answer": explicit_answers[pid],
                                "hit": explicit[pid],
                            },
                            **{
                                name: {
                                    "answer": answers[name][pid],
                                    "hit": policies[name][pid],
                                }
                                for name in policies
                            },
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
    parser.add_argument("--edge-cases", required=True)
    parser.add_argument("--focus-cases", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_transition_fresh")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--batch-problems", type=int, default=16)
    parser.add_argument("--bs-hidden", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
