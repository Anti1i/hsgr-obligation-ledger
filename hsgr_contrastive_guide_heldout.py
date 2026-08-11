"""Fresh held-out test for the HSGR Contrastive predecessor-copy Guide.

Development choices are frozen before this script touches the reserved n=320:

* SC@8 candidates use the original full-evidence MuSiQue prompt;
* lexical support weight = 0.75;
* exact predecessor-copy penalty = 0.75;
* hidden copy-risk penalty = 0.20;
* the hidden reader uses predecessor-context structured prompts and structural
  copy supervision only (candidate equals a verified predecessor value).

The experiment still uses oracle decomposition, predecessor values, and
support routing.  It is a held-out mechanism test, not an end-to-end method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

from hsgr_copy_guard_probe import (
    EXACT_GUARD_WEIGHT,
    SUPPORT_WEIGHT,
    dependency_state,
    metric,
    problem_scores,
)
from hsgr_dynamic_route_read import select_new_units
from hsgr_focus_route_ceiling import exact_mcnemar, unit_id
from hsgr_structured_hidden_verifier import (
    LAYERS,
    PROJECTION_DIM,
    SYSTEM as VERIFY_SYSTEM,
    build_units,
    projectors,
)
from mh_ceiling import (
    SYSTEM as SC_SYSTEM,
    USER as SC_USER,
    answers_match,
    evidence_from_row,
    extract_boxed,
    normalize,
)
from mh_e0 import hop_deps, load_rows
from mh_latent_rerank import auroc, fit_probe, within_problem_auroc
from pilot import Runner


EXPECTED_HELDOUT_SHA = (
    "3262ab2b75860743be0039aef399aad029c82b79d8f7998b154880934c3a1921"
)
HIDDEN_PENALTY = 0.20
DEV_SEED = 20260816


def sha_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def select_reserved_rows(args):
    _, test_units, selection = select_new_units(
        data=args.data,
        exclude_cases=[args.edge_cases, args.focus_cases],
        original_limit=200,
        calib_n=80,
        test_n=320,
        seed=20260814,
    )
    ids = [unit["id"] for unit in test_units]
    digest = sha_ids(ids)
    if digest != EXPECTED_HELDOUT_SHA:
        raise RuntimeError(f"reserved held-out hash mismatch: {digest}")
    all_rows = load_rows(args.data, 0, seed=0)
    by_id = {unit_id(row): row for row in all_rows}
    rows = [by_id[uid] for uid in ids]
    if len(rows) != 320:
        raise RuntimeError(f"expected 320 held-out rows, found {len(rows)}")
    return rows, selection


def sc_row(row, texts):
    candidates = []
    for text in texts:
        answer = extract_boxed(text)
        candidates.append({
            "ans": answer,
            "norm": normalize(answer) if answer else None,
            "text": text[:500],
        })
    gold = str(row["answer"])
    aliases = list(row.get("answer_aliases") or [])
    votes = Counter(cand["norm"] for cand in candidates if cand["norm"])
    top = votes.most_common(1)[0][0] if votes else None
    return {
        "id": row["_uid"],
        "gold": gold,
        "aliases": aliases,
        "n_hops": len(row["question_decomposition"]),
        "hit1": bool(
            candidates[0]["ans"]
            and answers_match(candidates[0]["ans"], gold, aliases)
        ),
        "sc": bool(top and answers_match(top, gold, aliases)),
        "oracle": any(
            cand["ans"] and answers_match(cand["ans"], gold, aliases)
            for cand in candidates
        ),
        "cands": candidates,
    }


def generate_sc8(runner, rows, out_path, batch_problems, max_new):
    completed = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as handle:
            completed = {
                row["id"]: row
                for row in (json.loads(line) for line in handle if line.strip())
            }
    ordered = []
    for start in range(0, len(rows), batch_problems):
        chunk = rows[start : start + batch_problems]
        missing = [row for row in chunk if row["_uid"] not in completed]
        if missing:
            prompts = [
                SC_USER.format(
                    evidence=evidence_from_row(row), question=row["question"]
                )
                for row in missing
            ]
            outputs = runner.chat_batch(
                prompts,
                system=SC_SYSTEM,
                max_new=max_new,
                temperature=0.7,
                n=8,
                bs=8,
            )
            for row, texts in zip(missing, outputs):
                completed[row["_uid"]] = sc_row(row, texts)
            with open(out_path, "w", encoding="utf-8") as handle:
                for source in rows:
                    if source["_uid"] in completed:
                        handle.write(
                            json.dumps(
                                completed[source["_uid"]], ensure_ascii=False
                            )
                            + "\n"
                        )
        ordered.extend(completed[row["_uid"]] for row in chunk)
        print(
            f"[sc8] {min(start + batch_problems, len(rows))}/{len(rows)}",
            flush=True,
        )
    if len(ordered) != len(rows):
        raise RuntimeError("incomplete held-out SC generation")
    return ordered


def extract_heldout_features(runner, units, batch_size, max_context):
    torch = runner.torch
    tokenizer = runner.tok
    matrices = projectors(torch, runner.model.config.hidden_size, "cuda")
    features = {layer: [] for layer in LAYERS}
    for start in range(0, len(units), batch_size):
        batch = units[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": unit["prompts"]["wrong"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for unit in batch
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
                f"[hidden] {min(start + batch_size, len(units))}/{len(units)}",
                flush=True,
            )
    return {layer: torch.cat(parts) for layer, parts in features.items()}


def dev_copy_labels(dev_metas, dev_state):
    return [
        int(
            bool(meta["norm"])
            and meta["norm"] in dev_state[meta["id"]]["values"]
        )
        for meta in dev_metas
    ]


def final_reader_predictions(
    dev_payload,
    dev_state,
    heldout_features,
    heldout_metas,
):
    import torch

    dev_metas = dev_payload["metas"]
    labels = torch.tensor(
        dev_copy_labels(dev_metas, dev_state), dtype=torch.float32
    )
    pids = sorted({meta["id"] for meta in dev_metas})
    rng = random.Random(DEV_SEED)
    rng.shuffle(pids)
    folds = [pids[index::5] for index in range(5)]
    model_predictions = []
    selected_layers = {}
    for fold_index, hold in enumerate(folds):
        hold_set = set(hold)
        train_pids = [pid for pid in pids if pid not in hold_set]
        candidates = []
        for layer in LAYERS:
            layer_rng = random.Random(
                DEV_SEED + 5000 + 31 * fold_index + layer
            )
            shuffled = list(train_pids)
            layer_rng.shuffle(shuffled)
            n_val = max(1, len(shuffled) // 7)
            val_set = set(shuffled[:n_val])
            train_idx = [
                i for i, meta in enumerate(dev_metas)
                if meta["id"] not in hold_set
                and meta["id"] not in val_set
            ]
            val_idx = [
                i for i, meta in enumerate(dev_metas)
                if meta["id"] in val_set
            ]
            X = dev_payload["features"]["wrong"][layer].float()
            scorer, criterion = fit_probe(
                X[train_idx],
                labels[train_idx],
                [dev_metas[i]["id"] for i in train_idx],
                X[val_idx],
                labels[val_idx],
                [dev_metas[i]["id"] for i in val_idx],
                use_rank=True,
                epochs=250,
                rank_w=1.0,
                seed=DEV_SEED + 5000 + fold_index,
            )
            candidates.append(
                (criterion, layer, scorer(heldout_features[layer].float()))
            )
        criterion, layer, prediction = max(candidates, key=lambda item: item[0])
        selected_layers[fold_index] = {
            "layer": layer,
            "internal_criterion": criterion,
        }
        # Normalize each ensemble member within held-out problem before
        # averaging, so arbitrary probe scale cannot dominate another fold.
        by_problem = defaultdict(list)
        for index, meta in enumerate(heldout_metas):
            by_problem[meta["id"]].append(index)
        normalized = [0.0] * len(prediction)
        for indices in by_problem.values():
            values = [prediction[index] for index in indices]
            center = sum(values) / len(values)
            variance = sum((value - center) ** 2 for value in values) / max(
                1, len(values) - 1
            )
            scale = math.sqrt(variance) if variance > 0 else 1.0
            for index in indices:
                normalized[index] = (prediction[index] - center) / scale
        model_predictions.append(normalized)
    ensemble = [
        sum(prediction[index] for prediction in model_predictions)
        / len(model_predictions)
        for index in range(len(heldout_metas))
    ]
    return ensemble, selected_layers


def nondependency_state(rows):
    state = {}
    for row in rows:
        decomp = row["question_decomposition"]
        deps = set(hop_deps(decomp)[-1])
        state[row["_uid"]] = {
            "values": {
                normalize(str(decomp[index]["answer"]))
                for index in range(len(decomp) - 1)
                if index not in deps
            },
            "n_hops": len(decomp),
        }
    return state


def main(args):
    import torch

    heldout_rows, selection = select_reserved_rows(args)
    heldout_ids = [row["_uid"] for row in heldout_rows]
    print(
        "[heldout] "
        + json.dumps(
            {
                "n": len(heldout_rows),
                "id_sha256": sha_ids(heldout_ids),
                "hop_counts": dict(
                    Counter(
                        len(row["question_decomposition"])
                        for row in heldout_rows
                    )
                ),
                "selection": selection,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "first_id": heldout_rows[0]["_uid"],
                    "first_question": heldout_rows[0]["question"],
                },
                ensure_ascii=False,
            )
        )
        return

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = Runner(args.model)
    sc_path = os.path.join(args.out_dir, "heldout_sc8.jsonl")
    sc_rows = generate_sc8(
        runner,
        heldout_rows,
        sc_path,
        args.batch_problems,
        args.max_new,
    )
    sc_by_id = {row["id"]: row for row in sc_rows}
    heldout_units = build_units(heldout_rows, sc_by_id)
    if len(heldout_units) != 320 * 8:
        raise RuntimeError(
            f"expected 2560 held-out candidate units, found {len(heldout_units)}"
        )
    heldout_features = extract_heldout_features(
        runner, heldout_units, args.bs_hidden, args.max_context
    )
    feature_path = os.path.join(args.out_dir, "heldout_hidden_features.pt")
    torch.save(
        {"features": heldout_features, "metas": heldout_units}, feature_path
    )

    dev_payload = torch.load(args.dev_features, map_location="cpu")
    dev_rows = load_rows(args.data, 200, seed=0)
    dev_state = dependency_state(dev_rows)
    hidden_scores, selected_layers = final_reader_predictions(
        dev_payload,
        dev_state,
        heldout_features,
        heldout_units,
    )
    heldout_state = dependency_state(heldout_rows)
    copy_labels = [
        int(
            bool(meta["norm"])
            and meta["norm"] in heldout_state[meta["id"]]["values"]
        )
        for meta in heldout_units
    ]
    copy_pooled = auroc(hidden_scores, copy_labels)
    copy_within, copy_within_n = within_problem_auroc(
        hidden_scores,
        copy_labels,
        [meta["id"] for meta in heldout_units],
    )

    zeros = [0.0] * len(heldout_units)
    lexical, lexical_answers = problem_scores(
        heldout_units, zeros, heldout_state, False, 0.0
    )
    explicit, explicit_answers = problem_scores(
        heldout_units, zeros, heldout_state, True, 0.0
    )
    hidden, hidden_answers = problem_scores(
        heldout_units, hidden_scores, heldout_state, False, HIDDEN_PENALTY
    )
    hybrid, hybrid_answers = problem_scores(
        heldout_units, hidden_scores, heldout_state, True, HIDDEN_PENALTY
    )
    sc_metas = [
        dict(meta, mentions={"correct": 0, "wrong": 0})
        for meta in heldout_units
    ]
    sc8, sc_answers = problem_scores(
        sc_metas, zeros, heldout_state, False, 0.0
    )
    nondep_state = nondependency_state(heldout_rows)
    nondep, _ = problem_scores(
        heldout_units, zeros, nondep_state, True, 0.0
    )

    policies = {
        "lexical": lexical,
        "explicit": explicit,
        "hidden": hidden,
        "hybrid": hybrid,
        "nondep_control": nondep,
    }
    report = {
        "experiment": "HSGR Contrastive Guide fresh held-out",
        "claim_boundary": (
            "Uses oracle decomposition, verified predecessor values, and gold "
            "support routing; establishes a held-out mechanism, not end-to-end use."
        ),
        "data": {
            "n": len(heldout_rows),
            "id_sha256": sha_ids(heldout_ids),
            "hop_counts": dict(
                Counter(
                    len(row["question_decomposition"])
                    for row in heldout_rows
                )
            ),
        },
        "fixed_weights": {
            "support": SUPPORT_WEIGHT,
            "exact_guard": EXACT_GUARD_WEIGHT,
            "hidden_guard": HIDDEN_PENALTY,
        },
        "sc8": sum(sc8.values()) / len(sc8),
        "oracle8": sum(row["oracle"] for row in sc_rows) / len(sc_rows),
        "copy_reader": {
            "positive_candidates": sum(copy_labels),
            "pooled_auroc": copy_pooled,
            "within_problem_auroc": copy_within,
            "within_problem_n": copy_within_n,
            "ensemble_layers": selected_layers,
        },
        "policy": {
            name: {
                **metric(hits, sc8),
                "vs_explicit": metric(hits, explicit),
            }
            for name, hits in policies.items()
        },
        "by_hop": {},
        "token_counts": {
            "generated": runner.n_new_tokens,
        },
    }
    row_by_id = {row["_uid"]: row for row in heldout_rows}
    for hop in (2, 3, 4):
        ids = [
            pid
            for pid in heldout_ids
            if len(row_by_id[pid]["question_decomposition"]) == hop
        ]
        if ids:
            report["by_hop"][str(hop)] = {
                "n": len(ids),
                **{
                    name: sum(hits[pid] for pid in ids) / len(ids)
                    for name, hits in {"sc8": sc8, **policies}.items()
                },
            }
    hybrid_depth_delta = {
        hop: values["hybrid"] - values["sc8"]
        for hop, values in report["by_hop"].items()
    }
    report["hybrid_depth_delta"] = hybrid_depth_delta
    report["gates"] = {
        "copy_reader_generalizes": (
            copy_pooled >= 0.90 and copy_within >= 0.80
        ),
        "explicit_guide": (
            report["policy"]["explicit"]["delta"] >= 0.06
            and report["policy"]["explicit"]["paired"]["p"] < 0.05
        ),
        "hidden_replaces_guard": (
            report["policy"]["hidden"]["delta"] >= 0.06
            and report["policy"]["hidden"]["paired"]["p"] < 0.05
            and report["policy"]["hidden"]["vs_explicit"]["delta"] >= -0.01
        ),
        "hybrid_nonnegative": (
            report["policy"]["hybrid"]["delta"] >= 0.06
            and report["policy"]["hybrid"]["paired"]["p"] < 0.05
            and report["policy"]["hybrid"]["vs_explicit"]["delta"] >= 0.0
        ),
        "depth_signature": (
            hybrid_depth_delta.get("4", float("-inf"))
            >= hybrid_depth_delta.get("2", float("inf")) - 0.01
            and min(hybrid_depth_delta.values()) >= 0.0
        ),
    }
    report["decision"] = (
        "PASS" if all(report["gates"].values()) else "GATE FAIL"
    )

    cases_path = os.path.join(args.out_dir, "heldout_policy_cases.jsonl")
    answer_maps = {
        "sc8": sc_answers,
        "lexical": lexical_answers,
        "explicit": explicit_answers,
        "hidden": hidden_answers,
        "hybrid": hybrid_answers,
    }
    with open(cases_path, "w", encoding="utf-8") as handle:
        for pid in heldout_ids:
            row = row_by_id[pid]
            handle.write(
                json.dumps(
                    {
                        "id": pid,
                        "n_hops": len(row["question_decomposition"]),
                        "gold": row["answer"],
                        "selected": {
                            name: {
                                "answer": answers[pid],
                                "hit": (
                                    sc8[pid]
                                    if name == "sc8"
                                    else policies[name][pid]
                                ),
                            }
                            for name, answers in answer_maps.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    report_path = os.path.join(
        args.out_dir, "hsgr_contrastive_guide_heldout_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--edge-cases", required=True)
    parser.add_argument("--focus-cases", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--out-dir", default="hsgr_contrastive_guide_heldout")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--batch-problems", type=int, default=16)
    parser.add_argument("--bs-hidden", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
