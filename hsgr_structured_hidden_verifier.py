"""Structured HSGR Guide verifier on the existing MuSiQue SC@8 pool.

This is a development-set mechanism test, not an end-to-end result.  For each
sampled final-answer candidate, an oracle-routed Guide supplies the compiled
final-hop goal, verified predecessor values, and the final-hop support block.
A frozen LM hidden state at the verdict slot is read by a problem-disjoint OOF
linear probe.  The probe can only adjust a safe selection score built from
SC vote mass and exact support in the routed block; lambda=0 is always present
and exactly recovers the lexical Guide policy.

Controls use a length-matched predecessor block in the same prompt format.
No answer candidate is generated or changed by this experiment.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

from hsgr_focus_route_ceiling import exact_mcnemar, support_for_step
from mh_ceiling import answers_match, normalize
from mh_e0 import hop_deps, load_rows, resolve_goal
from mh_latent_rerank import auroc, fit_probe, within_problem_auroc
from pilot import jread


LAYERS = (14, 21, 28)
ROUTES = ("correct", "wrong")
PROJECTION_DIM = 256
PROJECTION_SEED = 20263601
LEXICAL_WEIGHT = 0.75
HIDDEN_LAMBDAS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

SYSTEM = (
    "You verify exactly one current node in a typed multi-hop hierarchy. "
    "Use only the compiled goal, verified predecessor mappings, and the "
    "Guide-routed evidence."
)

VERIFY_USER = """Original question (context only): {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[COMPILED CURRENT GOAL] {compiled_goal}
[VERIFIED PREDECESSORS]
{dependencies}

[GUIDE-ROUTED EVIDENCE FOR CURRENT RELATION]
{evidence}

[PROPOSED CURRENT-NODE ANSWER]
{answer}

[GUIDE CONSISTENCY CHECK]
Decide whether the proposed answer satisfies the compiled current goal and is
supported by the routed evidence. Answer exactly VALID or INVALID."""


def find_sc(sc_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(sc_dir, "sc.s*.jsonl")))
    if not paths:
        raise SystemExit(f"no sc.s*.jsonl under {sc_dir}")
    return paths


def build_units(rows: list[dict], sc_by_id: dict) -> list[dict]:
    units = []
    for row in rows:
        uid = row["_uid"]
        sc = sc_by_id.get(uid)
        if not sc:
            continue
        decomp = row["question_decomposition"]
        hop = len(decomp) - 1
        deps = hop_deps(decomp)[hop]
        if not deps:
            continue
        blocks = [support_for_step(row, step) for step in decomp]
        if any(not block for block in blocks):
            continue
        pred_vals = {j: str(decomp[j]["answer"]) for j in deps}
        compiled_goal = resolve_goal(str(decomp[hop]["question"]), pred_vals)
        dependencies = "\n".join(
            f"  - #{j + 1} = {pred_vals[j]} (verified)" for j in deps
        )
        wrong_idx = min(
            range(hop),
            key=lambda j: (abs(len(blocks[j]) - len(blocks[hop])), j),
        )
        evidence = {"correct": blocks[hop], "wrong": blocks[wrong_idx]}
        gold = str(row["answer"])
        aliases = list(row.get("answer_aliases") or [])
        for cand_idx, cand in enumerate(sc["cands"]):
            answer = cand.get("ans") or ""
            answer_norm = cand.get("norm") or normalize(answer)
            common = {
                "id": uid,
                "cand": cand_idx,
                "ans": answer,
                "norm": answer_norm,
                "label": int(bool(answer) and answers_match(answer, gold, aliases)),
                "n_hops": len(decomp),
                "gold": gold,
                "aliases": aliases,
            }
            prompts = {}
            mentions = {}
            for route in ROUTES:
                prompts[route] = VERIFY_USER.format(
                    question=row["question"],
                    hop=hop + 1,
                    n_hops=len(decomp),
                    compiled_goal=compiled_goal,
                    dependencies=dependencies,
                    evidence=evidence[route],
                    answer=answer if answer else "(empty)",
                )
                ev_norm = normalize(evidence[route])
                mentions[route] = int(bool(answer_norm) and answer_norm in ev_norm)
            common["prompts"] = prompts
            common["mentions"] = mentions
            units.append(common)
    return units


def projectors(torch, hidden_size: int, device: str):
    out = {}
    for layer in LAYERS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(PROJECTION_SEED + layer)
        matrix = torch.randint(
            0,
            2,
            (hidden_size, PROJECTION_DIM),
            generator=generator,
            dtype=torch.int8,
        ).float()
        matrix.mul_(2.0).sub_(1.0).div_(math.sqrt(PROJECTION_DIM))
        out[layer] = matrix.to(device)
    return out


def extract_features(args, units: list[dict], feature_path: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    matrices = projectors(torch, model.config.hidden_size, "cuda")
    features = {
        route: {layer: [] for layer in LAYERS} for route in ROUTES
    }
    for route in ROUTES:
        print(f"[extract] route={route}", flush=True)
        for start in range(0, len(units), args.bs):
            batch = units[start : start + args.bs]
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": unit["prompts"][route]},
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
                max_length=args.max_context,
            ).to("cuda")
            with torch.no_grad():
                result = model(
                    **enc,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            for layer in LAYERS:
                hidden = result.hidden_states[layer][:, -1, :].float()
                projected = hidden @ matrices[layer]
                projected = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)
                features[route][layer].append(projected.half().cpu())
            del enc, result
            torch.cuda.empty_cache()
            if start == 0 or (start + args.bs) % 160 == 0:
                print(
                    f"[extract] {route} {min(start + args.bs, len(units))}/{len(units)}",
                    flush=True,
                )
    payload = {
        "layers": LAYERS,
        "projection_dim": PROJECTION_DIM,
        "features": {
            route: {
                layer: torch.cat(features[route][layer]) for layer in LAYERS
            }
            for route in ROUTES
        },
        "metas": [
            {
                key: unit[key]
                for key in ("id", "cand", "ans", "norm", "label", "n_hops", "gold", "aliases", "mentions")
            }
            for unit in units
        ],
    }
    torch.save(payload, feature_path)
    print(f"[extract] saved {feature_path}", flush=True)


def route_oof(payload: dict, route: str, folds: list[list[str]], seed: int):
    import torch

    metas = payload["metas"]
    pids = sorted({meta["id"] for meta in metas})
    labels = torch.tensor([meta["label"] for meta in metas], dtype=torch.float32)
    per_layer_oof = {layer: [None] * len(metas) for layer in LAYERS}
    fold_internal = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        X = payload["features"][route][layer].float()
        fold_scores = []
        for fold_idx, hold in enumerate(folds):
            hold_set = set(hold)
            test_idx = [i for i, meta in enumerate(metas) if meta["id"] in hold_set]
            train_pids = [pid for pid in pids if pid not in hold_set]
            rng = random.Random(seed + 31 * fold_idx + layer)
            rng.shuffle(train_pids)
            n_val = max(1, len(train_pids) // 7)
            val_set = set(train_pids[:n_val])
            train_idx = [
                i for i, meta in enumerate(metas)
                if meta["id"] not in hold_set and meta["id"] not in val_set
            ]
            val_idx = [i for i, meta in enumerate(metas) if meta["id"] in val_set]
            scorer, criterion = fit_probe(
                X[train_idx],
                labels[train_idx],
                [metas[i]["id"] for i in train_idx],
                X[val_idx],
                labels[val_idx],
                [metas[i]["id"] for i in val_idx],
                use_rank=True,
                epochs=250,
                seed=seed + fold_idx,
            )
            fold_scores.append(criterion)
            fold_internal[layer][fold_idx] = criterion
            for i, score in zip(test_idx, scorer(X[test_idx])):
                per_layer_oof[layer][i] = float(score)
        mean_internal = sum(fold_scores) / max(1, len(fold_scores))
        print(
            f"[probe] route={route} layer={layer} internal_wp={mean_internal:.4f}",
            flush=True,
        )
    selected_by_fold = {}
    scores = [None] * len(metas)
    for fold_idx, hold in enumerate(folds):
        selected_layer = max(
            LAYERS,
            key=lambda layer: fold_internal[layer].get(fold_idx, float("-inf")),
        )
        selected_by_fold[fold_idx] = selected_layer
        hold_set = set(hold)
        for index, meta in enumerate(metas):
            if meta["id"] in hold_set:
                scores[index] = per_layer_oof[selected_layer][index]
    if any(score is None for score in scores):
        raise RuntimeError(f"incomplete OOF scores for {route}")
    wp, wp_n = within_problem_auroc(
        scores,
        [meta["label"] for meta in metas],
        [meta["id"] for meta in metas],
    )
    pooled = auroc(scores, [meta["label"] for meta in metas])
    return scores, {
        "selected_layer_per_fold": selected_by_fold,
        "internal_layer_wp_per_fold": fold_internal,
        "oof_wp_auroc": wp,
        "oof_wp_problem_n": wp_n,
        "oof_pooled_auroc": pooled,
    }


def policy_by_lambda(metas, scores, route: str):
    by_problem = defaultdict(list)
    for index, meta in enumerate(metas):
        by_problem[meta["id"]].append((meta, float(scores[index])))
    outcomes = {}
    for pid, items in by_problem.items():
        raw_scores = [score for _, score in items]
        score_mean = sum(raw_scores) / len(raw_scores)
        variance = sum((score - score_mean) ** 2 for score in raw_scores) / max(1, len(raw_scores) - 1)
        score_sd = math.sqrt(variance) if variance > 0 else 1.0
        grouped = defaultdict(lambda: {"count": 0, "z": 0.0, "mention": 0.0})
        order = []
        for meta, score in items:
            key = meta["norm"]
            if not key:
                continue
            if key not in grouped:
                order.append(key)
            grouped[key]["count"] += 1
            grouped[key]["z"] += (score - score_mean) / score_sd
            grouped[key]["mention"] += float(meta["mentions"][route])
        gold, aliases = items[0][0]["gold"], items[0][0]["aliases"]
        for hidden_lambda in HIDDEN_LAMBDAS:
            best = max(
                order,
                key=lambda key: (
                    grouped[key]["count"] / len(items)
                    + LEXICAL_WEIGHT * grouped[key]["mention"] / grouped[key]["count"]
                    + hidden_lambda * grouped[key]["z"] / grouped[key]["count"]
                ),
                default=None,
            )
            outcomes[(pid, hidden_lambda)] = bool(
                best and answers_match(best, gold, aliases)
            )
    return by_problem, outcomes


def select_policy(metas, scores, route: str, folds: list[list[str]]):
    by_problem, outcomes = policy_by_lambda(metas, scores, route)
    pid_fold = {
        pid: fold_idx for fold_idx, fold in enumerate(folds) for pid in fold
    }
    selected = {}
    picked = {}
    for fold_idx, hold in enumerate(folds):
        training = [pid for pid in by_problem if pid_fold[pid] != fold_idx]
        candidates = []
        for hidden_lambda in HIDDEN_LAMBDAS:
            accuracy = sum(outcomes[(pid, hidden_lambda)] for pid in training) / len(training)
            candidates.append((accuracy, -hidden_lambda, hidden_lambda))
        _, _, best_lambda = max(candidates)
        picked[fold_idx] = best_lambda
        for pid in hold:
            selected[pid] = outcomes[(pid, best_lambda)]
    curve = {
        hidden_lambda: sum(outcomes[(pid, hidden_lambda)] for pid in by_problem) / len(by_problem)
        for hidden_lambda in HIDDEN_LAMBDAS
    }
    return selected, picked, curve, outcomes


def metric(hits: dict, baseline: dict) -> dict:
    ids = sorted(baseline)
    values = [hits[pid] for pid in ids]
    base = [baseline[pid] for pid in ids]
    return {
        "accuracy": sum(values) / len(values),
        "delta": sum(values) / len(values) - sum(base) / len(base),
        "paired": exact_mcnemar(values, base),
    }


def probe_and_report(args, feature_path: str, report_path: str) -> None:
    import torch

    payload = torch.load(feature_path, map_location="cpu")
    metas = payload["metas"]
    pids = sorted({meta["id"] for meta in metas})
    rng = random.Random(args.seed)
    rng.shuffle(pids)
    folds = [pids[index::args.folds] for index in range(args.folds)]
    sc8 = {}
    lexical = {}
    # Zero scores and lambda=0 recover vote + exact routed mention.
    zero_scores = [0.0] * len(metas)
    _, lexical_outcomes = policy_by_lambda(metas, zero_scores, "correct")
    _, sc_outcomes = policy_by_lambda(
        [dict(meta, mentions={"correct": 0, "wrong": 0}) for meta in metas],
        zero_scores,
        "correct",
    )
    for pid in pids:
        lexical[pid] = lexical_outcomes[(pid, 0.0)]
        sc8[pid] = sc_outcomes[(pid, 0.0)]

    scores = {}
    diagnostics = {}
    policies = {}
    picked = {}
    curves = {}
    for route in ROUTES:
        scores[route], diagnostics[route] = route_oof(
            payload, route, folds, args.seed + (0 if route == "correct" else 5000)
        )
        policies[route], picked[route], curves[route], _ = select_policy(
            metas, scores[route], route, folds
        )

    report = {
        "experiment": "HSGR structured Guide hidden verifier",
        "claim_boundary": (
            "Development-set OOF mechanism test with oracle final-hop routing and "
            "verified gold predecessor values; not an end-to-end held-out method."
        ),
        "n_problems": len(pids),
        "n_candidate_units": len(metas),
        "projection_dim": payload["projection_dim"],
        "sc8": sum(sc8.values()) / len(sc8),
        "lexical_correct_route": metric(lexical, sc8),
        "reader": diagnostics,
        "policy": {
            route: {
                **metric(policies[route], sc8),
                "lambda_per_fold": picked[route],
                "lambda_curve": curves[route],
            }
            for route in ROUTES
        },
    }
    report["hidden_increment_over_lexical"] = metric(policies["correct"], lexical)
    report["correct_vs_wrong"] = metric(policies["correct"], policies["wrong"])
    report["by_hop"] = {}
    meta_by_pid = {meta["id"]: meta for meta in metas}
    for hop in sorted({meta["n_hops"] for meta in metas}):
        ids = [pid for pid in pids if meta_by_pid[pid]["n_hops"] == hop]
        report["by_hop"][str(hop)] = {
            "n": len(ids),
            "sc8": sum(sc8[pid] for pid in ids) / len(ids),
            "lexical": sum(lexical[pid] for pid in ids) / len(ids),
            "hidden": sum(policies["correct"][pid] for pid in ids) / len(ids),
        }
    correct_policy = report["policy"]["correct"]
    hidden_increment = report["hidden_increment_over_lexical"]
    correct_wrong = report["correct_vs_wrong"]
    positive_lambdas = sum(value > 0 for value in picked["correct"].values())
    report["gates"] = {
        "overall_headroom": (
            correct_policy["delta"] >= 0.06
            and correct_policy["paired"]["p"] < 0.05
        ),
        "hidden_reader": (
            diagnostics["correct"]["oof_wp_auroc"] >= 0.75
            and diagnostics["correct"]["oof_wp_auroc"]
            >= diagnostics["wrong"]["oof_wp_auroc"] + 0.02
        ),
        "hidden_nontrivial": (
            hidden_increment["delta"] >= 0.01 and positive_lambdas >= 3
        ),
        "route_specificity": (
            correct_wrong["delta"] >= 0.03
            and correct_wrong["paired"]["p"] < 0.05
        ),
    }
    report["decision"] = "PASS" if all(report["gates"].values()) else "GATE FAIL"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


def main(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.data, args.limit, seed=0)
    sc_rows = []
    for path in find_sc(args.sc_dir):
        sc_rows.extend(jread(path))
    sc_by_id = {row["id"]: row for row in sc_rows}
    units = build_units(rows, sc_by_id)
    expected = len(rows) * 8
    if len(units) != expected:
        raise RuntimeError(f"expected {expected} candidate units, found {len(units)}")
    print(
        f"[data] problems={len(rows)} units={len(units)} "
        f"positive_candidates={sum(unit['label'] for unit in units)}",
        flush=True,
    )
    feature_path = os.path.join(args.out_dir, "structured_hidden_features.pt")
    report_path = os.path.join(args.out_dir, "hsgr_structured_hidden_report.json")
    if args.dry_run:
        print(json.dumps({
            "id": units[0]["id"],
            "label": units[0]["label"],
            "mentions": units[0]["mentions"],
            "correct_prompt": units[0]["prompts"]["correct"][:1800],
            "wrong_prompt": units[0]["prompts"]["wrong"][:1800],
        }, indent=1, ensure_ascii=False))
        return
    if not args.probe_only:
        extract_features(args, units, feature_path)
    if not os.path.isfile(feature_path):
        raise RuntimeError(f"missing {feature_path}")
    probe_and_report(args, feature_path, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sc-dir", required=True)
    parser.add_argument("--out-dir", default="hsgr_structured_hidden")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-context", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    main(parser.parse_args())
