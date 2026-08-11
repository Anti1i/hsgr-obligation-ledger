"""HSGR hidden-state Commit Guide: KEEP versus GENERIC_REPAIR.

The representation-steering line failed, so this experiment uses hidden state
as an observer for a discrete control action.  A base answer is generated first.
Before any gold outcome is available, answer-state hidden features choose one of
two low-dimensional actions:

* KEEP: return the base answer;
* GENERIC_REPAIR: execute one fixed repair prompt with the same complete
  evidence and compiled current goal.

Training supervision is counterfactual action utility
`repair_EM - base_EM` in {-1, 0, +1}, not answer correctness and not a ranking
among sampled candidates.  The policy reads only the base answer state and
token-confidence scalars; it never reads the repair candidate before acting.

An already consumed n=400 slice supplies nested five-fold OOF evaluation.  The
reserved n=320 held-out split is decoded only if the OOF hidden policy:

1. beats KEEP/base by >=3pp normalized EM with exact McNemar p<.05;
2. beats a confidence-only policy by >=2pp with exact McNemar p<.05;
3. has non-negative gain in both fixed OOF halves.

Held-out uses the same gates plus non-decreasing gain from 2-hop to 4-hop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from hsgr_dynamic_route_read import select_new_units
from hsgr_focus_route_ceiling import (
    SYSTEM,
    build_base_prompt,
    build_repair_prompt,
    exact_mcnemar,
    mean,
    paired_signflip_p,
    score_text,
    select_fresh_units,
)
from hsgr_hidden_route_guide import HiddenRouteRunner


FEATURE_LAYERS = (14, 21, 28)
RIDGE_ALPHAS = (10.0, 100.0, 1000.0)


def evaluate(units, texts):
    em, f1, answers = [], [], []
    for unit, text in zip(units, texts):
        hit, score, answer = score_text(unit, text)
        em.append(bool(hit))
        f1.append(float(score))
        answers.append(answer)
    return em, f1, answers


def sha_ids(units):
    import hashlib
    return hashlib.sha256(
        "\n".join(sorted(u["id"] for u in units)).encode("utf-8")
    ).hexdigest()


class CommitGuideRunner(HiddenRouteRunner):
    def full_answer_text(self, user: str, response: str):
        prefix_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        full_messages = prefix_messages + [
            {"role": "assistant", "content": response}
        ]
        prefix = self.tok.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        full = self.tok.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        if not full.startswith(prefix):
            raise RuntimeError("assistant chat template does not extend generation prefix")
        start = len(prefix)
        end = start + len(response)
        if full[start:end] != response:
            raise RuntimeError("failed to locate assistant response in chat template")
        return full, (start, end)

    def answer_features(self, users, responses, feature_layers, bs):
        torch = self.torch
        hidden_store = {layer: [] for layer in feature_layers}
        confidence_store = []
        for start in range(0, len(users), bs):
            chunk_users = users[start : start + bs]
            chunk_responses = responses[start : start + bs]
            rendered = [
                self.full_answer_text(user, response)
                for user, response in zip(chunk_users, chunk_responses)
            ]
            texts = [x[0] for x in rendered]
            char_spans = [x[1] for x in rendered]
            enc = self.encode_texts(texts)
            max_len = enc["input_ids"].shape[1]
            token_spans = []
            for text, chars in zip(texts, char_spans):
                single = self.tok(
                    text,
                    truncation=True,
                    max_length=self.max_context,
                    return_offsets_mapping=True,
                )
                pad = max_len - len(single["input_ids"])
                left, right = chars
                idxs = [
                    i + pad
                    for i, (a, b) in enumerate(single["offset_mapping"])
                    if b > left and a < right
                ]
                if not idxs:
                    raise RuntimeError("empty assistant-answer token span")
                token_spans.append(idxs)
            with torch.no_grad():
                out = self.model(
                    **enc, output_hidden_states=True, use_cache=False, return_dict=True
                )
            for b, idxs in enumerate(token_spans):
                last_idxs = idxs[-min(4, len(idxs)):]
                for layer in feature_layers:
                    feat = out.hidden_states[layer][b, last_idxs].float().mean(dim=0)
                    feat = feat / (feat.norm() + 1e-8)
                    hidden_store[layer].append(feat.cpu())
                pred_positions = [idx - 1 for idx in idxs if idx > 0]
                targets = enc["input_ids"][b, idxs[: len(pred_positions)]]
                logits = out.logits[b, pred_positions].float()
                logp = torch.log_softmax(logits, dim=-1)
                chosen = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
                confidence_store.append(torch.tensor([
                    float(chosen.mean().item()),
                    float(chosen.min().item()),
                    float(chosen[-1].item()),
                    math.log1p(len(idxs)),
                ], dtype=torch.float32))
            del enc, out
            torch.cuda.empty_cache()
            print(
                f"[answer-features] {min(start + bs, len(users))}/{len(users)}",
                flush=True,
            )
        return {
            "hidden": {
                layer: torch.stack(parts).numpy()
                for layer, parts in hidden_store.items()
            },
            "confidence": torch.stack(confidence_store).numpy(),
        }


def stratified_splits(labels, n_splits, seed):
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    counts = Counter(labels.tolist())
    if min(counts.values()) < n_splits:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(labels))))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def make_model(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), solver="lsqr", tol=1e-5),
    )


def feature_matrix(features, layer):
    conf = features["confidence"]
    if layer is None:
        return conf
    return np.concatenate([features["hidden"][int(layer)], conf], axis=1)


def threshold_candidates(scores):
    values = [float("inf")]
    for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        values.append(float(np.quantile(scores, q)))
    values.append(float("-inf"))
    return sorted(set(values))


def threshold_stats(scores, utility, threshold):
    actions = np.asarray(scores) > float(threshold)
    selected = int(actions.sum())
    gain = float(np.where(actions, utility, 0).mean())
    pos = int(((utility > 0) & actions).sum())
    neg = int(((utility < 0) & actions).sum())
    return {
        "threshold": float(threshold),
        "gain": gain,
        "action_rate": selected / max(1, len(actions)),
        "selected": selected,
        "positive": pos,
        "negative": neg,
    }


def choose_threshold(scores, utility):
    rows = [threshold_stats(scores, utility, t) for t in threshold_candidates(scores)]
    return max(
        rows,
        key=lambda r: (r["gain"], r["positive"] - r["negative"], -r["selected"]),
    )


def cv_predictions(X, utility, alpha, indices, n_splits, seed):
    indices = np.asarray(indices)
    local_labels = utility[indices]
    pred = np.zeros(len(indices), dtype=np.float64)
    for train_local, val_local in stratified_splits(local_labels, n_splits, seed):
        train_idx = indices[train_local]
        val_idx = indices[val_local]
        model = make_model(alpha)
        model.fit(X[train_idx], utility[train_idx])
        pred[val_local] = model.predict(X[val_idx])
    return pred


def select_config(features, utility, indices, layers, seed):
    indices = np.asarray(indices)
    rows = []
    for layer in layers:
        X = feature_matrix(features, layer)
        for alpha in RIDGE_ALPHAS:
            pred = cv_predictions(
                X, utility, alpha, indices, n_splits=3, seed=seed + int(alpha) + (layer or 0)
            )
            threshold = choose_threshold(pred, utility[indices])
            row = {
                "layer": layer,
                "alpha": alpha,
                **threshold,
            }
            rows.append(row)
    selected = max(
        rows,
        key=lambda r: (
            r["gain"], r["positive"] - r["negative"],
            -r["selected"], r["alpha"], -(r["layer"] or 0),
        ),
    )
    return selected, rows


def nested_oof(features, utility, layers, seed):
    n = len(utility)
    scores = np.zeros(n, dtype=np.float64)
    actions = np.zeros(n, dtype=bool)
    configs = []
    for fold, (train_idx, test_idx) in enumerate(
        stratified_splits(utility, n_splits=5, seed=seed)
    ):
        selected, _ = select_config(
            features, utility, train_idx, layers=layers, seed=seed + 100 * fold
        )
        X = feature_matrix(features, selected["layer"])
        model = make_model(selected["alpha"])
        model.fit(X[train_idx], utility[train_idx])
        fold_scores = model.predict(X[test_idx])
        scores[test_idx] = fold_scores
        actions[test_idx] = fold_scores > selected["threshold"]
        configs.append({"fold": fold, **selected})
        print("[oof-fold] " + json.dumps(configs[-1], sort_keys=True), flush=True)
    return scores, actions, configs


def policy_metrics(base_em, repair_em, actions):
    base = np.asarray(base_em, dtype=bool)
    repair = np.asarray(repair_em, dtype=bool)
    actions = np.asarray(actions, dtype=bool)
    hits = np.where(actions, repair, base).astype(bool)
    utility = repair.astype(int) - base.astype(int)
    return {
        "hits": hits.tolist(),
        "em": float(hits.mean()),
        "gain": float(hits.mean() - base.mean()),
        "action_rate": float(actions.mean()),
        "selected": int(actions.sum()),
        "positive": int(((utility > 0) & actions).sum()),
        "negative": int(((utility < 0) & actions).sum()),
        "paired_vs_base": exact_mcnemar(hits.tolist(), base.tolist()),
    }


def fit_final_ensemble(features, utility, layers, seed):
    indices = np.arange(len(utility))
    selected, grid = select_config(features, utility, indices, layers=layers, seed=seed)
    X = feature_matrix(features, selected["layer"])
    models = []
    oof = np.zeros(len(utility), dtype=np.float64)
    for train_idx, val_idx in stratified_splits(utility, n_splits=5, seed=seed + 909):
        model = make_model(selected["alpha"])
        model.fit(X[train_idx], utility[train_idx])
        oof[val_idx] = model.predict(X[val_idx])
        models.append(model)
    threshold = choose_threshold(oof, utility)
    selected.update({f"ensemble_{k}": v for k, v in threshold.items()})
    return selected, grid, models


def ensemble_predict(models, X):
    return np.mean([model.predict(X) for model in models], axis=0)


def main(args):
    train_units, train_meta = select_fresh_units(
        data=args.data,
        prior_cases=args.edge_cases,
        original_limit=200,
        prior_limit=400,
        prior_seed=20260811,
        new_limit=400,
        seed=20260812,
    )
    with open(args.focus_cases, encoding="utf-8") as handle:
        expected_train_ids = {json.loads(line)["id"] for line in handle if line.strip()}
    if {u["id"] for u in train_units} != expected_train_ids:
        raise RuntimeError("reconstructed training slice does not match focus cases")

    _, test_units, test_meta = select_new_units(
        data=args.data,
        exclude_cases=[args.edge_cases, args.focus_cases],
        original_limit=200,
        calib_n=80,
        test_n=320,
        seed=20260814,
    )
    meta = {
        "train_n": len(train_units),
        "train_id_sha256": sha_ids(train_units),
        "train_selection": train_meta,
        "reserved_test_n": len(test_units),
        "reserved_test_id_sha256": sha_ids(test_units),
        "reserved_test_hops": dict(Counter(u["n_hops"] for u in test_units)),
        "reserved_selection": test_meta,
    }
    print("[data] " + json.dumps(meta, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(json.dumps({
            "train_example": train_units[0]["id"],
            "test_example": test_units[0]["id"],
            "train_goal": train_units[0]["fields"]["compiled_goal"],
            "test_goal": test_units[0]["fields"]["compiled_goal"],
        }, ensure_ascii=False))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "hsgr_commit_guide_report.json")
    runner = CommitGuideRunner(args.model, args.max_context)
    train_users = [build_base_prompt(u) for u in train_units]
    train_base_texts = runner.generate(
        train_users, "train_base", args.bs_generate, args.max_new
    )
    train_base_em, train_base_f1, train_base_answers = evaluate(
        train_units, train_base_texts
    )
    train_repair_users = [
        build_repair_prompt(u, text, "neutral")
        for u, text in zip(train_units, train_base_texts)
    ]
    train_repair_texts = runner.generate(
        train_repair_users, "train_repair", args.bs_generate, args.max_new
    )
    train_repair_em, train_repair_f1, train_repair_answers = evaluate(
        train_units, train_repair_texts
    )
    train_features = runner.answer_features(
        train_users, train_base_texts, FEATURE_LAYERS, args.bs_hidden
    )
    utility = (
        np.asarray(train_repair_em, dtype=int) - np.asarray(train_base_em, dtype=int)
    )
    print("[utility] " + json.dumps({
        "positive": int((utility > 0).sum()),
        "zero": int((utility == 0).sum()),
        "negative": int((utility < 0).sum()),
        "base_em": mean(train_base_em),
        "always_repair_em": mean(train_repair_em),
    }), flush=True)

    hidden_scores, hidden_actions, hidden_configs = nested_oof(
        train_features, utility, layers=FEATURE_LAYERS, seed=args.seed
    )
    conf_scores, conf_actions, conf_configs = nested_oof(
        train_features, utility, layers=(None,), seed=args.seed + 5000
    )
    hidden_oof = policy_metrics(train_base_em, train_repair_em, hidden_actions)
    conf_oof = policy_metrics(train_base_em, train_repair_em, conf_actions)
    hidden_vs_conf = exact_mcnemar(hidden_oof["hits"], conf_oof["hits"])
    hidden_conf_delta = hidden_oof["em"] - conf_oof["em"]
    half = len(train_units) // 2
    half_gains = []
    for idxs in (np.arange(0, half), np.arange(half, len(train_units))):
        metrics = policy_metrics(
            np.asarray(train_base_em)[idxs],
            np.asarray(train_repair_em)[idxs],
            hidden_actions[idxs],
        )
        half_gains.append(metrics["gain"])
    oof_gates = {
        "gain_vs_base": (
            hidden_oof["gain"] >= 0.03
            and hidden_oof["paired_vs_base"]["p"] < 0.05
        ),
        "hidden_beats_confidence": (
            hidden_conf_delta >= 0.02 and hidden_vs_conf["p"] < 0.05
        ),
        "half_sign_stability": min(half_gains) >= -1e-12,
    }
    oof_pass = all(oof_gates.values())
    report = {
        "experiment": "HSGR answer-state hidden Commit Guide",
        "claim_boundary": (
            "The policy reads a completed base answer state and chooses KEEP or "
            "a fixed generic repair; it does not rank sampled candidates."
        ),
        "data": meta,
        "utility_counts": dict(Counter(utility.tolist())),
        "training": {
            "base_em": mean(train_base_em),
            "always_repair_em": mean(train_repair_em),
            "hidden_oof": hidden_oof,
            "confidence_oof": conf_oof,
            "hidden_vs_confidence_delta": hidden_conf_delta,
            "hidden_vs_confidence_paired": hidden_vs_conf,
            "hidden_half_gains": half_gains,
            "hidden_fold_configs": hidden_configs,
            "confidence_fold_configs": conf_configs,
            "gates": oof_gates,
            "advance_to_heldout": oof_pass,
        },
        "heldout_touched": False,
        "token_counts": dict(runner.token_counts),
        "model": args.model,
    }
    print("[oof] " + json.dumps(report["training"], ensure_ascii=False), flush=True)
    if not oof_pass:
        report["decision"] = "OOF GATE FAIL: reserved held-out was not decoded"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
        print(json.dumps({
            "advance_to_heldout": False,
            "hidden_oof": hidden_oof,
            "confidence_oof": conf_oof,
            "gates": oof_gates,
            "report": report_path,
        }, indent=1), flush=True)
        return

    hidden_final, hidden_grid, hidden_models = fit_final_ensemble(
        train_features, utility, FEATURE_LAYERS, args.seed + 10000
    )
    conf_final, conf_grid, conf_models = fit_final_ensemble(
        train_features, utility, (None,), args.seed + 15000
    )

    test_users = [build_base_prompt(u) for u in test_units]
    test_base_texts = runner.generate(
        test_users, "test_base", args.bs_generate, args.max_new
    )
    test_base_em, test_base_f1, test_base_answers = evaluate(test_units, test_base_texts)
    # The action decision is fixed from base-answer features before any repair
    # candidate is generated.
    test_features = runner.answer_features(
        test_users, test_base_texts, FEATURE_LAYERS, args.bs_hidden
    )
    hidden_X = feature_matrix(test_features, hidden_final["layer"])
    conf_X = feature_matrix(test_features, conf_final["layer"])
    hidden_test_scores = ensemble_predict(hidden_models, hidden_X)
    conf_test_scores = ensemble_predict(conf_models, conf_X)
    hidden_test_actions = hidden_test_scores > hidden_final["ensemble_threshold"]
    conf_test_actions = conf_test_scores > conf_final["ensemble_threshold"]

    test_repair_users = [
        build_repair_prompt(u, text, "neutral")
        for u, text in zip(test_units, test_base_texts)
    ]
    test_repair_texts = runner.generate(
        test_repair_users, "test_repair", args.bs_generate, args.max_new
    )
    test_repair_em, test_repair_f1, test_repair_answers = evaluate(
        test_units, test_repair_texts
    )
    hidden_test = policy_metrics(test_base_em, test_repair_em, hidden_test_actions)
    conf_test = policy_metrics(test_base_em, test_repair_em, conf_test_actions)
    hidden_vs_conf_test = exact_mcnemar(hidden_test["hits"], conf_test["hits"])
    depth_delta = {}
    for hop in sorted({u["n_hops"] for u in test_units}):
        idxs = [i for i, u in enumerate(test_units) if u["n_hops"] == hop]
        depth_delta[str(hop)] = mean([hidden_test["hits"][i] for i in idxs]) - mean(
            [test_base_em[i] for i in idxs]
        )
    depth_keys = [str(h) for h in (2, 3, 4) if str(h) in depth_delta]
    depth_non_decreasing = all(
        depth_delta[b] >= depth_delta[a] - 1e-12
        for a, b in zip(depth_keys, depth_keys[1:])
    )
    test_gates = {
        "gain_vs_base": (
            hidden_test["gain"] >= 0.03
            and hidden_test["paired_vs_base"]["p"] < 0.05
        ),
        "hidden_beats_confidence": (
            hidden_test["em"] - conf_test["em"] >= 0.02
            and hidden_vs_conf_test["p"] < 0.05
        ),
        "depth_non_decreasing": depth_non_decreasing,
    }

    cases_path = os.path.join(args.out_dir, "hsgr_commit_guide_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for i, unit in enumerate(test_units):
            handle.write(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "compiled_goal": unit["fields"]["compiled_goal"],
                "gold": unit["gold"],
                "base": {"answer": test_base_answers[i], "em": test_base_em[i]},
                "repair": {"answer": test_repair_answers[i], "em": test_repair_em[i]},
                "hidden_action_repair": bool(hidden_test_actions[i]),
                "confidence_action_repair": bool(conf_test_actions[i]),
                "hidden_score": float(hidden_test_scores[i]),
                "confidence_score": float(conf_test_scores[i]),
            }, ensure_ascii=False) + "\n")

    report.update({
        "heldout_touched": True,
        "final_policy": {
            "hidden": hidden_final,
            "confidence": conf_final,
            "hidden_grid": hidden_grid,
            "confidence_grid": conf_grid,
        },
        "test": {
            "base_em": mean(test_base_em),
            "always_repair_em": mean(test_repair_em),
            "hidden": hidden_test,
            "confidence": conf_test,
            "hidden_vs_confidence_delta": hidden_test["em"] - conf_test["em"],
            "hidden_vs_confidence_paired": hidden_vs_conf_test,
            "depth_hidden_vs_base": depth_delta,
            "gates": test_gates,
            "pass": all(test_gates.values()),
        },
        "token_counts": dict(runner.token_counts),
        "decision": "PASS" if all(test_gates.values()) else "HELD-OUT GATE FAIL",
    })
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print("\n== HSGR Commit Guide held-out result ==")
    print(json.dumps(report["test"], indent=1, ensure_ascii=False))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--edge-cases", required=True)
    ap.add_argument("--focus-cases", required=True)
    ap.add_argument("--out-dir", default="hsgr_commit_guide")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--bs-hidden", type=int, default=2)
    ap.add_argument("--bs-generate", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
