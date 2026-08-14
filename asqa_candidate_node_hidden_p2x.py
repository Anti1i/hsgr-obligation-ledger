"""ASQA answer-prefix candidate-node hidden Guide readout P2x."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from asqa_clean_fixed_support_p1x import (
    Case,
    aligned_clean_cases,
    build_decoy_mapping,
    render_user,
    select_cases,
)
from asqa_fixed_support_audit import exact_presence


SPLIT_SALT = "20260815-asqa-node-p2x"
FRACTIONS = (0.25, 0.50, 0.75)
LAYERS = (13, 20, 27)
PROJECTION_DIM = 64
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_half(record_id: str) -> int:
    return int(hash_key(f"asqa-node-p2x-half|{record_id}"), 16) % 2


def fold_for(record_id: str) -> int:
    return int(hash_key(f"asqa-node-p2x-fold|{record_id}"), 16) % 5


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def load_direct_generations(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"generation line {line_number} is not an object")
            if row.get("arm") != "fixed_direct":
                continue
            record_id, answer = str(row.get("id", "")), row.get("answer")
            if not record_id or not isinstance(answer, str):
                raise ValueError(f"invalid fixed_direct row at line {line_number}")
            if record_id in answers:
                raise ValueError(f"duplicate fixed_direct row for {record_id}")
            answers[record_id] = answer
    return answers


def load_p1x_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("P1x report is not an object")
    difficulty = report.get("difficulty_gates", {})
    if not isinstance(difficulty, dict) or not difficulty or not all(difficulty.values()):
        raise RuntimeError("P1x baseline-difficulty gates did not all pass; P2x is not licensed")
    return report


class WeightedRidge:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> "WeightedRidge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("non-positive training weight")
        self.x_mean = (x * weights[:, None]).sum(0) / total
        centered = x - self.x_mean
        variance = (centered**2 * weights[:, None]).sum(0) / total
        self.x_scale = np.sqrt(variance)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        standardized = centered / self.x_scale
        self.y_mean = float((y * weights).sum() / total)
        weighted_x = standardized * weights[:, None]
        gram = standardized.T @ weighted_x
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        self.coef = np.linalg.solve(
            gram,
            standardized.T @ (weights * (y - self.y_mean)),
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        return standardized @ self.coef + self.y_mean


def state_equal_weights(rows: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(rows[index]["state_id"] for index in indices)
    return np.asarray([1.0 / counts[rows[index]["state_id"]] for index in indices])


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive_total = float(weights[labels].sum())
    negative_total = float(weights[~labels].sum())
    if positive_total <= 0 or negative_total <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    concordant = 0.0
    negative_before = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group = order[start:end]
        positive = float(weights[group][labels[group]].sum())
        negative = float(weights[group][~labels[group]].sum())
        concordant += positive * negative_before + 0.5 * positive * negative
        negative_before += negative
        start = end
    return concordant / (positive_total * negative_total)


def weighted_average_precision(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    weights = np.asarray(weights, dtype=np.float64)[order]
    labels = labels[order]
    total_positive = float(weights[labels].sum())
    if total_positive <= 0:
        return float("nan")
    cumulative_weight = np.cumsum(weights)
    cumulative_positive = np.cumsum(weights * labels)
    precision = cumulative_positive / cumulative_weight
    return float((precision * weights * labels).sum() / total_positive)


def binary_ranking_metrics(
    rows: list[dict[str, Any]], indices: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    if len(indices) != len(scores):
        raise ValueError("indices/scores length mismatch")
    labels = np.asarray([bool(rows[index]["label"]) for index in indices])
    weights = state_equal_weights(rows, indices)
    by_state: dict[str, list[tuple[int, float, bool]]] = defaultdict(list)
    for index, score, label in zip(indices, scores, labels):
        by_state[rows[index]["state_id"]].append(
            (int(rows[index]["node_index"]), float(score), bool(label))
        )
    mixed = []
    for values in by_state.values():
        state_labels = {value[2] for value in values}
        if state_labels == {False, True}:
            mixed.append(values)
    top_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for values in mixed:
        ordered = sorted(values, key=lambda value: (-value[1], value[0]))
        top_hits.append(float(ordered[0][2]))
        reciprocal_ranks.append(
            1.0 / next(rank for rank, value in enumerate(ordered, 1) if value[2])
        )
    return {
        "rows": len(indices),
        "states": len(by_state),
        "mixed_states": len(mixed),
        "positive_prevalence_raw": float(labels.mean()) if len(labels) else float("nan"),
        "positive_prevalence_state_weighted": (
            float((weights * labels).sum() / weights.sum()) if weights.sum() else float("nan")
        ),
        "auroc_state_weighted": weighted_auc(labels, scores, weights),
        "average_precision_state_weighted": weighted_average_precision(labels, scores, weights),
        "mixed_state_recall_at_1": float(np.mean(top_hits)) if top_hits else float("nan"),
        "mixed_state_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
    }


def feature_surface(case: Case, prefix_text: str, fraction: float, node_index: int, prefix_tokens: int) -> np.ndarray:
    node_question = case.facet_questions[node_index]
    return np.asarray(
        [
            math.log1p(prefix_tokens),
            math.log1p(len(prefix_text.split())),
            fraction,
            node_index / max(1, len(case.alias_groups) - 1),
            math.log1p(len(case.alias_groups)),
            math.log1p(len(node_question.split())),
            jaccard(node_question, case.question),
            jaccard(node_question, prefix_text),
        ],
        dtype=np.float32,
    )


class ModelRunner:
    def __init__(self, model_id: str, layers: tuple[int, ...], projection_seed: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        self.torch = torch
        self.layers = layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda().eval()
        block_count = len(self.model.model.layers)
        if any(layer < 0 or layer >= block_count for layer in layers):
            raise RuntimeError(f"layers {layers} outside 0..{block_count - 1}")
        hidden_size = int(self.model.config.hidden_size)
        self.projectors: dict[int, np.ndarray] = {}
        for layer in layers:
            rng = np.random.default_rng(projection_seed + layer * 1009)
            self.projectors[layer] = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32),
                size=(hidden_size, PROJECTION_DIM),
            ) / math.sqrt(PROJECTION_DIM)
        self.token_counter = {"node_sequences": 0, "node_tokens": 0, "prefix_sequences": 0, "prefix_tokens": 0}
        print(
            f"[model] {model_id} blocks={block_count} hidden={hidden_size} "
            f"gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _batch_hidden(
        self,
        items: list[tuple[list[int], int, int]],
        batch_size: int,
        counter_prefix: str,
    ) -> list[dict[int, tuple[np.ndarray, np.ndarray]]]:
        torch = self.torch
        results: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            width = max(len(ids) for ids, _, _ in chunk)
            input_ids = torch.full(
                (len(chunk), width),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
                device="cuda",
            )
            attention = torch.zeros_like(input_ids)
            for row, (ids, _, _) in enumerate(chunk):
                tensor = torch.tensor(ids, dtype=torch.long, device="cuda")
                input_ids[row, : len(ids)] = tensor
                attention[row, : len(ids)] = 1
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    output_hidden_states=True,
                )
            for row, (_, span_start, span_end) in enumerate(chunk):
                per_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for layer in self.layers:
                    states = output.hidden_states[layer + 1][row, span_start:span_end].float()
                    last = states[-1].cpu().numpy() @ self.projectors[layer]
                    mean_state = states.mean(0).cpu().numpy() @ self.projectors[layer]
                    per_layer[layer] = (
                        np.asarray(last, dtype=np.float32),
                        np.asarray(mean_state, dtype=np.float32),
                    )
                results.append(per_layer)
            self.token_counter[f"{counter_prefix}_sequences"] += len(chunk)
            self.token_counter[f"{counter_prefix}_tokens"] += sum(len(ids) for ids, _, _ in chunk)
            print(
                f"[{counter_prefix}-hidden] {min(start + batch_size, len(items))}/{len(items)}",
                flush=True,
            )
            del output, input_ids, attention
            torch.cuda.empty_cache()
        return results

    def node_hidden(self, cases: list[Case], batch_size: int) -> dict[tuple[str, int], dict[int, np.ndarray]]:
        keys: list[tuple[str, int]] = []
        items: list[tuple[list[int], int, int]] = []
        for case in cases:
            for node_index, question in enumerate(case.facet_questions):
                ids = self.tokenizer.encode(
                    f"Facet question: {question}", add_special_tokens=False
                )
                if not ids:
                    raise RuntimeError(f"empty node tokenization for {case.id}/{node_index}")
                keys.append((case.id, node_index))
                items.append((ids, 0, len(ids)))
        hidden = self._batch_hidden(items, batch_size, "node")
        return {
            key: {layer: values[layer][1] for layer in self.layers}
            for key, values in zip(keys, hidden)
        }

    def prefix_hidden(
        self,
        cases: list[Case],
        answers: dict[str, str],
        batch_size: int,
    ) -> tuple[
        dict[tuple[str, float], dict[int, tuple[np.ndarray, np.ndarray]]],
        dict[tuple[str, float], tuple[str, int]],
    ]:
        keys: list[tuple[str, float]] = []
        items: list[tuple[list[int], int, int]] = []
        metadata: dict[tuple[str, float], tuple[str, int]] = {}
        for case in cases:
            prompt_ids = self.tokenizer.encode(
                self.chat_text(render_user(case, "fixed_direct")),
                add_special_tokens=False,
            )
            answer_ids = self.tokenizer.encode(answers[case.id], add_special_tokens=False)
            if not answer_ids:
                raise RuntimeError(f"empty direct answer for {case.id}")
            for fraction in FRACTIONS:
                count = max(1, int(math.floor(len(answer_ids) * fraction)))
                prefix_ids = answer_ids[:count]
                key = (case.id, fraction)
                keys.append(key)
                items.append((prompt_ids + prefix_ids, len(prompt_ids), len(prompt_ids) + count))
                metadata[key] = (
                    self.tokenizer.decode(prefix_ids, skip_special_tokens=True),
                    count,
                )
        hidden = self._batch_hidden(items, batch_size, "prefix")
        return dict(zip(keys, hidden)), metadata


def build_rows_and_features(
    cases: list[Case],
    answers: dict[str, str],
    prefix_hidden: dict[tuple[str, float], dict[int, tuple[np.ndarray, np.ndarray]]],
    prefix_metadata: dict[tuple[str, float], tuple[str, int]],
    node_hidden: dict[tuple[str, int], dict[int, np.ndarray]],
    wrong_cases: dict[str, Case],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    layer_features: dict[int, dict[str, list[np.ndarray]]] = {
        layer: defaultdict(list) for layer in LAYERS
    }
    for case in cases:
        answer = answers[case.id]
        wrong_case = wrong_cases[case.id]
        for fraction in FRACTIONS:
            prefix_text, prefix_tokens = prefix_metadata[(case.id, fraction)]
            for node_index, aliases in enumerate(case.alias_groups):
                if exact_presence(aliases, prefix_text):
                    continue
                label = not exact_presence(aliases, answer)
                surface = feature_surface(case, prefix_text, fraction, node_index, prefix_tokens)
                rows.append(
                    {
                        "id": case.id,
                        "state_id": f"{case.id}|{fraction:.2f}",
                        "fraction": fraction,
                        "node_index": node_index,
                        "node_count": len(case.alias_groups),
                        "label": bool(label),
                    }
                )
                for layer in LAYERS:
                    prefix_last, prefix_mean = prefix_hidden[(case.id, fraction)][layer]
                    node = node_hidden[(case.id, node_index)][layer]
                    wrong_node = node_hidden[(wrong_case.id, node_index)][layer]
                    layer_features[layer]["surface"].append(surface)
                    layer_features[layer]["prefix_only"].append(
                        np.concatenate([surface, prefix_last, prefix_mean])
                    )
                    layer_features[layer]["node_only"].append(
                        np.concatenate([surface, node])
                    )
                    layer_features[layer]["full"].append(
                        np.concatenate(
                            [
                                surface,
                                prefix_last,
                                prefix_mean,
                                node,
                                prefix_last * node,
                                np.abs(prefix_last - node),
                            ]
                        )
                    )
                    layer_features[layer]["wrong_node"].append(
                        np.concatenate(
                            [
                                surface,
                                prefix_last,
                                prefix_mean,
                                wrong_node,
                                prefix_last * wrong_node,
                                np.abs(prefix_last - wrong_node),
                            ]
                        )
                    )
    stacked = {
        layer: {
            family: np.asarray(values, dtype=np.float32)
            for family, values in families.items()
        }
        for layer, families in layer_features.items()
    }
    return rows, stacked


def select_alpha(
    x: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    calibration_indices: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    results: dict[str, Any] = {}
    for alpha in RIDGE_ALPHAS:
        predictions = np.full(len(calibration_indices), np.nan, dtype=np.float64)
        for fold in range(5):
            train_mask = np.asarray(
                [fold_for(rows[index]["id"]) != fold for index in calibration_indices]
            )
            valid_mask = ~train_mask
            if not train_mask.any() or not valid_mask.any():
                raise RuntimeError(f"empty calibration fold {fold}")
            train_indices = calibration_indices[train_mask]
            valid_indices = calibration_indices[valid_mask]
            weights = state_equal_weights(rows, train_indices)
            model = WeightedRidge(alpha).fit(
                x[train_indices], labels[train_indices], weights
            )
            predictions[valid_mask] = model.predict(x[valid_indices])
        metrics = binary_ranking_metrics(rows, calibration_indices, predictions)
        results[str(alpha)] = metrics
    best = max(
        RIDGE_ALPHAS,
        key=lambda alpha: (
            results[str(alpha)]["mixed_state_recall_at_1"],
            results[str(alpha)]["auroc_state_weighted"],
            -alpha,
        ),
    )
    return float(best), results


def train_and_score(
    x_train_feature: np.ndarray,
    x_test_feature: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> tuple[WeightedRidge, np.ndarray, dict[str, Any]]:
    weights = state_equal_weights(rows, calibration_indices)
    model = WeightedRidge(alpha).fit(
        x_train_feature[calibration_indices], labels[calibration_indices], weights
    )
    scores = model.predict(x_test_feature[test_indices])
    return model, scores, binary_ranking_metrics(rows, test_indices, scores)


def safe_number(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_number(item) for item in value]
    if isinstance(value, np.generic):
        return safe_number(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    p1x_report = load_p1x_report(args.p1x_report)
    eligible = aligned_clean_cases(args.alce, args.original)
    p1x_cases = select_cases(eligible, 192)
    direct_answers = load_direct_generations(args.generations)
    expected_ids = {case.id for case in p1x_cases}
    if set(direct_answers) != expected_ids:
        raise RuntimeError(
            f"fixed_direct ID mismatch: got {len(direct_answers)}, expected {len(expected_ids)}"
        )
    ordered = sorted(
        p1x_cases,
        key=lambda case: hash_key(f"{SPLIT_SALT}|{case.id}"),
    )
    calibration_ids = {case.id for case in ordered[:96]}
    test_ids = {case.id for case in ordered[96:]}
    wrong_cases = build_decoy_mapping(p1x_cases)

    runner = ModelRunner(args.model, LAYERS, args.projection_seed)
    node_hidden = runner.node_hidden(p1x_cases, args.node_batch_size)
    prefix_hidden, prefix_metadata = runner.prefix_hidden(
        p1x_cases, direct_answers, args.prefix_batch_size
    )
    rows, features = build_rows_and_features(
        p1x_cases,
        direct_answers,
        prefix_hidden,
        prefix_metadata,
        node_hidden,
        wrong_cases,
    )
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    calibration_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["id"] in calibration_ids],
        dtype=int,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["id"] in test_ids],
        dtype=int,
    )

    surface_x = features[LAYERS[0]]["surface"]
    surface_alpha, surface_cv = select_alpha(
        surface_x, labels, rows, calibration_indices
    )
    _, surface_scores, surface_metrics = train_and_score(
        surface_x,
        surface_x,
        labels,
        rows,
        calibration_indices,
        test_indices,
        surface_alpha,
    )

    layer_calibration: dict[str, Any] = {}
    for layer in LAYERS:
        alpha, cv = select_alpha(
            features[layer]["full"], labels, rows, calibration_indices
        )
        best_cv = cv[str(alpha)]
        layer_calibration[str(layer)] = {"alpha": alpha, "cv": cv, "best_cv": best_cv}
    selected_layer = max(
        LAYERS,
        key=lambda layer: (
            layer_calibration[str(layer)]["best_cv"]["mixed_state_recall_at_1"],
            layer_calibration[str(layer)]["best_cv"]["auroc_state_weighted"],
            -layer,
        ),
    )
    full_alpha = layer_calibration[str(selected_layer)]["alpha"]
    full_model, full_scores, full_metrics = train_and_score(
        features[selected_layer]["full"],
        features[selected_layer]["full"],
        labels,
        rows,
        calibration_indices,
        test_indices,
        full_alpha,
    )
    wrong_scores = full_model.predict(
        features[selected_layer]["wrong_node"][test_indices]
    )
    wrong_metrics = binary_ranking_metrics(rows, test_indices, wrong_scores)

    controls: dict[str, Any] = {}
    for family in ("prefix_only", "node_only"):
        alpha, cv = select_alpha(
            features[selected_layer][family], labels, rows, calibration_indices
        )
        _, scores, metrics = train_and_score(
            features[selected_layer][family],
            features[selected_layer][family],
            labels,
            rows,
            calibration_indices,
            test_indices,
            alpha,
        )
        controls[family] = {"alpha": alpha, "calibration_cv": cv, "heldout": metrics}

    subgroup_metrics: dict[str, Any] = {"fixed_halves": {}, "prefix_fractions": {}, "facet_counts": {}}
    for half in (0, 1):
        mask = np.asarray(
            [fixed_half(rows[index]["id"]) == half for index in test_indices]
        )
        indices = test_indices[mask]
        subgroup_metrics["fixed_halves"][str(half)] = {
            "surface": binary_ranking_metrics(rows, indices, surface_scores[mask]),
            "full": binary_ranking_metrics(rows, indices, full_scores[mask]),
            "wrong_node": binary_ranking_metrics(rows, indices, wrong_scores[mask]),
        }
    for fraction in FRACTIONS:
        mask = np.asarray(
            [rows[index]["fraction"] == fraction for index in test_indices]
        )
        indices = test_indices[mask]
        subgroup_metrics["prefix_fractions"][str(fraction)] = {
            "surface": binary_ranking_metrics(rows, indices, surface_scores[mask]),
            "full": binary_ranking_metrics(rows, indices, full_scores[mask]),
        }
    for count in sorted({row["node_count"] for row in rows}):
        mask = np.asarray(
            [rows[index]["node_count"] == count for index in test_indices]
        )
        indices = test_indices[mask]
        subgroup_metrics["facet_counts"][str(count)] = {
            "surface": binary_ranking_metrics(rows, indices, surface_scores[mask]),
            "full": binary_ranking_metrics(rows, indices, full_scores[mask]),
        }

    missing_test_problems = sum(
        any(not exact_presence(group, direct_answers[case.id]) for group in case.alias_groups)
        for case in p1x_cases
        if case.id in test_ids
    )
    apparatus_gates = {
        "g1_96_test_problems_and_at_least_250_rows": len(test_ids) == 96 and len(test_indices) >= 250,
        "g2_at_least_30_test_problems_with_final_missing_facet": missing_test_problems >= 30,
        "g3_at_least_60_mixed_test_states": full_metrics["mixed_states"] >= 60,
        "g4_future_omission_prevalence_between_10_and_60pct": 0.10 <= float(labels[test_indices].mean()) <= 0.60,
    }
    half_gains = []
    for half in (0, 1):
        values = subgroup_metrics["fixed_halves"][str(half)]
        half_gains.append(
            values["full"]["auroc_state_weighted"] > values["surface"]["auroc_state_weighted"]
            and values["full"]["mixed_state_recall_at_1"] > values["surface"]["mixed_state_recall_at_1"]
        )
    readout_gates = {
        "g5_full_heldout_auroc_at_least_0_70": full_metrics["auroc_state_weighted"] >= 0.70,
        "g6_auroc_gain_over_surface_at_least_0_05": full_metrics["auroc_state_weighted"] - surface_metrics["auroc_state_weighted"] >= 0.05,
        "g7_recall_at_1_gain_over_surface_at_least_0_10": full_metrics["mixed_state_recall_at_1"] - surface_metrics["mixed_state_recall_at_1"] >= 0.10,
        "g8_positive_auroc_and_recall_gains_in_both_halves": all(half_gains),
        "g9_wrong_node_drops_auroc_and_recall_by_at_least_0_05": (
            full_metrics["auroc_state_weighted"] - wrong_metrics["auroc_state_weighted"] >= 0.05
            and full_metrics["mixed_state_recall_at_1"] - wrong_metrics["mixed_state_recall_at_1"] >= 0.05
        ),
    }
    apparatus_pass = all(apparatus_gates.values())
    readout_pass = all(readout_gates.values())
    if apparatus_pass and readout_pass:
        outcome = "HIDDEN_CANDIDATE_NODE_SIGNAL_PASS"
    elif apparatus_pass:
        outcome = "HIDDEN_CANDIDATE_NODE_SIGNAL_FAIL"
    else:
        outcome = "APPARATUS_FAIL"

    report = safe_number(
        {
            "protocol": "EXPERIMENT_PROTOCOL_ASQA_CANDIDATE_NODE_HIDDEN_P2X.md",
            "outcome": outcome,
            "model": args.model,
            "p1x_outcome": p1x_report.get("outcome"),
            "counts": {
                "calibration_problems": len(calibration_ids),
                "heldout_problems": len(test_ids),
                "all_retained_rows": len(rows),
                "calibration_rows": len(calibration_indices),
                "heldout_rows": len(test_indices),
                "heldout_missing_facet_problems": missing_test_problems,
            },
            "token_accounting": runner.token_counter,
            "surface": {
                "alpha": surface_alpha,
                "calibration_cv": surface_cv,
                "heldout": surface_metrics,
            },
            "hidden": {
                "selected_layer": selected_layer,
                "selected_alpha": full_alpha,
                "layer_calibration": layer_calibration,
                "heldout": full_metrics,
            },
            "wrong_node_heldout": wrong_metrics,
            "controls": controls,
            "subgroups": subgroup_metrics,
            "apparatus_gates": apparatus_gates,
            "readout_gates": readout_gates,
            "protocol_match": {
                "split_salt": SPLIT_SALT == "20260815-asqa-node-p2x",
                "fractions": FRACTIONS == (0.25, 0.50, 0.75),
                "layers": LAYERS == (13, 20, 27),
                "projection_dim": PROJECTION_DIM == 64,
                "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            },
            "interpretation_guard": (
                "A predictive pass is not a causal Guide result. Gold facet questions are an "
                "oracle-node apparatus; aliases enter labels only, and causal steering requires "
                "a separately frozen experiment."
            ),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "asqa_candidate_node_hidden_p2x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "asqa_candidate_node_hidden_p2x_scores.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        score_lookup = {
            int(index): (float(surface), float(full), float(wrong))
            for index, surface, full, wrong in zip(
                test_indices, surface_scores, full_scores, wrong_scores
            )
        }
        for index, row in enumerate(rows):
            payload = dict(row)
            payload["partition"] = "calibration" if row["id"] in calibration_ids else "heldout"
            if index in score_lookup:
                payload["surface_score"], payload["hidden_score"], payload["wrong_node_score"] = score_lookup[index]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-report", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--projection-seed", type=int, default=20260815)
    parser.add_argument("--node-batch-size", type=int, default=16)
    parser.add_argument("--prefix-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
