"""CLAPnQ fixed-support evidence marginal and hidden Guide screen P1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PROJECTION_DIM = 64
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
ACTIVE_THRESHOLD = 0.005
RANGE_THRESHOLD = 0.010


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    sentences: tuple[str, ...]
    support_indices: tuple[int, ...]
    answer: str
    non_consecutive: bool


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else float("nan")


def hash_value(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()


def fixed_half(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(value)
    return records


def record_to_case(record: dict[str, Any]) -> Case | None:
    passages = record.get("passages")
    outputs = record.get("output")
    if not isinstance(passages, list) or not passages or not isinstance(passages[0], dict):
        return None
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        return None
    sentences = passages[0].get("sentences")
    answer = outputs[0].get("answer")
    selected = outputs[0].get("selected_sentences")
    question = record.get("input")
    meta = outputs[0].get("meta", {})
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str) or not answer.strip():
        return None
    if not isinstance(sentences, list) or not all(isinstance(x, str) for x in sentences):
        return None
    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        return None

    unique_selected = list(dict.fromkeys(selected))
    support_indices: list[int] = []
    used: set[int] = set()
    for support in unique_selected:
        matches = [i for i, sentence in enumerate(sentences) if sentence == support and i not in used]
        if not matches:
            return None
        support_indices.append(matches[0])
        used.add(matches[0])
    support_indices.sort()
    non_consecutive = bool(meta.get("non_consecutive", False)) if isinstance(meta, dict) else False

    if not 2 <= len(support_indices) <= 6:
        return None
    if not 30 <= len(answer.split()) <= 120:
        return None
    if not (non_consecutive or len(support_indices) >= 3):
        return None
    return Case(
        id=str(record.get("id", "")),
        question=question.strip(),
        sentences=tuple(sentences),
        support_indices=tuple(support_indices),
        answer=answer.strip(),
        non_consecutive=non_consecutive,
    )


def select_cases(records: list[dict[str, Any]], seed: int, n: int) -> list[Case]:
    cases = [case for record in records if (case := record_to_case(record)) is not None]
    cases.sort(key=lambda case: hash_value(case.id, seed))
    if len(cases) < n:
        raise RuntimeError(f"need {n} eligible records, found {len(cases)}")
    return cases[:n]


def render_user(case: Case, keep_indices: set[int]) -> tuple[str, dict[int, tuple[int, int]]]:
    prefix = (
        "Answer the question with a concise, cohesive answer using only the fixed "
        "passage. Combine all relevant facts, but do not mention sentence labels.\n\n"
        f"Question: {case.question}\n\nFixed passage:\n"
    )
    parts = [prefix]
    spans: dict[int, tuple[int, int]] = {}
    for index, sentence in enumerate(case.sentences):
        if index not in keep_indices:
            continue
        marker = f"[S{index:03d}] "
        parts.append(marker)
        start = sum(len(part) for part in parts)
        parts.append(sentence)
        end = start + len(sentence)
        spans[index] = (start, end)
        parts.append("\n")
    parts.append("\nAnswer:")
    return "".join(parts), spans


def surface_features(case: Case, support_index: int) -> list[float]:
    sentence = case.sentences[support_index]
    others = [
        case.sentences[index]
        for index in case.support_indices
        if index != support_index
    ]
    return [
        math.log1p(len(sentence.split())),
        support_index / max(1, len(case.sentences) - 1),
        math.log1p(len(case.sentences)),
        float(len(case.support_indices)),
        jaccard(case.question, sentence),
        mean(jaccard(sentence, other) for other in others) if others else 0.0,
    ]


class ModelRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        # The NUS PyTorch/cuDNN installation requires the non-cuDNN SDPA path.
        # This must be set in the model process; a separate Slurm smoke process
        # does not carry backend state into this interpreter.
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if not self.tok.is_fast:
            raise RuntimeError("a fast tokenizer is required for span offsets")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda().eval()
        self.context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        print(
            f"[model] {model_id} blocks={len(self.model.model.layers)} "
            f"hidden={self.model.config.hidden_size} context={self.context_limit}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_pair(self, prompt_text: str, answer: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tok.encode(answer, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            answer_ids = answer_ids + [int(self.tok.eos_token_id)]
        if not prompt_ids or not answer_ids:
            raise RuntimeError("empty prompt or answer tokenization")
        if len(prompt_ids) + len(answer_ids) > self.context_limit:
            raise RuntimeError(
                f"sequence exceeds model context: {len(prompt_ids) + len(answer_ids)} "
                f"> {self.context_limit}"
            )
        return prompt_ids, answer_ids

    def _batch_tensors(self, encoded: list[tuple[list[int], list[int]]]):
        torch = self.torch
        lengths = [len(prompt) + len(answer) for prompt, answer in encoded]
        width = max(lengths)
        input_ids = torch.full(
            (len(encoded), width),
            int(self.tok.pad_token_id),
            dtype=torch.long,
            device="cuda",
        )
        attention = torch.zeros_like(input_ids)
        for row, (prompt, answer) in enumerate(encoded):
            ids = torch.tensor(prompt + answer, dtype=torch.long, device="cuda")
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1
        return input_ids, attention

    def _nlls(self, logits, encoded: list[tuple[list[int], list[int]]]) -> list[float]:
        torch = self.torch
        values = []
        for row, (prompt, answer) in enumerate(encoded):
            start = len(prompt)
            scores = logits[row, start - 1 : start + len(answer) - 1].float()
            targets = torch.tensor(answer, dtype=torch.long, device=scores.device)
            loss = torch.nn.functional.cross_entropy(scores, targets, reduction="mean")
            values.append(float(loss.cpu()))
        return values

    def score(self, prompts: list[str], answers: list[str], batch_size: int) -> list[float]:
        torch = self.torch
        results: list[float] = []
        for start in range(0, len(prompts), batch_size):
            p = prompts[start : start + batch_size]
            a = answers[start : start + batch_size]
            encoded = [self.encode_pair(x, y) for x, y in zip(p, a)]
            input_ids, attention = self._batch_tensors(encoded)
            with torch.inference_mode():
                output = self.model(input_ids=input_ids, attention_mask=attention)
            results.extend(self._nlls(output.logits, encoded))
            print(f"[score] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return results

    def full_scores_and_features(
        self, cases: list[Case], layers: tuple[int, ...], batch_size: int
    ) -> tuple[list[float], dict[int, list[np.ndarray]]]:
        torch = self.torch
        block_count = len(self.model.model.layers)
        if any(layer < 0 or layer >= block_count for layer in layers):
            raise RuntimeError(f"layers {layers} outside 0..{block_count - 1}")
        scores: list[float] = []
        features = {layer: [] for layer in layers}

        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            rendered = [render_user(case, set(range(len(case.sentences)))) for case in chunk]
            prompt_texts = [self.chat_text(user) for user, _ in rendered]
            encoded = [self.encode_pair(text, case.answer) for text, case in zip(prompt_texts, chunk)]
            input_ids, attention = self._batch_tensors(encoded)
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    output_hidden_states=True,
                )
            scores.extend(self._nlls(output.logits, encoded))

            for row, (case, (_, user_spans), prompt_text) in enumerate(
                zip(chunk, rendered, prompt_texts)
            ):
                user_start = prompt_text.index(rendered[row][0])
                offsets = self.tok(
                    prompt_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )["offset_mapping"]
                for support_index in case.support_indices:
                    left, right = user_spans[support_index]
                    left += user_start
                    right += user_start
                    token_positions = [
                        i
                        for i, (a, b) in enumerate(offsets)
                        if b > left and a < right
                    ]
                    if not token_positions:
                        raise RuntimeError(f"no tokens for {case.id} support {support_index}")
                    for layer in layers:
                        # hidden_states[0] is the embedding output; block L is at L+1.
                        vector = output.hidden_states[layer + 1][
                            row, token_positions
                        ].float().mean(0).cpu().numpy()
                        features[layer].append(vector)
            del output
            torch.cuda.empty_cache()
            print(
                f"[full+hidden] {min(start + batch_size, len(cases))}/{len(cases)}",
                flush=True,
            )
        return scores, features


class Ridge:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_mean = x.mean(0)
        self.x_scale = x.std(0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        standardized = (x - self.x_mean) / self.x_scale
        self.y_mean = float(y.mean())
        gram = standardized.T @ standardized
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        self.coef = np.linalg.solve(gram, standardized.T @ (y - self.y_mean))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        return standardized @ self.coef + self.y_mean


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        result[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "mae": float(np.mean(np.abs(target - prediction))),
        "pearson": correlation(target, prediction),
        "spearman": correlation(ranks(target), ranks(prediction)),
    }


def fold_for(case_id: str) -> int:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 5


def select_alpha(x: np.ndarray, y: np.ndarray, groups: list[str]) -> tuple[float, dict[str, float]]:
    fold_ids = np.asarray([fold_for(group) for group in groups])
    candidate_rmse: dict[str, float] = {}
    for alpha in RIDGE_ALPHAS:
        predictions = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            train = fold_ids != fold
            valid = fold_ids == fold
            if not train.any() or not valid.any():
                raise RuntimeError(f"empty grouped CV fold {fold}")
            predictions[valid] = Ridge(alpha).fit(x[train], y[train]).predict(x[valid])
        candidate_rmse[str(alpha)] = regression_metrics(y, predictions)["rmse"]
    best = min(RIDGE_ALPHAS, key=lambda alpha: (candidate_rmse[str(alpha)], alpha))
    return float(best), candidate_rmse


def projected(hidden: np.ndarray, layer: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + layer * 1009)
    matrix = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(hidden.shape[1], PROJECTION_DIM),
    ) / math.sqrt(PROJECTION_DIM)
    return np.asarray(hidden, dtype=np.float32) @ matrix


def build_arm_scores(
    runner: ModelRunner,
    cases: list[Case],
    full_scores: list[float],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[str] = []
    answers: list[str] = []
    arm_keys: list[tuple[int, str, int | None]] = []
    for case_index, case in enumerate(cases):
        support = set(case.support_indices)
        for support_index in case.support_indices:
            keep = set(range(len(case.sentences))) - {support_index}
            user, _ = render_user(case, keep)
            prompts.append(runner.chat_text(user))
            answers.append(case.answer)
            arm_keys.append((case_index, "drop", support_index))
        for arm, keep in (
            ("drop_all", set(range(len(case.sentences))) - support),
            ("support_only", support),
        ):
            user, _ = render_user(case, keep)
            prompts.append(runner.chat_text(user))
            answers.append(case.answer)
            arm_keys.append((case_index, arm, None))

    values = runner.score(prompts, answers, batch_size)
    per_case = [
        {
            "id": case.id,
            "full_nll": float(full_scores[index]),
            "drop_all_nll": None,
            "support_only_nll": None,
            "marginals": {},
        }
        for index, case in enumerate(cases)
    ]
    for value, (case_index, arm, support_index) in zip(values, arm_keys):
        if arm == "drop":
            assert support_index is not None
            per_case[case_index]["marginals"][str(support_index)] = float(
                value - full_scores[case_index]
            )
        else:
            per_case[case_index][f"{arm}_nll"] = float(value)

    nodes: list[dict[str, Any]] = []
    for case, case_result in zip(cases, per_case):
        for support_index in case.support_indices:
            nodes.append(
                {
                    "case_id": case.id,
                    "support_index": support_index,
                    "target_utility": case_result["marginals"][str(support_index)],
                    "surface": surface_features(case, support_index),
                }
            )
    return per_case, nodes


def guide_readout(
    nodes: list[dict[str, Any]],
    hidden_by_layer: dict[int, list[np.ndarray]],
    calib_ids: set[str],
    test_ids: set[str],
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray([node["target_utility"] for node in nodes], dtype=np.float64)
    surface = np.asarray([node["surface"] for node in nodes], dtype=np.float64)
    groups = [node["case_id"] for node in nodes]
    calib = np.asarray([group in calib_ids for group in groups])
    test = np.asarray([group in test_ids for group in groups])
    if not calib.any() or not test.any() or np.any(calib & test):
        raise RuntimeError("invalid calibration/test node partition")

    surface_alpha, surface_cv = select_alpha(
        surface[calib], targets[calib], [groups[i] for i in np.flatnonzero(calib)]
    )
    surface_model = Ridge(surface_alpha).fit(surface[calib], targets[calib])
    surface_prediction = surface_model.predict(surface[test])
    surface_metrics = regression_metrics(targets[test], surface_prediction)

    layer_calibration: dict[str, Any] = {}
    layer_matrices: dict[int, np.ndarray] = {}
    for layer, vectors in hidden_by_layer.items():
        if len(vectors) != len(nodes):
            raise RuntimeError(f"layer {layer}: {len(vectors)} vectors for {len(nodes)} nodes")
        matrix = np.concatenate([projected(np.stack(vectors), layer, seed), surface], axis=1)
        layer_matrices[layer] = matrix
        alpha, cv = select_alpha(
            matrix[calib], targets[calib], [groups[i] for i in np.flatnonzero(calib)]
        )
        layer_calibration[str(layer)] = {
            "alpha": alpha,
            "cv_rmse_by_alpha": cv,
            "best_cv_rmse": cv[str(alpha)],
        }
    selected_layer = min(
        hidden_by_layer,
        key=lambda layer: (layer_calibration[str(layer)]["best_cv_rmse"], layer),
    )
    selected_alpha = layer_calibration[str(selected_layer)]["alpha"]
    hidden_model = Ridge(selected_alpha).fit(
        layer_matrices[selected_layer][calib], targets[calib]
    )
    hidden_prediction = hidden_model.predict(layer_matrices[selected_layer][test])
    hidden_metrics = regression_metrics(targets[test], hidden_prediction)

    test_groups = np.asarray([groups[i] for i in np.flatnonzero(test)])
    halves = {}
    for half in (0, 1):
        mask = np.asarray([fixed_half(group) == half for group in test_groups])
        halves[str(half)] = {
            "n_nodes": int(mask.sum()),
            **regression_metrics(targets[test][mask], hidden_prediction[mask]),
        }

    rmse_gain = (
        (surface_metrics["rmse"] - hidden_metrics["rmse"]) / surface_metrics["rmse"]
        if surface_metrics["rmse"] > 0
        else float("nan")
    )
    gates = {
        "heldout_spearman_at_least_0_20": hidden_metrics["spearman"] >= 0.20,
        "spearman_gain_over_surface_at_least_0_08": (
            hidden_metrics["spearman"] - surface_metrics["spearman"] >= 0.08
        ),
        "rmse_reduction_at_least_5pct": rmse_gain >= 0.05,
        "positive_spearman_in_both_halves": all(
            halves[str(half)]["spearman"] > 0 for half in (0, 1)
        ),
    }
    return {
        "calibration_nodes": int(calib.sum()),
        "heldout_nodes": int(test.sum()),
        "surface": {
            "alpha": surface_alpha,
            "cv_rmse_by_alpha": surface_cv,
            "heldout": surface_metrics,
        },
        "hidden_layer_calibration": layer_calibration,
        "selected_layer": selected_layer,
        "selected_alpha": selected_alpha,
        "hidden_heldout": hidden_metrics,
        "relative_rmse_reduction": rmse_gain,
        "fixed_halves": halves,
        "gates": gates,
        "readable": all(gates.values()),
    }


def model_relevance(case_results: list[dict[str, Any]], test_ids: set[str]) -> dict[str, Any]:
    heldout = [row for row in case_results if row["id"] in test_ids]
    total_gaps = [row["drop_all_nll"] - row["full_nll"] for row in heldout]
    support_only_gaps = [
        row["drop_all_nll"] - row["support_only_nll"] for row in heldout
    ]
    utilities = [
        value for row in heldout for value in row["marginals"].values()
    ]
    multi_active_rate = mean(
        sum(value >= ACTIVE_THRESHOLD for value in row["marginals"].values()) >= 2
        for row in heldout
    )
    differentiated_rate = mean(
        max(row["marginals"].values()) - min(row["marginals"].values())
        >= RANGE_THRESHOLD
        for row in heldout
    )
    all_values = [
        value
        for row in heldout
        for value in (
            [row["full_nll"], row["drop_all_nll"], row["support_only_nll"]]
            + list(row["marginals"].values())
        )
    ]
    finite = all(math.isfinite(value) for value in all_values)
    gates = {
        "all_scores_finite": finite,
        "median_total_evidence_sensitivity_at_least_0_020": median(total_gaps) >= 0.020,
        "positive_total_evidence_sensitivity_rate_at_least_60pct": mean(
            value > 0 for value in total_gaps
        )
        >= 0.60,
        "median_support_only_gain_over_drop_all_at_least_0_020": median(
            support_only_gaps
        )
        >= 0.020,
        "active_node_rate_at_least_35pct": mean(
            value >= ACTIVE_THRESHOLD for value in utilities
        )
        >= 0.35,
        "multi_active_case_rate_at_least_25pct": multi_active_rate >= 0.25,
        "differentiated_marginal_case_rate_at_least_30pct": differentiated_rate >= 0.30,
    }
    return {
        "heldout_cases": len(heldout),
        "heldout_nodes": len(utilities),
        "median_total_evidence_sensitivity": median(total_gaps),
        "positive_total_evidence_sensitivity_rate": mean(value > 0 for value in total_gaps),
        "median_support_only_gain_over_drop_all": median(support_only_gaps),
        "median_node_utility": median(utilities),
        "active_node_rate": mean(value >= ACTIVE_THRESHOLD for value in utilities),
        "multi_active_case_rate": multi_active_rate,
        "differentiated_marginal_case_rate": differentiated_rate,
        "node_utility_p10_p50_p90": [
            float(np.quantile(utilities, q)) for q in (0.10, 0.50, 0.90)
        ],
        "gates": gates,
        "model_relevant": all(gates.values()),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.data)
    cases = select_cases(records, args.seed, args.calib_n + args.test_n)
    calib_cases = cases[: args.calib_n]
    test_cases = cases[args.calib_n :]
    calib_ids = {case.id for case in calib_cases}
    test_ids = {case.id for case in test_cases}

    runner = ModelRunner(args.model)
    full_scores, hidden = runner.full_scores_and_features(
        cases, tuple(args.layers), args.hidden_batch_size
    )
    case_results, nodes = build_arm_scores(
        runner, cases, full_scores, args.score_batch_size
    )
    relevance = model_relevance(case_results, test_ids)
    readout = guide_readout(nodes, hidden, calib_ids, test_ids, args.seed)

    if relevance["model_relevant"] and readout["readable"]:
        outcome = "MODEL_RELEVANT_AND_READABLE"
    elif relevance["model_relevant"]:
        outcome = "MODEL_RELEVANT_NOT_READABLE"
    else:
        outcome = "MODEL_IRRELEVANT"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "clapnq_evidence_marginal_p1_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for case, result in zip(cases, case_results):
            payload = dict(result)
            payload["partition"] = "calibration" if case.id in calib_ids else "heldout"
            payload["support_indices"] = list(case.support_indices)
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")

    report = {
        "protocol": "EXPERIMENT_PROTOCOL_CLAPNQ_EVIDENCE_MARGINAL_P1.md",
        "outcome": outcome,
        "model": args.model,
        "seed": args.seed,
        "calibration_cases": len(calib_cases),
        "heldout_cases": len(test_cases),
        "layers": args.layers,
        "model_relevance": relevance,
        "guide_readout": readout,
        "protocol_match": {
            "seed": args.seed == 20260814,
            "calib_n": args.calib_n == 96,
            "test_n": args.test_n == 96,
            "layers": args.layers == [13, 20, 27],
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
        },
        "guard": (
            "P1 is a held-out marginal-utility/readout screen. It provides no causal "
            "Guide or generated-answer improvement claim."
        ),
    }
    report = json_safe(report)
    (args.out_dir / "clapnq_evidence_marginal_p1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data/longform_cache/clapnq_dev_answerable.jsonl")
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--calib-n", type=int, default=96)
    parser.add_argument("--test-n", type=int, default=96)
    parser.add_argument("--layers", type=int, nargs="+", default=[13, 20, 27])
    parser.add_argument("--hidden-batch-size", type=int, default=2)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("clapnq_evidence_marginal_p1"))
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
