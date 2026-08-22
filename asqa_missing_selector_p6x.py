"""Frozen ASQA automatic missing-facet selector and append screen P6x."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from asqa_clean_fixed_support_p1x import (
    Case,
    aligned_clean_cases,
    exact_mcnemar_p,
    hash_key,
    mean,
    median,
    render_documents,
    select_cases as select_p1x_cases,
)
from asqa_fixed_support_audit import facet_score
from asqa_obligation_repair_p5x import load_p3x_rows
from asqa_set_guide_patch_p4x import load_p1x_rows
from asqa_single_node_intervention_p3x import select_fresh_cases


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_MISSING_SELECTOR_P6X.md"
FOLD_SALT = "20260822-asqa-missing-selector-p6x-fold"
RANDOM_SALT = "20260822-asqa-missing-selector-p6x-random"
EXPECTED_ELIGIBLE = 427
EXPECTED_CASES = 192
EXPECTED_P1X_ROWS = 768
EXPECTED_P3X_ROWS = 1108
LAYERS = (13, 20, 27)
CS = (0.01, 0.1, 1.0)
ARMS = (
    "oracle_append",
    "hidden_probe_append",
    "logit_append",
    "lexical_append",
    "random_append",
    "generic_append",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class SelectorCase:
    case: Case
    direct_answer: str
    present: tuple[bool, ...]

    @property
    def missing_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.present) if not value)

    @property
    def mixed(self) -> bool:
        return bool(self.missing_indices) and any(self.present)

    @property
    def exactly_one_missing(self) -> bool:
        return self.mixed and len(self.missing_indices) == 1


@dataclass(frozen=True)
class Candidate:
    split: str
    case_id: str
    facet_index: int
    missing: bool
    prompt: str


def build_selector_cases(
    cases: list[Case], direct_rows: dict[str, dict[str, Any]]
) -> tuple[list[SelectorCase], bool]:
    result: list[SelectorCase] = []
    exact_rescore = True
    for case in cases:
        row = direct_rows[case.id]
        _, strict, present_raw = facet_score(list(case.alias_groups), str(row["answer"]))
        present = tuple(bool(value) for value in present_raw)
        exact_rescore = exact_rescore and bool(strict) == bool(row["str_hit"])
        saved = row.get("present_vector")
        if saved is not None:
            exact_rescore = exact_rescore and present == tuple(bool(value) for value in saved)
        result.append(SelectorCase(case, str(row["answer"]), present))
    return result, exact_rescore


def render_selector_prompt(item: SelectorCase, facet_index: int) -> str:
    return (
        "Check whether a saved answer explicitly covers one candidate interpretation. "
        "Reply with exactly one label and nothing else: A means COVERED; B means MISSING.\n\n"
        f"Question: {item.case.question}\n\n"
        f"Saved answer: {item.direct_answer}\n\n"
        f"Candidate interpretation: {item.case.facet_questions[facet_index]}\n\n"
        "Label:"
    )


def make_candidates(items: list[SelectorCase], split: str) -> list[Candidate]:
    return [
        Candidate(split, item.case.id, index, not item.present[index], render_selector_prompt(item, index))
        for item in items
        if item.mixed
        for index in range(len(item.case.facet_questions))
    ]


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def lexical_missing_score(question: str, answer: str) -> float:
    candidate = content_tokens(question)
    if not candidate:
        return 1.0
    answer_tokens = content_tokens(answer)
    return 1.0 - len(candidate & answer_tokens) / len(candidate)


def random_missing_score(case_id: str, facet_index: int) -> float:
    value = int(hash_key(f"{RANDOM_SALT}|{case_id}|{facet_index}")[:16], 16)
    return value / float(2**64 - 1)


def fold_id(case_id: str) -> int:
    return int(hash_key(f"{FOLD_SALT}|{case_id}")[:16], 16) % 5


class ModelRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()
        self.context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        a_ids = self.tokenizer.encode("A", add_special_tokens=False)
        b_ids = self.tokenizer.encode("B", add_special_tokens=False)
        if len(a_ids) != 1 or len(b_ids) != 1 or a_ids[0] == b_ids[0]:
            raise RuntimeError(f"A/B labels are not distinct single tokens: A={a_ids}, B={b_ids}")
        self.covered_token_id = a_ids[0]
        self.missing_token_id = b_ids[0]
        print(
            f"[model] {model_id} context={self.context_limit} "
            f"gpu={torch.cuda.get_device_name(0)} A={a_ids[0]} B={b_ids[0]}",
            flush=True,
        )

    def chat_text(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def extract_selector(
        self, prompts: list[str], layers: tuple[int, ...], batch_size: int
    ) -> tuple[dict[int, Any], Any]:
        import numpy as np

        torch = self.torch
        features: dict[int, list[Any]] = {layer: [] for layer in layers}
        logit_scores: list[Any] = []
        for start in range(0, len(prompts), batch_size):
            chunk = [self.chat_text(prompt) for prompt in prompts[start : start + batch_size]]
            encoded = self.tokenizer(
                chunk, padding=True, add_special_tokens=False, return_tensors="pt"
            )
            input_width = int(encoded["input_ids"].shape[1])
            if input_width + 2 > self.context_limit:
                raise RuntimeError(f"selector prompt exceeds context: {input_width}")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            captured: dict[int, Any] = {}
            handles = []

            def make_hook(layer: int):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer] = hidden[:, -1, :].detach().float().cpu()
                return hook

            for layer in layers:
                handles.append(self.model.model.layers[layer].register_forward_hook(make_hook(layer)))
            try:
                with torch.inference_mode():
                    base = self.model.model(**encoded, use_cache=False)
                    logits = self.model.lm_head(base.last_hidden_state[:, -1, :])
            finally:
                for handle in handles:
                    handle.remove()
            if any(layer not in captured for layer in layers):
                raise RuntimeError("missing captured selector hidden state")
            for layer in layers:
                features[layer].append(captured[layer].numpy())
            score = (
                logits[:, self.missing_token_id].float()
                - logits[:, self.covered_token_id].float()
            ).detach().cpu().numpy()
            logit_scores.append(score)
            print(f"[selector] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
            del encoded, base, logits, captured
            torch.cuda.empty_cache()
        return (
            {layer: np.concatenate(parts, axis=0) for layer, parts in features.items()},
            np.concatenate(logit_scores, axis=0),
        )

    def generate(self, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
        torch = self.torch
        answers: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = [self.chat_text(prompt) for prompt in prompts[start : start + batch_size]]
            encoded = self.tokenizer(
                chunk, padding=True, add_special_tokens=False, return_tensors="pt"
            )
            width = int(encoded["input_ids"].shape[1])
            if width + max_new_tokens > self.context_limit:
                raise RuntimeError(f"generation prompt exceeds context: {width}+{max_new_tokens}")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, width:]
            answers.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            print(f"[generate] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
            del encoded, generated, new_tokens
            torch.cuda.empty_cache()
        return answers


def target_selection_accuracy(
    candidates: list[Candidate], scores: Any, exact_case_ids: set[str]
) -> tuple[float, dict[str, int]]:
    grouped: dict[str, list[tuple[int, float, bool]]] = defaultdict(list)
    for candidate, score in zip(candidates, scores):
        if candidate.case_id in exact_case_ids:
            grouped[candidate.case_id].append((candidate.facet_index, float(score), candidate.missing))
    selected: dict[str, int] = {}
    correct = 0
    for case_id, rows in grouped.items():
        chosen = min(rows, key=lambda row: (-row[1], row[0]))
        selected[case_id] = chosen[0]
        correct += int(chosen[2])
    return correct / len(grouped), selected


def fit_hidden_probe(
    candidates: list[Candidate], features: dict[int, Any], exact_case_ids: set[str]
) -> tuple[dict[str, Any], Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([int(candidate.missing) for candidate in candidates], dtype=np.int64)
    folds = np.asarray([fold_id(candidate.case_id) for candidate in candidates], dtype=np.int64)
    cells: list[dict[str, Any]] = []
    cell_scores: dict[tuple[int, float], Any] = {}
    for layer in LAYERS:
        x = features[layer]
        for c_value in CS:
            oof = np.full(len(candidates), np.nan, dtype=np.float64)
            for fold in range(5):
                train = folds != fold
                valid = folds == fold
                model = make_pipeline(
                    StandardScaler(with_mean=False),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        solver="liblinear",
                        dual=True,
                        max_iter=1000,
                        random_state=0,
                    ),
                )
                model.fit(x[train], labels[train])
                oof[valid] = model.predict_proba(x[valid])[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError("non-finite OOF hidden scores")
            auc = float(roc_auc_score(labels, oof))
            selection_acc, _ = target_selection_accuracy(candidates, oof, exact_case_ids)
            cells.append(
                {"layer": layer, "C": c_value, "oof_candidate_auroc": auc,
                 "oof_exact_one_selection_accuracy": selection_acc}
            )
            cell_scores[(layer, c_value)] = oof
    selected = min(
        cells,
        key=lambda cell: (
            -cell["oof_exact_one_selection_accuracy"],
            -cell["oof_candidate_auroc"],
            cell["C"],
            cell["layer"],
        ),
    )
    layer, c_value = int(selected["layer"]), float(selected["C"])
    final_model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            dual=True,
            max_iter=1000,
            random_state=0,
        ),
    )
    final_model.fit(features[layer], labels)
    return {"cells": cells, "selected": selected}, final_model


def candidate_auroc(candidates: list[Candidate], scores: Any) -> float:
    from sklearn.metrics import roc_auc_score
    labels = [int(candidate.missing) for candidate in candidates]
    return float(roc_auc_score(labels, scores))


def paired_boolean(
    left_name: str, left: list[bool], right_name: str, right: list[bool]
) -> dict[str, Any]:
    p_value, left_only, right_only = exact_mcnemar_p(left, right)
    return {
        "left": left_name,
        "right": right_name,
        "left_rate": mean(float(value) for value in left),
        "right_rate": mean(float(value) for value in right),
        "delta": mean(float(value) for value in left) - mean(float(value) for value in right),
        "mcnemar_p_exact_two_sided": p_value,
        "left_only_successes": left_only,
        "right_only_successes": right_only,
    }


def render_append_prompt(item: SelectorCase, selected_index: int | None) -> str:
    common = (
        f"Question: {item.case.question}\n\nFixed documents:\n{render_documents(item.case)}\n\n"
        f"Saved answer:\n{item.direct_answer}\n\nTask:\n"
    )
    if selected_index is None:
        task = (
            "Return only a short one- or two-sentence addition to append to the saved answer. "
            "Add one important, factually supported interpretation that the saved answer does "
            "not yet cover, using only the fixed documents. Do not repeat, rewrite, or contradict "
            "the saved answer, and do not add a preface.\n\nAddition:"
        )
    else:
        task = (
            "Return only a short one- or two-sentence addition to append to the saved answer. "
            "Add the missing interpretation below using only the fixed documents. Do not repeat, "
            "rewrite, or contradict the saved answer, and do not add a preface.\n\n"
            f"Missing interpretation: {item.case.facet_questions[selected_index]}\n\nAddition:"
        )
    return common + task


def score_append(
    item: SelectorCase, arm: str, selected_index: int | None, generated: str
) -> dict[str, Any]:
    answer = f"{item.direct_answer.rstrip()} {generated.strip()}".strip()
    coverage, strict, present_raw = facet_score(list(item.case.alias_groups), answer)
    present = tuple(bool(value) for value in present_raw)
    target = item.missing_indices[0]
    original = [index for index, value in enumerate(item.present) if value]
    lost = sum(not present[index] for index in original)
    return {
        "id": item.case.id,
        "arm": arm,
        "selected_index": None if selected_index is None else selected_index + 1,
        "target_index": target + 1,
        "selection_correct": selected_index == target if selected_index is not None else None,
        "generated": generated,
        "answer": answer,
        "str_em": coverage,
        "str_hit": bool(strict),
        "target_recovered": present[target],
        "preservation_rate": sum(present[index] for index in original) / len(original),
        "all_original_present_preserved": lost == 0,
        "newly_lost_facets": lost,
        "generated_word_count": len(generated.split()),
        "answer_word_count": len(answer.split()),
    }


def append_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "str_em": mean(float(row["str_em"]) for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
        "target_recovery": mean(float(row["target_recovered"]) for row in rows),
        "all_present_preserved": mean(float(row["all_original_present_preserved"]) for row in rows),
        "mean_preservation": mean(float(row["preservation_rate"]) for row in rows),
        "median_generated_words": median(float(row["generated_word_count"]) for row in rows),
        "median_answer_words": median(float(row["answer_word_count"]) for row in rows),
    }


def paired_append(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    ids = sorted(left_by_id)
    left_hits = [bool(left_by_id[case_id]["str_hit"]) for case_id in ids]
    right_hits = [bool(right_by_id[case_id]["str_hit"]) for case_id in ids]
    result = paired_boolean(left[0]["arm"], left_hits, right[0]["arm"], right_hits)
    result["str_em_delta"] = append_metrics(left)["str_em"] - append_metrics(right)["str_em"]
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    eligible = aligned_clean_cases(args.alce, args.original)
    eval_cases = select_p1x_cases(eligible, EXPECTED_CASES)
    old_cases, fresh_pool, train_cases = select_fresh_cases(eligible, EXPECTED_CASES)
    if [case.id for case in old_cases] != [case.id for case in eval_cases]:
        raise RuntimeError("P1x reconstruction mismatch")

    p1x = load_p1x_rows(args.p1x_generations, eval_cases)
    p3x_rows = load_p3x_rows(args.p3x_generations, train_cases)
    p1_direct = {case.id: p1x[(case.id, "fixed_direct")] for case in eval_cases}
    p3_direct = {row["id"]: row for row in p3x_rows if row["arm"] == "fixed_direct"}
    train_items, train_rescore = build_selector_cases(train_cases, p3_direct)
    eval_items, eval_rescore = build_selector_cases(eval_cases, p1_direct)
    train_mixed = [item for item in train_items if item.mixed]
    eval_mixed = [item for item in eval_items if item.mixed]
    train_repairs = [item for item in train_items if item.exactly_one_missing]
    eval_repairs = [item for item in eval_items if item.exactly_one_missing]
    train_candidates = make_candidates(train_mixed, "train_p3x")
    eval_candidates = make_candidates(eval_mixed, "eval_p1x")
    print(
        f"[apparatus] eligible={len(eligible)} train_cases={len(train_cases)} "
        f"eval_cases={len(eval_cases)} train_mixed={len(train_mixed)} "
        f"eval_mixed={len(eval_mixed)} train_repairs={len(train_repairs)} "
        f"eval_repairs={len(eval_repairs)} train_candidates={len(train_candidates)} "
        f"eval_candidates={len(eval_candidates)}",
        flush=True,
    )

    runner = ModelRunner(args.model)
    train_features, train_logits = runner.extract_selector(
        [candidate.prompt for candidate in train_candidates], LAYERS, args.selector_batch_size
    )
    probe_selection, probe = fit_hidden_probe(
        train_candidates, train_features, {item.case.id for item in train_repairs}
    )
    selected_layer = int(probe_selection["selected"]["layer"])
    del train_features

    eval_features, eval_logits = runner.extract_selector(
        [candidate.prompt for candidate in eval_candidates], LAYERS, args.selector_batch_size
    )
    hidden_scores = probe.predict_proba(eval_features[selected_layer])[:, 1]
    lexical_scores = np.asarray(
        [
            lexical_missing_score(
                next(item for item in eval_mixed if item.case.id == candidate.case_id).case.facet_questions[candidate.facet_index],
                next(item for item in eval_mixed if item.case.id == candidate.case_id).direct_answer,
            )
            for candidate in eval_candidates
        ],
        dtype=np.float64,
    )
    random_scores = np.asarray(
        [random_missing_score(candidate.case_id, candidate.facet_index) for candidate in eval_candidates],
        dtype=np.float64,
    )
    if not all(
        np.isfinite(array).all()
        for array in [train_logits, eval_logits, hidden_scores, lexical_scores, random_scores]
    ):
        raise RuntimeError("non-finite selector score")

    exact_eval_ids = {item.case.id for item in eval_repairs}
    selector_scores = {
        "hidden_probe": hidden_scores,
        "logit": eval_logits,
        "lexical": lexical_scores,
        "random": random_scores,
    }
    selector_absolute: dict[str, Any] = {}
    selector_mapping: dict[str, dict[str, int]] = {}
    selector_correct: dict[str, list[bool]] = {}
    ordered_eval_ids = [item.case.id for item in eval_repairs]
    target_by_id = {item.case.id: item.missing_indices[0] for item in eval_repairs}
    for name, scores in selector_scores.items():
        selection_accuracy, mapping = target_selection_accuracy(eval_candidates, scores, exact_eval_ids)
        selector_mapping[name] = mapping
        correct = [mapping[case_id] == target_by_id[case_id] for case_id in ordered_eval_ids]
        selector_correct[name] = correct
        selector_absolute[name] = {
            "candidate_auroc": candidate_auroc(eval_candidates, scores),
            "exact_one_target_selection_accuracy": selection_accuracy,
        }
    selector_paired = {
        "hidden_vs_logit": paired_boolean(
            "hidden_probe", selector_correct["hidden_probe"], "logit", selector_correct["logit"]
        ),
        "hidden_vs_random": paired_boolean(
            "hidden_probe", selector_correct["hidden_probe"], "random", selector_correct["random"]
        ),
        "logit_vs_random": paired_boolean(
            "logit", selector_correct["logit"], "random", selector_correct["random"]
        ),
    }

    arm_mapping: dict[str, dict[str, int | None]] = {
        "oracle_append": target_by_id,
        "hidden_probe_append": selector_mapping["hidden_probe"],
        "logit_append": selector_mapping["logit"],
        "lexical_append": selector_mapping["lexical"],
        "random_append": selector_mapping["random"],
        "generic_append": {case_id: None for case_id in ordered_eval_ids},
    }
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        mapping = arm_mapping[arm]
        prompts = [render_append_prompt(item, mapping[item.case.id]) for item in eval_repairs]
        generated = runner.generate(prompts, args.generation_batch_size, 96)
        rows.extend(
            score_append(item, arm, mapping[item.case.id], output)
            for item, output in zip(eval_repairs, generated)
        )
        print(f"[arm] {arm} complete", flush=True)

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    absolute = {arm: append_metrics(by_arm[arm]) for arm in ARMS}
    paired_results = {
        "oracle_vs_generic": paired_append(by_arm["oracle_append"], by_arm["generic_append"]),
        "hidden_vs_generic": paired_append(by_arm["hidden_probe_append"], by_arm["generic_append"]),
        "logit_vs_generic": paired_append(by_arm["logit_append"], by_arm["generic_append"]),
        "hidden_vs_logit": paired_append(by_arm["hidden_probe_append"], by_arm["logit_append"]),
    }

    fold_label_counts = {}
    for fold in range(5):
        fold_rows = [candidate for candidate in train_candidates if fold_id(candidate.case_id) == fold]
        fold_label_counts[str(fold)] = {
            "candidates": len(fold_rows),
            "missing": sum(candidate.missing for candidate in fold_rows),
            "covered": sum(not candidate.missing for candidate in fold_rows),
            "cases": len({candidate.case_id for candidate in fold_rows}),
        }
    selector_prompt_has_documents = any("Fixed documents:" in candidate.prompt for candidate in train_candidates + eval_candidates)

    apparatus_gates = {
        "g1_exact_counts_rows_and_zero_overlap": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(train_cases) == EXPECTED_CASES
            and len(eval_cases) == EXPECTED_CASES
            and len(fresh_pool) == 235
            and len(p1x) == EXPECTED_P1X_ROWS
            and len(p3x_rows) == EXPECTED_P3X_ROWS
            and not ({case.id for case in train_cases} & {case.id for case in eval_cases})
        ),
        "g2_eval_repairs_at_least_40_and_all_folds_mixed": (
            len(eval_repairs) >= 40
            and all(counts["missing"] > 0 and counts["covered"] > 0 for counts in fold_label_counts.values())
        ),
        "g3_exact_rescore_and_finite_scores": train_rescore and eval_rescore,
        "g4_single_tokens_and_no_documents_in_selector": (
            runner.covered_token_id != runner.missing_token_id and not selector_prompt_has_documents
        ),
    }
    oracle_vs_generic = paired_results["oracle_vs_generic"]
    oracle_gates = {
        "g5_oracle_action_replication": (
            absolute["oracle_append"]["str_hit"] >= 0.30
            and absolute["oracle_append"]["all_present_preserved"] >= 0.98
            and oracle_vs_generic["delta"] >= 0.20
            and oracle_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
        )
    }
    logit_vs_random = selector_paired["logit_vs_random"]
    logit_vs_generic = paired_results["logit_vs_generic"]
    oracle_hit = absolute["oracle_append"]["str_hit"]
    surface_gates = {
        "g6_logit_selection_at_least_50pct": selector_absolute["logit"]["exact_one_target_selection_accuracy"] >= 0.50,
        "g7_logit_beats_random_by_10_points_significantly": (
            logit_vs_random["delta"] >= 0.10
            and logit_vs_random["mcnemar_p_exact_two_sided"] < 0.05
            and logit_vs_random["left_only_successes"] > logit_vs_random["right_only_successes"]
        ),
        "g8_logit_append_recovers_half_oracle_and_preserves": (
            absolute["logit_append"]["str_hit"] >= 0.20
            and absolute["logit_append"]["str_hit"] >= 0.50 * oracle_hit
            and absolute["logit_append"]["all_present_preserved"] >= 0.98
        ),
        "g9_logit_append_beats_generic_by_10_points_significantly": (
            logit_vs_generic["delta"] >= 0.10
            and logit_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and logit_vs_generic["left_only_successes"] > logit_vs_generic["right_only_successes"]
        ),
    }
    hidden_vs_logit_selection = selector_paired["hidden_vs_logit"]
    hidden_vs_logit_append = paired_results["hidden_vs_logit"]
    hidden_vs_generic = paired_results["hidden_vs_generic"]
    hidden_gates = {
        "g10_hidden_auc_and_selection": (
            selector_absolute["hidden_probe"]["candidate_auroc"] >= 0.70
            and selector_absolute["hidden_probe"]["exact_one_target_selection_accuracy"] >= 0.50
        ),
        "g11_hidden_selection_beats_logit_by_5_points": hidden_vs_logit_selection["delta"] >= 0.05,
        "g12_hidden_append_beats_logit_by_5_points": hidden_vs_logit_append["delta"] >= 0.05,
        "g13_hidden_end_to_end_action": (
            absolute["hidden_probe_append"]["str_hit"] >= 0.20
            and absolute["hidden_probe_append"]["str_hit"] >= 0.50 * oracle_hit
            and absolute["hidden_probe_append"]["all_present_preserved"] >= 0.98
            and hidden_vs_generic["delta"] >= 0.10
            and hidden_vs_generic["mcnemar_p_exact_two_sided"] < 0.05
            and hidden_vs_generic["left_only_successes"] > hidden_vs_generic["right_only_successes"]
        ),
    }
    apparatus_pass = all(apparatus_gates.values()) and all(oracle_gates.values())
    surface_pass = apparatus_pass and all(surface_gates.values())
    hidden_pass = surface_pass and all(hidden_gates.values())
    if not apparatus_pass:
        outcome = "APPARATUS_FAIL"
    elif hidden_pass:
        outcome = "HIDDEN_SELECTOR_PASS"
    elif surface_pass:
        outcome = "SURFACE_SELECTOR_ONLY_PASS"
    else:
        outcome = "SELECTOR_FAIL"

    selector_rows = []
    for index, candidate in enumerate(eval_candidates):
        selector_rows.append(
            {
                "split": candidate.split,
                "id": candidate.case_id,
                "facet_index": candidate.facet_index + 1,
                "missing_label": candidate.missing,
                "hidden_probe_score": float(hidden_scores[index]),
                "logit_score": float(eval_logits[index]),
                "lexical_score": float(lexical_scores[index]),
                "random_score": float(random_scores[index]),
            }
        )
    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "eligible": len(eligible),
            "train_cases": len(train_cases),
            "eval_cases": len(eval_cases),
            "train_mixed_cases": len(train_mixed),
            "eval_mixed_cases": len(eval_mixed),
            "train_exact_one_missing": len(train_repairs),
            "eval_exact_one_missing": len(eval_repairs),
            "train_candidates": len(train_candidates),
            "eval_candidates": len(eval_candidates),
            "generated_rows": len(rows),
        },
        "probe_selection": probe_selection,
        "fold_label_counts": fold_label_counts,
        "eval_selector_absolute": selector_absolute,
        "eval_selector_paired": selector_paired,
        "append_absolute": absolute,
        "append_paired": paired_results,
        "apparatus_gates": apparatus_gates,
        "oracle_gates": oracle_gates,
        "surface_selector_gates": surface_gates,
        "hidden_specific_gates": hidden_gates,
        "protocol_match": {
            "layers": list(LAYERS) == [13, 20, 27],
            "Cs": list(CS) == [0.01, 0.1, 1.0],
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            "arms": list(ARMS),
            "generation_max_new_tokens": 96,
        },
        "interpretation_guard": (
            "P6x supplies gold facet questions and trains on scorer-derived labels. A pass does "
            "not establish automatic facet induction, hierarchy, novelty, or a natural unprompted "
            "hidden controller."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_missing_selector_p6x_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in selector_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_missing_selector_p6x_generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_missing_selector_p6x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--p3x-generations", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--selector-batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
