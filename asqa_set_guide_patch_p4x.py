"""Frozen ASQA set-level answer-boundary latent-mediation screen P4x."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from asqa_clean_fixed_support_p1x import (
    Case,
    aligned_clean_cases,
    build_decoy_mapping,
    exact_mcnemar_p,
    fixed_half,
    render_user,
    score_output,
    select_cases,
)


PROTOCOL = "EXPERIMENT_PROTOCOL_ASQA_SET_GUIDE_PATCH_P4X.md"
SPLIT_SALT = "20260815-asqa-set-guide-p4x"
WRONG_SALT = "20260815-asqa-set-guide-p4x-wrong"
RANDOM_SALT = "20260815-asqa-set-guide-p4x-random"
LAYERS = (13, 20, 27)
ALPHAS = (0.5, 1.0, 2.0)
EXPECTED_ELIGIBLE = 427
EXPECTED_CASES = 192
CALIBRATION_N = 64
EXPECTED_P1X_ROWS = 768
P1X_ARMS = ("closedbook", "fixed_direct", "true_facets", "decoy_facets")


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def split_cases(cases: list[Case]) -> tuple[list[Case], list[Case]]:
    ordered = sorted(cases, key=lambda case: hash_key(f"{SPLIT_SALT}|{case.id}"))
    if len(ordered) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} P1x cases, found {len(ordered)}")
    return ordered[:CALIBRATION_N], ordered[CALIBRATION_N:]


def build_wrong_mapping(cases: list[Case]) -> dict[str, Case]:
    by_count: dict[int, list[Case]] = defaultdict(list)
    for case in cases:
        by_count[len(case.facet_questions)].append(case)
    mapping: dict[str, Case] = {}
    for case in cases:
        candidates = [
            other
            for other in by_count[len(case.facet_questions)]
            if other.id != case.id
        ]
        if not candidates:
            raise RuntimeError(f"no same-count wrong Guide for {case.id}")
        mapping[case.id] = min(
            candidates,
            key=lambda other: hash_key(f"{WRONG_SALT}|{case.id}|{other.id}"),
        )
    return mapping


def load_p1x_rows(path: Path, cases: list[Case]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    case_ids = {case.id for case in cases}
    expected = {(case.id, arm) for case in cases for arm in P1X_ARMS}
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("id", "")), str(row.get("arm", "")))
        if key in keyed:
            raise RuntimeError(f"duplicate P1x row: {key}")
        keyed[key] = row
    if len(rows) != EXPECTED_P1X_ROWS or set(keyed) != expected:
        missing = sorted(expected - set(keyed))[:5]
        extra = sorted(set(keyed) - expected)[:5]
        raise RuntimeError(
            f"P1x row alignment failed: rows={len(rows)} missing={missing} extra={extra}"
        )
    if {row["id"] for row in rows} != case_ids:
        raise RuntimeError("P1x generation IDs do not match reconstructed cases")
    return keyed


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "str_em": mean(float(row["str_em"]) for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
        "median_words": median(float(row["word_count"]) for row in rows),
        "mean_words": mean(float(row["word_count"]) for row in rows),
    }


def p1x_metrics(
    p1x: dict[tuple[str, str], dict[str, Any]], cases: list[Case], arm: str
) -> dict[str, float | int]:
    return metric_rows([p1x[(case.id, arm)] for case in cases])


def select_calibration_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if not cells:
        raise ValueError("no calibration cells")
    return min(
        cells,
        key=lambda cell: (
            -float(cell["metrics"]["str_hit"]),
            -float(cell["metrics"]["str_em"]),
            float(cell["alpha"]),
            int(cell["layer"]),
        ),
    )


def rescale_to_norm(vector, reference, torch):
    target_norm = reference.float().norm()
    vector_norm = vector.float().norm()
    if not torch.isfinite(target_norm) or not torch.isfinite(vector_norm):
        raise RuntimeError("non-finite Guide norm")
    if float(target_norm) <= 0.0 or float(vector_norm) <= 0.0:
        raise RuntimeError("zero Guide norm")
    return vector.float() * (target_norm / vector_norm)


def random_guide(reference, case_id: str, layer: int, torch):
    seed = int(hash_key(f"{RANDOM_SALT}|{case_id}|{layer}")[:16], 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return rescale_to_norm(vector, reference, torch)


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
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda().eval()
        self.context_limit = int(
            getattr(self.model.config, "max_position_embeddings", 32768)
        )
        self.hook_applied_sequences = 0
        self.hook_applied_batches = 0
        print(
            f"[model] {model_id} context={self.context_limit} "
            f"layers={len(self.model.model.layers)} gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, prompts: list[str]):
        encoded = self.tokenizer(
            [self.chat_text(prompt) for prompt in prompts],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return {key: value.cuda() for key, value in encoded.items()}

    def extract_guides(
        self,
        cases: list[Case],
        decoys: dict[str, Case],
        layers: tuple[int, ...],
        batch_size: int,
    ) -> dict[int, dict[str, Any]]:
        torch = self.torch
        guides: dict[int, dict[str, Any]] = {layer: {} for layer in layers}
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            prompts = [render_user(case, "true_facets") for case in chunk]
            prompts += [
                render_user(case, "decoy_facets", decoys[case.id]) for case in chunk
            ]
            encoded = self._encode(prompts)
            if int(encoded["input_ids"].shape[1]) > self.context_limit:
                raise RuntimeError("source prompt exceeds context limit")
            captured: dict[int, Any] = {}
            handles = []
            for layer in layers:
                if not 0 <= layer < len(self.model.model.layers):
                    raise RuntimeError(f"invalid transformer block {layer}")

                def hook(_module, _inputs, output, layer=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer] = hidden[:, -1, :].detach().float().cpu()

                handles.append(self.model.model.layers[layer].register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    output = self.model(**encoded, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            del output, encoded
            for layer in layers:
                states = captured.get(layer)
                if states is None or states.shape[0] != 2 * len(chunk):
                    raise RuntimeError(f"missing captured source states at layer {layer}")
                for row, case in enumerate(chunk):
                    guide = states[row] - states[len(chunk) + row]
                    if not bool(torch.isfinite(guide).all()) or float(guide.norm()) <= 0.0:
                        raise RuntimeError(f"invalid Guide for {case.id} at layer {layer}")
                    guides[layer][case.id] = guide
            print(
                f"[guides] {min(start + batch_size, len(cases))}/{len(cases)}",
                flush=True,
            )
            torch.cuda.empty_cache()
        return guides

    def generate_patched(
        self,
        prompts: list[str],
        vectors: list[Any],
        layer: int,
        alpha: float,
        batch_size: int,
        max_new_tokens: int,
        label: str,
    ) -> list[str]:
        if len(prompts) != len(vectors):
            raise ValueError("prompt/vector count mismatch")
        torch = self.torch
        answers: list[str] = []
        block = self.model.model.layers[layer]
        for start in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[start : start + batch_size]
            chunk_vectors = vectors[start : start + batch_size]
            encoded = self._encode(chunk_prompts)
            input_width = int(encoded["input_ids"].shape[1])
            if input_width + max_new_tokens > self.context_limit:
                raise RuntimeError(
                    f"target batch exceeds context: {input_width}+{max_new_tokens}>"
                    f"{self.context_limit}"
                )
            patch = torch.stack(chunk_vectors).cuda()
            applied = False

            def hook(_module, _inputs, output):
                nonlocal applied
                if applied:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[0] != patch.shape[0] or hidden.shape[1] != input_width:
                    raise RuntimeError(
                        f"first hook call is not the full prefill: {tuple(hidden.shape)}"
                    )
                changed = hidden.clone()
                changed[:, -1, :] = changed[:, -1, :] + alpha * patch.to(hidden.dtype)
                applied = True
                if isinstance(output, tuple):
                    return (changed,) + output[1:]
                return changed

            handle = block.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            finally:
                handle.remove()
            if not applied:
                raise RuntimeError("Guide hook was never applied")
            new_tokens = generated[:, input_width:]
            answers.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            self.hook_applied_sequences += len(chunk_prompts)
            self.hook_applied_batches += 1
            print(
                f"[{label}] {min(start + batch_size, len(prompts))}/{len(prompts)}",
                flush=True,
            )
            del encoded, patch, generated, new_tokens
            torch.cuda.empty_cache()
        return answers


def score_new_rows(cases: list[Case], answers: list[str], arm: str, **extra) -> list[dict[str, Any]]:
    if len(cases) != len(answers):
        raise RuntimeError(f"{arm}: generated {len(answers)} answers for {len(cases)} cases")
    return [
        {"id": case.id, "arm": arm, "answer": answer, **extra, **score_output(case, answer)}
        for case, answer in zip(cases, answers)
    ]


def paired_summary(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {row["id"]: row for row in left_rows}
    right = {row["id"]: row for row in right_rows}
    if set(left) != set(right):
        raise RuntimeError("paired summary ID mismatch")
    ids = sorted(left)
    left_hits = [bool(left[case_id]["str_hit"]) for case_id in ids]
    right_hits = [bool(right[case_id]["str_hit"]) for case_id in ids]
    p_value, left_only, right_only = exact_mcnemar_p(left_hits, right_hits)
    left_metrics, right_metrics = metric_rows(left_rows), metric_rows(right_rows)
    return {
        "left_arm": left_rows[0]["arm"],
        "right_arm": right_rows[0]["arm"],
        "str_hit_delta": float(left_metrics["str_hit"]) - float(right_metrics["str_hit"]),
        "str_em_delta": float(left_metrics["str_em"]) - float(right_metrics["str_em"]),
        "mcnemar_p_exact_two_sided": p_value,
        "left_only_successes": left_only,
        "right_only_successes": right_only,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eligible = aligned_clean_cases(args.alce, args.original)
    cases = select_cases(eligible, EXPECTED_CASES)
    calibration, heldout = split_cases(cases)
    decoys = build_decoy_mapping(cases)
    p1x = load_p1x_rows(args.p1x_generations, cases)
    wrong = build_wrong_mapping(heldout)
    print(
        f"[apparatus] eligible={len(eligible)} cases={len(cases)} "
        f"calibration={len(calibration)} heldout={len(heldout)} "
        f"facets={dict(sorted(Counter(len(case.facet_questions) for case in cases).items()))}",
        flush=True,
    )

    runner = ModelRunner(args.model)
    guides = runner.extract_guides(cases, decoys, LAYERS, args.source_batch_size)
    direct_prompts = {case.id: render_user(case, "fixed_direct") for case in cases}
    target_prompt_exact = all(
        direct_prompts[case.id] == render_user(case, "fixed_direct")
        and "Coverage checklist" not in direct_prompts[case.id]
        for case in cases
    )

    calibration_rows: list[dict[str, Any]] = []
    calibration_cells: list[dict[str, Any]] = []
    for layer in LAYERS:
        for alpha in ALPHAS:
            answers = runner.generate_patched(
                [direct_prompts[case.id] for case in calibration],
                [guides[layer][case.id] for case in calibration],
                layer,
                alpha,
                args.batch_size,
                args.max_new_tokens,
                f"cal-l{layer}-a{alpha:g}",
            )
            rows = score_new_rows(
                calibration, answers, "correct_guide", layer=layer, alpha=alpha, split="calibration"
            )
            calibration_rows.extend(rows)
            calibration_cells.append(
                {"layer": layer, "alpha": alpha, "metrics": metric_rows(rows)}
            )
    selected = select_calibration_cell(calibration_cells)
    selected_layer = int(selected["layer"])
    selected_alpha = float(selected["alpha"])
    print(
        f"[calibration] selected layer={selected_layer} alpha={selected_alpha:g} "
        f"hit={selected['metrics']['str_hit']:.6f} em={selected['metrics']['str_em']:.6f}",
        flush=True,
    )

    correct_vectors = [guides[selected_layer][case.id] for case in heldout]
    wrong_vectors = [
        rescale_to_norm(
            guides[selected_layer][wrong[case.id].id],
            guides[selected_layer][case.id],
            runner.torch,
        )
        for case in heldout
    ]
    random_vectors = [
        random_guide(guides[selected_layer][case.id], case.id, selected_layer, runner.torch)
        for case in heldout
    ]
    heldout_rows: list[dict[str, Any]] = []
    for arm, vectors in (
        ("correct_guide", correct_vectors),
        ("wrong_guide", wrong_vectors),
        ("random_guide", random_vectors),
    ):
        answers = runner.generate_patched(
            [direct_prompts[case.id] for case in heldout],
            vectors,
            selected_layer,
            selected_alpha,
            args.batch_size,
            args.max_new_tokens,
            f"heldout-{arm}",
        )
        heldout_rows.extend(
            score_new_rows(
                heldout,
                answers,
                arm,
                layer=selected_layer,
                alpha=selected_alpha,
                split="heldout",
            )
        )

    new_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heldout_rows:
        new_by_arm[row["arm"]].append(row)
    baseline_rows = {
        arm: [p1x[(case.id, arm)] for case in heldout]
        for arm in ("fixed_direct", "true_facets", "decoy_facets")
    }
    heldout_metrics = {
        **{arm: metric_rows(rows) for arm, rows in baseline_rows.items()},
        **{arm: metric_rows(rows) for arm, rows in new_by_arm.items()},
    }
    correct_vs_direct = paired_summary(
        new_by_arm["correct_guide"], baseline_rows["fixed_direct"]
    )
    correct_vs_wrong = paired_summary(
        new_by_arm["correct_guide"], new_by_arm["wrong_guide"]
    )
    correct_vs_random = paired_summary(
        new_by_arm["correct_guide"], new_by_arm["random_guide"]
    )
    textual_hit_gain = float(heldout_metrics["true_facets"]["str_hit"]) - float(
        heldout_metrics["fixed_direct"]["str_hit"]
    )
    recovered_fraction = (
        correct_vs_direct["str_hit_delta"] / textual_hit_gain
        if textual_hit_gain > 0
        else float("nan")
    )

    correct_by_id = {row["id"]: row for row in new_by_arm["correct_guide"]}
    half_results = {}
    for half in (0, 1):
        half_cases = [case for case in heldout if fixed_half(case.id) == half]
        direct = [p1x[(case.id, "fixed_direct")] for case in half_cases]
        correct = [correct_by_id[case.id] for case in half_cases]
        half_results[str(half)] = {
            "n": len(half_cases),
            "direct": metric_rows(direct),
            "correct_guide": metric_rows(correct),
            "correct_minus_direct_str_hit": float(metric_rows(correct)["str_hit"])
            - float(metric_rows(direct)["str_hit"]),
        }
    facet_strata = {}
    for count in sorted({len(case.facet_questions) for case in heldout}):
        stratum = [case for case in heldout if len(case.facet_questions) == count]
        facet_strata[str(count)] = {
            "n": len(stratum),
            "direct": metric_rows([p1x[(case.id, "fixed_direct")] for case in stratum]),
            "correct_guide": metric_rows([correct_by_id[case.id] for case in stratum]),
        }

    full_direct = p1x_metrics(p1x, cases, "fixed_direct")
    full_true = p1x_metrics(p1x, cases, "true_facets")
    guide_norms = {
        str(layer): {
            "min": min(float(guides[layer][case.id].norm()) for case in cases),
            "median": median(float(guides[layer][case.id].norm()) for case in cases),
            "mean": mean(float(guides[layer][case.id].norm()) for case in cases),
            "max": max(float(guides[layer][case.id].norm()) for case in cases),
        }
        for layer in LAYERS
    }
    expected_hook_sequences = len(calibration) * len(LAYERS) * len(ALPHAS) + len(
        heldout
    ) * 3
    apparatus_gates = {
        "g1_exact_counts_split_and_p1x_alignment": (
            len(eligible) == EXPECTED_ELIGIBLE
            and len(cases) == EXPECTED_CASES
            and len(calibration) == CALIBRATION_N
            and len(heldout) == EXPECTED_CASES - CALIBRATION_N
            and not ({case.id for case in calibration} & {case.id for case in heldout})
            and len(p1x) == EXPECTED_P1X_ROWS
        ),
        "g2_valid_cases_decoys_and_finite_hidden_states": (
            all(
                2 <= len(case.facet_questions) <= 6
                and len(case.documents) == 5
                and decoys[case.id].id != case.id
                and len(decoys[case.id].facet_questions) == len(case.facet_questions)
                for case in cases
            )
            and all(
                bool(runner.torch.isfinite(guides[layer][case.id]).all())
                for layer in LAYERS
                for case in cases
            )
        ),
        "g3_full_p1x_static_structure_replication": (
            float(full_true["str_hit"]) - float(full_direct["str_hit"]) >= 0.05
            and float(full_true["str_em"]) - float(full_direct["str_em"]) >= 0.03
        ),
        "g4_exact_direct_targets_and_once_only_prefill_patch": (
            target_prompt_exact and runner.hook_applied_sequences == expected_hook_sequences
        ),
    }
    mediation_gates = {
        "g5_correct_beats_direct_hit_by_5_points": correct_vs_direct["str_hit_delta"] >= 0.05,
        "g6_correct_beats_direct_em_by_3_points": correct_vs_direct["str_em_delta"] >= 0.03,
        "g7_recovers_half_of_positive_textual_hit_gain": (
            math.isfinite(recovered_fraction) and recovered_fraction >= 0.50
        ),
        "g8_correct_beats_wrong_hit_by_5_points": correct_vs_wrong["str_hit_delta"] >= 0.05,
        "g9_correct_beats_random_hit_by_5_points": correct_vs_random["str_hit_delta"] >= 0.05,
        "g10_correct_vs_direct_mcnemar_significant_and_correct_only_wins": (
            correct_vs_direct["mcnemar_p_exact_two_sided"] < 0.05
            and correct_vs_direct["left_only_successes"]
            > correct_vs_direct["right_only_successes"]
        ),
        "g11_positive_hit_delta_in_both_fixed_halves": all(
            result["correct_minus_direct_str_hit"] > 0.0
            for result in half_results.values()
        ),
        "g12_length_in_range_and_within_40_words_of_direct": (
            30 <= float(heldout_metrics["correct_guide"]["median_words"]) <= 160
            and abs(
                float(heldout_metrics["correct_guide"]["median_words"])
                - float(heldout_metrics["fixed_direct"]["median_words"])
            )
            <= 40
        ),
    }
    if not all(apparatus_gates.values()):
        outcome = "APPARATUS_FAIL"
    elif all(mediation_gates.values()):
        outcome = "SET_LATENT_MEDIATION_PASS"
    else:
        outcome = "SET_LATENT_MEDIATION_FAIL"

    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "counts": {
            "eligible": len(eligible),
            "p1x_cases": len(cases),
            "p1x_rows": len(p1x),
            "calibration": len(calibration),
            "heldout": len(heldout),
            "hook_applied_sequences": runner.hook_applied_sequences,
            "expected_hook_applied_sequences": expected_hook_sequences,
            "hook_applied_batches": runner.hook_applied_batches,
        },
        "selection": {
            "split_salt": SPLIT_SALT,
            "all_id_sha256": hash_key("\n".join(case.id for case in cases) + "\n"),
            "calibration_id_sha256": hash_key(
                "\n".join(case.id for case in calibration) + "\n"
            ),
            "heldout_id_sha256": hash_key("\n".join(case.id for case in heldout) + "\n"),
            "selected_layer": selected_layer,
            "selected_alpha": selected_alpha,
            "rule": "max STR-HIT, max STR-EM, smaller alpha, shallower layer",
        },
        "full_p1x_static_replication": {
            "fixed_direct": full_direct,
            "true_facets": full_true,
            "true_minus_direct_str_hit": float(full_true["str_hit"])
            - float(full_direct["str_hit"]),
            "true_minus_direct_str_em": float(full_true["str_em"])
            - float(full_direct["str_em"]),
        },
        "calibration_cells": calibration_cells,
        "heldout_absolute": heldout_metrics,
        "heldout_paired": {
            "correct_vs_direct": correct_vs_direct,
            "correct_vs_wrong": correct_vs_wrong,
            "correct_vs_random": correct_vs_random,
            "textual_true_minus_direct_str_hit": textual_hit_gain,
            "correct_recovered_fraction_of_textual_hit_gain": recovered_fraction,
        },
        "guide_norms": guide_norms,
        "wrong_guide_mapping": {case.id: wrong[case.id].id for case in heldout},
        "fixed_id_hash_halves": half_results,
        "facet_count_strata": facet_strata,
        "apparatus_gates": apparatus_gates,
        "mediation_gates": mediation_gates,
        "generation": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "source_batch_size": args.source_batch_size,
            "target_prompt_has_checklist": not target_prompt_exact,
            "patch_position": "last prompt token after selected block, prefill only",
        },
        "protocol_match": {
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            "max_new_tokens": args.max_new_tokens == 192,
            "layers": list(LAYERS) == [13, 20, 27],
            "alphas": list(ALPHAS) == [0.5, 1.0, 2.0],
            "calibration_n": CALIBRATION_N == 64,
        },
        "interpretation_guard": (
            "P4x is a gold-facet Oracle latent-mediation screen. A pass supports only "
            "a causal set-level hidden mediator; it does not establish novelty, an "
            "automatic Guide, or a deployable HSGR method."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_set_guide_patch_p4x_calibration.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in calibration_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "asqa_set_guide_patch_p4x_heldout.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in heldout_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_set_guide_patch_p4x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "calibration_ids.txt").write_text(
        "\n".join(case.id for case in calibration) + "\n", encoding="utf-8"
    )
    (args.out_dir / "heldout_ids.txt").write_text(
        "\n".join(case.id for case in heldout) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--p1x-generations", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
