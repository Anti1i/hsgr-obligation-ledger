"""Exploratory ASQA clean fixed-support generation screen P1x."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from asqa_fixed_support_audit import (
    alignment_signature,
    extract_row,
    facet_groups,
    facet_score,
    load_alce,
    load_original_dev,
)


SELECTION_SALT = "20260815-clean-p1x"
ARMS = ("closedbook", "fixed_direct", "true_facets", "decoy_facets")


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    facet_questions: tuple[str, ...]
    alias_groups: tuple[tuple[str, ...], ...]
    documents: tuple[tuple[str, str], ...]

    @property
    def facet_word_count(self) -> int:
        return sum(len(question.split()) for question in self.facet_questions)


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_half(record_id: str) -> int:
    return int(hash_key(f"asqa-p1x-half|{record_id}"), 16) % 2


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def case_from_record(record: dict[str, Any]) -> Case | None:
    record_id = str(record.get("sample_id", ""))
    question = record.get("question")
    qa_pairs = record.get("qa_pairs")
    docs = record.get("docs")
    if not record_id or not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(qa_pairs, list) or not isinstance(docs, list):
        return None

    row = extract_row(record_id, record)
    if (
        row["invalid_facet_groups"]
        or row["has_duplicate_facets"]
        or not 2 <= row["facet_count"] <= 6
        or row["total_doc_count"] != 5
        or row["nonempty_fixed_doc_count"] != 5
        or not row["passage_strict"]
        or not row["best_human_strict"]
        or row["verbatim_human_answer_leak"]
    ):
        return None

    groups = facet_groups(record)
    facet_questions: list[str] = []
    for pair in qa_pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("question"), str):
            return None
        facet_questions.append(pair["question"].strip())
    if len(facet_questions) != len(groups):
        return None

    documents: list[tuple[str, str]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            return None
        title, text = doc.get("title", ""), doc.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
            return None
        documents.append((title.strip(), text.strip()))
    return Case(
        id=record_id,
        question=question.strip(),
        facet_questions=tuple(facet_questions),
        alias_groups=tuple(groups),
        documents=tuple(documents),
    )


def aligned_clean_cases(alce_path: Path, original_path: Path) -> list[Case]:
    original = load_original_dev(original_path)
    alce = load_alce(alce_path)
    alce_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in alce:
        alce_by_id[str(record.get("sample_id", ""))].append(record)

    cases: list[Case] = []
    for record_id, original_record in original.items():
        matches = alce_by_id.get(record_id, [])
        if len(matches) != 1:
            continue
        record = matches[0]
        if alignment_signature(original_record) != alignment_signature(record):
            continue
        case = case_from_record(record)
        if case is not None:
            cases.append(case)
    return cases


def select_cases(cases: list[Case], n: int = 192) -> list[Case]:
    ordered = sorted(cases, key=lambda case: hash_key(f"{SELECTION_SALT}|{case.id}"))
    if len(ordered) < n:
        raise RuntimeError(f"need {n} clean eligible cases, found {len(ordered)}")
    return ordered[:n]


def build_decoy_mapping(cases: list[Case]) -> dict[str, Case]:
    by_count: dict[int, list[Case]] = defaultdict(list)
    for case in cases:
        by_count[len(case.alias_groups)].append(case)

    mapping: dict[str, Case] = {}
    for case in cases:
        candidates = [other for other in by_count[len(case.alias_groups)] if other.id != case.id]
        if not candidates:
            raise RuntimeError(f"no same-count decoy for {case.id}")
        mapping[case.id] = min(
            candidates,
            key=lambda other: (
                abs(other.facet_word_count - case.facet_word_count),
                hash_key(f"decoy|{case.id}|{other.id}"),
            ),
        )
    return mapping


def render_documents(case: Case) -> str:
    return "\n\n".join(
        f"[Document {index}]\nTitle: {title}\n{text}"
        for index, (title, text) in enumerate(case.documents, 1)
    )


def render_checklist(questions: tuple[str, ...]) -> str:
    return "\n".join(f"- {question}" for question in questions)


def render_user(case: Case, arm: str, decoy: Case | None = None) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    common = (
        "Write one coherent long-form answer to the question. The question may have "
        "multiple interpretations, so cover every relevant interpretation you can "
        "identify. State the facts directly and do not discuss your reasoning process."
    )
    if arm == "closedbook":
        return f"{common}\n\nQuestion: {case.question}\n\nAnswer:"

    prompt = (
        f"{common} Use only the fixed documents below.\n\n"
        f"Question: {case.question}\n\nFixed documents:\n{render_documents(case)}"
    )
    if arm in {"true_facets", "decoy_facets"}:
        if arm == "true_facets":
            checklist = case.facet_questions
        else:
            if decoy is None:
                raise ValueError("decoy_facets requires a decoy case")
            checklist = decoy.facet_questions
        prompt += (
            "\n\nCoverage checklist (use these questions only to decide which "
            "interpretations to cover; do not mention the checklist):\n"
            f"{render_checklist(checklist)}"
        )
    return prompt + "\n\nAnswer:"


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
        self.context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        print(
            f"[model] {model_id} context={self.context_limit} "
            f"gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
        torch = self.torch
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = [self.chat_text(prompt) for prompt in prompts[start : start + batch_size]]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_width = int(encoded["input_ids"].shape[1])
            if input_width + max_new_tokens > self.context_limit:
                raise RuntimeError(
                    f"batch exceeds context: {input_width}+{max_new_tokens}>{self.context_limit}"
                )
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, input_width:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            print(
                f"[generate] {min(start + batch_size, len(prompts))}/{len(prompts)}",
                flush=True,
            )
            del encoded, generated, new_tokens
            torch.cuda.empty_cache()
        return outputs


def exact_mcnemar_p(true_hits: list[bool], decoy_hits: list[bool]) -> tuple[float, int, int]:
    if len(true_hits) != len(decoy_hits):
        raise ValueError("paired hit lists differ in length")
    true_only = sum(left and not right for left, right in zip(true_hits, decoy_hits))
    decoy_only = sum(right and not left for left, right in zip(true_hits, decoy_hits))
    discordant = true_only + decoy_only
    if discordant == 0:
        return 1.0, true_only, decoy_only
    tail = sum(math.comb(discordant, k) for k in range(min(true_only, decoy_only) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return p_value, true_only, decoy_only


def score_output(case: Case, answer: str) -> dict[str, Any]:
    coverage, strict, present = facet_score(list(case.alias_groups), answer)
    return {
        "str_em": coverage,
        "str_hit": strict,
        "present_facets": sum(present),
        "facet_count": len(present),
        "word_count": len(answer.split()),
    }


def arm_metrics(rows: list[dict[str, Any]], ids: set[str] | None = None) -> dict[str, Any]:
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
    return {
        "n": len(rows),
        "str_em": mean(row["str_em"] for row in rows),
        "str_hit": mean(float(row["str_hit"]) for row in rows),
        "median_words": median(row["word_count"] for row in rows),
        "mean_words": mean(row["word_count"] for row in rows),
    }


def grouped_metrics(rows: list[dict[str, Any]], cases: list[Case]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    absolute = {arm: arm_metrics(by_arm[arm]) for arm in ARMS}

    result_by_key = {(row["id"], row["arm"]): row for row in rows}
    true_hits = [bool(result_by_key[(case.id, "true_facets")]["str_hit"]) for case in cases]
    decoy_hits = [bool(result_by_key[(case.id, "decoy_facets")]["str_hit"]) for case in cases]
    mcnemar_p, true_only, decoy_only = exact_mcnemar_p(true_hits, decoy_hits)

    paired_changes = {
        "fixed_direct_minus_closedbook_str_em": absolute["fixed_direct"]["str_em"] - absolute["closedbook"]["str_em"],
        "fixed_direct_minus_closedbook_str_hit": absolute["fixed_direct"]["str_hit"] - absolute["closedbook"]["str_hit"],
        "true_minus_decoy_str_em": absolute["true_facets"]["str_em"] - absolute["decoy_facets"]["str_em"],
        "true_minus_decoy_str_hit": absolute["true_facets"]["str_hit"] - absolute["decoy_facets"]["str_hit"],
        "true_minus_direct_str_hit": absolute["true_facets"]["str_hit"] - absolute["fixed_direct"]["str_hit"],
        "decoy_minus_direct_str_hit": absolute["decoy_facets"]["str_hit"] - absolute["fixed_direct"]["str_hit"],
    }
    difficulty_gates = {
        "g1_fixed_direct_hit_between_40_and_75pct": 0.40 <= absolute["fixed_direct"]["str_hit"] <= 0.75,
        "g2_fixed_beats_closedbook_by_10_points_hit_or_em": (
            paired_changes["fixed_direct_minus_closedbook_str_hit"] >= 0.10
            or paired_changes["fixed_direct_minus_closedbook_str_em"] >= 0.10
        ),
        "g3_fixed_direct_median_words_between_30_and_160": 30 <= absolute["fixed_direct"]["median_words"] <= 160,
    }
    structure_gates = {
        "g4_true_beats_decoy_hit_by_8_points": paired_changes["true_minus_decoy_str_hit"] >= 0.08,
        "g5_true_beats_decoy_em_by_5_points": paired_changes["true_minus_decoy_str_em"] >= 0.05,
        "g6_true_vs_decoy_mcnemar_p_below_0_05": mcnemar_p < 0.05,
        "g7_true_beats_direct_hit_by_5_points": paired_changes["true_minus_direct_str_hit"] >= 0.05,
        "g8_decoy_minus_direct_hit_at_most_3_points": paired_changes["decoy_minus_direct_str_hit"] <= 0.03,
    }

    halves = {}
    for half in (0, 1):
        ids = {case.id for case in cases if fixed_half(case.id) == half}
        halves[str(half)] = {arm: arm_metrics(by_arm[arm], ids) for arm in ARMS}
    strata = {}
    for count in sorted({len(case.alias_groups) for case in cases}):
        ids = {case.id for case in cases if len(case.alias_groups) == count}
        strata[str(count)] = {arm: arm_metrics(by_arm[arm], ids) for arm in ARMS}

    difficulty_pass = all(difficulty_gates.values())
    structure_pass = all(structure_gates.values())
    if difficulty_pass and structure_pass:
        outcome = "HEADROOM_AND_STRUCTURE_PASS"
    elif difficulty_pass:
        outcome = "HEADROOM_PASS_STRUCTURE_FAIL"
    elif structure_pass:
        outcome = "STRUCTURE_PASS_DIFFICULTY_FAIL"
    else:
        outcome = "FAIL"
    return {
        "outcome": outcome,
        "absolute": absolute,
        "paired_changes": paired_changes,
        "mcnemar": {
            "p_exact_two_sided": mcnemar_p,
            "true_only_successes": true_only,
            "decoy_only_successes": decoy_only,
        },
        "difficulty_gates": difficulty_gates,
        "structure_gates": structure_gates,
        "fixed_id_hash_halves": halves,
        "facet_count_strata": strata,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eligible = aligned_clean_cases(args.alce, args.original)
    cases = select_cases(eligible, args.n)
    decoys = build_decoy_mapping(cases)
    print(
        f"[apparatus] clean_eligible={len(eligible)} selected={len(cases)} "
        f"facets={dict(sorted(Counter(len(case.alias_groups) for case in cases).items()))}",
        flush=True,
    )

    prompts: list[str] = []
    keys: list[tuple[Case, str]] = []
    for arm in ARMS:
        for case in cases:
            prompts.append(render_user(case, arm, decoys.get(case.id)))
            keys.append((case, arm))

    runner = ModelRunner(args.model)
    answers = runner.generate(prompts, args.batch_size, args.max_new_tokens)
    rows: list[dict[str, Any]] = []
    for (case, arm), answer in zip(keys, answers):
        rows.append(
            {
                "id": case.id,
                "arm": arm,
                "decoy_id": decoys[case.id].id if arm == "decoy_facets" else None,
                "answer": answer,
                **score_output(case, answer),
            }
        )

    metrics = grouped_metrics(rows, cases)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_ASQA_CLEAN_FIXED_SUPPORT_P1X.md",
        "outcome": metrics["outcome"],
        "model": args.model,
        "clean_eligible_examples": len(eligible),
        "selected_examples": len(cases),
        "selection_salt": SELECTION_SALT,
        "selected_id_sha256": hash_key("\n".join(case.id for case in cases) + "\n"),
        "selected_facet_histogram": dict(sorted(Counter(len(case.alias_groups) for case in cases).items())),
        "generation": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        **metrics,
        "protocol_match": {
            "model": args.model == "Qwen/Qwen2.5-7B-Instruct",
            "n": args.n == 192,
            "max_new_tokens": args.max_new_tokens == 192,
            "selection_salt": SELECTION_SALT == "20260815-clean-p1x",
        },
        "interpretation_guard": (
            "This separately frozen P1x is an exploratory oracle textual-structure ceiling. "
            "It does not retroactively pass ASQA P0 or establish hidden-state/causal Guide value."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "asqa_clean_fixed_support_p1x_generations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "asqa_clean_fixed_support_p1x_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "selected_ids.txt").write_text(
        "\n".join(case.id for case in cases) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alce", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
