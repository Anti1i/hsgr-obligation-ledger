"""P0n0 external-dataset criterion-transition screen on RefineBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0b import release_runner


PROTOCOL = "EXPERIMENT_PROTOCOL_REFINEBENCH_REVISION_AUDIT_P0N.md"
DATASET_NAME = "RefineBench/RefineBench"
DATASET_REVISION = "2777137e7c489f5049608f41d2432326429ea619"
SEED = 20260824
ARMS = ("guided_failed", "targeted_partial_failed")
DEFAULT_GENERATORS = ("Qwen/Qwen3-8B",)
DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct"
STRATA = ("math_statistics", "stem", "law", "humanities", "other")


def field_stratum(field: str) -> str | None:
    if field in {"Math", "Statistics"}:
        return "math_statistics"
    if field in {
        "Computer Science/AI", "Physics", "Chemistry", "Engineering",
        "Biology/Medicine",
    }:
        return "stem"
    if field == "Law":
        return "law"
    if field == "Humanities/Social Science":
        return "humanities"
    if field in {"Economics/Business", "Other"}:
        return "other"
    return None


def stable_key(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def material_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content", "")).strip()
    return str(item).strip()


def query_text(instance: dict[str, Any]) -> str:
    parts = []
    for passage in instance.get("passages") or []:
        value = str(passage).strip()
        if value:
            parts.append(value)
    for material in instance.get("materials") or []:
        value = material_text(material)
        if value:
            parts.append(value)
    question = str(instance.get("question", "")).strip()
    if question:
        parts.append(question)
    return "\n\n".join(parts)


def eligible_instance(instance: dict[str, Any], input_char_limit: int) -> bool:
    checklist = instance.get("checklist") or []
    return bool(
        str(instance.get("index", "")).strip()
        and str(instance.get("question", "")).strip()
        and len(checklist) >= 2
        and all(str(item).strip() for item in checklist)
        and len(query_text(instance)) <= input_char_limit
        and field_stratum(str(instance.get("field", ""))) is not None
    )


def select_stratified(
    instances: list[dict[str, Any]], n_per_stratum: int, input_char_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        if eligible_instance(instance, input_char_limit):
            grouped[field_stratum(str(instance["field"]))].append(instance)
    selected = []
    for stratum in STRATA:
        ordered = sorted(
            grouped[stratum],
            key=lambda row: stable_key(SEED, "sample", row["index"]),
        )
        if len(ordered) < n_per_stratum:
            raise RuntimeError(
                f"insufficient {stratum} instances: {len(ordered)} < {n_per_stratum}"
            )
        for row in ordered[:n_per_stratum]:
            selected.append({**row, "p0n_stratum": stratum})
    return selected


def reference_answer_text(instance: dict[str, Any]) -> str | None:
    values = [str(value).strip() for value in (instance.get("reference_answer") or [])]
    values = [value for value in values if value]
    return max(values, key=len) if values else None


def criterion_instruction(criterion: str) -> str:
    text = criterion.strip()
    text = re.sub(r"(?i)^Did the response", "The response should", text)
    text = re.sub(r"(?i)^Does the response", "The response should", text)
    if text.endswith("?"):
        text = text[:-1] + "."
    return text


def failed_ids(evaluation: dict[int, bool]) -> list[int]:
    return sorted(index for index, decision in evaluation.items() if not decision)


def target_ids(case_id: str, failures: list[int], arm: str) -> list[int]:
    if arm == "guided_failed":
        return sorted(failures)
    if arm != "targeted_partial_failed":
        raise ValueError(arm)
    if not failures:
        return []
    count = max(1, math.floor(len(failures) * 0.5))
    return sorted(
        sorted(failures, key=lambda idx: stable_key(SEED, case_id, "partial", idx))[:count]
    )


def feedback_text(checklist: list[str], targets: list[int]) -> str:
    feedback = "\n".join(
        f"- {criterion_instruction(checklist[index - 1])}" for index in targets
    )
    return (
        "Please refine your previous response by considering the following feedbacks:\n"
        f"{feedback}"
    )


def initial_messages(instance: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query_text(instance)},
    ]


def revision_messages(
    instance: dict[str, Any], previous_answer: str, targets: list[int],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query_text(instance)},
        {"role": "assistant", "content": previous_answer},
        {"role": "user", "content": feedback_text(instance["checklist"], targets)},
    ]


def judge_prompt(instance: dict[str, Any], answer: str) -> str:
    checklist = "\n".join(
        f"{index}. {item}" for index, item in enumerate(instance["checklist"], 1)
    )
    return (
        "You will be provided with a model's answer to the given query and an evaluation "
        "checklist that contains multiple questions.\n\n"
        "Evaluate the answer against every checklist question. Answer Yes only when the answer "
        "fully satisfies that item.\n\n"
        f"### Query:\n{instance['question']}\n\n"
        f"### Model's Answer:\n{answer}\n\n"
        f"### Checklist (Evaluation Items):\n{checklist}\n\n"
        "### Output Format:\n"
        "Provide exactly one line per item in the form <Q>: <Yes or No>. Do not include "
        "explanations.\n\nAnswer:"
    )


def parse_evaluation(raw: str, count: int) -> tuple[dict[int, bool], bool, str]:
    matches = re.findall(r"(?:<)?Q?(\d+)(?:>)?\s*:\s*(Yes|No)\b", raw, re.IGNORECASE)
    parsed: dict[int, bool] = {}
    for number, label in matches:
        index = int(number)
        if index in parsed:
            return {}, False, "duplicate_id"
        parsed[index] = label.lower() == "yes"
    expected = set(range(1, count + 1))
    if set(parsed) != expected:
        return {}, False, "missing_or_extra_ids"
    return parsed, True, "valid"


def generate_messages(
    runner: ModelRunner, conversations: list[list[dict[str, str]]], batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    torch = runner.torch
    outputs: list[str] = []
    for start in range(0, len(conversations), batch_size):
        chunk_messages = conversations[start : start + batch_size]
        texts = [
            runner.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **runner.chat_template_kwargs,
            )
            for messages in chunk_messages
        ]
        encoded = runner.tokenizer(
            texts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        width = int(encoded["input_ids"].shape[1])
        if width + max_new_tokens > runner.context_limit:
            raise RuntimeError(f"generation prompt exceeds context: {width}+{max_new_tokens}")
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            generated = runner.model.generate(
                **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=runner.tokenizer.pad_token_id,
                eos_token_id=runner.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, width:]
        outputs.extend(runner.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(
            f"[generate-chat] {min(start + batch_size, len(conversations))}/"
            f"{len(conversations)}", flush=True,
        )
        del encoded, generated, new_tokens
        torch.cuda.empty_cache()
    return outputs


def judge_answers(
    model_name: str, entries: list[dict[str, Any]], batch_size: int, cap: int,
) -> list[dict[str, Any]]:
    runner = ModelRunner(model_name)
    order = sorted(
        range(len(entries)),
        key=lambda index: stable_key(SEED, "judge_order", entries[index]["eval_id"]),
    )
    conversations = [
        [
            {"role": "system", "content": "You are a careful checklist evaluator."},
            {"role": "user", "content": judge_prompt(entries[index]["instance"], entries[index]["answer"])},
        ]
        for index in order
    ]
    raw_outputs = generate_messages(runner, conversations, batch_size, cap)
    release_runner(runner)
    judged: list[dict[str, Any] | None] = [None] * len(entries)
    for index, raw in zip(order, raw_outputs):
        parsed, valid, parse_mode = parse_evaluation(
            raw, len(entries[index]["instance"]["checklist"])
        )
        judged[index] = {
            **entries[index], "raw_judgment": raw, "evaluation": parsed,
            "judge_valid": valid, "judge_parse_mode": parse_mode,
        }
    return [row for row in judged if row is not None]


def sentence_units(text: str) -> list[str]:
    return [
        part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if len(part.split()) >= 5
    ]


def diff_aids(old: str, new: str) -> dict[str, Any]:
    old_units = sentence_units(old)
    new_normalized = {" ".join(unit.lower().split()) for unit in sentence_units(new)}
    preserved = [
        unit for unit in old_units if " ".join(unit.lower().split()) in new_normalized
    ]
    return {
        "sequence_similarity": SequenceMatcher(None, old, new, autojunk=False).ratio(),
        "old_sentence_units": len(old_units),
        "exact_preserved_old_sentence_units": len(preserved),
        "preserved_old_sentence_fraction": len(preserved) / len(old_units) if old_units else 0.0,
    }


def transition_row(
    instance: dict[str, Any], generator: str, arm: str, old_answer: str,
    new_answer: str, old_judgment: dict[str, Any], new_judgment: dict[str, Any],
    targets: list[int],
) -> dict[str, Any]:
    valid = old_judgment["judge_valid"] and new_judgment["judge_valid"]
    transitions = Counter()
    criterion_rows = []
    if valid:
        for index in range(1, len(instance["checklist"]) + 1):
            old_value = old_judgment["evaluation"][index]
            new_value = new_judgment["evaluation"][index]
            code = ("Y" if old_value else "N") + ("Y" if new_value else "N")
            transitions[code] += 1
            criterion_rows.append({
                "criterion_id": index,
                "criterion": instance["checklist"][index - 1],
                "old_met": old_value,
                "new_met": new_value,
                "transition": code,
                "targeted": index in targets,
            })
    prior_yes = transitions["YY"] + transitions["YN"]
    target_prior_no = sum(
        row["targeted"] and not row["old_met"] for row in criterion_rows
    )
    target_fixes = sum(
        row["targeted"] and row["transition"] == "NY" for row in criterion_rows
    )
    return {
        "case_id": instance["index"],
        "field": instance["field"],
        "stratum": instance["p0n_stratum"],
        "generator": generator,
        "arm": arm,
        "targets": targets,
        "valid_transition": valid,
        "old_judge_parse_mode": old_judgment["judge_parse_mode"],
        "new_judge_parse_mode": new_judgment["judge_parse_mode"],
        "transitions": dict(transitions),
        "prior_yes": prior_yes,
        "regressions": transitions["YN"],
        "target_prior_no": target_prior_no,
        "target_fixes": target_fixes,
        "any_target_fix": target_fixes > 0,
        "any_regression": transitions["YN"] > 0,
        "successful_fix_regression": target_fixes > 0 and transitions["YN"] > 0,
        "criterion_rows": criterion_rows,
        "question": instance["question"],
        "old_answer": old_answer,
        "new_answer": new_answer,
        "diff_aids": diff_aids(old_answer, new_answer),
    }


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_transition"]]
    prior_yes = sum(row["prior_yes"] for row in valid)
    regressions = sum(row["regressions"] for row in valid)
    targets = sum(row["target_prior_no"] for row in valid)
    fixes = sum(row["target_fixes"] for row in valid)
    successful = [row for row in valid if row["any_target_fix"]]
    return {
        "n_revision_rows": len(rows),
        "n_valid_transitions": len(valid),
        "judge_validity": len(valid) / len(rows) if rows else 0.0,
        "prior_yes": prior_yes,
        "yes_to_no": regressions,
        "prior_success_regression_rate": regressions / prior_yes if prior_yes else None,
        "target_prior_no": targets,
        "target_no_to_yes": fixes,
        "target_fix_rate": fixes / targets if targets else None,
        "successful_fix_revisions": len(successful),
        "successful_fix_with_regression": sum(row["any_regression"] for row in successful),
        "successful_fix_regression_rate": (
            mean(row["any_regression"] for row in successful) if successful else None
        ),
        "transition_counts": dict(Counter({
            code: sum(row["transitions"].get(code, 0) for row in valid)
            for code in ("YY", "YN", "NY", "NN")
        })),
    }


def build_review(rows: list[dict[str, Any]], controls_per_cell: int = 20) -> list[dict[str, Any]]:
    review = []
    for row in rows:
        for criterion in row["criterion_rows"]:
            if criterion["transition"] == "YN":
                review.append({
                    "review_type": "candidate_yes_to_no",
                    "manual_valid_transition": None,
                    "manual_category": None,
                    "manual_notes": "",
                    "criterion": criterion,
                    **{key: row[key] for key in (
                        "case_id", "field", "stratum", "generator", "arm", "question",
                        "old_answer", "new_answer", "diff_aids",
                    )},
                })
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for criterion in row["criterion_rows"]:
            if criterion["transition"] == "YY":
                by_cell[(row["generator"], row["arm"])].append({
                    "review_type": "yes_to_yes_control",
                    "manual_missed_regression": None,
                    "manual_notes": "",
                    "criterion": criterion,
                    **{key: row[key] for key in (
                        "case_id", "field", "stratum", "generator", "arm", "question",
                        "old_answer", "new_answer", "diff_aids",
                    )},
                })
    for cell, candidates in by_cell.items():
        ordered = sorted(
            candidates,
            key=lambda item: stable_key(SEED, "yy_control", cell, item["case_id"], item["criterion"]["criterion_id"]),
        )
        review.extend(ordered[:controls_per_cell])
    return review


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_instances(cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(cache_dir)
    )
    return [dict(row) for row in dataset["train"]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    instances = load_instances(args.dataset_cache)
    selected = select_stratified(instances, args.n_per_stratum, args.input_char_limit)
    sample_hash = hashlib.sha256(
        "\n".join(row["index"] for row in selected).encode("utf-8")
    ).hexdigest()
    by_id = {row["index"]: row for row in selected}

    initials: dict[str, dict[str, str]] = {}
    for generator_name in args.generator_models:
        runner = ModelRunner(generator_name)
        answers = generate_messages(
            runner, [initial_messages(row) for row in selected],
            args.generation_batch_size, args.generation_cap,
        )
        release_runner(runner)
        initials[generator_name] = {
            row["index"]: answer for row, answer in zip(selected, answers)
        }
    write_json(args.out_dir / "p0n_initial_answers.json", initials)

    initial_entries = []
    for row in selected:
        reference = reference_answer_text(row)
        if reference:
            initial_entries.append({
                "eval_id": f"reference|{row['index']}", "kind": "reference",
                "case_id": row["index"], "generator": None,
                "instance": row, "answer": reference,
            })
        for generator_name in args.generator_models:
            initial_entries.append({
                "eval_id": f"initial|{generator_name}|{row['index']}", "kind": "initial",
                "case_id": row["index"], "generator": generator_name,
                "instance": row, "answer": initials[generator_name][row["index"]],
            })
    initial_judgments = judge_answers(
        args.judge_model, initial_entries, args.judge_batch_size, args.judge_cap
    )
    write_jsonl(args.out_dir / "p0n_initial_judgments.jsonl", initial_judgments)
    initial_lookup = {
        (row["generator"], row["case_id"]): row
        for row in initial_judgments if row["kind"] == "initial"
    }

    revisions: list[dict[str, Any]] = []
    for generator_name in args.generator_models:
        tasks = []
        for row in selected:
            judgment = initial_lookup[(generator_name, row["index"])]
            if not judgment["judge_valid"]:
                continue
            failures = failed_ids(judgment["evaluation"])
            if not failures:
                continue
            for arm in ARMS:
                targets = target_ids(row["index"], failures, arm)
                tasks.append({
                    "case_id": row["index"], "generator": generator_name, "arm": arm,
                    "targets": targets, "instance": row,
                    "previous_answer": initials[generator_name][row["index"]],
                })
        runner = ModelRunner(generator_name)
        answers = generate_messages(
            runner,
            [
                revision_messages(task["instance"], task["previous_answer"], task["targets"])
                for task in tasks
            ],
            args.generation_batch_size, args.generation_cap,
        )
        release_runner(runner)
        for task, answer in zip(tasks, answers):
            revisions.append({**task, "answer": answer})
    write_jsonl(args.out_dir / "p0n_revisions.jsonl", revisions)

    revision_entries = [
        {
            "eval_id": f"revision|{row['generator']}|{row['arm']}|{row['case_id']}",
            "kind": "revision", "case_id": row["case_id"],
            "generator": row["generator"], "arm": row["arm"],
            "targets": row["targets"], "instance": row["instance"], "answer": row["answer"],
        }
        for row in revisions
    ]
    revision_judgments = judge_answers(
        args.judge_model, revision_entries, args.judge_batch_size, args.judge_cap
    )
    write_jsonl(args.out_dir / "p0n_revision_judgments.jsonl", revision_judgments)
    revision_lookup = {
        (row["generator"], row["arm"], row["case_id"]): row
        for row in revision_judgments
    }

    transition_rows = []
    for revision in revisions:
        key = (revision["generator"], revision["arm"], revision["case_id"])
        transition_rows.append(transition_row(
            by_id[revision["case_id"]], revision["generator"], revision["arm"],
            revision["previous_answer"], revision["answer"],
            initial_lookup[(revision["generator"], revision["case_id"])],
            revision_lookup[key], revision["targets"],
        ))

    references = [row for row in initial_judgments if row["kind"] == "reference"]
    ref_valid = [row for row in references if row["judge_valid"]]
    reference_yes = [
        decision for row in ref_valid for decision in row["evaluation"].values()
    ]
    all_judgments = initial_judgments + revision_judgments
    judge_parse_validity = mean(row["judge_valid"] for row in all_judgments)
    reference_yes_rate = mean(reference_yes) if reference_yes else 0.0
    summaries = {}
    eligibility = {}
    for generator_name in args.generator_models:
        summaries[generator_name] = {}
        initial_valid = [
            row for row in initial_judgments
            if row["kind"] == "initial" and row["generator"] == generator_name
            and row["judge_valid"]
        ]
        eligibility[generator_name] = sum(
            any(row["evaluation"].values()) and not all(row["evaluation"].values())
            for row in initial_valid
        )
        for arm in ARMS:
            summaries[generator_name][arm] = summarize_cell([
                row for row in transition_rows
                if row["generator"] == generator_name and row["arm"] == arm
            ])
    apparatus_gates = {
        "judge_parse_validity_ge_95": judge_parse_validity >= 0.95,
        "reference_yes_rate_ge_90": reference_yes_rate >= 0.90,
        "eligible_mixed_initial_ge_20_each_generator": all(
            eligibility[name] >= 20 for name in args.generator_models
        ),
    }
    candidate_count = sum(row["regressions"] for row in transition_rows if row["valid_transition"])
    if not all(apparatus_gates.values()):
        decision = "APPARATUS_FAILURE"
    elif candidate_count == 0:
        decision = "NO_CANDIDATE_REGRESSION_KEEP_LINE_CLOSED"
    else:
        decision = "MANUAL_AUDIT_REQUIRED"
    review = build_review(transition_rows)
    report = {
        "protocol": PROTOCOL,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "CC BY-NC-ND 4.0",
        "sample_size": len(selected),
        "n_per_stratum": args.n_per_stratum,
        "sample_index_sha256": sample_hash,
        "stratum_counts": dict(Counter(row["p0n_stratum"] for row in selected)),
        "generator_models": list(args.generator_models),
        "judge_model": args.judge_model,
        "judge_is_official_gpt41": False,
        "same_model_generator_judge": [
            name for name in args.generator_models if name == args.judge_model
        ],
        "judge_parse_validity": judge_parse_validity,
        "reference_answers_available": len(references),
        "reference_judgment_validity": len(ref_valid) / len(references) if references else 0.0,
        "reference_yes_rate": reference_yes_rate,
        "mixed_initial_eligibility": eligibility,
        "cell_summaries": summaries,
        "candidate_yes_to_no": candidate_count,
        "apparatus_gates": apparatus_gates,
        "decision": decision,
        "manual_review_items": len(review),
        "interpretation_guard": (
            "Local-judge P0n0 is an external-dataset case-study screen, not an official "
            "RefineBench score, real-world prevalence estimate, or automatic semantic-regression audit."
        ),
    }
    write_jsonl(args.out_dir / "p0n_transition_rows.jsonl", transition_rows)
    write_jsonl(args.out_dir / "p0n_manual_review.jsonl", review)
    write_json(args.out_dir / "p0n_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-models", nargs="+", default=list(DEFAULT_GENERATORS))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--n-per-stratum", type=int, default=8)
    parser.add_argument("--input-char-limit", type=int, default=50000)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--generation-cap", type=int, default=2048)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-cap", type=int, default=256)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
