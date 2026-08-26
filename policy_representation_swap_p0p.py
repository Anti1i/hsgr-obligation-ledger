"""P0p cross-policy task-representation swap case study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from refinebench_revision_audit_p0n import (
    DATASET_NAME,
    DATASET_REVISION,
    STRATA,
    field_stratum,
    query_text,
    reference_answer_text,
    stable_key,
)


PROTOCOL = "EXPERIMENT_PROTOCOL_POLICY_REPRESENTATION_SWAP_P0P.md"
SEED = 20260826
MODEL_SPECS = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
}
DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct"
REFERENCE_CHAR_MIN = 400
REFERENCE_CHAR_MAX = 12000
EXCLUDED_CASES = {
    "refinebench-000010", "refinebench-000100", "refinebench-000147",
    "refinebench-000347", "refinebench-000577", "refinebench-000581",
    "refinebench-000827", "refinebench-000833",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_array(text: str) -> Any:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array")
    return json.loads(text[start : end + 1])


def parse_native_checklist(text: str, expected_count: int) -> list[str]:
    value = extract_json_array(text)
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"expected {expected_count} checklist items")
    items = []
    for item in value:
        if not isinstance(item, str) or len(item.strip()) < 8:
            raise ValueError("checklist items must be nontrivial strings")
        items.append(" ".join(item.strip().split()))
    normalized = {item.casefold() for item in items}
    if len(normalized) != len(items):
        raise ValueError("duplicate checklist item")
    return items


def parse_structural_plan(text: str, item_count: int) -> list[list[int]]:
    value = extract_json_array(text)
    if not isinstance(value, list) or not 2 <= len(value) <= min(6, item_count):
        raise ValueError("plan must contain between 2 and 6 nonempty phases")
    groups: list[list[int]] = []
    for group in value:
        if not isinstance(group, list) or not group:
            raise ValueError("each phase must be a nonempty list")
        parsed = []
        for item in group:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("requirement IDs must be integers")
            parsed.append(item)
        groups.append(parsed)
    flattened = [item for group in groups for item in group]
    if sorted(flattened) != list(range(1, item_count + 1)):
        raise ValueError("every requirement ID must occur exactly once")
    return groups


def parse_yes_no_lines(text: str, count: int) -> tuple[dict[int, bool], bool]:
    matches = re.findall(r"(?:<)?[QC]?(\d+)(?:>)?\s*:\s*(Yes|No)\b", text, re.I)
    parsed: dict[int, bool] = {}
    for number, label in matches:
        index = int(number)
        if index in parsed:
            return {}, False
        parsed[index] = label.casefold() == "yes"
    expected = set(range(1, count + 1))
    return (parsed, True) if set(parsed) == expected else ({}, False)


def render_requirements(items: list[str], prefix: str = "C") -> str:
    return "\n".join(f"{prefix}{index}: {item}" for index, item in enumerate(items, 1))


def render_structural_plan(items: list[str], groups: list[list[int]]) -> str:
    lines = []
    for phase_index, group in enumerate(groups, 1):
        lines.append(f"Phase {phase_index}:")
        for item_id in group:
            lines.append(f"- C{item_id}: {items[item_id - 1]}")
    return "\n".join(lines)


def native_checklist_prompt(task: str, item_count: int) -> str:
    return f"""Read the task and compile an execution checklist for a language model that will answer it.

TASK:
{task}

Return exactly {item_count} atomic, nonredundant requirements. Include only requirements that follow from the task. Do not answer the task. Return only a JSON array of {item_count} strings."""


def native_repair_prompt(task: str, item_count: int, previous: str) -> str:
    return f"""Your previous checklist did not match the required JSON format.

TASK:
{task}

PREVIOUS OUTPUT:
{previous}

Return only a valid JSON array containing exactly {item_count} distinct, nonempty string requirements. Do not answer the task."""


def structural_plan_prompt(task: str, canonical: list[str]) -> str:
    return f"""Organize the canonical requirements into an execution order for answering the task.

TASK:
{task}

CANONICAL REQUIREMENTS:
{render_requirements(canonical)}

Return only a JSON array of 2 to {min(6, len(canonical))} nonempty arrays of integer requirement IDs. Every integer from 1 through {len(canonical)} must occur exactly once. You may only group and reorder IDs; do not add, remove, paraphrase, or answer requirements.

Example format: [[1, 3], [2, 4, 5]]"""


def structural_repair_prompt(task: str, canonical: list[str], previous: str) -> str:
    return f"""Repair the invalid plan format below.

TASK:
{task}

CANONICAL REQUIREMENT IDS: 1 through {len(canonical)}
INVALID OUTPUT:
{previous}

Return only a JSON array of 2 to {min(6, len(canonical))} nonempty integer arrays. Every ID from 1 through {len(canonical)} must occur exactly once."""


def answer_prompt(task: str, representation: str) -> str:
    return f"""Answer the task directly and completely. Use the planning aid to ensure that all relevant requirements are satisfied. Do not mention the planning aid or checklist in the answer.

TASK:
{task}

PLANNING AID:
{representation}

ANSWER:"""


def evaluation_prompt(instance: dict[str, Any], answer: str) -> str:
    checklist = render_requirements([str(item) for item in instance["checklist"]], "Q")
    return f"""Evaluate the answer against every canonical requirement. Use the full task context. Answer Yes only if the answer fully satisfies that requirement.

FULL TASK:
{query_text(instance)}

ANSWER TO EVALUATE:
{answer}

CANONICAL REQUIREMENTS:
{checklist}

Return exactly one line per requirement in the form Q<number>: Yes or Q<number>: No. Do not explain."""


def coverage_prompt(canonical: list[str], native: list[str]) -> str:
    return f"""For each canonical requirement, decide whether the candidate checklist contains an explicit semantically equivalent requirement. Judge coverage, not whether the candidate answer itself satisfies the task.

CANONICAL REQUIREMENTS:
{render_requirements(canonical, 'C')}

CANDIDATE CHECKLIST:
{render_requirements(native, 'N')}

Return exactly one line per canonical requirement: C<number>: Yes or C<number>: No. Do not explain."""


class TextRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        self.torch = torch
        self.model_id = model_id
        self.chat_template_kwargs = (
            {"enable_thinking": False} if model_id.startswith("Qwen/Qwen3") else {}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).cuda().eval()
        self.context_limit = int(getattr(self.model.config, "max_position_embeddings", 4096))
        print(
            f"[model] {model_id} context={self.context_limit} "
            f"gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def _chat_text(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def generate(
        self, prompts: list[str], batch_size: int, max_new_tokens: int, label: str,
    ) -> list[str]:
        torch = self.torch
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = [self._chat_text(prompt) for prompt in prompts[start : start + batch_size]]
            encoded = self.tokenizer(
                chunk, padding=True, add_special_tokens=False, return_tensors="pt"
            )
            width = int(encoded["input_ids"].shape[1])
            if width + max_new_tokens > self.context_limit:
                raise RuntimeError(
                    f"{self.model_id} prompt exceeds context: {width}+{max_new_tokens}>"
                    f"{self.context_limit}"
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
            new_tokens = generated[:, width:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            print(
                f"[{label}] {min(start + batch_size, len(prompts))}/{len(prompts)}",
                flush=True,
            )
            del encoded, generated, new_tokens
            torch.cuda.empty_cache()
        return outputs


def release_runner(runner: TextRunner) -> None:
    torch = runner.torch
    del runner.model
    del runner.tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def eligible_task(instance: dict[str, Any], query_char_limit: int) -> bool:
    checklist = [str(item).strip() for item in (instance.get("checklist") or [])]
    reference = reference_answer_text(instance)
    return bool(
        str(instance.get("index", "")).strip()
        and instance.get("index") not in EXCLUDED_CASES
        and field_stratum(str(instance.get("field", ""))) in STRATA
        and 5 <= len(checklist) <= 12
        and all(checklist)
        and reference
        and REFERENCE_CHAR_MIN <= len(reference) <= REFERENCE_CHAR_MAX
        and len(query_text(instance)) <= query_char_limit
    )


def selection_audit(
    instances: list[dict[str, Any]], query_char_limit: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        candidates = [
            instance for instance in instances
            if field_stratum(str(instance.get("field", ""))) == stratum
        ]
        funnel = {
            "field_total": len(candidates),
            "id_present_and_not_excluded": 0,
            "checklist_count_5_to_12": 0,
            "nonempty_checklist_items": 0,
            "reference_present": 0,
            "reference_chars_within_bounds": 0,
            "query_within_char_limit": 0,
            "all_conditions": 0,
        }
        metadata = []
        for instance in candidates:
            checklist = [str(item).strip() for item in (instance.get("checklist") or [])]
            reference = reference_answer_text(instance)
            flags = {
                "id_present_and_not_excluded": bool(
                    str(instance.get("index", "")).strip()
                    and instance.get("index") not in EXCLUDED_CASES
                ),
                "checklist_count_5_to_12": 5 <= len(checklist) <= 12,
                "nonempty_checklist_items": bool(checklist and all(checklist)),
                "reference_present": bool(reference),
                "reference_chars_within_bounds": bool(
                    reference
                    and REFERENCE_CHAR_MIN <= len(reference) <= REFERENCE_CHAR_MAX
                ),
                "query_within_char_limit": len(query_text(instance)) <= query_char_limit,
            }
            for name, value in flags.items():
                funnel[name] += int(value)
            valid = all(flags.values())
            funnel["all_conditions"] += int(valid)
            metadata.append({
                "task_id": instance.get("index"),
                "checklist_count": len(checklist),
                "reference_chars": len(reference) if reference else 0,
                "query_chars": len(query_text(instance)),
                "eligible": valid,
                "failed_conditions": [name for name, value in flags.items() if not value],
            })
        rows[stratum] = {"funnel_marginal_counts": funnel, "candidate_metadata": metadata}
    return {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "query_char_limit": query_char_limit,
        "reference_char_bounds": [REFERENCE_CHAR_MIN, REFERENCE_CHAR_MAX],
        "strata": rows,
    }


def select_tasks(
    instances: list[dict[str, Any]], per_stratum: int, query_char_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        if eligible_task(instance, query_char_limit):
            grouped[field_stratum(str(instance["field"]))].append(instance)
    selected = []
    for stratum in STRATA:
        ordered = sorted(
            grouped[stratum], key=lambda row: stable_key(SEED, "p0p_task", row["index"])
        )
        if len(ordered) < per_stratum:
            raise RuntimeError(f"insufficient eligible tasks for {stratum}")
        for row in ordered[:per_stratum]:
            selected.append({**row, "p0p_stratum": stratum})
    return selected


def representation_key(task_id: str, kind: str, source: str) -> tuple[str, str, str]:
    return task_id, kind, source


def create_representations(
    tasks: list[dict[str, Any]], model_specs: dict[str, str], batch_size: int,
    checklist_cap: int, plan_cap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, model_id in model_specs.items():
        runner = TextRunner(model_id)
        native_prompts = [
            native_checklist_prompt(query_text(task), len(task["checklist"])) for task in tasks
        ]
        native_raw = runner.generate(native_prompts, batch_size, checklist_cap, f"native-{source}")
        plan_prompts = [
            structural_plan_prompt(query_text(task), list(task["checklist"])) for task in tasks
        ]
        plan_raw = runner.generate(plan_prompts, batch_size, plan_cap, f"plan-{source}")

        for task, raw in zip(tasks, native_raw):
            repaired = False
            try:
                parsed = parse_native_checklist(raw, len(task["checklist"]))
                valid = True
                error = ""
            except Exception as exc:
                repaired = True
                retry = runner.generate(
                    [native_repair_prompt(query_text(task), len(task["checklist"]), raw)],
                    1, checklist_cap, f"native-repair-{source}",
                )[0]
                raw = retry
                try:
                    parsed = parse_native_checklist(raw, len(task["checklist"]))
                    valid, error = True, ""
                except Exception as retry_exc:
                    parsed, valid, error = [], False, str(retry_exc)
            rows.append({
                "task_id": task["index"], "source": source, "source_model": model_id,
                "kind": "native", "raw": raw, "parsed": parsed, "valid": valid,
                "repaired": repaired, "error": error,
            })

        for task, raw in zip(tasks, plan_raw):
            repaired = False
            try:
                parsed = parse_structural_plan(raw, len(task["checklist"]))
                valid = True
                error = ""
            except Exception:
                repaired = True
                retry = runner.generate(
                    [structural_repair_prompt(query_text(task), list(task["checklist"]), raw)],
                    1, plan_cap, f"plan-repair-{source}",
                )[0]
                raw = retry
                try:
                    parsed = parse_structural_plan(raw, len(task["checklist"]))
                    valid, error = True, ""
                except Exception as retry_exc:
                    parsed, valid, error = [], False, str(retry_exc)
            rows.append({
                "task_id": task["index"], "source": source, "source_model": model_id,
                "kind": "structural", "raw": raw, "parsed": parsed, "valid": valid,
                "repaired": repaired, "error": error,
            })
        release_runner(runner)
    return rows


def representation_text(
    task: dict[str, Any], row: dict[str, Any] | None, kind: str,
) -> str:
    canonical = list(task["checklist"])
    if kind == "canonical":
        return render_requirements(canonical)
    if row is None or not row["valid"]:
        raise ValueError("invalid representation")
    if kind == "native":
        return render_requirements(row["parsed"], "R")
    if kind == "structural":
        return render_structural_plan(canonical, row["parsed"])
    raise ValueError(kind)


def generate_answers(
    tasks: list[dict[str, Any]], representations: list[dict[str, Any]],
    model_specs: dict[str, str], batch_size: int, answer_cap: int,
) -> list[dict[str, Any]]:
    lookup = {
        representation_key(row["task_id"], row["kind"], row["source"]): row
        for row in representations
    }
    tasks_by_id = {task["index"]: task for task in tasks}
    rows: list[dict[str, Any]] = []
    for target, model_id in model_specs.items():
        requests = []
        for task_id in sorted(tasks_by_id, key=lambda value: stable_key(SEED, "answer", value)):
            task = tasks_by_id[task_id]
            requests.append((task, "canonical", "canonical", None))
            for kind in ("native", "structural"):
                for source in model_specs:
                    rep = lookup[representation_key(task_id, kind, source)]
                    requests.append((task, kind, source, rep))
        prompts = [
            answer_prompt(query_text(task), representation_text(task, rep, kind))
            for task, kind, _source, rep in requests
        ]
        runner = TextRunner(model_id)
        outputs = runner.generate(prompts, batch_size, answer_cap, f"answer-{target}")
        release_runner(runner)
        for (task, kind, source, _rep), answer in zip(requests, outputs):
            rows.append({
                "answer_id": f"{task['index']}|{target}|{kind}|{source}",
                "task_id": task["index"], "target": target, "target_model": model_id,
                "kind": kind, "source": source, "answer": answer,
            })
    return rows


def judge_all(
    tasks: list[dict[str, Any]], answers: list[dict[str, Any]],
    representations: list[dict[str, Any]], judge_model: str,
    batch_size: int, judge_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks_by_id = {task["index"]: task for task in tasks}
    eval_entries = []
    for task in tasks:
        eval_entries.append({
            "eval_id": f"reference|{task['index']}", "kind": "reference",
            "task_id": task["index"], "answer": reference_answer_text(task),
        })
    for row in answers:
        eval_entries.append({**row, "eval_id": f"answer|{row['answer_id']}"})
    order = sorted(
        range(len(eval_entries)),
        key=lambda index: stable_key(SEED, "eval_order", eval_entries[index]["eval_id"]),
    )
    prompts = [
        evaluation_prompt(tasks_by_id[eval_entries[index]["task_id"]], eval_entries[index]["answer"])
        for index in order
    ]

    valid_native = [row for row in representations if row["kind"] == "native" and row["valid"]]
    coverage_order = sorted(
        range(len(valid_native)),
        key=lambda index: stable_key(
            SEED, "coverage_order", valid_native[index]["task_id"], valid_native[index]["source"]
        ),
    )
    coverage_prompts = [
        coverage_prompt(
            list(tasks_by_id[valid_native[index]["task_id"]]["checklist"]),
            valid_native[index]["parsed"],
        )
        for index in coverage_order
    ]

    runner = TextRunner(judge_model)
    raw_eval = runner.generate(prompts, batch_size, judge_cap, "judge-answer")
    raw_coverage = runner.generate(coverage_prompts, batch_size, judge_cap, "judge-coverage")
    release_runner(runner)

    judged: list[dict[str, Any] | None] = [None] * len(eval_entries)
    for index, raw in zip(order, raw_eval):
        task = tasks_by_id[eval_entries[index]["task_id"]]
        parsed, valid = parse_yes_no_lines(raw, len(task["checklist"]))
        judged[index] = {
            **eval_entries[index], "raw_judgment": raw,
            "evaluation": parsed, "judge_valid": valid,
        }
    coverage_rows: list[dict[str, Any] | None] = [None] * len(valid_native)
    for index, raw in zip(coverage_order, raw_coverage):
        task = tasks_by_id[valid_native[index]["task_id"]]
        parsed, valid = parse_yes_no_lines(raw, len(task["checklist"]))
        coverage_rows[index] = {
            "task_id": valid_native[index]["task_id"],
            "source": valid_native[index]["source"],
            "raw_judgment": raw, "coverage": parsed, "judge_valid": valid,
        }
    return (
        [row for row in judged if row is not None],
        [row for row in coverage_rows if row is not None],
    )


def answer_score(row: dict[str, Any]) -> float:
    return mean(row["evaluation"].values())


def matrix_statistic(
    score_map: dict[tuple[str, str, str], float], task_ids: list[str],
    policies: list[str], source_permutations: dict[str, list[str]] | None = None,
) -> float:
    diagonal = []
    off_diagonal = []
    for task_id in task_ids:
        mapping = source_permutations.get(task_id, policies) if source_permutations else policies
        source_for_label = dict(zip(policies, mapping))
        for target in policies:
            matched_source = source_for_label[target]
            diagonal.append(score_map[(task_id, target, matched_source)])
            for source in policies:
                if source != matched_source:
                    off_diagonal.append(score_map[(task_id, target, source)])
    return mean(diagonal) - mean(off_diagonal)


def matrix_summary(
    kind: str, judged_answers: list[dict[str, Any]], task_ids: list[str],
    policies: list[str], permutations: int,
) -> dict[str, Any]:
    rows = [
        row for row in judged_answers
        if row.get("kind") == kind and row["judge_valid"]
    ]
    score_map = {
        (row["task_id"], row["target"], row["source"]): answer_score(row) for row in rows
    }
    expected = len(task_ids) * len(policies) * len(policies)
    if len(score_map) != expected:
        return {"valid": False, "expected_cells": expected, "observed_cells": len(score_map)}
    cells = {
        target: {
            source: mean(score_map[(task_id, target, source)] for task_id in task_ids)
            for source in policies
        }
        for target in policies
    }
    per_policy = {
        target: (
            mean(score_map[(task_id, target, target)] for task_id in task_ids)
            - mean(
                score_map[(task_id, target, source)]
                for task_id in task_ids for source in policies if source != target
            )
        )
        for target in policies
    }
    observed = matrix_statistic(score_map, task_ids, policies)
    rng = random.Random(SEED + (1 if kind == "structural" else 0))
    null_values = []
    for _ in range(permutations):
        permuted = {}
        for task_id in task_ids:
            values = policies.copy()
            rng.shuffle(values)
            permuted[task_id] = values
        null_values.append(matrix_statistic(score_map, task_ids, policies, permuted))
    p_value = (1 + sum(value >= observed for value in null_values)) / (1 + permutations)

    bootstrap_rng = random.Random(SEED + (11 if kind == "structural" else 10))
    bootstrap = []
    for _ in range(2000):
        sampled = [bootstrap_rng.choice(task_ids) for _ in task_ids]
        # Duplicate task IDs need independent aliases to preserve multiplicity.
        diagonal = []
        off = []
        for task_id in sampled:
            for target in policies:
                diagonal.append(score_map[(task_id, target, target)])
                off.extend(
                    score_map[(task_id, target, source)]
                    for source in policies if source != target
                )
        bootstrap.append(mean(diagonal) - mean(off))
    bootstrap.sort()
    return {
        "valid": True,
        "cells": cells,
        "diagonal_advantage": observed,
        "per_policy_diagonal_advantage": per_policy,
        "positive_policy_count": sum(value > 0 for value in per_policy.values()),
        "permutation_p_one_sided": p_value,
        "bootstrap_95_interval": [bootstrap[49], bootstrap[1949]],
    }


def canonical_summary(
    judged_answers: list[dict[str, Any]], task_ids: list[str], policies: list[str],
) -> dict[str, float | None]:
    lookup = {
        (row["task_id"], row["target"]): answer_score(row)
        for row in judged_answers if row.get("kind") == "canonical" and row["judge_valid"]
    }
    result: dict[str, float | None] = {}
    for target in policies:
        values = [lookup[(task_id, target)] for task_id in task_ids if (task_id, target) in lookup]
        result[target] = mean(values) if len(values) == len(task_ids) else None
    return result


def build_report(
    tasks: list[dict[str, Any]], representations: list[dict[str, Any]],
    judged: list[dict[str, Any]], coverage_rows: list[dict[str, Any]],
    model_specs: dict[str, str], judge_model: str, permutations: int,
) -> dict[str, Any]:
    policies = list(model_specs)
    task_ids = [task["index"] for task in tasks]
    reference_rows = [row for row in judged if row["kind"] == "reference"]
    answer_rows = [row for row in judged if row["kind"] != "reference"]
    reference_values = [
        value for row in reference_rows if row["judge_valid"]
        for value in row["evaluation"].values()
    ]
    coverage_by_source: dict[str, float | None] = {}
    for source in policies:
        source_rows = [
            row for row in coverage_rows if row["source"] == source and row["judge_valid"]
        ]
        values = [value for row in source_rows for value in row["coverage"].values()]
        coverage_by_source[source] = (
            mean(values) if len(source_rows) == len(task_ids) and values else None
        )
    coverage_values = [value for value in coverage_by_source.values() if value is not None]
    native_quality_gate = bool(
        len(coverage_values) == len(policies)
        and min(coverage_values) >= 0.75
        and max(coverage_values) - min(coverage_values) <= 0.10
    )
    native = matrix_summary("native", answer_rows, task_ids, policies, permutations)
    structural = matrix_summary("structural", answer_rows, task_ids, policies, permutations)
    canonical = canonical_summary(answer_rows, task_ids, policies)
    own_structural_vs_canonical = {
        target: structural["cells"][target][target] - canonical[target]
        for target in policies
    } if structural.get("valid") and all(canonical[target] is not None for target in policies) else {}
    own_gain = mean(own_structural_vs_canonical.values()) if own_structural_vs_canonical else None

    mandatory_apparatus = {
        "ten_tasks_selected": len(tasks) == 10,
        "all_representations_parse": all(row["valid"] for row in representations),
        "all_structural_plans_exactly_cover_ids": all(
            row["valid"] for row in representations if row["kind"] == "structural"
        ),
        "all_evaluator_outputs_parse": all(row["judge_valid"] for row in judged + coverage_rows),
        "reference_yes_rate_ge_90": mean(reference_values) >= 0.90 if reference_values else False,
    }
    structural_gates = {
        "diagonal_advantage_ge_3pp": structural.get("diagonal_advantage", -1) >= 0.03,
        "positive_for_at_least_two_policies": structural.get("positive_policy_count", 0) >= 2,
        "permutation_p_le_10pct": structural.get("permutation_p_one_sided", 1) <= 0.10,
        "own_structural_vs_canonical_gain_ge_3pp": own_gain is not None and own_gain >= 0.03,
    }
    native_gates = {
        "coverage_matched": native_quality_gate,
        "diagonal_advantage_ge_5pp": native.get("diagonal_advantage", -1) >= 0.05,
        "positive_for_at_least_two_policies": native.get("positive_policy_count", 0) >= 2,
        "permutation_p_le_10pct": native.get("permutation_p_one_sided", 1) <= 0.10,
    }
    if not all(mandatory_apparatus.values()):
        decision = "APPARATUS_FAILURE"
    elif all(structural_gates.values()):
        decision = "PROVISIONAL_STRUCTURAL_PASS_REVIEW_REQUIRED"
    elif all(native_gates.values()):
        decision = "CONTENT_OR_WORDING_CONFOUNDED"
    else:
        decision = "STOP_NO_ALIGNMENT_SIGNAL"
    sample_hash = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()
    return {
        "protocol": PROTOCOL,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "task_ids": task_ids,
        "task_index_sha256": sample_hash,
        "stratum_counts": {
            stratum: sum(task["p0p_stratum"] == stratum for task in tasks) for stratum in STRATA
        },
        "model_specs": model_specs,
        "judge_model": judge_model,
        "judge_is_official_refinebench_gpt41": False,
        "answer_generations": len(answer_rows),
        "reference_yes_rate": mean(reference_values) if reference_values else None,
        "native_coverage_by_source": coverage_by_source,
        "native_quality_gate": native_quality_gate,
        "canonical_scores": canonical,
        "native_matrix": native,
        "structural_matrix": structural,
        "own_structural_vs_canonical": own_structural_vs_canonical,
        "mean_own_structural_vs_canonical": own_gain,
        "apparatus_gates": mandatory_apparatus,
        "native_support_gates": native_gates,
        "structural_mechanism_gates": structural_gates,
        "decision": decision,
        "interpretation_guard": (
            "Small deterministic cross-model case study with a local Qwen judge. A pass requires "
            "manual audit and does not establish hidden-state alignment, architectural necessity, "
            "model-native language, novelty, or population-level generality."
        ),
    }


def build_manual_review(
    tasks: list[dict[str, Any]], judged: list[dict[str, Any]], policies: list[str],
) -> list[dict[str, Any]]:
    tasks_by_id = {task["index"]: task for task in tasks}
    rows = [row for row in judged if row.get("kind") in {"native", "structural"}]
    lookup = {
        (row["task_id"], row["target"], row["kind"], row["source"]): row for row in rows
    }
    review = []
    for task_id in tasks_by_id:
        for target in policies:
            for kind in ("native", "structural"):
                own = lookup[(task_id, target, kind, target)]
                others = [
                    lookup[(task_id, target, kind, source)]
                    for source in policies if source != target
                ]
                own_score = answer_score(own)
                other_score = mean(answer_score(row) for row in others)
                if own_score != other_score:
                    review.append({
                        "task_id": task_id, "target": target, "kind": kind,
                        "own_score": own_score, "other_mean_score": other_score,
                        "manual_valid_difference": None, "manual_notes": "",
                        "question": tasks_by_id[task_id]["question"],
                        "own_answer": own["answer"],
                        "other_answers": [
                            {"source": row["source"], "score": answer_score(row), "answer": row["answer"]}
                            for row in others
                        ],
                    })
    return review


def run(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_NAME, revision=DATASET_REVISION, split="train", cache_dir=str(args.dataset_cache)
    )
    instances = list(dataset)
    audit = selection_audit(instances, args.query_char_limit)
    write_json(args.out_dir / "p0p_selection_audit.json", audit)
    if args.selection_audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return audit
    tasks = select_tasks(instances, args.per_stratum, args.query_char_limit)
    write_jsonl(args.out_dir / "p0p_selected_tasks.jsonl", tasks)
    model_specs = dict(MODEL_SPECS)

    representations = create_representations(
        tasks, model_specs, args.representation_batch_size,
        args.checklist_cap, args.plan_cap,
    )
    write_jsonl(args.out_dir / "p0p_representations.jsonl", representations)
    if not all(row["valid"] for row in representations):
        report = {
            "protocol": PROTOCOL, "task_ids": [task["index"] for task in tasks],
            "model_specs": model_specs, "decision": "APPARATUS_FAILURE",
            "reason": "one or more representations failed parsing after one repair",
        }
        write_json(args.out_dir / "p0p_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report

    answers = generate_answers(
        tasks, representations, model_specs, args.answer_batch_size, args.answer_cap
    )
    write_jsonl(args.out_dir / "p0p_answers.jsonl", answers)
    judged, coverage_rows = judge_all(
        tasks, answers, representations, args.judge_model,
        args.judge_batch_size, args.judge_cap,
    )
    write_jsonl(args.out_dir / "p0p_judgments.jsonl", judged)
    write_jsonl(args.out_dir / "p0p_native_coverage.jsonl", coverage_rows)
    report = build_report(
        tasks, representations, judged, coverage_rows,
        model_specs, args.judge_model, args.permutations,
    )
    write_json(args.out_dir / "p0p_report.json", report)
    review = build_manual_review(tasks, judged, list(model_specs))
    write_jsonl(args.out_dir / "p0p_manual_review.jsonl", review)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--query-char-limit", type=int, default=8000)
    parser.add_argument("--representation-batch-size", type=int, default=1)
    parser.add_argument("--answer-batch-size", type=int, default=2)
    parser.add_argument("--judge-batch-size", type=int, default=3)
    parser.add_argument("--checklist-cap", type=int, default=768)
    parser.add_argument("--plan-cap", type=int, default=384)
    parser.add_argument("--answer-cap", type=int, default=1024)
    parser.add_argument("--judge-cap", type=int, default=384)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--selection-audit-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
