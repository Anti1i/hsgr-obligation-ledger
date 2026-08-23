"""Frozen GAMUT typed ordered-process minimal-repair case study P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from asqa_missing_selector_p6x import ModelRunner


PROTOCOL = "EXPERIMENT_PROTOCOL_GAMUT_PROCESS_REPAIR_P0.md"
SELECTION_SALT = "20260823-gamut-process-repair-p0"
ARMS = (
    "flat_full_rewrite",
    "typed_full_rewrite",
    "flat_span_patch",
    "typed_span_patch",
)
PROCESS_TRIGGER = re.compile(
    r"(?:relative\s+order|chronological\s+(?:order|sequence)|mandatory\s+sequence|"
    r"correct\s+(?:order|sequence)|ordered\s+(?:steps|process)|sequence\s+of\s+steps)",
    re.IGNORECASE,
)
STOP_AFTER_LIST = re.compile(
    r"\b(?:Missing steps|Extra unlisted steps|It passes|The element passes|The answer passes)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Requirement:
    handle: str
    ingredient: str


@dataclass(frozen=True)
class ProcessCase:
    id: str
    question: str
    evidence: str
    target: Requirement
    steps: tuple[str, ...]
    answer_critical: tuple[Requirement, ...]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_step(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,;:.")


def extract_ordered_steps(ingredient: str) -> tuple[str, ...]:
    """Mechanically parse an explicitly numbered ordered list."""
    if not PROCESS_TRIGGER.search(ingredient):
        return ()
    lowered = ingredient.casefold()
    anchors = ["master list:", "ordered steps:", "sequence:", "order:"]
    starts = [lowered.find(anchor) for anchor in anchors if lowered.find(anchor) >= 0]
    tail = ingredient[min(starts) :] if starts else ingredient
    stop = STOP_AFTER_LIST.search(tail)
    if stop:
        tail = tail[: stop.start()]
    matches = list(re.finditer(r"(?<!\d)(\d+)[.)]\s+", tail))
    if len(matches) < 2:
        return ()
    steps: list[str] = []
    expected = 1
    for index, match in enumerate(matches):
        number = int(match.group(1))
        if number != expected:
            return ()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tail)
        step = clean_step(tail[match.end() : end])
        if not step:
            return ()
        steps.append(step)
        expected += 1
    return tuple(steps)


def _tier(rubrics: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = rubrics.get(name, rubrics.get(name.replace("_", " "), []))
    return value if isinstance(value, list) else []


def _requirements(rubrics: dict[str, Any]) -> tuple[Requirement, ...]:
    result = []
    for raw in _tier(rubrics, "Answer_Critical"):
        if not isinstance(raw, dict):
            continue
        handle, ingredient = raw.get("Handle"), raw.get("Ingredient")
        if isinstance(handle, str) and isinstance(ingredient, str) and ingredient.strip():
            result.append(Requirement(handle.strip(), ingredient.strip()))
    return tuple(result)


def _evidence(rubrics: dict[str, Any], max_snippets: int = 12, max_chars: int = 550) -> str:
    tiers = (
        _tier(rubrics, "Answer_Critical"),
        _tier(rubrics, "Valuable"),
        _tier(rubrics, "Context"),
    )
    cited: list[str] = []
    for tier in tiers:
        for raw in tier:
            if not isinstance(raw, dict):
                continue
            for specific in raw.get("Specifics", []) or []:
                if isinstance(specific, dict) and isinstance(specific.get("Citation"), str):
                    cited.append(specific["Citation"])
    cited_order = {value: index for index, value in enumerate(dict.fromkeys(cited))}
    snippets = rubrics.get("Snippets", [])
    snippets = snippets if isinstance(snippets, list) else []
    ranked = sorted(
        (snippet for snippet in snippets if isinstance(snippet, dict)),
        key=lambda snippet: (
            0 if str(snippet.get("id", "")) in cited_order else 1,
            cited_order.get(str(snippet.get("id", "")), 10**9),
            str(snippet.get("id", "")),
        ),
    )[:max_snippets]
    lines = []
    for snippet in ranked:
        text = re.sub(r"\s+", " ", str(snippet.get("Text", ""))).strip()
        if not text:
            continue
        title = re.sub(r"\s+", " ", str(snippet.get("Title", ""))).strip()
        sid = str(snippet.get("id", "?"))
        lines.append(f"[{sid}] {title}\n{text[:max_chars]}")
    return "\n\n".join(lines)


def load_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    path = Path(dataset)
    if path.suffix.casefold() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    from datasets import load_dataset

    if path.suffix.casefold() == ".parquet":
        loaded = load_dataset("parquet", data_files=str(path), split="train")
    else:
        loaded = load_dataset(dataset, split=split)
    return [dict(row) for row in loaded]


def build_process_cases(rows: list[dict[str, Any]], n: int) -> tuple[list[ProcessCase], dict[str, Any]]:
    by_id: dict[str, list[ProcessCase]] = defaultdict(list)
    process_elements = 0
    parseable_elements = 0
    for row_index, row in enumerate(rows):
        rubrics = row.get("rubrics")
        if not isinstance(rubrics, dict):
            continue
        requirements = _requirements(rubrics)
        candidates = []
        for requirement in requirements:
            if PROCESS_TRIGGER.search(requirement.ingredient):
                process_elements += 1
            steps = extract_ordered_steps(requirement.ingredient)
            if steps:
                parseable_elements += 1
                candidates.append((requirement, steps))
        if not candidates:
            continue
        case_id = str(row.get("session_id", row.get("id", f"row_{row_index}")))
        question = str(row.get("question", rubrics.get("Question", ""))).strip()
        evidence = _evidence(rubrics)
        if not question or not evidence:
            continue
        requirement, steps = min(
            candidates,
            key=lambda item: hash_key(f"{SELECTION_SALT}|target|{case_id}|{item[0].handle}"),
        )
        by_id[case_id].append(
            ProcessCase(case_id, question, evidence, requirement, steps, requirements)
        )
    unique = [values[0] for _, values in sorted(by_id.items())]
    ordered = sorted(unique, key=lambda case: hash_key(f"{SELECTION_SALT}|case|{case.id}"))
    selected = ordered[:n]
    audit = {
        "dataset_rows": len(rows),
        "triggered_process_elements": process_elements,
        "parseable_process_elements": parseable_elements,
        "unique_parseable_cases": len(ordered),
        "selected_cases": len(selected),
        "selected_id_sha256": hash_key("\n".join(case.id for case in selected) + "\n"),
    }
    return selected, audit


def baseline_prompt(case: ProcessCase) -> str:
    return (
        "Answer the question using only the fixed evidence. Write one coherent, useful answer "
        "of roughly 140 to 220 words. Include concrete details and explain any procedure in a "
        "reader-usable way. Do not mention the evidence labels or these instructions.\n\n"
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\nAnswer:"
    )


def judge_prompt(case: ProcessCase, answer: str, handle: str, requirement: str) -> str:
    return (
        "Evaluate one requirement against an answer using the fixed evidence. Reply with exactly "
        "one label and nothing else. A means the answer fully and correctly meets the requirement. "
        "B means it only partially meets it, misses it, or materially contradicts it. Judge meaning, "
        "not exact wording.\n\n"
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n"
        f"Answer:\n{answer}\n\nRequirement [{handle}]: {requirement}\n\nLabel:"
    )


def component_requirement(step: str) -> str:
    return f"The answer explicitly states this process step: {step}."


def positive_process_answer(steps: tuple[str, ...]) -> str:
    connectors = ["First", "Next", "Then", "After that", "Finally"]
    return " ".join(
        f"{connectors[min(index, len(connectors) - 1)]}, {step.rstrip('.')} ."
        for index, step in enumerate(steps)
    ).replace(" .", ".")


def negative_process_answer(steps: tuple[str, ...]) -> str:
    """Mention the same components in reverse so the sequence check must fail."""
    return positive_process_answer(tuple(reversed(steps)))


def typed_graph(case: ProcessCase) -> str:
    nodes = "\n".join(f"- P{index}: {step}" for index, step in enumerate(case.steps, 1))
    edges = "\n".join(
        f"- P{index} -> P{index + 1} (must occur before)"
        for index in range(1, len(case.steps))
    )
    return f"Type: ORDERED_PROCESS\nNodes:\n{nodes}\nDirected constraints:\n{edges}"


def preservation_text(requirements: list[Requirement]) -> str:
    if not requirements:
        return "- None detected by the frozen baseline check."
    return "\n".join(f"- [{item.handle}] {item.ingredient}" for item in requirements)


def repair_prompt(
    case: ProcessCase,
    answer: str,
    arm: str,
    preserved: list[Requirement],
) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    typed = arm.startswith("typed_")
    patch = arm.endswith("span_patch")
    target = (
        f"Failed requirement in ordinary text:\n{case.target.ingredient}"
        if not typed
        else f"Failed typed obligation:\n{typed_graph(case)}\n\nOriginal wording:\n{case.target.ingredient}"
    )
    output = (
        "Return only one JSON object with exactly two string fields: old_text and new_text. "
        "old_text must be one exact, nonempty, contiguous span copied from the saved answer and "
        "must occur there exactly once. new_text is its corrected replacement. Make the smallest "
        "replacement that fixes the failed requirement while preserving all satisfied requirements. "
        "Do not return markdown or commentary."
        if patch
        else "Return only the complete revised answer. Make no unnecessary changes and do not mention these instructions."
    )
    return (
        f"Question: {case.question}\n\nFixed evidence:\n{case.evidence}\n\n"
        f"Saved answer:\n{answer}\n\n{target}\n\n"
        "Requirements already satisfied in the saved answer and required to remain true:\n"
        f"{preservation_text(preserved)}\n\nTask:\n{output}"
    )


def parse_patch(text: str, answer: str) -> tuple[str, bool, str]:
    stripped = text.strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return answer, False, "no_json_object"
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return answer, False, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"old_text", "new_text"}:
        return answer, False, "wrong_fields"
    old, new = value["old_text"], value["new_text"]
    if not isinstance(old, str) or not isinstance(new, str) or not old or not new.strip():
        return answer, False, "empty_span"
    if answer.count(old) != 1:
        return answer, False, "old_text_not_unique"
    if len(new.split()) > 180:
        return answer, False, "replacement_too_long"
    return answer.replace(old, new, 1), True, "valid"


def edit_ratio(before: str, after: str) -> float:
    return 1.0 - SequenceMatcher(a=before.split(), b=after.split(), autojunk=False).ratio()


def _score_prompts(runner: ModelRunner, prompts: list[str], batch_size: int) -> list[float]:
    if not prompts:
        return []
    torch = runner.torch
    values: list[float] = []
    for start in range(0, len(prompts), batch_size):
        chunk = [runner.chat_text(prompt) for prompt in prompts[start : start + batch_size]]
        encoded = runner.tokenizer(
            chunk, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        width = int(encoded["input_ids"].shape[1])
        if width + 2 > runner.context_limit:
            raise RuntimeError(f"judge prompt exceeds context: {width}")
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            base = runner.model.model(**encoded, use_cache=False)
            logits = runner.model.lm_head(base.last_hidden_state[:, -1, :])
        scores = (
            logits[:, runner.missing_token_id].float()
            - logits[:, runner.covered_token_id].float()
        )
        values.extend(float(value) for value in scores.detach().cpu().tolist())
        print(f"[judge] {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
        del encoded, base, logits, scores
        torch.cuda.empty_cache()
    return values


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_recovery": mean(float(row["target_recovered"]) for row in rows),
        "mean_preservation": mean(float(row["preservation_rate"]) for row in rows),
        "no_regression": mean(float(row["no_regression"]) for row in rows),
        "safe_success": mean(float(row["safe_success"]) for row in rows),
        "patch_valid": mean(float(row["patch_valid"]) for row in rows),
        "median_edit_ratio": median(float(row["edit_ratio"]) for row in rows),
        "mean_edit_ratio": mean(float(row["edit_ratio"]) for row in rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.dataset, args.split)
    cases, audit = build_process_cases(rows, args.n)
    print(f"[audit] {json.dumps(audit, sort_keys=True)}", flush=True)
    runner = ModelRunner(args.model)

    baseline_answers = runner.generate(
        [baseline_prompt(case) for case in cases], args.generation_batch_size, 256
    )
    baseline_by_id = {case.id: answer.strip() for case, answer in zip(cases, baseline_answers)}

    checks: list[tuple[str, str, str, str]] = []
    prompts: list[str] = []
    for case in cases:
        answer = baseline_by_id[case.id]
        for index, requirement in enumerate(case.answer_critical):
            checks.append((case.id, "ac", str(index), requirement.handle))
            prompts.append(judge_prompt(case, answer, requirement.handle, requirement.ingredient))
        for index, step in enumerate(case.steps):
            checks.append((case.id, "component", str(index), f"P{index + 1}"))
            prompts.append(judge_prompt(case, answer, f"P{index + 1}", component_requirement(step)))
        checks.extend(
            [
                (case.id, "control", "positive", case.target.handle),
                (case.id, "control", "negative", case.target.handle),
            ]
        )
        prompts.extend(
            [
                judge_prompt(case, positive_process_answer(case.steps), case.target.handle, case.target.ingredient),
                judge_prompt(case, negative_process_answer(case.steps), case.target.handle, case.target.ingredient),
            ]
        )
    baseline_scores = _score_prompts(runner, prompts, args.judge_batch_size)
    score_map = {key: value for key, value in zip(checks, baseline_scores)}

    baseline_rows: list[dict[str, Any]] = []
    eligible: list[tuple[ProcessCase, list[int], bool]] = []
    positive_ok = negative_ok = 0
    for case in cases:
        target_index = case.answer_critical.index(case.target)
        ac_scores = [score_map[(case.id, "ac", str(index), req.handle)] for index, req in enumerate(case.answer_critical)]
        component_scores = [score_map[(case.id, "component", str(index), f"P{index + 1}")] for index in range(len(case.steps))]
        target_met = ac_scores[target_index] < 0
        component_met = [value < 0 for value in component_scores]
        originally_met = [index for index, value in enumerate(ac_scores) if value < 0 and index != target_index]
        relation_only = not target_met and all(component_met)
        repair_eligible = (
            not target_met
            and sum(component_met) >= min(2, len(component_met))
            and sum(not value for value in component_met) <= 1
            and len(originally_met) >= 1
        )
        positive_ok += int(score_map[(case.id, "control", "positive", case.target.handle)] < 0)
        negative_ok += int(score_map[(case.id, "control", "negative", case.target.handle)] >= 0)
        if repair_eligible:
            eligible.append((case, originally_met, relation_only))
        baseline_rows.append(
            {
                "id": case.id,
                "question": case.question,
                "target_handle": case.target.handle,
                "target_requirement": case.target.ingredient,
                "steps": list(case.steps),
                "answer": baseline_by_id[case.id],
                "target_score": ac_scores[target_index],
                "target_met": target_met,
                "component_scores": component_scores,
                "component_met": component_met,
                "originally_met_indices": originally_met,
                "relation_only_failure": relation_only,
                "repair_eligible": repair_eligible,
            }
        )
    relation_only_count = sum(item[2] for item in eligible)
    print(
        f"[baseline] selected={len(cases)} eligible={len(eligible)} relation_only={relation_only_count}",
        flush=True,
    )

    generation_keys: list[tuple[str, str]] = []
    repair_prompts: list[str] = []
    eligible_by_id: dict[str, tuple[ProcessCase, list[int], bool]] = {}
    for case, met_indices, relation_only in eligible:
        eligible_by_id[case.id] = (case, met_indices, relation_only)
        preserved = [case.answer_critical[index] for index in met_indices]
        for arm in ARMS:
            generation_keys.append((case.id, arm))
            repair_prompts.append(repair_prompt(case, baseline_by_id[case.id], arm, preserved))
    raw_outputs = runner.generate(repair_prompts, args.generation_batch_size, 256) if repair_prompts else []

    candidates: list[dict[str, Any]] = []
    for (case_id, arm), raw in zip(generation_keys, raw_outputs):
        case, met_indices, relation_only = eligible_by_id[case_id]
        baseline = baseline_by_id[case_id]
        if arm.endswith("span_patch"):
            answer, valid, parse_mode = parse_patch(raw, baseline)
        else:
            answer, valid, parse_mode = raw.strip(), bool(raw.strip()), "full_answer"
            if not valid:
                answer = baseline
        candidates.append(
            {
                "id": case_id,
                "arm": arm,
                "relation_only_failure": relation_only,
                "raw_output": raw,
                "answer": answer,
                "patch_valid": valid,
                "parse_mode": parse_mode,
                "edit_ratio": edit_ratio(baseline, answer),
                "met_indices": met_indices,
            }
        )

    final_specs: list[tuple[int, str, int]] = []
    final_prompts: list[str] = []
    for row_index, candidate in enumerate(candidates):
        case, met_indices, _ = eligible_by_id[candidate["id"]]
        target_index = case.answer_critical.index(case.target)
        final_specs.append((row_index, "target", target_index))
        final_prompts.append(judge_prompt(case, candidate["answer"], case.target.handle, case.target.ingredient))
        for index in met_indices:
            requirement = case.answer_critical[index]
            final_specs.append((row_index, "preserve", index))
            final_prompts.append(judge_prompt(case, candidate["answer"], requirement.handle, requirement.ingredient))
    final_scores = _score_prompts(runner, final_prompts, args.judge_batch_size)
    scored: dict[int, dict[str, Any]] = defaultdict(lambda: {"preserve": []})
    for (row_index, kind, index), value in zip(final_specs, final_scores):
        if kind == "target":
            scored[row_index]["target"] = value
        else:
            scored[row_index]["preserve"].append((index, value))

    for row_index, candidate in enumerate(candidates):
        target_recovered = scored[row_index].get("target", 1.0) < 0
        preservation = [value < 0 for _, value in scored[row_index]["preserve"]]
        preservation_rate = mean(float(value) for value in preservation)
        no_regression = all(preservation)
        candidate.update(
            {
                "target_score": scored[row_index].get("target"),
                "target_recovered": target_recovered,
                "preservation_scores": [value for _, value in scored[row_index]["preserve"]],
                "preservation_rate": preservation_rate,
                "no_regression": no_regression,
                "safe_success": target_recovered and no_regression,
            }
        )

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_arm[candidate["arm"]].append(candidate)
        by_case[candidate["id"]].append(candidate)
    absolute = {arm: metrics(by_arm[arm]) for arm in ARMS}
    relation_only = {
        arm: metrics([row for row in by_arm[arm] if row["relation_only_failure"]])
        for arm in ARMS
    }
    oracle_hits = {
        case_id: any(row["safe_success"] for row in rows_for_case)
        for case_id, rows_for_case in by_case.items()
    }
    disagreement_cases = sum(
        len({bool(row["safe_success"]) for row in rows_for_case}) > 1
        for rows_for_case in by_case.values()
    )
    distinct_candidates = [
        len({re.sub(r"\s+", " ", row["answer"]).strip() for row in rows_for_case})
        for rows_for_case in by_case.values()
    ]
    oracle_rate = mean(float(value) for value in oracle_hits.values())
    best_fixed = max((absolute[arm]["safe_success"] for arm in ARMS), default=0.0)

    apparatus_gates = {
        "at_least_24_structural_targets": len(cases) >= 24,
        "positive_control_at_least_90pct": positive_ok / max(1, len(cases)) >= 0.90,
        "negative_control_at_least_90pct": negative_ok / max(1, len(cases)) >= 0.90,
        "complete_candidate_denominator": all(len(by_arm[arm]) == len(eligible) for arm in ARMS),
    }
    problem_gate = relation_only_count >= 4
    action_gate = oracle_rate >= 0.30
    flat_full_edit = absolute["flat_full_rewrite"]["median_edit_ratio"]
    typed_patch = absolute["typed_span_patch"]
    typed_gate = (
        typed_patch["safe_success"] >= 0.25
        and typed_patch["no_regression"] >= 0.85
        and typed_patch["safe_success"] - absolute["flat_span_patch"]["safe_success"] >= 0.05
        and (flat_full_edit == 0.0 or typed_patch["median_edit_ratio"] <= 0.50 * flat_full_edit)
    )
    hidden_gate = oracle_rate - best_fixed >= 0.10 and disagreement_cases >= 3
    if not all(apparatus_gates.values()):
        outcome = "APPARATUS_FAIL"
    elif not problem_gate:
        outcome = "PROBLEM_NOT_ESTABLISHED"
    elif not action_gate:
        outcome = "ACTION_SPACE_FAIL"
    elif typed_gate:
        outcome = "STRUCTURED_REPAIR_P0_PASS"
    else:
        outcome = "REPAIR_WORKS_STRUCTURE_NOT_ADDED"

    report = {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "model": args.model,
        "audit": audit,
        "counts": {
            "baseline_cases": len(cases),
            "repair_cases": len(eligible),
            "relation_only_failures": relation_only_count,
            "disagreement_cases": disagreement_cases,
        },
        "controls": {
            "positive_accuracy": positive_ok / max(1, len(cases)),
            "negative_accuracy": negative_ok / max(1, len(cases)),
        },
        "absolute": absolute,
        "relation_only": relation_only,
        "action_oracle": {
            "safe_success": oracle_rate,
            "best_fixed_safe_success": best_fixed,
            "oracle_minus_best_fixed": oracle_rate - best_fixed,
            "mean_distinct_candidates": mean(float(value) for value in distinct_candidates),
        },
        "apparatus_gates": apparatus_gates,
        "problem_gate_at_least_four_relation_only": problem_gate,
        "action_space_gate_oracle_at_least_30pct": action_gate,
        "typed_minimal_gate": typed_gate,
        "hidden_selector_gate": hidden_gate,
        "interpretation_guard": (
            "This is a same-model, text-only, one-relation case-study screen. It does not establish "
            "benchmark performance, independent judge validity, hidden-state control, or novelty."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, output_rows in [
        ("gamut_process_repair_p0_baselines.jsonl", baseline_rows),
        ("gamut_process_repair_p0_candidates.jsonl", candidates),
    ]:
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "gamut_process_repair_p0_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test_text_only")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
