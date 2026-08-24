"""P0o-R1 selected-case obligation carry-forward rescue screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from asqa_missing_selector_p6x import ModelRunner
from gamut_process_repair_p0b import release_runner
from refinebench_revision_audit_p0n import (
    criterion_instruction,
    diff_aids,
    failed_ids,
    feedback_text,
    judge_answers,
    reference_answer_text,
    stable_key,
    write_json,
    write_jsonl,
)


PROTOCOL = "EXPERIMENT_PROTOCOL_OBLIGATION_CARRY_FORWARD_P0O.md"
SEED = 20260824
ARMS = (
    "failed_only",
    "all_checklist_no_status",
    "full_ledger",
    "shuffled_status",
)
RESCUE_CASE_IDS = (
    "refinebench-000010",
    "refinebench-000100",
    "refinebench-000147",
    "refinebench-000347",
    "refinebench-000577",
    "refinebench-000581",
    "refinebench-000827",
    "refinebench-000833",
)
DEFAULT_GENERATOR = "Qwen/Qwen3-8B"
DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def source_initials(p0n_dir: Path, generator: str) -> list[dict[str, Any]]:
    rows = read_jsonl(p0n_dir / "p0n_initial_judgments.jsonl")
    by_id = {
        row["case_id"]: row for row in rows
        if row["kind"] == "initial" and row["generator"] == generator
    }
    missing = [case_id for case_id in RESCUE_CASE_IDS if case_id not in by_id]
    if missing:
        raise RuntimeError(f"missing P0n source cases: {missing}")
    return [by_id[case_id] for case_id in RESCUE_CASE_IDS]


def exact_judge_key(entry: dict[str, Any]) -> str:
    instance = entry["instance"]
    payload = json.dumps(
        {
            "question": instance["question"],
            "checklist": instance["checklist"],
            "answer": entry["answer"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def judge_deduplicated(
    model_name: str, entries: list[dict[str, Any]], cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    unique: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = exact_judge_key(entry)
        if key not in unique:
            unique[key] = {
                **entry,
                "eval_id": f"unique|{key}",
                "judge_cache_key": key,
            }
    judged_unique = judge_answers(model_name, list(unique.values()), 1, cap)
    cache = {row["judge_cache_key"]: row for row in judged_unique}
    judged = []
    for entry in entries:
        key = exact_judge_key(entry)
        result = cache[key]
        judged.append({
            **entry,
            "judge_cache_key": key,
            "raw_judgment": result["raw_judgment"],
            "evaluation": result["evaluation"],
            "judge_valid": result["judge_valid"],
            "judge_parse_mode": result["judge_parse_mode"],
        })
    return judged, {"requested": len(entries), "unique": len(unique)}


def shuffled_evaluation(case_id: str, evaluation: dict[int, bool]) -> dict[int, bool]:
    indices = sorted(evaluation)
    labels = [evaluation[index] for index in indices]
    if all(labels) or not any(labels):
        raise ValueError("shuffled status requires a mixed state")
    for attempt in range(1000):
        candidate = labels.copy()
        rng = random.Random(int(stable_key(SEED, case_id, "shuffle", attempt)[:16], 16))
        rng.shuffle(candidate)
        result = dict(zip(indices, candidate))
        pass_to_fail = any(evaluation[index] and not result[index] for index in indices)
        fail_to_pass = any(not evaluation[index] and result[index] for index in indices)
        if pass_to_fail and fail_to_pass:
            return result
    raise RuntimeError(f"could not construct shuffled state for {case_id}")


def checklist_lines(checklist: list[str], indices: list[int]) -> str:
    return "\n".join(
        f"- O{index}: {criterion_instruction(checklist[index - 1])}"
        for index in indices
    )


def arm_feedback(
    case_id: str, checklist: list[str], evaluation: dict[int, bool], arm: str,
) -> str:
    failed = failed_ids(evaluation)
    if arm == "failed_only":
        return feedback_text(checklist, failed)
    if arm == "all_checklist_no_status":
        return (
            "Please revise the previous response while considering every requirement below.\n\n"
            f"REQUIREMENTS:\n{checklist_lines(checklist, list(range(1, len(checklist) + 1)))}\n\n"
            "Return one improved answer."
        )
    state = evaluation if arm == "full_ledger" else shuffled_evaluation(case_id, evaluation)
    if arm not in {"full_ledger", "shuffled_status"}:
        raise ValueError(arm)
    preserve = sorted(index for index, value in state.items() if value)
    repair = sorted(index for index, value in state.items() if not value)
    return (
        "Current obligation ledger\n\n"
        f"PRESERVE:\n{checklist_lines(checklist, preserve)}\n\n"
        f"REPAIR:\n{checklist_lines(checklist, repair)}\n\n"
        "Revise the previous response so that REPAIR obligations are fixed while "
        "PRESERVE obligations remain satisfied. Return one improved answer."
    )


def revision_messages(
    instance: dict[str, Any], previous_answer: str, feedback: str,
) -> list[dict[str, str]]:
    from refinebench_revision_audit_p0n import query_text

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query_text(instance)},
        {"role": "assistant", "content": previous_answer},
        {"role": "user", "content": feedback},
    ]


def sampling_seed(case_id: str, replicate: int) -> int:
    return int(stable_key(SEED, "p0o_sample", case_id, replicate)[:8], 16)


def sample_tasks(
    runner: ModelRunner, tasks: list[dict[str, Any]], max_new_tokens: int,
    temperature: float, top_p: float,
) -> list[str]:
    torch = runner.torch
    answers = []
    for position, task in enumerate(tasks, 1):
        seed = sampling_seed(task["case_id"], task["replicate"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text = runner.tokenizer.apply_chat_template(
            task["messages"], tokenize=False, add_generation_prompt=True,
            **runner.chat_template_kwargs,
        )
        encoded = runner.tokenizer(
            [text], padding=True, add_special_tokens=False, return_tensors="pt"
        )
        width = int(encoded["input_ids"].shape[1])
        if width + max_new_tokens > runner.context_limit:
            raise RuntimeError(f"generation prompt exceeds context: {width}+{max_new_tokens}")
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            generated = runner.model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=runner.tokenizer.pad_token_id,
                eos_token_id=runner.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, width:]
        answers.append(runner.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0])
        print(f"[sample-revision] {position}/{len(tasks)}", flush=True)
        del encoded, generated, new_tokens
        torch.cuda.empty_cache()
    return answers


def revision_row(
    task: dict[str, Any], answer: str, judgment: dict[str, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    valid = initial["judge_valid"] and judgment["judge_valid"]
    criteria = []
    transitions = Counter()
    if valid:
        for index in sorted(initial["evaluation"]):
            old_value = initial["evaluation"][index]
            new_value = judgment["evaluation"][index]
            transition = ("Y" if old_value else "N") + ("Y" if new_value else "N")
            transitions[transition] += 1
            criteria.append({
                "criterion_id": index,
                "criterion": task["instance"]["checklist"][index - 1],
                "old_met": old_value,
                "new_met": new_value,
                "transition": transition,
                "targeted": not old_value,
            })
    prior_yes = transitions["YY"] + transitions["YN"]
    prior_no = transitions["NY"] + transitions["NN"]
    fixes = transitions["NY"]
    regressions = transitions["YN"]
    return {
        "case_id": task["case_id"],
        "field": task["instance"]["field"],
        "stratum": task["instance"].get("p0n_stratum"),
        "arm": task["arm"],
        "replicate": task["replicate"],
        "sample_seed": sampling_seed(task["case_id"], task["replicate"]),
        "valid_transition": valid,
        "judge_parse_mode": judgment["judge_parse_mode"],
        "prior_yes": prior_yes,
        "prior_no": prior_no,
        "fixes": fixes,
        "regressions": regressions,
        "any_fix": fixes > 0,
        "all_targets_fixed": prior_no > 0 and fixes == prior_no,
        "any_regression": regressions > 0,
        "all_preserved": regressions == 0,
        "joint_success_any": fixes > 0 and regressions == 0,
        "strict_joint_success": prior_no > 0 and fixes == prior_no and regressions == 0,
        "transitions": dict(transitions),
        "criterion_rows": criteria,
        "question": task["instance"]["question"],
        "old_answer": task["previous_answer"],
        "new_answer": answer,
        "diff_aids": diff_aids(task["previous_answer"], answer),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_transition"]]
    prior_yes = sum(row["prior_yes"] for row in valid)
    prior_no = sum(row["prior_no"] for row in valid)
    fixes = sum(row["fixes"] for row in valid)
    regressions = sum(row["regressions"] for row in valid)
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "judge_validity": len(valid) / len(rows) if rows else 0.0,
        "target_fix_rate": fixes / prior_no if prior_no else None,
        "preserve_rate": 1.0 - regressions / prior_yes if prior_yes else None,
        "any_regression_rate": mean(row["any_regression"] for row in valid) if valid else None,
        "any_fix_rate": mean(row["any_fix"] for row in valid) if valid else None,
        "all_targets_fixed_rate": mean(row["all_targets_fixed"] for row in valid) if valid else None,
        "all_preserved_rate": mean(row["all_preserved"] for row in valid) if valid else None,
        "joint_success_any_rate": mean(row["joint_success_any"] for row in valid) if valid else None,
        "strict_joint_success_rate": mean(row["strict_joint_success"] for row in valid) if valid else None,
        "criterion_transitions": {
            code: sum(row["transitions"].get(code, 0) for row in valid)
            for code in ("YY", "YN", "NY", "NN")
        },
    }


def gate_decision(
    summaries: dict[str, dict[str, Any]], apparatus: dict[str, bool],
) -> tuple[dict[str, bool], str]:
    if not all(apparatus.values()):
        return {}, "APPARATUS_FAILURE"
    failed = summaries["failed_only"]
    full = summaries["full_ledger"]
    all_list = summaries["all_checklist_no_status"]
    shuffled = summaries["shuffled_status"]
    baseline_regression = failed["any_regression_rate"]
    full_regression = full["any_regression_rate"]
    relative_reduction = (
        (baseline_regression - full_regression) / baseline_regression
        if baseline_regression else 0.0
    )
    gates = {
        "g1_regression_relative_ge_50_and_absolute_ge_10pp": (
            relative_reduction >= 0.50 and baseline_regression - full_regression >= 0.10
        ),
        "g2_fix_rate_drop_le_10pp": (
            full["target_fix_rate"] >= failed["target_fix_rate"] - 0.10
        ),
        "g3_status_specificity_ge_10pp_each": (
            full["joint_success_any_rate"] >= all_list["joint_success_any_rate"] + 0.10
            and full["joint_success_any_rate"] >= shuffled["joint_success_any_rate"] + 0.10
        ),
        "g4_joint_gain_vs_failed_ge_15pp": (
            full["joint_success_any_rate"] >= failed["joint_success_any_rate"] + 0.15
        ),
    }
    return gates, "PROVISIONAL_PASS_REVIEW_REQUIRED" if all(gates.values()) else "STOP_NO_P0O_R2"


def build_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review = []
    for row in rows:
        for criterion in row["criterion_rows"]:
            if criterion["transition"] == "YN":
                review.append({
                    "review_type": "candidate_yes_to_no",
                    "manual_valid_transition": None,
                    "manual_notes": "",
                    "criterion": criterion,
                    **{key: row[key] for key in (
                        "case_id", "field", "stratum", "arm", "replicate",
                        "question", "old_answer", "new_answer", "diff_aids",
                    )},
                })
    return review


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = source_initials(args.p0n_result_dir, args.generator_model)
    source_hash = hashlib.sha256(
        "\n".join(
            f"{row['case_id']}\t{hashlib.sha256(row['answer'].encode('utf-8')).hexdigest()}"
            for row in source
        ).encode("utf-8")
    ).hexdigest()

    initial_entries = []
    for row in source:
        initial_entries.append({
            "eval_id": f"initial|{row['case_id']}",
            "kind": "initial", "case_id": row["case_id"],
            "instance": row["instance"], "answer": row["answer"],
        })
        reference = reference_answer_text(row["instance"])
        if reference:
            initial_entries.append({
                "eval_id": f"reference|{row['case_id']}",
                "kind": "reference", "case_id": row["case_id"],
                "instance": row["instance"], "answer": reference,
            })
    initial_judgments, initial_cache = judge_deduplicated(
        args.judge_model, initial_entries, args.judge_cap
    )
    write_jsonl(args.out_dir / "p0o_initial_judgments.jsonl", initial_judgments)
    initial_by_id = {
        row["case_id"]: row for row in initial_judgments if row["kind"] == "initial"
    }
    references = [row for row in initial_judgments if row["kind"] == "reference"]
    reference_values = [
        value for row in references if row["judge_valid"]
        for value in row["evaluation"].values()
    ]
    mixed_cases = [
        case_id for case_id, row in initial_by_id.items()
        if row["judge_valid"] and any(row["evaluation"].values())
        and not all(row["evaluation"].values())
    ]
    initial_apparatus = {
        "initial_and_reference_judgments_parse_valid": all(
            row["judge_valid"] for row in initial_judgments
        ),
        "reference_yes_rate_ge_90": mean(reference_values) >= 0.90 if reference_values else False,
        "all_eight_initial_states_mixed": len(mixed_cases) == len(RESCUE_CASE_IDS),
    }
    if not all(initial_apparatus.values()):
        report = {
            "protocol": PROTOCOL,
            "source_p0n_result_dir": str(args.p0n_result_dir),
            "source_case_answer_sha256": source_hash,
            "case_ids": list(RESCUE_CASE_IDS),
            "distinct_cases": len(RESCUE_CASE_IDS),
            "replicates": args.replicates,
            "arms": list(ARMS),
            "total_revisions": 0,
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "judge_is_official_gpt41": False,
            "initial_judge_cache": initial_cache,
            "reference_yes_rate": mean(reference_values) if reference_values else None,
            "mixed_initial_case_ids": mixed_cases,
            "apparatus_gates": initial_apparatus,
            "mechanism_gates": {},
            "decision": "APPARATUS_FAILURE",
            "interpretation_guard": "Initial apparatus failed before revision generation.",
        }
        write_json(args.out_dir / "p0o_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report

    tasks = []
    for row in source:
        initial = initial_by_id[row["case_id"]]
        if not initial["judge_valid"]:
            continue
        for replicate in range(args.replicates):
            for arm in ARMS:
                feedback = arm_feedback(
                    row["case_id"], row["instance"]["checklist"],
                    initial["evaluation"], arm,
                )
                tasks.append({
                    "case_id": row["case_id"], "arm": arm, "replicate": replicate,
                    "instance": row["instance"], "previous_answer": row["answer"],
                    "feedback": feedback,
                    "messages": revision_messages(row["instance"], row["answer"], feedback),
                })
    runner = ModelRunner(args.generator_model)
    answers = sample_tasks(
        runner, tasks, args.generation_cap, args.temperature, args.top_p
    )
    release_runner(runner)
    revisions = [
        {**{key: value for key, value in task.items() if key != "messages"}, "answer": answer}
        for task, answer in zip(tasks, answers)
    ]
    write_jsonl(args.out_dir / "p0o_revisions.jsonl", revisions)

    revision_entries = [
        {
            "eval_id": f"revision|{row['case_id']}|{row['arm']}|{row['replicate']}",
            "kind": "revision", "case_id": row["case_id"], "arm": row["arm"],
            "replicate": row["replicate"], "instance": row["instance"],
            "answer": row["answer"],
        }
        for row in revisions
    ]
    revision_judgments, revision_cache = judge_deduplicated(
        args.judge_model, revision_entries, args.judge_cap
    )
    write_jsonl(args.out_dir / "p0o_revision_judgments.jsonl", revision_judgments)
    judgment_lookup = {
        (row["case_id"], row["arm"], row["replicate"]): row
        for row in revision_judgments
    }
    transition_rows = []
    for task, answer in zip(tasks, answers):
        key = (task["case_id"], task["arm"], task["replicate"])
        transition_rows.append(revision_row(
            task, answer, judgment_lookup[key], initial_by_id[task["case_id"]]
        ))
    write_jsonl(args.out_dir / "p0o_transition_rows.jsonl", transition_rows)
    write_jsonl(args.out_dir / "p0o_manual_review.jsonl", build_review(transition_rows))

    summaries = {
        arm: summarize([row for row in transition_rows if row["arm"] == arm])
        for arm in ARMS
    }
    failed_rows = [row for row in transition_rows if row["arm"] == "failed_only"]
    failed_regression_cases = len({
        row["case_id"] for row in failed_rows if row["valid_transition"] and row["any_regression"]
    })
    apparatus = {
        **initial_apparatus,
        "all_revision_judgments_parse_valid": all(
            row["judge_valid"] for row in revision_judgments
        ),
        "failed_only_regression_cases_ge_3": failed_regression_cases >= 3,
        "failed_only_regression_rate_ge_20": (
            summaries["failed_only"]["any_regression_rate"] >= 0.20
        ),
    }
    mechanism_gates, decision = gate_decision(summaries, apparatus)
    report = {
        "protocol": PROTOCOL,
        "source_p0n_result_dir": str(args.p0n_result_dir),
        "source_case_answer_sha256": source_hash,
        "case_ids": list(RESCUE_CASE_IDS),
        "distinct_cases": len(RESCUE_CASE_IDS),
        "selection": "P0n0 successful all-failed revisions with manually confirmed direct omission",
        "replicates": args.replicates,
        "arms": list(ARMS),
        "total_revisions": len(revisions),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "judge_is_official_gpt41": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "initial_judge_cache": initial_cache,
        "revision_judge_cache": revision_cache,
        "reference_yes_rate": mean(reference_values) if reference_values else None,
        "mixed_initial_case_ids": mixed_cases,
        "failed_only_regression_cases": failed_regression_cases,
        "arm_summaries": summaries,
        "apparatus_gates": apparatus,
        "mechanism_gates": mechanism_gates,
        "decision": decision,
        "manual_review_candidates": sum(
            row["regressions"] for row in transition_rows if row["valid_transition"]
        ),
        "interpretation_guard": (
            "Selected-case local-judge rescue screen; not an official RefineBench score, "
            "fresh-set estimate, novelty result, relation-aware result, hidden-state result, "
            "or RL result. Automated gate passage requires researcher review."
        ),
    }
    write_json(args.out_dir / "p0o_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"result_dir={args.out_dir}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0n-result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--generation-cap", type=int, default=2048)
    parser.add_argument("--judge-cap", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
