"""Four-call multi-view parent candidate action-space pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

from answer_check import answers_equal, extract_boxed, normalize_answer
from pilot import JWriter, Runner, jread
from structural_hardness_screen import modal_candidate, read_rows
from hsgr_join_provenance_ceiling import BASE_SYSTEM, PARENT_USER, split_questions


ARMS = ("equation", "verify", "check")

EQUATION_USER = """Solve this arithmetic word problem by first writing the
minimal equations, then calculating carefully. Check units and arithmetic.
End with exactly one final answer in \\boxed{{...}}.

Problem: {question}"""

VERIFY_USER = """Solve this arithmetic word problem independently. Before the
final answer, verify every arithmetic operation by substitution or an
alternative calculation. End with exactly one final answer in
\\boxed{{...}}.

Problem: {question}"""

CHECK_USER = """A previous solver proposed the answer shown below. Do not trust
it. Solve the problem from scratch, compare the result, and correct it if
necessary. End with exactly one final answer in \\boxed{{...}}.

Problem: {question}
Previous proposed answer: {proposal}"""


def stable_half(pid: int) -> int:
    digest = hashlib.sha256(f"multiview-half|{pid}".encode()).digest()
    return digest[0] % 2


def load_baseline(path: str) -> dict[tuple[int, int], dict]:
    rows = jread(path)
    baseline = {(int(row["id"]), int(row["slot"])): row for row in rows}
    if len(baseline) != 800:
        raise SystemExit(f"expected 800 baseline parent rows, found {len(baseline)}")
    return baseline


def node_units(data_path: str, baseline: dict) -> list[dict]:
    rows = read_rows(data_path)
    units = []
    for row in rows:
        questions = split_questions(row["problem"])
        for slot in (0, 1):
            base = baseline[(row["id"], slot)]["candidates"][0]
            units.append({
                "id": int(row["id"]),
                "slot": slot,
                "question": questions[slot],
                "gold": str(row["parent_answers"][slot]),
                "proposal": base.get("answer") or "UNKNOWN",
            })
    return units


def arm_prompt(unit: dict, arm: str) -> str:
    if arm == "equation":
        return EQUATION_USER.format(question=unit["question"])
    if arm == "verify":
        return VERIFY_USER.format(question=unit["question"])
    if arm == "check":
        return CHECK_USER.format(
            question=unit["question"], proposal=unit["proposal"]
        )
    raise ValueError(arm)


def run_arm(runner, units: list[dict], arm: str, out_path: str, batch_size: int):
    done = {
        (int(row["id"]), int(row["slot"])): row for row in jread(out_path)
    }
    writer = JWriter(out_path)
    todo = [unit for unit in units if (unit["id"], unit["slot"]) not in done]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        before = runner.n_new_tokens
        outputs = runner.chat_batch(
            [arm_prompt(unit, arm) for unit in batch],
            system=BASE_SYSTEM,
            max_new=384,
            bs=batch_size,
        )
        batch_generated = runner.n_new_tokens - before
        for unit, output in zip(batch, outputs):
            text = output[0]
            answer = extract_boxed(text)
            record = {
                "id": unit["id"], "slot": unit["slot"], "arm": arm,
                "text": text, "answer": answer,
                "norm": normalize_answer(answer),
                "prompt": arm_prompt(unit, arm),
                "generated_tokens_batch": batch_generated,
            }
            writer.write(record)
            done[(unit["id"], unit["slot"])] = record
        print(f"[{arm}] {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
    return done


def analyze(units: list[dict], baseline: dict, arms: dict[str, dict]) -> dict:
    arm_valid = Counter()
    old_modal_hits = old_oracle_hits = new_oracle_hits = 0
    recovered_nodes = set()
    recovered_graphs = set()
    added_wrong_on_old_correct = 0
    unique_distribution = Counter()
    noncollapsed_graphs = set()
    half_delta = defaultdict(lambda: [0, 0])
    slot_delta = defaultdict(lambda: [0, 0])
    rows = []
    for unit in units:
        key = (unit["id"], unit["slot"])
        old_candidates = baseline[key]["candidates"]
        base = old_candidates[0]
        new_candidates = [base]
        for arm in ARMS:
            record = arms[arm][key]
            arm_valid[arm] += record["norm"] is not None
            new_candidates.append(record)
        old_modal = modal_candidate(old_candidates)
        old_modal_hit = answers_equal(old_modal.get("answer"), unit["gold"])
        old_oracle = any(
            answers_equal(candidate.get("answer"), unit["gold"])
            for candidate in old_candidates
        )
        new_oracle = any(
            answers_equal(candidate.get("answer"), unit["gold"])
            for candidate in new_candidates
        )
        norms = {
            candidate.get("norm") for candidate in new_candidates
            if candidate.get("norm") is not None
        }
        unique_distribution[len(norms)] += 1
        if len(norms) > 1:
            noncollapsed_graphs.add(unit["id"])
        if not old_modal_hit and new_oracle:
            recovered_nodes.add(key)
            recovered_graphs.add(unit["id"])
        if old_modal_hit and any(
            candidate.get("norm") is not None
            and not answers_equal(candidate.get("answer"), unit["gold"])
            for candidate in new_candidates[1:]
        ):
            added_wrong_on_old_correct += 1
        old_modal_hits += old_modal_hit
        old_oracle_hits += old_oracle
        new_oracle_hits += new_oracle
        half = stable_half(unit["id"])
        half_delta[half][0] += int(new_oracle) - int(old_oracle)
        half_delta[half][1] += 1
        slot_delta[unit["slot"]][0] += int(new_oracle) - int(old_oracle)
        slot_delta[unit["slot"]][1] += 1
        rows.append({
            "id": unit["id"], "slot": unit["slot"],
            "old_modal_hit": old_modal_hit,
            "old_oracle": old_oracle, "new_oracle": new_oracle,
            "unique_new_values": len(norms),
        })
    n = len(units)
    summary = {
        "n_nodes": n,
        "arm_parse_validity": {arm: arm_valid[arm] / n for arm in ARMS},
        "old_modal_accuracy": old_modal_hits / n,
        "old_candidate_gold_coverage": old_oracle_hits / n,
        "new_candidate_gold_coverage": new_oracle_hits / n,
        "coverage_gain": (new_oracle_hits - old_oracle_hits) / n,
        "unique_value_distribution": dict(sorted(unique_distribution.items())),
        "n_noncollapsed_graphs": len(noncollapsed_graphs),
        "noncollapsed_graph_rate": len(noncollapsed_graphs) / 400,
        "n_recovered_nodes": len(recovered_nodes),
        "n_recovered_graphs": len(recovered_graphs),
        "n_added_wrong_on_old_correct_nodes": added_wrong_on_old_correct,
        "hash_half_coverage_deltas": {
            str(key): value[0] / value[1] for key, value in half_delta.items()
        },
        "slot_coverage_deltas": {
            str(key): value[0] / value[1] for key, value in slot_delta.items()
        },
    }
    checks = {
        "n_nodes_800": n == 800,
        "new_arm_parse_ge_95pct": all(
            summary["arm_parse_validity"][arm] >= 0.95 for arm in ARMS
        ),
        "coverage_gain_ge_10pp": summary["coverage_gain"] >= 0.10,
        "noncollapsed_graph_rate_ge_15pct": (
            summary["noncollapsed_graph_rate"] >= 0.15
        ),
        "recovered_nodes_ge_40": summary["n_recovered_nodes"] >= 40,
        "recovered_graphs_ge_20": summary["n_recovered_graphs"] >= 20,
        "nonnegative_both_hash_halves": all(
            value >= 0 for value in summary["hash_half_coverage_deltas"].values()
        ),
        "nonnegative_both_slots": all(
            value >= 0 for value in summary["slot_coverage_deltas"].values()
        ),
    }
    return {"summary": summary, "checks": checks, "gate_pass": all(checks.values()), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_join_train.jsonl")
    parser.add_argument("--baseline-parents", required=True)
    parser.add_argument("--out-dir", default="parent_multiview_action_space")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unit = {"question": "What is 1+1?", "proposal": "3"}
        assert "1+1" in arm_prompt(unit, "equation")
        assert "Previous proposed answer: 3" in arm_prompt(unit, "check")
        assert stable_half(1) == stable_half(1)
        print("SELF_TEST_OK")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    baseline = load_baseline(args.baseline_parents)
    units = node_units(args.data, baseline)
    runner = Runner(args.model)
    arm_rows = {}
    counters = {"calls": 0, "generated_tokens": 0}
    for arm in ARMS:
        before = runner.n_new_tokens
        arm_rows[arm] = run_arm(
            runner, units, arm, os.path.join(args.out_dir, f"{arm}.jsonl"),
            args.batch_size,
        )
        counters["calls"] += len(units)
        counters["generated_tokens"] += runner.n_new_tokens - before
    report = analyze(units, baseline, arm_rows)
    report["accounting"] = counters
    report["protocol"] = "EXPERIMENT_PROTOCOL_PARENT_MULTIVIEW_ACTION_SPACE_V0.md"
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps({
        "summary": report["summary"], "checks": report["checks"],
        "gate_pass": report["gate_pass"], "accounting": counters,
    }, indent=1), flush=True)
    print(f"MULTIVIEW_ACTION_SPACE={'PASS' if report['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
