"""Progressive greedy viability screen for the frozen easy GSM join."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from answer_check import answers_equal, extract_boxed
from hsgr_join_provenance_ceiling import (
    BASE_SYSTEM,
    PARENT_USER,
    ROOT_USER,
    bind_root,
    split_questions,
)
from pilot import JWriter, Runner, jread
from structural_hardness_screen import DIRECT_USER, generated_tokens


PROTOCOL = "EXPERIMENT_PROTOCOL_JOIN_VIABILITY_EASY_V2.md"
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
CALIBRATION_N = 96
CONFIRMATION_N = 128
EXPECTED_SHA256 = {
    "calibration": "576fcbf7d6cee0d0d3c9e4a1cf059c0d474b17f675da183c1f47e41df30ae129",
    "confirmation": "5d274c136e149481e47b7ebe534599f7f899efc8e9ae073cce757b3c95c4e1ad",
}


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        payload = handle.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def stable_rank(split: str, problem: str) -> str:
    value = f"join-viability-easy-v2|{split}|{problem}"
    return hashlib.sha256(value.encode()).hexdigest()


def load_subset(path: str, split: str, limit: int) -> list[dict]:
    actual_hash = file_sha256(path)
    if actual_hash != EXPECTED_SHA256[split]:
        raise ValueError(
            f"{split} data hash mismatch: {actual_hash} != {EXPECTED_SHA256[split]}"
        )
    rows = jread(path)
    if len(rows) < limit:
        raise ValueError(f"{split} needs {limit} rows, found {len(rows)}")
    ranked = sorted(
        enumerate(rows), key=lambda pair: stable_rank(split, pair[1]["problem"])
    )[:limit]
    selected = []
    for pid, source in sorted(ranked):
        row = dict(source)
        row["id"] = pid
        if row.get("graph", {}).get("edges") != [
            ["parent_0", "root"], ["parent_1", "root"]
        ]:
            raise ValueError(f"unexpected graph at {split} id={pid}")
        if int(row["root_step_count"]) > 2:
            raise ValueError(f"root cap violated at {split} id={pid}")
        if max(int(value) for value in row["parent_step_counts"]) > 3:
            raise ValueError(f"parent cap violated at {split} id={pid}")
        selected.append(row)
    return selected


def subset_id_hash(rows: list[dict]) -> str:
    payload = ",".join(str(row["id"]) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def prompt_tokens(runner: Runner, user: str) -> int:
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": user},
    ]
    rendered = runner.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(runner.tok.encode(rendered, add_special_tokens=False))


def run_units(
    runner: Runner, rows: list[dict], out_dir: str, batch_size: int,
) -> dict[tuple[int, str], dict]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calls.jsonl")
    done = {(int(row["id"]), row["arm"]): row for row in jread(path)}
    units = []
    for row in rows:
        q0, q1, root = split_questions(row["problem"])
        direct = DIRECT_USER.format(problem=row["problem"])
        arms = [
            ("direct", direct, 512),
            ("parent_0", PARENT_USER.format(question=q0), 192),
            ("parent_1", PARENT_USER.format(question=q1), 192),
        ]
        gold = [str(value) for value in row["parent_answers"]]
        bound = bind_root(root, gold[0], gold[1])
        arms.append((
            "gold_root",
            ROOT_USER.format(parent_0=gold[0], parent_1=gold[1], root=bound),
            192,
        ))
        for arm, user, max_new in arms:
            if (row["id"], arm) not in done:
                units.append((row, arm, user, max_new))

    writer = JWriter(path)
    for max_new in (512, 192):
        selected = [unit for unit in units if unit[3] == max_new]
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            outputs = runner.chat_batch(
                [unit[2] for unit in batch], system=BASE_SYSTEM,
                max_new=max_new, bs=batch_size,
            )
            for (row, arm, user, _), output in zip(batch, outputs):
                text = output[0]
                record = {
                    "id": row["id"], "arm": arm, "text": text,
                    "answer": extract_boxed(text),
                    "prompt_tokens": prompt_tokens(runner, user),
                    "generated_tokens": generated_tokens(runner, text),
                    "calls": 1, "max_new": max_new,
                }
                writer.write(record)
                done[(row["id"], arm)] = record
            print(
                f"[{os.path.basename(out_dir)}:{max_new}] "
                f"{min(start + batch_size, len(selected))}/{len(selected)}",
                flush=True,
            )
    return done


def fraction(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze(
    rows: list[dict], calls: dict[tuple[int, str], dict], split: str,
) -> dict:
    expected_arms = ("direct", "parent_0", "parent_1", "gold_root")
    complete = all((row["id"], arm) in calls for row in rows for arm in expected_arms)
    direct_hits = []
    parent_hits = []
    root_hits = []
    validity = {arm: [] for arm in expected_arms}
    accounting = []
    for row in rows:
        pid = row["id"]
        direct = calls[(pid, "direct")]
        direct_hits.append(answers_equal(direct["answer"], row["answer"]))
        root = calls[(pid, "gold_root")]
        root_hits.append(answers_equal(root["answer"], row["answer"]))
        for slot in (0, 1):
            arm = f"parent_{slot}"
            parent_hits.append(answers_equal(
                calls[(pid, arm)]["answer"], row["parent_answers"][slot]
            ))
        for arm in expected_arms:
            record = calls[(pid, arm)]
            validity[arm].append(record["answer"] is not None)
            accounting.append(
                record.get("calls") == 1
                and record.get("max_new") == (512 if arm == "direct" else 192)
            )
    direct_accuracy = fraction(direct_hits)
    parent_accuracy = fraction(parent_hits)
    gold_root_accuracy = fraction(root_hits)
    parse_validity = {arm: fraction(values) for arm, values in validity.items()}
    thresholds = (
        {"direct_low": 0.30, "direct_high": 0.70, "parent": 0.70,
         "root": 0.70, "gap": 0.10, "n": CALIBRATION_N}
        if split == "calibration"
        else {"direct_low": 0.25, "direct_high": 0.75, "parent": 0.65,
              "root": 0.65, "gap": 0.08, "n": CONFIRMATION_N}
    )
    gates = {
        "complete_n_times_4": complete and len(calls) == len(rows) * 4
        and len(rows) == thresholds["n"],
        "direct_in_range": thresholds["direct_low"] <= direct_accuracy
        <= thresholds["direct_high"],
        "parent_accuracy": parent_accuracy >= thresholds["parent"],
        "gold_root_accuracy": gold_root_accuracy >= thresholds["root"],
        "gold_root_gap": gold_root_accuracy - direct_accuracy >= thresholds["gap"],
        "parse_validity": min(parse_validity.values()) >= 0.95,
        "one_call_accounting": all(accounting),
    }
    return {
        "protocol": PROTOCOL,
        "split": split,
        "n": len(rows),
        "ids": [row["id"] for row in rows],
        "id_sha256": subset_id_hash(rows),
        "accuracy": {
            "direct": direct_accuracy,
            "mean_parent": parent_accuracy,
            "gold_bound_root": gold_root_accuracy,
            "gold_root_minus_direct": gold_root_accuracy - direct_accuracy,
        },
        "parse_validity": parse_validity,
        "cost": {
            "calls": len(rows) * 4,
            "prompt_tokens": sum(record["prompt_tokens"] for record in calls.values()),
            "generated_tokens": sum(record["generated_tokens"] for record in calls.values()),
        },
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


def print_report(report: dict) -> None:
    acc = report["accuracy"]
    print(
        f"{report['split']} n={report['n']} direct={acc['direct']:.3f} "
        f"parent={acc['mean_parent']:.3f} root={acc['gold_bound_root']:.3f} "
        f"gap={acc['gold_root_minus_direct']:+.3f}",
        flush=True,
    )
    for name, passed in report["gates"].items():
        print(f"GATE {report['split']} {name}: {'PASS' if passed else 'FAIL'}")
    print(
        f"JOIN_VIABILITY_EASY_{report['split'].upper()}="
        f"{'PASS' if report['gate_pass'] else 'FAIL'}",
        flush=True,
    )


def write_report(path: str, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)


def self_test() -> None:
    assert CALIBRATION_N == 96 and CONFIRMATION_N == 128
    assert stable_rank("calibration", "x") == stable_rank("calibration", "x")
    assert stable_rank("calibration", "x") != stable_rank("confirmation", "x")
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-data", default="data/gsm_join_easy_train.jsonl")
    parser.add_argument("--confirmation-data", default="data/gsm_join_easy_test.jsonl")
    parser.add_argument("--out-dir", default="join_viability_easy_v2")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    calibration = load_subset(
        args.calibration_data, "calibration", CALIBRATION_N
    )
    confirmation = load_subset(
        args.confirmation_data, "confirmation", CONFIRMATION_N
    )
    os.makedirs(args.out_dir, exist_ok=True)
    runner = Runner(args.model)
    calibration_dir = os.path.join(args.out_dir, "calibration")
    calibration_calls = run_units(
        runner, calibration, calibration_dir, args.batch_size
    )
    calibration_report = analyze(calibration, calibration_calls, "calibration")
    write_report(
        os.path.join(args.out_dir, "calibration_report.json"), calibration_report
    )
    print_report(calibration_report)
    if not calibration_report["gate_pass"]:
        write_report(os.path.join(args.out_dir, "report.json"), {
            "protocol": PROTOCOL,
            "calibration": calibration_report,
            "confirmation": None,
            "gate_pass": False,
        })
        return
    confirmation_dir = os.path.join(args.out_dir, "confirmation")
    confirmation_calls = run_units(
        runner, confirmation, confirmation_dir, args.batch_size
    )
    confirmation_report = analyze(
        confirmation, confirmation_calls, "confirmation"
    )
    write_report(
        os.path.join(args.out_dir, "confirmation_report.json"),
        confirmation_report,
    )
    print_report(confirmation_report)
    write_report(os.path.join(args.out_dir, "report.json"), {
        "protocol": PROTOCOL,
        "calibration": calibration_report,
        "confirmation": confirmation_report,
        "gate_pass": confirmation_report["gate_pass"],
    })


if __name__ == "__main__":
    main()
