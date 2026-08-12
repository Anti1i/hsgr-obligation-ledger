"""Metadata-only FinQA Stage-A structure and gold-executor audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from typing import Any, Dict, List, Sequence

from finqa_program import (
    ALL_OPS,
    execute_program,
    execution_matches,
    parse_program,
    structure_metrics,
)


PROTOCOL = "EXPERIMENT_PROTOCOL_FINQA_PROGRAM_INTERVENTION_P0.md"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def histogram(values: Sequence[int]) -> Dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(collections.Counter(values).items())
    }


def fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit_split(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    cases: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, str]] = []
    execution_errors: List[Dict[str, Any]] = []
    operations = collections.Counter()
    for position, row in enumerate(rows):
        qa = row.get("qa", {})
        uid = str(row.get("id", position))
        try:
            steps = parse_program(qa.get("program"))
        except Exception as exc:
            parse_errors.append({"id": uid, "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        operations.update(step.op for step in steps)
        metrics = structure_metrics(steps)
        result = execute_program(steps, row.get("table", []))
        match = result.valid and execution_matches(result.value, qa.get("exe_ans"))
        case = {
            "id": uid,
            "position": position,
            "execution_valid": result.valid,
            "execution_match": match,
            **metrics,
        }
        cases.append(case)
        if not match:
            execution_errors.append({
                "id": uid,
                "program": qa.get("program"),
                "gold": qa.get("exe_ans"),
                "executed": result.value,
                "valid": result.valid,
                "error": result.error,
            })
    total = len(rows)
    parsed = len(cases)
    matching = sum(case["execution_match"] for case in cases)
    counts = {
        name: sum(bool(case[name]) for case in cases)
        for name in ("deep", "join", "deep_join", "has_branch")
    }
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "n": total,
        "parsed": parsed,
        "parse_coverage": fraction(parsed, total),
        "execution_valid": sum(case["execution_valid"] for case in cases),
        "execution_matches": matching,
        "execution_agreement": fraction(matching, total),
        "subset_counts": counts,
        "subset_fractions": {key: fraction(value, total) for key, value in counts.items()},
        "step_histogram": histogram([case["n_steps"] for case in cases]),
        "depth_histogram": histogram([case["depth"] for case in cases]),
        "reference_edge_histogram": histogram([case["reference_edges"] for case in cases]),
        "operation_counts": {key: operations.get(key, 0) for key in ALL_OPS},
        "parse_errors": parse_errors[:50],
        "execution_errors": execution_errors[:50],
        "cases": cases,
    }


def build_report(paths: Sequence[str], source_commit: str) -> Dict[str, Any]:
    splits = {os.path.splitext(os.path.basename(path))[0]: audit_split(path) for path in paths}
    test = splits.get("test")
    checks: Dict[str, bool] = {}
    for split, metrics in splits.items():
        checks["%s_parse_ge_095" % split] = metrics["parse_coverage"] >= 0.95
        checks["%s_exec_ge_095" % split] = metrics["execution_agreement"] >= 0.95
    checks["test_present"] = test is not None
    checks["test_deep_ge_150"] = bool(test and test["subset_counts"]["deep"] >= 150)
    checks["test_join_ge_100"] = bool(test and test["subset_counts"]["join"] >= 100)
    return {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "splits": splits,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("protocol=%s source_commit=%s" % (report["protocol"], report["source_commit"]))
    for split, metrics in report["splits"].items():
        subsets = metrics["subset_counts"]
        print(
            "%s n=%d parse=%.4f exec=%.4f deep=%d join=%d deep_join=%d branch=%d sha256=%s"
            % (
                split, metrics["n"], metrics["parse_coverage"],
                metrics["execution_agreement"], subsets["deep"], subsets["join"],
                subsets["deep_join"], subsets["has_branch"], metrics["sha256"],
            )
        )
    print("checks=%s" % json.dumps(report["checks"], sort_keys=True))
    print("STAGE_A=%s" % ("PASS" if report["gate_pass"] else "FAIL"))


def self_test() -> None:
    from finqa_program import Step, canonical_program

    steps = parse_program("divide(9413, 20.01), subtract(#0, const_1)")
    assert canonical_program(steps) == "divide(9413, 20.01), subtract(#0, const_1)"
    metrics = structure_metrics(steps)
    assert metrics["n_steps"] == 2 and metrics["depth"] == 2
    result = execute_program(steps, [])
    assert result.valid and execution_matches(result.value, round(9413 / 20.01 - 1, 5))
    join = [
        Step("divide", ("10", "2")),
        Step("subtract", ("#0", "3")),
        Step("multiply", ("4", "5")),
        Step("add", ("#1", "#2")),
    ]
    assert structure_metrics(join)["deep_join"]
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/FinQA/dataset")
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--out", default="finqa_structure_audit_report.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    paths = [
        os.path.join(args.data_dir, "%s.json" % split.strip())
        for split in args.splits.split(",") if split.strip()
    ]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    report = build_report(paths, args.source_commit)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print_summary(report)


if __name__ == "__main__":
    main()
