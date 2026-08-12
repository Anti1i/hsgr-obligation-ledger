"""Truncation-controlled corrective replication of provenance ceiling v1."""
from __future__ import annotations

import argparse
import json
import os

import hsgr_error_provenance_ceiling as v1
from pilot import Runner, jread


MAX_NEW = 512
PROTOCOL = "EXPERIMENT_PROTOCOL_ERROR_PROVENANCE_GUIDE_V1_1.md"


def parse_validity(base: list[dict], repairs: list[dict]) -> dict:
    by_arm = {
        arm: [row for row in repairs if row["arm"] == arm]
        for arm in v1.ACTIONS
    }

    def rate(rows: list[dict], key: str) -> float:
        return sum(row.get(key) not in (None, "") for row in rows) / max(1, len(rows))

    repair_parse = {arm: rate(rows, "answer") for arm, rows in by_arm.items()}
    repair_near_cap = {
        arm: sum(int(row.get("generated_tokens", 0)) >= MAX_NEW - 1 for row in rows)
        / max(1, len(rows))
        for arm, rows in by_arm.items()
    }
    return {
        "base_q1_parse_rate": rate(base, "q1_answer"),
        "base_q2_parse_rate": rate(base, "q2_answer"),
        "repair_parse_rate": repair_parse,
        "repair_near_cap_rate": repair_near_cap,
    }


def analyze(rows: list[dict], base: list[dict], repairs: list[dict]) -> dict:
    report = v1.analyze(rows, base, repairs)
    validity = parse_validity(base, repairs)
    parse_rates = [
        validity["base_q1_parse_rate"],
        validity["base_q2_parse_rate"],
        *validity["repair_parse_rate"].values(),
    ]
    report["protocol"] = PROTOCOL
    report["corrective_change"] = {"common_max_new_tokens": MAX_NEW}
    report["output_validity"] = validity
    report["gates"]["all_output_parse_rates_ge_95pct"] = min(parse_rates) >= 0.95
    report["gate_pass"] = all(report["gates"].values())
    return report


def print_report(report: dict) -> None:
    v1.print_report(report)
    validity = report["output_validity"]
    print(
        "parse rates: "
        f"base_q1={validity['base_q1_parse_rate']:.4f} "
        f"base_q2={validity['base_q2_parse_rate']:.4f} "
        f"repairs={validity['repair_parse_rate']}"
    )
    print(f"repair near-cap rates={validity['repair_near_cap_rate']}")


def self_test() -> None:
    v1.self_test()
    base = [{"q1_answer": "1", "q2_answer": "2"}]
    repairs = [
        {"arm": arm, "answer": "2", "generated_tokens": 4}
        for arm in v1.ACTIONS
    ]
    validity = parse_validity(base, repairs)
    assert validity["base_q1_parse_rate"] == 1.0
    assert min(validity["repair_parse_rate"].values()) == 1.0
    print("SELF_TEST_V1_1_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_chain_test.jsonl")
    parser.add_argument("--out-dir", default="hsgr_error_provenance_v1_1")
    parser.add_argument("--model", default=v1.MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    v1.MAX_NEW_BASE = MAX_NEW
    v1.MAX_NEW_REPAIR = MAX_NEW
    if args.self_test:
        self_test()
        return

    data_path = args.data if os.path.isabs(args.data) else os.path.join(v1.HERE, args.data)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(v1.HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows = v1.read_rows(data_path, args.limit)
    if args.analyze_only:
        base = jread(os.path.join(out_dir, "base.jsonl"))
        repairs = jread(os.path.join(out_dir, "repairs.jsonl"))
    else:
        runner = Runner(args.model)
        base = v1.run_base(runner, rows, out_dir, args.batch_size)
        repairs = v1.run_repairs(runner, rows, base, out_dir, args.batch_size)
    if len(base) != len(rows) or len(repairs) != len(rows) * len(v1.ACTIONS):
        raise RuntimeError(
            f"incomplete outputs: rows={len(rows)} base={len(base)} repairs={len(repairs)}"
        )

    report = analyze(rows, base, repairs)
    cases = report.pop("cases")
    report_path = os.path.join(out_dir, "hsgr_error_provenance_v1_1_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(
        os.path.join(out_dir, "hsgr_error_provenance_v1_1_cases.jsonl"),
        "w",
        encoding="utf-8",
    ) as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()

