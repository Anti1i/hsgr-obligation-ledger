"""Length-repaired progressive viability screen for the frozen easy join."""
from __future__ import annotations

import argparse
import os

import join_viability_easy_v2 as v2


PROTOCOL = "EXPERIMENT_PROTOCOL_JOIN_VIABILITY_EASY_V2_1.md"
MAX_NEW_BY_ARM = {
    "direct": 1536,
    "parent_0": 512,
    "parent_1": 512,
    "gold_root": 512,
}
MAX_TOKEN_CAP_RATE = 0.05


def analyze(rows: list[dict], calls: dict[tuple[int, str], dict], split: str) -> dict:
    return v2.analyze(
        rows,
        calls,
        split,
        expected_max_new_by_arm=MAX_NEW_BY_ARM,
        protocol=PROTOCOL,
        max_token_cap_rate=MAX_TOKEN_CAP_RATE,
    )


def self_test() -> None:
    assert MAX_NEW_BY_ARM == {
        "direct": 1536,
        "parent_0": 512,
        "parent_1": 512,
        "gold_root": 512,
    }
    assert MAX_TOKEN_CAP_RATE == 0.05
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-data", default="data/gsm_join_easy_train.jsonl")
    parser.add_argument("--confirmation-data", default="data/gsm_join_easy_test.jsonl")
    parser.add_argument("--out-dir", default="join_viability_easy_v2_1")
    parser.add_argument("--model", default=v2.MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    calibration = v2.load_subset(
        args.calibration_data, "calibration", v2.CALIBRATION_N
    )
    confirmation = v2.load_subset(
        args.confirmation_data, "confirmation", v2.CONFIRMATION_N
    )
    os.makedirs(args.out_dir, exist_ok=True)
    runner = v2.Runner(args.model)

    calibration_calls = v2.run_units(
        runner,
        calibration,
        os.path.join(args.out_dir, "calibration"),
        args.batch_size,
        max_new_by_arm=MAX_NEW_BY_ARM,
    )
    calibration_report = analyze(calibration, calibration_calls, "calibration")
    v2.write_report(
        os.path.join(args.out_dir, "calibration_report.json"), calibration_report
    )
    v2.print_report(calibration_report)
    if not calibration_report["gate_pass"]:
        v2.write_report(
            os.path.join(args.out_dir, "report.json"),
            {
                "protocol": PROTOCOL,
                "calibration": calibration_report,
                "confirmation": None,
                "gate_pass": False,
            },
        )
        return

    confirmation_calls = v2.run_units(
        runner,
        confirmation,
        os.path.join(args.out_dir, "confirmation"),
        args.batch_size,
        max_new_by_arm=MAX_NEW_BY_ARM,
    )
    confirmation_report = analyze(confirmation, confirmation_calls, "confirmation")
    v2.write_report(
        os.path.join(args.out_dir, "confirmation_report.json"),
        confirmation_report,
    )
    v2.print_report(confirmation_report)
    v2.write_report(
        os.path.join(args.out_dir, "report.json"),
        {
            "protocol": PROTOCOL,
            "calibration": calibration_report,
            "confirmation": confirmation_report,
            "gate_pass": confirmation_report["gate_pass"],
        },
    )


if __name__ == "__main__":
    main()
