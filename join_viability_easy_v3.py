"""Frozen 14B progressive screen on the unchanged easy-join benchmark."""
from __future__ import annotations

import argparse
import os

import join_viability_easy_v2 as v2
import join_viability_easy_v2_1 as v21


PROTOCOL = "EXPERIMENT_PROTOCOL_JOIN_VIABILITY_EASY_V3.md"
MODEL_REPO = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL_CACHE_DIR = "models--Qwen--Qwen2.5-14B-Instruct"


def model_snapshot(hf_home: str) -> str:
    return os.path.join(
        hf_home, "hub", MODEL_CACHE_DIR, "snapshots", MODEL_REVISION
    )


def analyze(rows: list[dict], calls: dict[tuple[int, str], dict], split: str) -> dict:
    report = v21.analyze(rows, calls, split, protocol=PROTOCOL)
    report["model"] = {"repo": MODEL_REPO, "revision": MODEL_REVISION}
    return report


def self_test() -> None:
    assert len(MODEL_REVISION) == 40
    assert set(MODEL_REVISION) <= set("0123456789abcdef")
    assert v21.MAX_NEW_BY_ARM == {
        "direct": 1536,
        "parent_0": 512,
        "parent_1": 512,
        "gold_root": 512,
    }
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-data", default="data/gsm_join_easy_train.jsonl")
    parser.add_argument("--confirmation-data", default="data/gsm_join_easy_test.jsonl")
    parser.add_argument("--out-dir", default="join_viability_easy_v3")
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", ""))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.hf_home:
        raise ValueError("--hf-home or HF_HOME is required")
    snapshot = model_snapshot(args.hf_home)
    if not os.path.isdir(snapshot):
        raise FileNotFoundError(f"frozen model snapshot not found: {snapshot}")

    calibration = v2.load_subset(
        args.calibration_data, "calibration", v2.CALIBRATION_N
    )
    confirmation = v2.load_subset(
        args.confirmation_data, "confirmation", v2.CONFIRMATION_N
    )
    os.makedirs(args.out_dir, exist_ok=True)
    runner = v2.Runner(snapshot)

    calibration_calls = v2.run_units(
        runner,
        calibration,
        os.path.join(args.out_dir, "calibration"),
        args.batch_size,
        max_new_by_arm=v21.MAX_NEW_BY_ARM,
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
                "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
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
        max_new_by_arm=v21.MAX_NEW_BY_ARM,
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
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "calibration": calibration_report,
            "confirmation": confirmation_report,
            "gate_pass": confirmation_report["gate_pass"],
        },
    )


if __name__ == "__main__":
    main()
