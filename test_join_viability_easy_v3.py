import os
import unittest

from join_viability_easy_v2 import CALIBRATION_N
from join_viability_easy_v2_1 import MAX_NEW_BY_ARM
from join_viability_easy_v3 import MODEL_REVISION, PROTOCOL, analyze, model_snapshot


def fake_rows(n):
    return [
        {
            "id": i,
            "answer": str(i + 10),
            "parent_answers": [str(i + 1), str(i + 2)],
        }
        for i in range(n)
    ]


def fake_calls(rows):
    calls = {}
    for index, row in enumerate(rows):
        values = {
            "direct": row["answer"] if index < 48 else "wrong",
            "parent_0": row["parent_answers"][0],
            "parent_1": row["parent_answers"][1] if index < 48 else "wrong",
            "gold_root": row["answer"] if index < 72 else "wrong",
        }
        for arm, answer in values.items():
            calls[(row["id"], arm)] = {
                "answer": answer,
                "calls": 1,
                "max_new": MAX_NEW_BY_ARM[arm],
                "prompt_tokens": 1,
                "generated_tokens": 1,
            }
    return calls


class JoinViabilityEasyV3Test(unittest.TestCase):
    def test_snapshot_path_is_revision_locked(self):
        path = model_snapshot("/project/hf")
        self.assertEqual(
            path,
            os.path.join(
                "/project/hf",
                "hub",
                "models--Qwen--Qwen2.5-14B-Instruct",
                "snapshots",
                MODEL_REVISION,
            ),
        )

    def test_report_records_v3_protocol_and_model_revision(self):
        rows = fake_rows(CALIBRATION_N)
        report = analyze(rows, fake_calls(rows), "calibration")
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["protocol"], PROTOCOL)
        self.assertEqual(report["model"]["revision"], MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
