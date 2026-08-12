import unittest

from join_viability_easy_v2 import (
    CALIBRATION_N,
    analyze,
    stable_rank,
)


def fake_rows(n):
    return [
        {
            "id": i,
            "answer": str(i + 10),
            "parent_answers": [str(i + 1), str(i + 2)],
        }
        for i in range(n)
    ]


def fake_calls(rows, direct_correct, parent_correct, root_correct):
    calls = {}
    for index, row in enumerate(rows):
        values = {
            "direct": row["answer"] if index < direct_correct else "wrong",
            "parent_0": row["parent_answers"][0]
            if index * 2 < parent_correct else "wrong",
            "parent_1": row["parent_answers"][1]
            if index * 2 + 1 < parent_correct else "wrong",
            "gold_root": row["answer"] if index < root_correct else "wrong",
        }
        for arm, answer in values.items():
            calls[(row["id"], arm)] = {
                "answer": answer,
                "calls": 1,
                "max_new": 512 if arm == "direct" else 192,
                "prompt_tokens": 1,
                "generated_tokens": 1,
            }
    return calls


class JoinViabilityEasyV2Test(unittest.TestCase):
    def test_selection_rank_is_split_specific(self):
        self.assertEqual(stable_rank("calibration", "p"), stable_rank("calibration", "p"))
        self.assertNotEqual(stable_rank("calibration", "p"), stable_rank("confirmation", "p"))

    def test_calibration_gate_passes_only_complete_strong_regime(self):
        rows = fake_rows(CALIBRATION_N)
        report = analyze(
            rows,
            fake_calls(rows, direct_correct=48, parent_correct=144, root_correct=72),
            "calibration",
        )
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["accuracy"]["direct"], 0.5)
        self.assertEqual(report["accuracy"]["mean_parent"], 0.75)
        self.assertEqual(report["accuracy"]["gold_bound_root"], 0.75)

    def test_calibration_rejects_small_root_gap(self):
        rows = fake_rows(CALIBRATION_N)
        report = analyze(
            rows,
            fake_calls(rows, direct_correct=58, parent_correct=144, root_correct=64),
            "calibration",
        )
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["gates"]["gold_root_gap"])


if __name__ == "__main__":
    unittest.main()

