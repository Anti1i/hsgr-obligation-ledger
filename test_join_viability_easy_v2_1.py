import unittest

from join_viability_easy_v2 import CALIBRATION_N
from join_viability_easy_v2_1 import MAX_NEW_BY_ARM, analyze


def fake_rows(n):
    return [
        {
            "id": i,
            "answer": str(i + 10),
            "parent_answers": [str(i + 1), str(i + 2)],
        }
        for i in range(n)
    ]


def fake_calls(rows, capped=False):
    calls = {}
    for index, row in enumerate(rows):
        values = {
            "direct": row["answer"] if index < 48 else "wrong",
            "parent_0": row["parent_answers"][0],
            "parent_1": row["parent_answers"][1]
            if index < 48 else "wrong",
            "gold_root": row["answer"] if index < 72 else "wrong",
        }
        for arm, answer in values.items():
            calls[(row["id"], arm)] = {
                "answer": answer,
                "calls": 1,
                "max_new": MAX_NEW_BY_ARM[arm],
                "prompt_tokens": 1,
                "generated_tokens": MAX_NEW_BY_ARM[arm] if capped else 1,
            }
    return calls


class JoinViabilityEasyV21Test(unittest.TestCase):
    def test_complete_uncapped_regime_passes(self):
        rows = fake_rows(CALIBRATION_N)
        report = analyze(rows, fake_calls(rows), "calibration")
        self.assertTrue(report["gate_pass"])
        self.assertTrue(report["gates"]["token_cap_rate"])

    def test_token_cap_gate_rejects_truncated_regime(self):
        rows = fake_rows(CALIBRATION_N)
        report = analyze(rows, fake_calls(rows, capped=True), "calibration")
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["gates"]["token_cap_rate"])

    def test_old_generation_budgets_fail_accounting(self):
        rows = fake_rows(CALIBRATION_N)
        calls = fake_calls(rows)
        calls[(0, "gold_root")]["max_new"] = 192
        report = analyze(rows, calls, "calibration")
        self.assertFalse(report["gates"]["one_call_accounting"])


if __name__ == "__main__":
    unittest.main()
