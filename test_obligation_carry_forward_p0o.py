import unittest
from collections import Counter

from obligation_carry_forward_p0o import (
    ARMS,
    RESCUE_CASE_IDS,
    arm_feedback,
    gate_decision,
    shuffled_evaluation,
    summarize,
)


class ObligationCarryForwardP0OTests(unittest.TestCase):
    def test_frozen_rescue_set_is_eight_distinct_cases(self):
        self.assertEqual(len(RESCUE_CASE_IDS), 8)
        self.assertEqual(len(set(RESCUE_CASE_IDS)), 8)

    def test_shuffled_state_preserves_counts_and_flips_both_directions(self):
        evaluation = {1: True, 2: True, 3: False, 4: True, 5: False}
        shuffled = shuffled_evaluation("case-a", evaluation)
        self.assertEqual(Counter(evaluation.values()), Counter(shuffled.values()))
        self.assertTrue(any(evaluation[i] and not shuffled[i] for i in evaluation))
        self.assertTrue(any(not evaluation[i] and shuffled[i] for i in evaluation))
        self.assertEqual(shuffled, shuffled_evaluation("case-a", evaluation))

    def test_prompt_controls_expose_expected_information(self):
        checklist = ["Does the response mention A?", "Does the response mention B?"]
        evaluation = {1: True, 2: False}
        failed = arm_feedback("case-a", checklist, evaluation, "failed_only")
        all_list = arm_feedback("case-a", checklist, evaluation, "all_checklist_no_status")
        full = arm_feedback("case-a", checklist, evaluation, "full_ledger")
        shuffled = arm_feedback("case-a", checklist, evaluation, "shuffled_status")
        self.assertNotIn("mention A", failed)
        self.assertIn("mention B", failed)
        self.assertIn("mention A", all_list)
        self.assertIn("mention B", all_list)
        self.assertIn("PRESERVE", full)
        self.assertIn("REPAIR", full)
        self.assertNotEqual(full, shuffled)
        self.assertEqual(len(full), len(shuffled))

    def test_summary_distinguishes_fix_preservation_and_joint(self):
        rows = [
            {
                "valid_transition": True, "prior_yes": 2, "prior_no": 2,
                "fixes": 2, "regressions": 0, "any_fix": True,
                "all_targets_fixed": True, "any_regression": False,
                "all_preserved": True, "joint_success_any": True,
                "strict_joint_success": True,
                "transitions": {"YY": 2, "NY": 2},
            },
            {
                "valid_transition": True, "prior_yes": 2, "prior_no": 2,
                "fixes": 1, "regressions": 1, "any_fix": True,
                "all_targets_fixed": False, "any_regression": True,
                "all_preserved": False, "joint_success_any": False,
                "strict_joint_success": False,
                "transitions": {"YY": 1, "YN": 1, "NY": 1, "NN": 1},
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["target_fix_rate"], 0.75)
        self.assertEqual(result["preserve_rate"], 0.75)
        self.assertEqual(result["any_regression_rate"], 0.5)
        self.assertEqual(result["joint_success_any_rate"], 0.5)
        self.assertEqual(result["strict_joint_success_rate"], 0.5)

    def test_gate_requires_state_specific_joint_gain(self):
        base = {
            "target_fix_rate": 0.90,
            "any_regression_rate": 0.30,
            "joint_success_any_rate": 0.60,
        }
        summaries = {
            "failed_only": base,
            "all_checklist_no_status": {**base, "joint_success_any_rate": 0.68},
            "shuffled_status": {**base, "joint_success_any_rate": 0.65},
            "full_ledger": {
                "target_fix_rate": 0.86,
                "any_regression_rate": 0.10,
                "joint_success_any_rate": 0.82,
            },
        }
        gates, decision = gate_decision(summaries, {"ok": True})
        self.assertTrue(all(gates.values()))
        self.assertEqual(decision, "PROVISIONAL_PASS_REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
