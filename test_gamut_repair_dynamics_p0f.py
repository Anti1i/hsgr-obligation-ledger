import unittest

from gamut_repair_dynamics_p0f import arm_summary, transition_row


class RepairDynamicsP0fTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "id": "x",
            "valid": True,
            "relation_met": False,
            "all_components_present": True,
            "matched_order": ["P2", "P1"],
        }
        self.candidate = {
            "id": "x",
            "arm": "flat_full_rewrite",
            "question": "q",
            "steps": ["one", "two"],
            "original_answer": "two then one",
            "answer": "one then two",
            "patch_valid": True,
            "sentence_span": None,
            "edit_ratio": 0.2,
            "valid_extraction": True,
            "matched_order": ["P1", "P2"],
            "relation_met": True,
        }

    def test_monotonic_targeted_repair(self):
        row = transition_row(self.candidate, self.baseline)
        self.assertTrue(row["relation_sorted"])
        self.assertTrue(row["complete_target_recovered"])
        self.assertFalse(row["any_component_regression"])
        self.assertTrue(row["automatic_monotonic_success"])

    def test_fix_one_break_another(self):
        self.candidate["matched_order"] = ["P1"]
        row = transition_row(self.candidate, self.baseline)
        self.assertEqual(row["component_regressions"], ["P2"])
        self.assertTrue(row["destructive_repair_attempt"])
        self.assertFalse(row["complete_target_recovered"])

    def test_invalid_extraction_is_unknown_not_regression(self):
        self.candidate["valid_extraction"] = False
        self.candidate["matched_order"] = []
        self.candidate["relation_met"] = False
        row = transition_row(self.candidate, self.baseline)
        self.assertIsNone(row["component_regressions"])
        self.assertIsNone(row["any_component_regression"])

    def test_summary(self):
        row = transition_row(self.candidate, self.baseline)
        summary = arm_summary([row])
        self.assertEqual(summary["relation_sorted_rate"], 1.0)
        self.assertEqual(summary["complete_target_recovery_rate"], 1.0)
        self.assertEqual(summary["automatic_monotonic_success_rate"], 1.0)

    def test_precondition_fails_closed(self):
        self.baseline["relation_met"] = True
        with self.assertRaises(ValueError):
            transition_row(self.candidate, self.baseline)


if __name__ == "__main__":
    unittest.main()
