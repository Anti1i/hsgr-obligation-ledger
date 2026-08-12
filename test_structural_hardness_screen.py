import unittest

from structural_hardness_screen import modal_candidate, rule_specs


class StructuralHardnessScreenTest(unittest.TestCase):
    def test_modal_tie_prefers_greedy(self):
        candidates = [
            {"kind": "greedy", "answer": "2", "norm": "2", "text": ""},
            {"kind": "sample", "answer": "3", "norm": "3", "text": ""},
            {"kind": "sample", "answer": "3", "norm": "3", "text": ""},
            {"kind": "sample", "answer": "2", "norm": "2", "text": ""},
        ]
        self.assertEqual(modal_candidate(candidates)["norm"], "2")

    def test_rules_use_metadata_only(self):
        rules = dict(rule_specs())
        case = {"n_steps": 14, "root_step_count": 3, "parent_step_counts": [4, 2]}
        self.assertTrue(rules["total_ge_12__root_ge_3"](case))
        self.assertFalse(rules["min_parent_ge_3"](case))


if __name__ == "__main__":
    unittest.main()
