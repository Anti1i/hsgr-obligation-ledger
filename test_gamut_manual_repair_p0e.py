import unittest

from gamut_manual_repair_p0e import arm_metrics, flat_guide, repair_prompt
from gamut_process_repair_p0 import ProcessCase, Requirement


class ManualRepairP0eTests(unittest.TestCase):
    def setUp(self):
        target = Requirement("order", "one before two")
        self.case = ProcessCase("x", "q", "evidence", target, ("one", "two"), (target,))

    def test_flat_guide_contains_same_numbered_nodes(self):
        self.assertEqual(flat_guide(("one", "two")), "Required stages in canonical order:\n1. one\n2. two")

    def test_typed_prompt_has_edges_but_flat_does_not(self):
        flat = repair_prompt(self.case, "saved", "flat_full_rewrite")
        typed = repair_prompt(self.case, "saved", "typed_full_rewrite")
        self.assertNotIn("Directed constraints", flat)
        self.assertIn("P1 -> P2", typed)
        self.assertIn("one", flat)
        self.assertIn("one", typed)

    def test_arm_metrics(self):
        rows = [{
            "valid_extraction": True,
            "all_components_present": True,
            "structural_safe_success": True,
            "patch_valid": True,
            "edit_ratio": 0.1,
        }]
        self.assertEqual(arm_metrics(rows)["structural_safe_success"], 1)


if __name__ == "__main__":
    unittest.main()
