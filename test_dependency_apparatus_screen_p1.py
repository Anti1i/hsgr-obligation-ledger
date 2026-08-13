import unittest

from dependency_apparatus_screen_p1 import (
    FAMILY_ORDER,
    build_cases,
    select_family,
)


class DependencyApparatusScreenP1Test(unittest.TestCase):
    def test_all_families_are_sensitive_and_single_character_corruptions(self):
        for family in FAMILY_ORDER:
            cases = build_cases(family, 32, 20260815)
            self.assertEqual(len(cases), 32)
            for case in cases:
                self.assertNotEqual(case.clean_root, case.corrupt_root)
                self.assertEqual(len(case.clean_user), len(case.corrupt_user))
                diffs = [
                    i
                    for i, (clean, corrupt) in enumerate(
                        zip(case.clean_user, case.corrupt_user)
                    )
                    if clean != corrupt
                ]
                self.assertEqual(len(diffs), 1)

    def test_copy_families_copy_checkpoint_value(self):
        for family in ("chain3_copy", "chain1_copy"):
            for case in build_cases(family, 32, 20260815):
                clean_section = case.clean_user.split("Checkpoint values:\n", 1)[1]
                digit_lines = [line for line in clean_section.splitlines() if " = " in line]
                values = [int(line.rsplit(" = ", 1)[1]) for line in digit_lines]
                self.assertIn(case.clean_root, values)

    def test_order_is_counterbalanced(self):
        for family in FAMILY_ORDER:
            cases = build_cases(family, 96, 20260815)
            count = sum(case.p_first for case in cases)
            self.assertGreaterEqual(count, 32)
            self.assertLessEqual(count, 64)

    def test_selection_uses_fixed_complexity_order(self):
        reports = {family: {"gate_pass": False} for family in FAMILY_ORDER}
        reports["chain3_copy"]["gate_pass"] = True
        reports["chain1_copy"]["gate_pass"] = True
        self.assertEqual(select_family(reports), "chain3_copy")
        reports["dag_add"]["gate_pass"] = True
        self.assertEqual(select_family(reports), "dag_add")

    def test_no_pass_returns_none(self):
        reports = {family: {"gate_pass": False} for family in FAMILY_ORDER}
        self.assertIsNone(select_family(reports))


if __name__ == "__main__":
    unittest.main()

