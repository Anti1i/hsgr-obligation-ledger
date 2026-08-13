import unittest

from dependency_route_apparatus_p3a import TEMPLATE_ORDER, build_cases, select_template


class DependencyRouteApparatusP3ATest(unittest.TestCase):
    def test_templates_are_frozen(self):
        self.assertEqual(TEMPLATE_ORDER, ("original", "explicit_select", "concise_select"))

    def test_values_are_distinct(self):
        for template in TEMPLATE_ORDER:
            for case in build_cases(template, 96, 20260819):
                self.assertEqual(len({case.clean_root, case.corrupt_root, case.decoy_root}), 3)

    def test_route_pairs_change_one_print_character(self):
        for template in TEMPLATE_ORDER:
            for case in build_cases(template, 32, 20260819):
                self.assertEqual(len(case.route_on_user), len(case.route_off_user))
                diffs = [i for i, pair in enumerate(zip(case.route_on_user, case.route_off_user)) if pair[0] != pair[1]]
                self.assertEqual(len(diffs), 1)
                self.assertLess(diffs[0], case.route_on_user.index("Checkpoint values:"))

    def test_corruption_changes_one_checkpoint_character(self):
        for template in TEMPLATE_ORDER:
            for case in build_cases(template, 32, 20260819):
                self.assertEqual(len(case.route_on_user), len(case.corrupt_user))
                diffs = [i for i, pair in enumerate(zip(case.route_on_user, case.corrupt_user)) if pair[0] != pair[1]]
                self.assertEqual(len(diffs), 1)
                self.assertGreater(diffs[0], case.route_on_user.index("Checkpoint values:"))

    def test_selection_uses_first_passing_template(self):
        reports = {template: {"gate_pass": False} for template in TEMPLATE_ORDER}
        reports["explicit_select"]["gate_pass"] = True
        reports["concise_select"]["gate_pass"] = True
        self.assertEqual(select_template(reports), "explicit_select")

    def test_no_pass_returns_none(self):
        reports = {template: {"gate_pass": False} for template in TEMPLATE_ORDER}
        self.assertIsNone(select_template(reports))


if __name__ == "__main__":
    unittest.main()
