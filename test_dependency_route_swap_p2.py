import unittest

from dependency_route_swap_p2 import LOCKED_LAYER, build_cases, choose_donors


class DependencyRouteSwapP2Test(unittest.TestCase):
    def test_new_seed_and_one_hop_answers(self):
        cases = build_cases(288, 20260817)
        self.assertEqual(len(cases), 288)
        self.assertEqual(LOCKED_LAYER, 21)
        self.assertTrue(all(case.clean_root == case.clean_p for case in cases))
        self.assertTrue(all(case.corrupt_root == case.corrupt_p for case in cases))
        self.assertTrue(all(case.clean_p != case.corrupt_p for case in cases))

    def test_route_pair_differs_only_at_print_target(self):
        for case in build_cases(96, 20260817):
            self.assertEqual(len(case.route_on_user), len(case.route_off_user))
            diffs = [
                i for i, pair in enumerate(zip(case.route_on_user, case.route_off_user))
                if pair[0] != pair[1]
            ]
            self.assertEqual(len(diffs), 1)
            self.assertLess(diffs[0], case.route_on_user.index("Checkpoint values:"))
            self.assertEqual(case.route_on_user[diffs[0]], case.labels["root"])
            self.assertEqual(case.route_off_user[diffs[0]], case.labels["decoy"])

    def test_corruption_changes_only_p_digit(self):
        for case in build_cases(96, 20260817):
            self.assertEqual(len(case.route_on_user), len(case.corrupt_user))
            diffs = [
                i for i, pair in enumerate(zip(case.route_on_user, case.corrupt_user))
                if pair[0] != pair[1]
            ]
            self.assertEqual(len(diffs), 1)
            self.assertGreater(diffs[0], case.route_on_user.index("Checkpoint values:"))

    def test_same_checkpoint_context(self):
        for case in build_cases(96, 20260817):
            start = case.route_on_user.index("Checkpoint values:")
            self.assertEqual(case.route_on_user[start:], case.route_off_user[start:])

    def test_cross_problem_donors_are_distinct_and_value_matched(self):
        cases = build_cases(288, 20260817)
        donors = choose_donors(cases)
        for index, donor in enumerate(donors):
            self.assertNotEqual(index, donor)
            self.assertEqual(cases[index].clean_p, cases[donor].clean_p)

    def test_order_is_counterbalanced(self):
        count = sum(case.p_first for case in build_cases(288, 20260817))
        self.assertGreaterEqual(count, 112)
        self.assertLessEqual(count, 176)


if __name__ == "__main__":
    unittest.main()
