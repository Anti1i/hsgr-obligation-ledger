import unittest

from dependency_patch_p1 import build_cases, choose_donors


class DependencyPatchP1Test(unittest.TestCase):
    def test_cases_are_new_one_hop_identity_cases(self):
        cases = build_cases(288, 20260816)
        self.assertEqual(len(cases), 288)
        self.assertTrue(all(case.family == "chain1_copy" for case in cases))
        self.assertTrue(all(case.clean_root == case.clean_p for case in cases))
        self.assertTrue(all(case.corrupt_root == case.corrupt_p for case in cases))
        self.assertTrue(all(case.clean_p != case.corrupt_p for case in cases))

    def test_prompt_corruption_is_one_character(self):
        for case in build_cases(64, 20260816):
            self.assertEqual(len(case.clean_user), len(case.corrupt_user))
            diffs = [
                i
                for i, (clean, corrupt) in enumerate(
                    zip(case.clean_user, case.corrupt_user)
                )
                if clean != corrupt
            ]
            self.assertEqual(len(diffs), 1)

    def test_donors_are_distinct_and_value_matched(self):
        cases = build_cases(288, 20260816)
        donors = choose_donors(cases)
        self.assertEqual(len(donors), len(cases))
        for index, donor in enumerate(donors):
            self.assertNotEqual(index, donor)
            self.assertEqual(cases[index].clean_p, cases[donor].clean_p)

    def test_order_is_counterbalanced(self):
        cases = build_cases(288, 20260816)
        count = sum(case.p_first for case in cases)
        self.assertGreaterEqual(count, 112)
        self.assertLessEqual(count, 176)


if __name__ == "__main__":
    unittest.main()

