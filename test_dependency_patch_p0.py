import unittest

from dependency_patch_p0 import (
    apply_op,
    auc,
    build_cases,
    choose_donors,
    execute,
    exact_sign_p,
)


class DependencyPatchP0Test(unittest.TestCase):
    def test_ops(self):
        self.assertEqual(apply_op(8, "+", 7), 5)
        self.assertEqual(apply_op(2, "-", 9), 3)
        self.assertEqual(apply_op(7, "*", 8), 6)

    def test_generated_cases_are_sensitive_and_matched(self):
        cases = build_cases(80, 20260813)
        self.assertEqual(len(cases), 80)
        self.assertGreater(len({c.id for c in cases}), 79)
        for case in cases:
            self.assertEqual(case.clean_values["p"], case.clean_values["x"])
            self.assertNotEqual(case.clean_root, case.corrupt_root)
            corrupt = execute(
                {
                    "p": case.corrupt_p,
                    "x": case.clean_values["x"],
                    "q": case.clean_values["q"],
                    "t": case.clean_values["t"],
                    "qx": case.clean_values["qx"],
                    "tx": case.clean_values["tx"],
                },
                case.ops,
            )
            self.assertEqual(corrupt["root"], case.corrupt_root)
            self.assertEqual(case.clean_values["r"], case.clean_values["rd"])
            self.assertEqual(case.clean_values["s"], case.clean_values["sd"])
            self.assertEqual(case.clean_values["u"], case.clean_values["ud"])
            self.assertEqual(
                case.clean_values["root"], case.clean_values["decoy_root"]
            )
            self.assertIn("Checkpoint values:", case.clean_user)
            self.assertEqual(len(case.clean_user), len(case.corrupt_user))

    def test_order_is_counterbalanced(self):
        cases = build_cases(200, 20260813)
        rate = sum(c.p_first for c in cases) / len(cases)
        self.assertGreater(rate, 0.35)
        self.assertLess(rate, 0.65)

    def test_donors_are_distinct_and_value_matched(self):
        cases = build_cases(200, 20260813)
        donors = choose_donors(cases)
        for i, j in enumerate(donors):
            self.assertNotEqual(i, j)
            self.assertEqual(cases[i].clean_values["p"], cases[j].clean_values["p"])

    def test_auc_and_sign_test(self):
        self.assertAlmostEqual(auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertAlmostEqual(exact_sign_p(3, 0), 0.125)


if __name__ == "__main__":
    unittest.main()
