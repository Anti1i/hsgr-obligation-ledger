import unittest

from audit_marginal_root_utility_guide import choose, exact_mcnemar, summarize_pair


class TestMarginalRootUtilityGuide(unittest.TestCase):
    def test_choose_uses_frequency_then_frozen_order(self):
        classes = [{"freq": 0.25}, {"freq": 0.75}, {"freq": 0.75}]
        self.assertEqual(choose(classes, [1.0, 1.0, 1.0]), 1)

    def test_exact_mcnemar_symmetric(self):
        a = [True] * 8 + [False] * 2
        b = [False] * 8 + [True] * 2
        x = exact_mcnemar(a, b)
        y = exact_mcnemar(b, a)
        self.assertEqual((x["wins"], x["losses"]), (8, 2))
        self.assertEqual(x["p"], y["p"])

    def test_pair_mask(self):
        r = summarize_pair([True, False, True], [False, True, True], [True, True, False])
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["delta"], 0.0)
        self.assertEqual(r["mcnemar"]["wins"], 1)
        self.assertEqual(r["mcnemar"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()

