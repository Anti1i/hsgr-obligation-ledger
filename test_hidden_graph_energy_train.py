import unittest

from hidden_graph_energy_train import (
    expected_calibration_error,
    high_confidence_negative_rate,
    holm_adjust,
    nonhidden_row,
    selection_outcomes,
    simple_numeric,
)


class HiddenGraphEnergyTrainTest(unittest.TestCase):
    def test_surface_features_are_finite(self):
        row = nonhidden_row({
            "bindings": ["1,000", "-2.5"],
            "frequencies": [0.75, 0.25],
            "counts": [3, 1],
            "is_modal": True,
        })
        self.assertEqual(len(row), 11)
        self.assertGreater(simple_numeric("1,000"), simple_numeric("10"))
        self.assertEqual(simple_numeric("not numeric"), 0.0)

    def test_selection_is_problem_level(self):
        pids, chosen, modal = selection_outcomes(
            [0.1, 0.9, 0.5], [0, 1, 0], [1, 1, 2], [True, False, True]
        )
        self.assertEqual(pids, [1, 2])
        self.assertEqual(chosen, [True, False])
        self.assertEqual(modal, [False, False])

    def test_calibration_diagnostics(self):
        self.assertAlmostEqual(expected_calibration_error([0.1, 0.9], [0, 1]), 0.1)
        self.assertEqual(high_confidence_negative_rate([0.95, 0.2], [0, 0]), 0.5)

    def test_holm_is_monotone(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.08)
        self.assertAlmostEqual(adjusted["c"], 0.2)


if __name__ == "__main__":
    unittest.main()
