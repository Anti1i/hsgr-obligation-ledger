import unittest

try:
    import torch
except ModuleNotFoundError:  # Local lightweight test environments may omit torch.
    torch = None

from dependency_route_subspace_p3 import (
    PRIMARY_LAYER,
    WINDOW_LAYERS,
    P3Runner,
    build_cases,
    fit_route_directions,
)


class DependencyRouteSubspaceP3Test(unittest.TestCase):
    def test_three_distinct_answers_and_locked_layers(self):
        cases = build_cases(288, 20260818)
        self.assertEqual(PRIMARY_LAYER, 21)
        self.assertEqual(WINDOW_LAYERS, (19, 20, 21))
        for case in cases:
            self.assertEqual(len({case.clean_root, case.corrupt_root, case.decoy_root}), 3)

    def test_route_pair_differs_only_at_print_target(self):
        for case in build_cases(96, 20260818):
            self.assertEqual(len(case.route_on_user), len(case.route_off_user))
            diffs = [i for i, pair in enumerate(zip(case.route_on_user, case.route_off_user)) if pair[0] != pair[1]]
            self.assertEqual(len(diffs), 1)
            self.assertLess(diffs[0], case.route_on_user.index("Checkpoint values:"))

    def test_corruption_changes_only_p_digit(self):
        for case in build_cases(96, 20260818):
            self.assertEqual(len(case.route_on_user), len(case.corrupt_user))
            diffs = [i for i, pair in enumerate(zip(case.route_on_user, case.corrupt_user)) if pair[0] != pair[1]]
            self.assertEqual(len(diffs), 1)
            self.assertGreater(diffs[0], case.route_on_user.index("Checkpoint values:"))

    def test_all_digits_present_in_calibration(self):
        cases = build_cases(288, 20260818)
        self.assertEqual(set(case.clean_p for case in cases[:96]), set(range(10)))
        self.assertEqual(set(case.decoy_x for case in cases[:96]), set(range(10)))

    @unittest.skipUnless(torch is not None, "torch is required for direction algebra")
    def test_direction_is_value_orthogonal_and_oriented(self):
        cases = build_cases(288, 20260818)
        hidden = 32
        generator = torch.Generator().manual_seed(7)
        value = torch.randn(10, hidden, generator=generator)
        route = torch.randn(hidden, generator=generator)
        centered = value - value.mean(0, keepdim=True)
        _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = int((singular > singular.max() * 1e-6).sum())
        basis = vh[:rank].T
        route = route - basis @ (basis.T @ route)
        route = torch.nn.functional.normalize(route, dim=0)
        feats = {}
        for layer in WINDOW_LAYERS:
            midpoint = torch.stack([value[case.clean_p] for case in cases])
            on = midpoint + 0.5 * route
            off = midpoint - 0.5 * route
            feats[layer] = {"p": on, "x": off}
        directions, shams, report = fit_route_directions(feats, cases, 96, WINDOW_LAYERS, 20260818)
        self.assertTrue(report["gate_pass"])
        for layer in WINDOW_LAYERS:
            self.assertGreater(float(torch.dot(directions[layer], route)), 0.8)
            self.assertLess(abs(float(torch.dot(directions[layer], shams[layer]))), 1e-5)

    @unittest.skipUnless(torch is not None, "torch is required for intervention algebra")
    def test_arm_edits_have_matched_norm_and_route_projection(self):
        generator = torch.Generator().manual_seed(11)
        on = torch.randn(8, 16, generator=generator)
        off = torch.randn(8, 16, generator=generator)
        direction = torch.nn.functional.normalize(torch.randn(16, generator=generator), dim=0)
        sham = torch.randn(16, generator=generator)
        sham = sham - torch.dot(sham, direction) * direction
        sham = torch.nn.functional.normalize(sham, dim=0)
        arms = P3Runner.arm_vectors(on, off, direction, sham)
        route_shift = arms["route_swap"] - on
        plus_shift = arms["sham_plus"] - on
        minus_shift = arms["sham_minus"] - on
        self.assertTrue(torch.allclose(route_shift.norm(dim=1), plus_shift.norm(dim=1), atol=1e-5))
        self.assertTrue(torch.allclose(route_shift.norm(dim=1), minus_shift.norm(dim=1), atol=1e-5))
        self.assertTrue(torch.allclose(arms["route_swap"] @ direction, off @ direction, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
