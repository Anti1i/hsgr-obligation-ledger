import unittest

import numpy as np

from clapnq_evidence_marginal_p1 import (
    Case,
    Ridge,
    correlation,
    record_to_case,
    regression_metrics,
    render_user,
    select_alpha,
    surface_features,
)


class ClapnqEvidenceMarginalP1Test(unittest.TestCase):
    def sample_record(self):
        answer = " ".join(["answer"] * 35)
        return {
            "id": "case-1",
            "input": "What combines alpha and omega?",
            "passages": [
                {
                    "sentences": [
                        "Alpha is the first fact.",
                        "This is an unrelated sentence.",
                        "Omega is the second fact.",
                    ]
                }
            ],
            "output": [
                {
                    "answer": answer,
                    "selected_sentences": [
                        "Alpha is the first fact.",
                        "Omega is the second fact.",
                    ],
                    "meta": {"non_consecutive": True},
                }
            ],
        }

    def test_record_and_prompt(self):
        case = record_to_case(self.sample_record())
        self.assertIsNotNone(case)
        assert case is not None
        self.assertEqual(case.support_indices, (0, 2))
        user, spans = render_user(case, {0, 1, 2})
        for index in case.support_indices:
            left, right = spans[index]
            self.assertEqual(user[left:right], case.sentences[index])

    def test_drop_prompt_preserves_stable_labels(self):
        case = record_to_case(self.sample_record())
        assert case is not None
        user, spans = render_user(case, {1, 2})
        self.assertNotIn("[S000]", user)
        self.assertIn("[S002]", user)
        left, right = spans[2]
        self.assertEqual(user[left:right], case.sentences[2])

    def test_surface_features_are_finite(self):
        case = record_to_case(self.sample_record())
        assert case is not None
        self.assertTrue(np.isfinite(surface_features(case, 0)).all())

    def test_ridge_and_metrics(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 3))
        y = x[:, 0] - 0.5 * x[:, 1]
        prediction = Ridge(1.0).fit(x, y).predict(x)
        metrics = regression_metrics(y, prediction)
        self.assertGreater(metrics["spearman"], 0.95)
        self.assertGreater(correlation(y, prediction), 0.95)

    def test_grouped_alpha_selection(self):
        rng = np.random.default_rng(1)
        groups = [f"case-{i // 2}" for i in range(50)]
        # Make sure every deterministic fold has examples.
        while len({__import__('clapnq_evidence_marginal_p1').fold_for(g) for g in groups}) < 5:
            groups.append(f"extra-{len(groups)}")
        x = rng.normal(size=(len(groups), 2))
        y = x[:, 0] + rng.normal(scale=0.01, size=len(groups))
        alpha, scores = select_alpha(x, y, groups)
        self.assertIn(alpha, (0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(len(scores), 5)


if __name__ == "__main__":
    unittest.main()

