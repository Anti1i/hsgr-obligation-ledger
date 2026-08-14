import unittest

import numpy as np

from asqa_candidate_node_hidden_p2x import (
    WeightedRidge,
    binary_ranking_metrics,
    weighted_auc,
)


class ASQACandidateNodeHiddenP2XTest(unittest.TestCase):
    def test_weighted_auc_perfect_and_reversed(self):
        labels = np.asarray([0, 0, 1, 1], dtype=bool)
        weights = np.ones(4)
        self.assertEqual(weighted_auc(labels, np.asarray([0, 1, 2, 3]), weights), 1.0)
        self.assertEqual(weighted_auc(labels, np.asarray([3, 2, 1, 0]), weights), 0.0)

    def test_state_ranking_metrics(self):
        rows = [
            {"state_id": "a", "node_index": 0, "label": False},
            {"state_id": "a", "node_index": 1, "label": True},
            {"state_id": "b", "node_index": 0, "label": True},
            {"state_id": "b", "node_index": 1, "label": True},
        ]
        metrics = binary_ranking_metrics(
            rows,
            np.arange(4),
            np.asarray([0.1, 0.9, 0.2, 0.1]),
        )
        self.assertEqual(metrics["mixed_states"], 1)
        self.assertEqual(metrics["mixed_state_recall_at_1"], 1.0)
        self.assertEqual(metrics["mixed_state_mrr"], 1.0)

    def test_weighted_ridge_fits_order(self):
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.asarray([0.0, 0.0, 1.0, 1.0])
        model = WeightedRidge(0.1).fit(x, y, np.ones(4))
        prediction = model.predict(x)
        self.assertTrue(np.all(np.diff(prediction) > 0))


if __name__ == "__main__":
    unittest.main()
