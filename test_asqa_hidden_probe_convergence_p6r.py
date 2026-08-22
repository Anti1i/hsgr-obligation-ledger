import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asqa_hidden_probe_convergence_p6r import load_p6x_scores, outcome_for


class HiddenProbeConvergenceP6RTests(unittest.TestCase):
    def test_outcome_precedence(self):
        self.assertEqual(outcome_for(False, True, True), "APPARATUS_FAIL")
        self.assertEqual(outcome_for(True, False, True), "SOLVER_STILL_FAIL")
        self.assertEqual(outcome_for(True, True, True), "CONVERGED_HIDDEN_RECOVERY")
        self.assertEqual(outcome_for(True, True, False), "CONVERGED_HIDDEN_NO_ADVANTAGE")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is provided by the cluster venv")
    def test_frozen_score_alignment(self):
        candidates = [SimpleNamespace(case_id="a", facet_index=0, missing=True)]
        row = {
            "id": "a", "facet_index": 1, "missing_label": True,
            "hidden_probe_score": 0.1, "logit_score": 0.2, "random_score": 0.3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            scores = load_p6x_scores(path, candidates)
        self.assertAlmostEqual(float(scores["logit"][0]), 0.2)


if __name__ == "__main__":
    unittest.main()
