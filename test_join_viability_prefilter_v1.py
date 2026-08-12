import unittest

from join_viability_prefilter_v1 import (
    is_preeligible,
    modal_candidate,
    rule_specs,
)


class JoinViabilityPrefilterV1Test(unittest.TestCase):
    def test_frozen_rule_count_and_metadata_only_behavior(self):
        rules = rule_specs()
        self.assertEqual(len(rules), 36)
        easy = {
            "n_steps": 8,
            "root_step_count": 2,
            "parent_step_counts": [3, 3],
        }
        hard = {
            "n_steps": 13,
            "root_step_count": 5,
            "parent_step_counts": [5, 6],
        }
        decisions_easy = [predicate(easy) for _, predicate in rules]
        decisions_hard = [predicate(hard) for _, predicate in rules]
        easy_with_outcomes = dict(easy, answer="wrong", model_correct=False)
        hard_with_outcomes = dict(hard, answer="right", model_correct=True)
        self.assertEqual(
            decisions_easy,
            [predicate(easy_with_outcomes) for _, predicate in rules],
        )
        self.assertEqual(
            decisions_hard,
            [predicate(hard_with_outcomes) for _, predicate in rules],
        )

    def test_modal_tie_prefers_greedy(self):
        candidates = [
            {"answer": "5", "norm": "5"},
            {"answer": "6", "norm": "6"},
            {"answer": "6", "norm": "6"},
            {"answer": "5", "norm": "5"},
        ]
        self.assertEqual(modal_candidate(candidates)["answer"], "5")

    def test_phase_a_boundaries_are_frozen(self):
        passing = {
            "n": 100,
            "direct": {"sc1": 0.30, "sc8": 0.70},
            "parent_greedy_accuracy": 0.70,
        }
        self.assertTrue(is_preeligible(passing))
        for key, value in (
            ("n", 99),
            ("parent_greedy_accuracy", 0.699),
        ):
            failing = {
                "n": passing["n"],
                "direct": dict(passing["direct"]),
                "parent_greedy_accuracy": passing["parent_greedy_accuracy"],
            }
            failing[key] = value
            self.assertFalse(is_preeligible(failing))
        for direct_key, value in (("sc1", 0.299), ("sc8", 0.701)):
            failing = {
                "n": passing["n"],
                "direct": dict(passing["direct"]),
                "parent_greedy_accuracy": passing["parent_greedy_accuracy"],
            }
            failing["direct"][direct_key] = value
            self.assertFalse(is_preeligible(failing))


if __name__ == "__main__":
    unittest.main()
