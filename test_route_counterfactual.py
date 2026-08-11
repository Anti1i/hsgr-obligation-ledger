import unittest

from hsgr_route_counterfactual_eval import holm_adjust, make_model
from hsgr_structured_hidden_verifier import LAYERS
from hsgr_route_counterfactual_features import (
    FALLBACK_FOIL,
    FALLBACK_PARENT,
    donor_map,
    parse_plan,
    state_block,
)


class RouteCounterfactualTest(unittest.TestCase):
    def test_plan_parser_rejects_original_question_and_deduplicates(self):
        original = "Who founded the company?"
        raw = (
            'prefix {"root_goal":"Find the founder",'
            '"parent_questions":["Who founded the company?",'
            '"Which company is discussed?","Which company is discussed?"],'
            '"foil_question":"Where is the company based?"} suffix'
        )
        plan = parse_plan(raw, original)
        self.assertEqual(plan["parent_questions"], ["Which company is discussed?"])
        self.assertEqual(plan["foil_question"], "Where is the company based?")

    def test_plan_parser_has_no_gold_dependent_fallback(self):
        plan = parse_plan("not json", "An arbitrary original question")
        self.assertEqual(plan["parent_questions"], [FALLBACK_PARENT])
        self.assertEqual(plan["foil_question"], FALLBACK_FOIL)

    def test_role_intervention_preserves_all_node_text(self):
        state = {
            "parents": [
                {"question": "parent one", "answer": "value one"},
                {"question": "parent two", "answer": "value two"},
            ],
            "foil": {"question": "foil", "answer": "foil value"},
        }
        matched = state_block(state, "matched")
        counter = state_block(state, "counterfactual")
        for text in ("parent one", "value one", "parent two", "value two", "foil", "foil value"):
            self.assertEqual(matched.count(text), counter.count(text))
        self.assertEqual(matched.count("ROLE=PARENT"), counter.count("ROLE=PARENT"))
        self.assertEqual(matched.count("ROLE=NONPARENT"), counter.count("ROLE=NONPARENT"))
        self.assertNotEqual(matched, counter)

    def test_mismatch_donors_are_depth_matched_and_deranged(self):
        states = {
            "a": {"predicted_depth": 2, "parents": [{"question": "a", "answer": "1"}], "foil": {"question": "fa", "answer": "2"}},
            "b": {"predicted_depth": 2, "parents": [{"question": "b", "answer": "1"}], "foil": {"question": "fb", "answer": "2"}},
            "c": {"predicted_depth": 3, "parents": [{"question": "c1", "answer": "1"}, {"question": "c2", "answer": "2"}], "foil": {"question": "fc", "answer": "3"}},
            "d": {"predicted_depth": 3, "parents": [{"question": "d1", "answer": "1"}, {"question": "d2", "answer": "2"}], "foil": {"question": "fd", "answer": "3"}},
        }
        donors = donor_map(states)
        for pid, donor in donors.items():
            self.assertNotEqual(pid, donor)
            self.assertEqual(states[pid]["predicted_depth"], states[donor]["predicted_depth"])

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)

    def test_all_reader_views_have_valid_shape_when_torch_is_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is available only in the cluster project environment")
        n = 4
        data = {
            "scalar": torch.randn(n, 7),
            "scalar_permuted": torch.randn(n, 7),
            "features": {
                view: {layer: torch.randn(n, 256) for layer in LAYERS}
                for view in (
                    "matched",
                    "counterfactual",
                    "mismatch",
                    "matched_start",
                )
            },
        }
        for kind in ("guide", "ordinary", "activation_delta", "nonhidden"):
            model = make_model(torch, kind)
            self.assertEqual(tuple(model(data).shape), (n,))
            if kind == "guide":
                for mode in ("swap", "mismatch", "state_permute"):
                    self.assertEqual(tuple(model(data, mode=mode).shape), (n,))


if __name__ == "__main__":
    unittest.main()
