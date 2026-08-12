import unittest

from hidden_graph_energy_guide import (
    assignment_specs,
    modal_norm,
    response_spans,
    stable_bucket,
    structural_assignment_gate,
    teacher_force_ids,
    value_classes,
)


class HiddenGraphEnergyGuideTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"kind": "greedy", "answer": "2", "norm": "2", "text": "a"},
            {"kind": "sample", "answer": "3", "norm": "3", "text": "b"},
            {"kind": "sample", "answer": "3.0", "norm": "3", "text": "c"},
            {"kind": "sample", "answer": "2.0", "norm": "2", "text": "d"},
        ]

    def test_value_classes_preserve_mass_and_greedy_tie_break(self):
        classes = value_classes(self.candidates)
        self.assertEqual([item["norm"] for item in classes], ["2", "3"])
        self.assertEqual([item["count"] for item in classes], [2, 2])
        self.assertEqual(modal_norm(self.candidates), "2")

    def test_invalid_candidates_form_unknown_class(self):
        candidates = [
            {"kind": "greedy", "answer": None, "norm": None, "text": "bad"},
            {"kind": "sample", "answer": "", "norm": None, "text": "empty"},
        ]
        classes = value_classes(candidates)
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["norm"], "__UNKNOWN__")
        self.assertEqual(classes[0]["answer"], "UNKNOWN")
        self.assertEqual(modal_norm(candidates), "__UNKNOWN__")

    def test_cartesian_domain_has_exactly_one_modal_tuple(self):
        assignments = assignment_specs(self.candidates, self.candidates)
        self.assertEqual(len(assignments), 4)
        self.assertEqual(sum(item["is_modal"] for item in assignments), 1)
        self.assertEqual(assignments[0]["norms"], ["2", "2"])

    def test_response_spans_account_for_left_padding(self):
        self.assertEqual(response_spans([3, 2], [5, 4], 5), [(3, 5), (3, 5)])
        with self.assertRaises(ValueError):
            response_spans([3], [3], 5)

    def test_teacher_force_truncation_keeps_exact_response(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.assert_call = (messages, tokenize, add_generation_prompt)
                return [1, 2, 3, 4, 5]

            def __call__(self, text, add_special_tokens):
                return {"input_ids": [8, 9] if text == "answer" else [7]}

        ids, span, raw_prompt, raw_response = teacher_force_ids(
            FakeTokenizer(), "system", "user", "answer", 5
        )
        self.assertEqual(ids, [3, 4, 5, 8, 9])
        self.assertEqual(span, (3, 5))
        self.assertEqual((raw_prompt, raw_response), (5, 2))

    def test_hash_split_is_stable(self):
        values = [stable_bucket(pid, 5, "outer") for pid in range(20)]
        self.assertEqual(values, [stable_bucket(pid, 5, "outer") for pid in range(20)])

    def test_assignment_gate_is_strict(self):
        summary = {
            "n_graphs": 100,
            "n_actionable": 20,
            "n_mixed_label_graphs": 40,
            "oracle_gap": 0.10,
        }
        self.assertTrue(all(structural_assignment_gate(summary).values()))
        summary["n_actionable"] = 19
        self.assertFalse(all(structural_assignment_gate(summary).values()))


if __name__ == "__main__":
    unittest.main()
