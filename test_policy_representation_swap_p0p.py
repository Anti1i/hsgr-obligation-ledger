import unittest

from policy_representation_swap_p0p import (
    build_report,
    matrix_statistic,
    parse_native_checklist,
    parse_structural_plan,
    parse_yes_no_lines,
    render_structural_plan,
    selection_audit,
)


class PolicyRepresentationSwapTests(unittest.TestCase):
    def test_parse_native_checklist(self):
        value = parse_native_checklist('["Explain the cause clearly", "State the final result"]', 2)
        self.assertEqual(len(value), 2)

    def test_parse_native_rejects_wrong_count(self):
        with self.assertRaises(ValueError):
            parse_native_checklist('["Only one valid requirement"]', 2)

    def test_plan_requires_exact_cover(self):
        self.assertEqual(parse_structural_plan("[[1, 3], [2, 4]]", 4), [[1, 3], [2, 4]])
        with self.assertRaises(ValueError):
            parse_structural_plan("[[1, 3], [2, 3]]", 4)

    def test_render_plan_preserves_canonical_wording(self):
        items = ["Alpha requirement", "Beta requirement", "Gamma requirement"]
        rendered = render_structural_plan(items, [[2], [1, 3]])
        self.assertIn("C2: Beta requirement", rendered)
        self.assertIn("C1: Alpha requirement", rendered)
        self.assertIn("C3: Gamma requirement", rendered)

    def test_parse_yes_no_lines(self):
        parsed, valid = parse_yes_no_lines("Q1: Yes\nQ2: No", 2)
        self.assertTrue(valid)
        self.assertEqual(parsed, {1: True, 2: False})

    def test_balanced_diagonal_statistic(self):
        tasks = ["t1", "t2"]
        policies = ["a", "b", "c"]
        score_map = {}
        for task in tasks:
            for target in policies:
                for source in policies:
                    score_map[(task, target, source)] = 0.8 if target == source else 0.6
        self.assertAlmostEqual(matrix_statistic(score_map, tasks, policies), 0.2)

    def test_source_main_effect_cancels(self):
        tasks = ["t1"]
        policies = ["a", "b", "c"]
        source_effect = {"a": 0.9, "b": 0.6, "c": 0.3}
        score_map = {
            (task, target, source): source_effect[source]
            for task in tasks for target in policies for source in policies
        }
        self.assertAlmostEqual(matrix_statistic(score_map, tasks, policies), 0.0)

    def test_invalid_judge_output_becomes_apparatus_failure(self):
        tasks = [
            {
                "index": "t1",
                "checklist": ["Criterion one", "Criterion two"],
                "p0p_stratum": "other",
            }
        ]
        model_specs = {"a": "model-a", "b": "model-b", "c": "model-c"}
        representations = [
            {"kind": kind, "valid": True}
            for kind in ("native", "structural")
            for _source in model_specs
        ]
        judged = [
            {
                "kind": "reference",
                "task_id": "t1",
                "judge_valid": False,
                "evaluation": {},
            }
        ]
        report = build_report(
            tasks, representations, judged, [], model_specs, "judge", permutations=10
        )
        self.assertEqual(report["decision"], "APPARATUS_FAILURE")
        self.assertFalse(report["apparatus_gates"]["all_evaluator_outputs_parse"])

    def test_selection_audit_exposes_failed_filter(self):
        instance = {
            "index": "law-1",
            "field": "Law",
            "question": "Explain the issue.",
            "checklist": [f"Criterion {index}" for index in range(5)],
            "reference_answer": ["short"],
        }
        audit = selection_audit([instance], query_char_limit=1000)
        law = audit["strata"]["law"]
        self.assertEqual(law["funnel_marginal_counts"]["field_total"], 1)
        self.assertEqual(law["funnel_marginal_counts"]["all_conditions"], 0)
        self.assertIn(
            "reference_chars_400_to_8000",
            law["candidate_metadata"][0]["failed_conditions"],
        )


if __name__ == "__main__":
    unittest.main()
