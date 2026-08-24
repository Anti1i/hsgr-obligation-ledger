import json
import unittest

from persistence_recomputation_p0m import (
    ARMS,
    CACHED_CONTEXT,
    FRESH_CONTEXT,
    assess_structured,
    build_records,
    execute_operator,
    oracle_trace,
    operands_match,
    parse_structured,
)
from semantic_staleness_p0k import build_cases


class PersistenceRecomputationP0MTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_cases()
        self.records = build_records()

    def test_balanced_two_by_two_matrix_with_controls(self):
        self.assertEqual(len(self.records), 320)
        self.assertEqual(len({row["record_id"] for row in self.records}), 320)
        for arm in ARMS:
            self.assertEqual(sum(row["arm"] == arm for row in self.records), 80)
        self.assertEqual(sum(row["state"] == "dependency_edit" for row in self.records), 160)
        self.assertEqual(sum(row["state"] == "harmless_edit" for row in self.records), 160)

    def test_cache_prompts_differ_only_in_context_within_mode(self):
        grouped = {}
        for row in self.records:
            grouped.setdefault((row["case_id"], row["state"], row["mode"]), {})[row["cache"]] = row
        for cells in grouped.values():
            fresh = cells["fresh"]["prompt"]
            cached = cells["cached"]["prompt"]
            self.assertNotIn("older version", fresh)
            self.assertIn("older version: SAT", cached)
            self.assertEqual(cached.replace(CACHED_CONTEXT, "", 1), fresh)

    def test_prompts_do_not_expose_arm_or_expected_label(self):
        for row in self.records:
            self.assertNotIn(row["arm"], row["prompt"])
            self.assertNotIn(row["state"], row["prompt"])

    def test_all_oracle_traces_execute_to_labels(self):
        for case in self.cases:
            stale = oracle_trace(case, "dependency_edit")
            harmless = oracle_trace(case, "harmless_edit")
            self.assertFalse(execute_operator(stale["operator"], stale["operands"]))
            self.assertTrue(execute_operator(harmless["operator"], harmless["operands"]))
            self.assertIn(stale["record_source_id"], stale["source_ids"])

    def test_operator_execution(self):
        self.assertFalse(execute_operator("less_than", [4, 3]))
        self.assertTrue(execute_operator("claim_matches", ["Claim A", "claim a"]))
        self.assertTrue(execute_operator("subtract_equals", [145, 115, 30]))
        self.assertFalse(execute_operator("before", [2022, 2020]))
        self.assertTrue(execute_operator("above_threshold", [80, 60]))
        self.assertIsNone(execute_operator("unknown", [1, 2]))

    def test_structured_parser_and_assessment(self):
        case = self.cases[0]
        oracle = oracle_trace(case, "dependency_edit")
        raw = json.dumps({
            "source_ids": [f"S{value}" for value in oracle["source_ids"]],
            "operator": oracle["operator"],
            "operands": oracle["operands"],
            "computed_met": False,
            "met": True,
        })
        parsed, valid, mode = parse_structured(raw)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        assessment = assess_structured(parsed, valid, oracle)
        self.assertTrue(assessment["trace_complete"])
        self.assertTrue(assessment["checker_correct"])
        self.assertTrue(assessment["computed_matches_checker"])
        self.assertTrue(assessment["final_override"])
        self.assertEqual(assessment["failure_stage"], "final_override")

    def test_complete_trace_requires_every_dependency_source(self):
        case = next(case for case in self.cases if len(case.dependency_sentence_ids) == 2)
        oracle = oracle_trace(case, "dependency_edit")
        raw = json.dumps({
            "source_ids": [oracle["record_source_id"]],
            "operator": oracle["operator"],
            "operands": oracle["operands"],
            "computed_met": False,
            "met": True,
        })
        parsed, valid, _ = parse_structured(raw)
        assessment = assess_structured(parsed, valid, oracle)
        self.assertTrue(assessment["source_record_found"])
        self.assertFalse(assessment["source_exact"])
        self.assertFalse(assessment["trace_complete"])
        self.assertFalse(assessment["final_override"])
        self.assertEqual(assessment["failure_stage"], "dependency_coverage")

    def test_operand_matching_is_typed_and_ordered(self):
        self.assertTrue(operands_match(["20", 10], [20, 10]))
        self.assertFalse(operands_match([10, 20], [20, 10]))
        self.assertTrue(operands_match(["Claim A", "claim a"], ["claim a", "Claim A"]))


if __name__ == "__main__":
    unittest.main()
