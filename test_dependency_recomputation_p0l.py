import unittest

from dependency_recomputation_p0l import (
    ARMS,
    RELATION_GUIDES,
    build_records,
    closest_wrong_mechanism,
    exact_mcnemar,
    guide_for,
    irrelevant_source_ids,
    parse_met,
    summarize_model,
)
from semantic_staleness_p0k import build_cases, tokens


class DependencyRecomputationP0LTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_cases()
        self.records = build_records()

    def test_balanced_five_arm_matrix(self):
        self.assertEqual(len(self.records), 400)
        self.assertEqual(len({row["record_id"] for row in self.records}), 400)
        for arm in ARMS:
            self.assertEqual(sum(row["arm"] == arm for row in self.records), 80)
        for state in ("dependency_edit", "harmless_edit"):
            self.assertEqual(sum(row["state"] == state for row in self.records), 200)

    def test_every_arm_has_identical_cached_verdict_and_document(self):
        grouped = {}
        for row in self.records:
            grouped.setdefault((row["case_id"], row["state"]), []).append(row)
        for rows in grouped.values():
            self.assertEqual(len(rows), 5)
            for row in rows:
                self.assertIn("marked this conclusion SAT on an older document version", row["prompt"])
                self.assertIn("cached verdict is not evidence", row["prompt"])
            prefixes = [row["prompt"].split("Guide:\n", 1)[0] for row in rows]
            self.assertEqual(len(set(prefixes)), 1)

    def test_correct_guides_do_not_contain_answer_labels(self):
        for case in self.cases:
            for arm in ("source_only", "relation_only", "source_relation"):
                guide = guide_for(case, arm)["text"].lower()
                for forbidden in ("true", "false", "stale", "safe", "correct", "incorrect"):
                    self.assertNotIn(forbidden, guide)

    def test_shuffled_guides_are_wrong_and_length_matched(self):
        for case in self.cases:
            guide = guide_for(case, "shuffled_guide")
            self.assertNotEqual(guide["relation_mechanism"], case.mechanism)
            self.assertTrue(set(guide["source_ids"]).isdisjoint(case.dependency_sentence_ids))
            self.assertNotIn(case.conclusion_sentence, guide["source_ids"])
            gap = abs(
                len(tokens(RELATION_GUIDES[case.mechanism]))
                - len(tokens(RELATION_GUIDES[guide["relation_mechanism"]]))
            )
            self.assertLessEqual(gap, 2)

    def test_source_ids_are_valid_and_matched_count(self):
        for case in self.cases:
            wrong = irrelevant_source_ids(case)
            self.assertEqual(len(wrong), len(case.dependency_sentence_ids))
            self.assertTrue(all(1 <= index <= 10 for index in wrong))

    def test_parser(self):
        self.assertEqual(parse_met('{"met": false}'), (False, True, "json"))
        self.assertEqual(parse_met('```json\n{"met": true}\n```'), (True, True, "json"))
        self.assertEqual(parse_met("{'met': false}"), (False, True, "explicit_boolean_recovery"))
        self.assertFalse(parse_met("SAT")[1])

    def test_exact_mcnemar(self):
        result = exact_mcnemar([True] * 10, [False] * 10)
        self.assertEqual(result["left_wins"], 10)
        self.assertEqual(result["right_wins"], 0)
        self.assertLess(result["two_sided_exact_p"], 0.01)

    def test_full_summary_and_gates_run(self):
        rows = []
        for record in self.records:
            correct = record["state"] == "harmless_edit" or record["arm"] in (
                "relation_only", "source_relation"
            )
            rows.append({**record, "valid": True, "correct": correct})
        summary = summarize_model(rows, 0)
        self.assertEqual(summary["arms"]["source_relation"]["stale_recall"], 1.0)
        self.assertEqual(summary["arms"]["flat"]["stale_recall"], 0.0)
        self.assertTrue(summary["passes_all_model_gates"])


if __name__ == "__main__":
    unittest.main()
