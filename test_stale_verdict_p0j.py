import json
import importlib.util
import unittest

from stale_verdict_p0j import (
    FEATURE_NAMES,
    OBLIGATION_IDS,
    OPERATORS,
    TARGET_TYPES,
    analyze_transitions,
    apply_candidate,
    build_cases,
    build_scenarios,
    frozen_policy_selections,
    nested_grouped_predictions,
    parse_judgment,
    policy_metrics,
    section_span,
    sentence_diff,
    transition_features,
)


class StaleVerdictP0JTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_cases()
        self.case = self.cases[0]

    def test_frozen_matrix_and_single_failure(self):
        self.assertEqual(len(build_scenarios()), 8)
        self.assertEqual(len(self.cases), 24)
        self.assertEqual({case.target_type for case in self.cases}, set(TARGET_TYPES))
        self.assertEqual(len({case.id for case in self.cases}), 24)
        self.assertGreater(len({case.target_sentence for case in self.cases}), 3)
        for case in self.cases:
            self.assertEqual(len(case.obligations), 12)
            self.assertEqual(sum(not value for value in case.expected_before.values()), 1)
            self.assertFalse(case.expected_before[case.target_id])
            self.assertNotIn(case.target_id, case.witness_sentences)
            self.assertEqual(len(case.witness_sentences), 11)

    def test_section_is_fixed_four_sentence_block(self):
        for target in range(1, 13):
            first, last = section_span(target)
            self.assertEqual(last - first + 1, 4)
            self.assertLessEqual(first, target)
            self.assertGreaterEqual(last, target)

    def test_forced_target_and_section_parsers(self):
        raw = json.dumps({"replacement": "A corrected complete finding [E4]."})
        revised, valid, _, span = apply_candidate(self.case, "target_sentence", raw)
        self.assertTrue(valid)
        self.assertEqual(span, (self.case.target_sentence, self.case.target_sentence))
        self.assertIn("A corrected complete finding", revised)
        revised, valid, _, span = apply_candidate(self.case, "section_patch", raw)
        self.assertTrue(valid)
        self.assertEqual(span, section_span(self.case.target_sentence))

    def test_target_never_becomes_prediction_feature_row(self):
        oid = next(oid for oid in OBLIGATION_IDS if oid != self.case.target_id)
        features = transition_features(
            self.case,
            oid,
            self.case.clean_answer,
            "full_rewrite",
            "Qwen/Qwen3-8B",
            True,
        )
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["generator_qwen3"], 1.0)
        with self.assertRaises(ValueError):
            transition_features(
                self.case,
                self.case.target_id,
                self.case.clean_answer,
                "full_rewrite",
                "Qwen/Qwen3-8B",
                True,
            )

    def test_actual_diff_does_not_assume_declared_full_rewrite_scope(self):
        old = "Alpha remains true. Beta is wrong. Gamma remains true."
        new = "Alpha remains true. Beta is corrected. Gamma remains true."
        diff = sentence_diff(old, new)
        self.assertEqual(diff["changed_old_sentence_ids"], [2])
        self.assertEqual(diff["changed_new_sentence_ids"], [2])

    def test_judge_parser_requires_all_twelve_in_order(self):
        items = [
            {"id": oid, "met": True, "witness_sentences": [index]}
            for index, oid in enumerate(OBLIGATION_IDS, 1)
        ]
        parsed, valid, mode = parse_judgment(json.dumps({"items": items}), 12)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(tuple(parsed), OBLIGATION_IDS)
        parsed, valid, _ = parse_judgment(json.dumps({"items": items[:-1]}), 12)
        self.assertFalse(valid)
        self.assertEqual(parsed, {})

    def test_verification_saving_always_charges_target(self):
        rows = [
            {"stale": True, "features": {"witness_touched": 1.0}},
            {"stale": False, "features": {"witness_touched": 0.0}},
        ]
        metric = policy_metrics(rows, [True, False], revision_count=1)
        self.assertEqual(metric["total_rechecks_including_target"], 2)
        self.assertAlmostEqual(metric["verification_saving"], 10 / 12)
        self.assertEqual(frozen_policy_selections(rows, "witness_overlap"), [True, False])

    def test_no_classifier_claim_without_positive_apparatus(self):
        rows = []
        for scenario in ("a", "b", "c", "d"):
            for index in range(11):
                features = {name: 0.0 for name in FEATURE_NAMES}
                rows.append({
                    "scenario_id": scenario,
                    "revision_id": scenario,
                    "stale": False,
                    "features": features,
                })
        report = analyze_transitions(rows, revision_count=4)
        self.assertFalse(report["learned"]["available"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is required")
    def test_nested_prediction_holds_out_whole_scenarios(self):
        rows = []
        for scenario_index in range(8):
            scenario = f"s{scenario_index}"
            for index in range(6):
                stale = index < 2
                features = {name: 0.0 for name in FEATURE_NAMES}
                features["witness_touched"] = float(stale)
                rows.append({
                    "scenario_id": scenario,
                    "revision_id": f"{scenario}-{index}",
                    "stale": stale,
                    "features": features,
                })
        selected, scores, folds = nested_grouped_predictions(rows)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(len(scores), len(rows))
        self.assertEqual({fold["held_out_scenario"] for fold in folds}, {f"s{i}" for i in range(8)})
        self.assertGreaterEqual(
            sum(row["stale"] and keep for row, keep in zip(rows, selected))
            / sum(row["stale"] for row in rows),
            0.90,
        )


if __name__ == "__main__":
    unittest.main()
