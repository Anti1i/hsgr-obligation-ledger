import importlib.util
import json
import unittest

from gamut_process_repair_p0b import split_sentences
from semantic_staleness_p0k import (
    FEATURE_NAMES,
    MECHANISMS,
    build_cases,
    changed_sentence,
    heuristic_selection,
    judge_prompt,
    nested_surface_predictions,
    parse_judgment,
    surface_features,
)


class SemanticStalenessP0KTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_cases()

    def test_crossed_executable_construction(self):
        self.assertEqual(len(self.cases), 40)
        self.assertEqual(len({case.id for case in self.cases}), 40)
        self.assertEqual(
            {mechanism: sum(case.mechanism == mechanism for case in self.cases) for mechanism in MECHANISMS},
            {mechanism: 8 for mechanism in MECHANISMS},
        )
        self.assertEqual(sum(case.hard_alias for case in self.cases), 20)
        for case in self.cases:
            self.assertTrue(case.oracle_old_met)
            self.assertFalse(case.oracle_dependency_met)
            self.assertTrue(case.oracle_harmless_met)

    def test_witness_is_byte_unchanged_and_edit_is_disjoint(self):
        for case in self.cases:
            old = split_sentences(case.old_document)
            for revised in (case.dependency_document, case.harmless_document):
                new = split_sentences(revised)
                self.assertEqual(old[case.conclusion_sentence - 1], new[case.conclusion_sentence - 1])
                source_id, _, _ = changed_sentence(case.old_document, revised)
                self.assertNotEqual(source_id, case.conclusion_sentence)

    def test_paired_edits_share_counterbalanced_source_position(self):
        dependency_positions = []
        harmless_positions = []
        for case in self.cases:
            dependency_positions.append(changed_sentence(case.old_document, case.dependency_document)[0])
            harmless_positions.append(changed_sentence(case.old_document, case.harmless_document)[0])
        self.assertEqual(dependency_positions, harmless_positions)
        self.assertEqual(set(dependency_positions), {3, 6})
        self.assertEqual(set(harmless_positions), {3, 6})
        self.assertEqual(sum(position < 5 for position in harmless_positions), 20)

    def test_feature_schema_has_no_label_or_mechanism(self):
        case = self.cases[0]
        features = surface_features(case, case.dependency_document)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        for forbidden in ("arm", "label", "stale", "mechanism", "domain"):
            self.assertNotIn(forbidden, features)

    def test_direct_witness_overlap_always_selects_none(self):
        rows = []
        for case in self.cases:
            rows.append({"features": surface_features(case, case.dependency_document)})
            rows.append({"features": surface_features(case, case.harmless_document)})
        self.assertFalse(any(heuristic_selection(rows, "witness_overlap")))

    def test_judge_parser_and_prompt(self):
        case = self.cases[0]
        raw = json.dumps({
            "met": True,
            "conclusion_sentence": case.conclusion_sentence,
            "dependency_sentences": list(case.dependency_sentence_ids),
        })
        parsed, valid, mode = parse_judgment(raw, 10)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertTrue(parsed["met"])
        prompt = judge_prompt(case, case.dependency_document)
        self.assertIn("unchanged conclusion can be false", prompt)
        self.assertNotIn("dependency_edit", prompt)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is required")
    def test_domain_grouped_surface_prediction_runs(self):
        rows = []
        for case in self.cases:
            for arm, document, stale in (
                ("dependency", case.dependency_document, True),
                ("harmless", case.harmless_document, False),
            ):
                rows.append({
                    "case_id": case.id,
                    "domain_id": case.domain_id,
                    "stale": stale,
                    "features": surface_features(case, document),
                })
        selected, scores, folds = nested_surface_predictions(rows)
        self.assertEqual(len(selected), 80)
        self.assertEqual(len(scores), 80)
        self.assertEqual(len(folds), 8)
        self.assertEqual(len({fold["held_out_domain"] for fold in folds}), 8)


if __name__ == "__main__":
    unittest.main()
