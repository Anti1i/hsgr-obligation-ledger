import json
import unittest

from support_entanglement_p0h import (
    ARMS,
    BLOCKS,
    LAYOUTS,
    OBLIGATION_IDS,
    build_cases,
    exact_one_sided_sign_p,
    judge_prompt,
    operator_interaction_summary,
    paired_summary,
    parse_forced_sentence_replacement,
    parse_judgment,
    parse_one_sentence_patch,
    repair_prompt,
    split_sentences,
    target_sentence_id,
)


class SupportEntanglementP0hTests(unittest.TestCase):
    def test_matched_pair_construction(self):
        cases = build_cases()
        self.assertEqual(len(BLOCKS), 18)
        self.assertEqual(len(cases), 36)
        by_block = {}
        for case in cases:
            by_block.setdefault(case.block_id, {})[case.layout] = case
            self.assertEqual(case.expected_before, {
                "O_LEFT": True, "O_TARGET": False, "O_RIGHT": True
            })
        for pair in by_block.values():
            self.assertEqual(set(pair), set(LAYOUTS))
            self.assertEqual(pair["entangled"].semantic_atoms, pair["disentangled"].semantic_atoms)
            self.assertEqual(pair["entangled"].evidence, pair["disentangled"].evidence)
            self.assertEqual(len(split_sentences(pair["entangled"].baseline_answer)), 1)
            self.assertEqual(len(split_sentences(pair["disentangled"].baseline_answer)), 3)
            self.assertEqual(pair["entangled"].witness_sentences, {
                "O_LEFT": (1,), "O_RIGHT": (1,)
            })
            self.assertEqual(pair["disentangled"].witness_sentences, {
                "O_LEFT": (1,), "O_RIGHT": (3,)
            })

    def test_target_type_blocks_are_balanced(self):
        counts = {kind: sum(block.target_type == kind for block in BLOCKS) for kind in {
            "numeric", "attribution", "ordering"
        }}
        self.assertEqual(counts, {"numeric": 6, "attribution": 6, "ordering": 6})

    def test_patch_prompt_requires_exactly_one_sentence(self):
        case = build_cases()[0]
        prompt = repair_prompt(case, "sentence_patch")
        self.assertIn("Replace exactly one source sentence", prompt)
        self.assertIn(case.obligations["O_TARGET"], prompt)
        self.assertIn(case.obligations["O_LEFT"], prompt)
        self.assertIn(case.obligations["O_RIGHT"], prompt)
        for arm in ARMS:
            self.assertTrue(repair_prompt(case, arm))

    def test_r1_forces_frozen_target_unit(self):
        cases = build_cases()
        entangled = next(case for case in cases if case.layout == "entangled")
        disentangled = next(case for case in cases if case.layout == "disentangled")
        self.assertEqual(target_sentence_id(entangled), 1)
        self.assertEqual(target_sentence_id(disentangled), 2)
        for case, marker in ((entangled, "[S1]"), (disentangled, "[S2]")):
            prompt = repair_prompt(case, "sentence_patch", forced_target_unit=True)
            self.assertIn(marker, prompt)
            self.assertIn("single field replacement", prompt)
            self.assertNotIn("start_sentence", prompt)

    def test_r1_replacement_parser_applies_frozen_unit(self):
        answer = "One. Wrong two. Three."
        raw = json.dumps({"replacement": "Fixed two."})
        revised, valid, mode, span = parse_forced_sentence_replacement(raw, answer, 2)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(span, (2, 2))
        self.assertEqual(revised, "One. Fixed two. Three.")

    def test_patch_parser_accepts_one_and_rejects_two(self):
        answer = "One. Two. Three."
        good = json.dumps({
            "start_sentence": 2, "end_sentence": 2, "replacement": "Fixed two."
        })
        revised, valid, mode, span = parse_one_sentence_patch(good, answer)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(span, (2, 2))
        self.assertEqual(revised, "One. Fixed two. Three.")
        bad = json.dumps({
            "start_sentence": 1, "end_sentence": 2, "replacement": "Merged."
        })
        revised, valid, mode, span = parse_one_sentence_patch(bad, answer)
        self.assertFalse(valid)
        self.assertEqual(mode, "not_one_sentence")
        self.assertIsNone(span)

    def test_judgment_parser_and_prompt(self):
        raw = json.dumps({"items": [
            {"id": oid, "met": True, "witness_sentences": [1]} for oid in OBLIGATION_IDS
        ]})
        parsed, valid, mode = parse_judgment(raw, 3)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(tuple(parsed), OBLIGATION_IDS)
        prompt = judge_prompt(build_cases()[0], build_cases()[0].baseline_answer)
        self.assertIn("Do not infer or silently correct", prompt)
        strict = judge_prompt(build_cases()[0], build_cases()[0].baseline_answer, strict=True)
        self.assertIn("JSON boolean true or false", strict)
        self.assertIn("reversed event order is false", strict)

    def test_exact_sign_probability(self):
        self.assertEqual(exact_one_sided_sign_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_one_sided_sign_p(5, 0), 0.03125)
        self.assertAlmostEqual(exact_one_sided_sign_p(5, 1), 7 / 64)

    def test_paired_summary_uses_joint_target_success(self):
        rows = []
        for block, outcomes in {
            "a": (True, False), "b": (True, False), "c": (False, False)
        }.items():
            for layout, regression in zip(LAYOUTS, outcomes):
                rows.append({
                    "block_id": block, "arm": "sentence_patch", "layout": layout,
                    "judge_valid": True, "target_recovered": block != "c",
                    "any_regression": regression,
                })
        summary = paired_summary(rows, "sentence_patch")
        self.assertEqual(summary["joint_target_success_pairs"], 2)
        self.assertEqual(summary["entangled_only_discordant"], 2)
        self.assertEqual(summary["disentangled_only_discordant"], 0)
        self.assertEqual(summary["paired_regression_risk_difference"], 1.0)

    def test_operator_interaction_uses_same_four_cell_blocks(self):
        rows = []
        for block in ("kept", "excluded"):
            for arm in ARMS:
                for layout in LAYOUTS:
                    rows.append({
                        "block_id": block, "arm": arm, "layout": layout,
                        "judge_valid": True,
                        "target_recovered": not (
                            block == "excluded" and arm == "full_rewrite" and layout == "entangled"
                        ),
                        "any_regression": (
                            block == "kept" and arm == "sentence_patch" and layout == "entangled"
                        ),
                    })
        summary = operator_interaction_summary(rows)
        self.assertEqual(summary["common_four_cell_target_success_blocks"], 1)
        self.assertEqual(summary["sentence_patch_layout_effect"], 1.0)
        self.assertEqual(summary["full_rewrite_layout_effect"], 0.0)
        self.assertEqual(summary["difference_in_layout_effects"], 1.0)


if __name__ == "__main__":
    unittest.main()
