import json
import unittest

from witness_interference_p0g import (
    ARMS,
    FAILURE_TYPES,
    OBLIGATION_IDS,
    build_cases,
    judge_prompt,
    parse_judgment,
    parse_two_sentence_patch,
    repair_prompt,
    split_sentences,
)


class WitnessInterferenceP0gTests(unittest.TestCase):
    def test_frozen_crossed_design_and_preconditions(self):
        cases = build_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case.id for case in cases}), 24)
        for case in cases:
            self.assertEqual(sum(not value for value in case.expected_before.values()), 1)
            self.assertFalse(case.expected_before[case.target_id])
            self.assertEqual(len(case.witness_sentences), 3)
            sentences = split_sentences(case.baseline_answer)
            for witness_ids in case.witness_sentences.values():
                self.assertTrue(all(1 <= sid <= len(sentences) for sid in witness_ids))
        for failure in FAILURE_TYPES:
            self.assertEqual(sum(case.failure_type == failure for case in cases), 6)

    def test_arms_isolate_obligations_from_witness_locations(self):
        case = build_cases()[0]
        ordinary = repair_prompt(case, "local_patch")
        obligation = repair_prompt(case, "obligation_patch")
        witness = repair_prompt(case, "witness_patch")
        self.assertNotIn("Already-satisfied obligations", ordinary)
        self.assertIn("Already-satisfied obligations", obligation)
        self.assertNotIn("Frozen witness", obligation)
        self.assertIn("Frozen witness [S", witness)
        self.assertIn(case.obligations[next(iter(case.witness_sentences))], obligation)

    def test_patch_rejects_more_than_two_source_sentences(self):
        answer = "One. Two. Three."
        raw = json.dumps({"start_sentence": 1, "end_sentence": 3, "replacement": "Fixed."})
        revised, valid, mode, span = parse_two_sentence_patch(raw, answer)
        self.assertEqual(revised, answer)
        self.assertFalse(valid)
        self.assertEqual(mode, "span_over_two_sentences")
        self.assertIsNone(span)

    def test_judgment_parser(self):
        raw = json.dumps({"items": [
            {"id": oid, "met": True, "witness_sentences": [1]} for oid in OBLIGATION_IDS
        ]})
        parsed, valid, mode = parse_judgment(raw, 2)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(tuple(parsed), OBLIGATION_IDS)

    def test_judgment_rejects_false_item_with_witness(self):
        items = [{"id": oid, "met": True, "witness_sentences": [1]} for oid in OBLIGATION_IDS]
        items[0] = {"id": OBLIGATION_IDS[0], "met": False, "witness_sentences": [1]}
        _, valid, mode = parse_judgment(json.dumps({"items": items}), 2)
        self.assertFalse(valid)
        self.assertEqual(mode, "inconsistent_witness")

    def test_judgment_rejects_true_item_without_witness(self):
        items = [{"id": oid, "met": True, "witness_sentences": [1]} for oid in OBLIGATION_IDS]
        items[0] = {"id": OBLIGATION_IDS[0], "met": True, "witness_sentences": []}
        _, valid, mode = parse_judgment(json.dumps({"items": items}), 2)
        self.assertFalse(valid)
        self.assertEqual(mode, "inconsistent_witness")

    def test_judge_prompt_has_all_requirements(self):
        case = build_cases()[0]
        prompt = judge_prompt(case, case.baseline_answer)
        for oid in OBLIGATION_IDS:
            self.assertIn(oid, prompt)
        self.assertIn("never silently repair", prompt)
        self.assertIn("attaches [B]", prompt)
        for arm in ARMS:
            self.assertTrue(repair_prompt(case, arm))


if __name__ == "__main__":
    unittest.main()
