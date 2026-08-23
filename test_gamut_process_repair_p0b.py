import json
import unittest

from gamut_process_repair_p0b import (
    numbered_sentences,
    parse_sentence_patch,
    split_sentences,
)


class GamutProcessRepairP0bTest(unittest.TestCase):
    def test_sentence_numbering(self):
        answer = "First step. Second step! Third step?"
        self.assertEqual(split_sentences(answer), ["First step.", "Second step!", "Third step?"])
        self.assertIn("[S2] Second step!", numbered_sentences(answer))

    def test_valid_sentence_patch(self):
        answer = "First step. Wrong second step. Third step. Safety remains."
        raw = json.dumps({
            "start_sentence": 2,
            "end_sentence": 3,
            "replacement": "Correct second step. Then perform the third step.",
        })
        revised, valid, mode, span = parse_sentence_patch(raw, answer)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertEqual(span, (2, 3))
        self.assertIn("Safety remains.", revised)

    def test_rejects_large_or_bad_span(self):
        answer = "One. Two. Three. Four. Five."
        over = json.dumps({"start_sentence": 1, "end_sentence": 5, "replacement": "New."})
        self.assertFalse(parse_sentence_patch(over, answer)[1])
        bad = json.dumps({"start_sentence": 0, "end_sentence": 1, "replacement": "New."})
        self.assertFalse(parse_sentence_patch(bad, answer)[1])


if __name__ == "__main__":
    unittest.main()
