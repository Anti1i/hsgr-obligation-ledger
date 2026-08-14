import unittest

from longform_structure_audit import extract_clapnq, jaccard, percentile


class LongformStructureAuditTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard("a b c", "b c d"), 0.5)

    def test_extracts_noncollapsed_record(self):
        record = {
            "id": "x",
            "passages": [
                {
                    "text": "Alpha evidence. Middle. Omega evidence.",
                    "sentences": ["Alpha evidence.", "Middle.", "Omega evidence."],
                }
            ],
            "output": [
                {
                    "answer": "word " * 30,
                    "selected_sentences": ["Alpha evidence.", "Omega evidence."],
                    "meta": {"non_consecutive": True},
                }
            ],
        }
        row, error = extract_clapnq(record)
        self.assertIsNone(error)
        assert row is not None
        self.assertEqual(row["support_count"], 2)
        self.assertEqual(row["subset_state_capacity"], 4)
        self.assertTrue(row["structural_headroom"])
        self.assertTrue(row["pilot_eligible"])

    def test_missing_selected_sentence_is_integrity_failure(self):
        record = {
            "passages": [{"text": "Alpha.", "sentences": ["Alpha."]}],
            "output": [
                {
                    "answer": "A valid answer.",
                    "selected_sentences": ["Not in passage."],
                    "meta": {},
                }
            ],
        }
        row, error = extract_clapnq(record)
        self.assertIsNone(error)
        assert row is not None
        self.assertEqual(row["missing_support_count"], 1)
        self.assertFalse(row["pilot_eligible"])


if __name__ == "__main__":
    unittest.main()

