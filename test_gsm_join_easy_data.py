import os
import tempfile
import unittest

from data_prep import build_gsm_join, read_jsonl


HERE = os.path.dirname(os.path.abspath(__file__))


class GSMJoinEasyDataTest(unittest.TestCase):
    def test_easy_builder_is_deterministic_and_respects_step_caps(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            source = os.path.join(HERE, "data", "gsm8k_test.jsonl")
            for directory in (first, second):
                target = os.path.join(directory, "gsm8k_test.jsonl")
                with open(source, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())
            a = build_gsm_join(
                first, "test", limit=12, seed=43,
                max_parent_steps=3, max_root_steps=2,
                output_stem="gsm_join_easy",
            )
            b = build_gsm_join(
                second, "test", limit=12, seed=43,
                max_parent_steps=3, max_root_steps=2,
                output_stem="gsm_join_easy",
            )
            self.assertEqual(a, b)
            self.assertEqual(len(a), 12)
            self.assertTrue(all(row["root_step_count"] <= 2 for row in a))
            self.assertTrue(all(max(row["parent_step_counts"]) <= 3 for row in a))
            self.assertEqual(
                a,
                read_jsonl(os.path.join(first, "gsm_join_easy_test.jsonl")),
            )


if __name__ == "__main__":
    unittest.main()
