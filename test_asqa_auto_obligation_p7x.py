import unittest

from asqa_auto_obligation_p7x import parse_obligations, random_index


class AutoObligationP7XTests(unittest.TestCase):
    def test_json_parser(self):
        parsed, mode = parse_obligations('["What is A?", "What is B?", "What is C?", "What is D?"]')
        self.assertEqual(mode, "json")
        self.assertEqual(len(parsed), 4)

    def test_numbered_fallback(self):
        parsed, mode = parse_obligations("1. What is A?\n2. What is B?\n3. What is C?\n4. What is D?")
        self.assertEqual(mode, "numbered")
        self.assertEqual(len(parsed), 4)

    def test_duplicate_is_invalid(self):
        parsed, mode = parse_obligations('["A?", "A?", "B?", "C?"]')
        self.assertIsNone(parsed)
        self.assertEqual(mode, "invalid")

    def test_random_index_is_bounded_and_deterministic(self):
        self.assertEqual(random_index("case", 4), random_index("case", 4))
        self.assertIn(random_index("case", 4), range(4))


if __name__ == "__main__":
    unittest.main()
