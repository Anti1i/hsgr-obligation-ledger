import unittest

from asqa_auto_obligation_p7r import parse_obligations_repaired


class AutoObligationP7RTests(unittest.TestCase):
    def test_repairs_four_singleton_arrays(self):
        raw = '["A question?"]\n["B question?"]\n["C question?"]\n["D question?"]'
        parsed, mode = parse_obligations_repaired(raw)
        self.assertEqual(mode, "multi_array")
        self.assertEqual(parsed, ["A question?", "B question?", "C question?", "D question?"])

    def test_keeps_original_json_parser(self):
        raw = '["A question?", "B question?", "C question?", "D question?"]'
        parsed, mode = parse_obligations_repaired(raw)
        self.assertEqual(mode, "json")
        self.assertEqual(len(parsed), 4)

    def test_rejects_non_four_lines(self):
        parsed, mode = parse_obligations_repaired('["A?"]\n["B?"]')
        self.assertIsNone(parsed)
        self.assertEqual(mode, "invalid")


if __name__ == "__main__":
    unittest.main()
