import unittest

from gamut_long_baseline_p0d import paired_summary, summarize


class LongBaselineP0dTests(unittest.TestCase):
    def test_summarize(self):
        rows = [
            {"id": "a", "valid": True, "output_tokens": 256, "all_components_present": True, "relation_met": False},
            {"id": "b", "valid": True, "output_tokens": 100, "all_components_present": False, "relation_met": True},
        ]
        result = summarize(rows, 256)
        self.assertEqual(result["relation_only_count"], 1)
        self.assertEqual(result["cap_hit_proxy_rate"], 0.5)

    def test_paired_summary(self):
        old = [
            {"id": "a", "valid": True, "all_components_present": False, "relation_met": False},
            {"id": "b", "valid": True, "all_components_present": True, "relation_met": True},
        ]
        new = [
            {"id": "a", "valid": True, "all_components_present": True, "relation_met": True},
            {"id": "b", "valid": True, "all_components_present": True, "relation_met": False},
        ]
        result = paired_summary(old, new)
        self.assertEqual(result["all_components_gained"], 1)
        self.assertEqual(result["old_relation_failure_fixed"], 1)
        self.assertEqual(result["new_relation_failure_created"], 1)


if __name__ == "__main__":
    unittest.main()

