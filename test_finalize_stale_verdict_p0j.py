import unittest

from stale_verdict_p0j import FEATURE_NAMES


class ManualAuditInvariantTests(unittest.TestCase):
    def test_feature_schema_has_no_target_indicator(self):
        self.assertNotIn("is_repair_target", FEATURE_NAMES)
        self.assertNotIn("old_verdict", FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
