import unittest

from gamut_relation_judge_p0c import (
    combine_counterbalanced,
    direct_prompt,
    parse_matched_order,
    relation_met,
)
from gamut_process_repair_p0 import ProcessCase, Requirement


class RelationJudgeP0cTests(unittest.TestCase):
    def test_counterbalanced_margin_removes_shared_label_bias(self):
        self.assertGreater(combine_counterbalanced(-3.0, 1.0), 0)
        self.assertLess(combine_counterbalanced(2.0, -4.0), 0)

    def test_parse_and_relation_check(self):
        order, valid, mode = parse_matched_order('{"matched_order":["P1","P3"]}', 3)
        self.assertEqual((order, valid, mode), (["P1", "P3"], True, "valid"))
        self.assertTrue(relation_met(order))
        self.assertFalse(relation_met(["P2", "P1"]))
        self.assertTrue(relation_met(["P2"]))

    def test_duplicate_is_invalid(self):
        self.assertFalse(parse_matched_order('{"matched_order":["P1","P1"]}', 2)[1])

    def test_answer_only_prompt_has_no_gold_evidence(self):
        case = ProcessCase(
            id="x",
            question="q",
            evidence="SECRET GOLD",
            answer_critical=(Requirement("r", "P1 before P2"),),
            target=Requirement("r", "P1 before P2"),
            steps=("one", "two"),
        )
        prompt = direct_prompt(case, "answer", swapped=False, include_evidence=False)
        self.assertNotIn("SECRET GOLD", prompt)
        self.assertIn("A means MET", prompt)


if __name__ == "__main__":
    unittest.main()

