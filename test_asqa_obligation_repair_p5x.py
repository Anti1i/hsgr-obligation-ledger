import unittest

from asqa_clean_fixed_support_p1x import Case
from asqa_obligation_repair_p5x import (
    RepairCase,
    final_answer,
    ledger_lines,
    render_prompt,
    score_arm,
)


class ObligationRepairP5xTests(unittest.TestCase):
    def setUp(self):
        case = Case(
            id="case-1",
            question="Which Mercury is meant?",
            facet_questions=("What is the planet Mercury?", "Who was the god Mercury?"),
            alias_groups=(("planet",), ("god",)),
            documents=(("Astronomy", "Mercury is a planet."),) * 5,
        )
        self.repair = RepairCase(
            case=case,
            direct_answer="Mercury is a planet near the Sun.",
            original_present=(True, False),
            target_index=1,
            swap_index=0,
            all_true_row={"str_em": 1.0, "str_hit": True, "word_count": 12},
        )

    def test_ledger_has_same_questions_and_swaps_status(self):
        correct = ledger_lines(self.repair, False)
        swapped = ledger_lines(self.repair, True)
        self.assertEqual(len(correct), len(swapped))
        for question in self.repair.case.facet_questions:
            self.assertEqual(sum(question in line for line in correct), 1)
            self.assertEqual(sum(question in line for line in swapped), 1)
        self.assertIn("UNSATISFIED", correct[1])
        self.assertIn("UNSATISFIED", swapped[0])

    def test_append_is_deterministic_and_preserves_original(self):
        answer = final_answer(self.repair, "target_append", "Mercury was also a Roman god.")
        self.assertTrue(answer.startswith(self.repair.direct_answer))
        row = score_arm(self.repair, "target_append", "Mercury was also a Roman god.")
        self.assertTrue(row["str_hit"])
        self.assertTrue(row["target_recovered"])
        self.assertTrue(row["all_original_present_preserved"])

    def test_prompts_have_expected_state_information(self):
        target = render_prompt(self.repair, "target_append")
        generic = render_prompt(self.repair, "generic_append")
        correct = render_prompt(self.repair, "correct_ledger_rewrite")
        swapped = render_prompt(self.repair, "swapped_ledger_rewrite")
        self.assertIn(self.repair.case.facet_questions[1], target)
        self.assertNotIn("Missing interpretation:", generic)
        self.assertIn("UNSATISFIED", correct)
        self.assertIn("UNSATISFIED", swapped)


if __name__ == "__main__":
    unittest.main()
