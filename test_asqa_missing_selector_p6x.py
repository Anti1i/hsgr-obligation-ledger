import unittest

from asqa_clean_fixed_support_p1x import Case
from asqa_missing_selector_p6x import (
    SelectorCase,
    fold_id,
    lexical_missing_score,
    random_missing_score,
    render_append_prompt,
    render_selector_prompt,
    score_append,
)


class MissingSelectorP6xTests(unittest.TestCase):
    def setUp(self):
        self.case = Case(
            id="case-p6",
            question="Which Mercury is meant?",
            facet_questions=("What is the planet Mercury?", "Who was the god Mercury?"),
            alias_groups=(("planet",), ("god",)),
            documents=(("Mercury", "Mercury is a planet and also the name of a Roman god."),) * 5,
        )
        self.item = SelectorCase(
            self.case, "Mercury is a planet near the Sun.", (True, False)
        )

    def test_selector_prompt_has_candidate_but_no_documents(self):
        prompt = render_selector_prompt(self.item, 1)
        self.assertIn("Who was the god Mercury?", prompt)
        self.assertIn("A means COVERED; B means MISSING", prompt)
        self.assertNotIn("Fixed documents:", prompt)

    def test_lexical_score_favors_unmentioned_content(self):
        present = lexical_missing_score("What is the planet Mercury?", self.item.direct_answer)
        missing = lexical_missing_score("Who was the Roman god Mercury?", self.item.direct_answer)
        self.assertGreater(missing, present)

    def test_hash_controls_are_deterministic(self):
        self.assertEqual(fold_id(self.case.id), fold_id(self.case.id))
        self.assertEqual(random_missing_score(self.case.id, 1), random_missing_score(self.case.id, 1))
        self.assertNotEqual(random_missing_score(self.case.id, 0), random_missing_score(self.case.id, 1))

    def test_selected_append_preserves_saved_answer(self):
        prompt = render_append_prompt(self.item, 1)
        self.assertIn("Missing interpretation", prompt)
        row = score_append(self.item, "oracle_append", 1, "Mercury was also a Roman god.")
        self.assertTrue(row["str_hit"])
        self.assertTrue(row["selection_correct"])
        self.assertTrue(row["all_original_present_preserved"])


if __name__ == "__main__":
    unittest.main()
