import unittest

from asqa_clean_fixed_support_p1x import (
    Case,
    build_decoy_mapping,
    case_from_record,
    exact_mcnemar_p,
    render_user,
)


def record(record_id="a", duplicate=False):
    qa_pairs = [
        {"question": "Which alpha?", "short_answers": ["alpha answer"]},
        {"question": "Which beta?", "short_answers": ["beta answer"]},
    ]
    if duplicate:
        qa_pairs[1]["short_answers"] = ["alpha answer"]
    return {
        "sample_id": record_id,
        "question": "What is the answer?",
        "qa_pairs": qa_pairs,
        "annotations": [{"long_answer": "The alpha answer and beta answer are both relevant."}],
        "docs": [
            {"title": f"Doc {index}", "text": "The alpha answer and beta answer are here."}
            for index in range(5)
        ],
    }


def case(record_id, facet_words):
    questions = tuple("word " * words for words in facet_words)
    return Case(
        id=record_id,
        question=f"Question {record_id}?",
        facet_questions=questions,
        alias_groups=tuple((f"answer {record_id} {i}",) for i in range(len(questions))),
        documents=tuple((f"Title {i}", "Evidence") for i in range(5)),
    )


class ASQACleanFixedSupportP1XTest(unittest.TestCase):
    def test_clean_record_is_kept_and_duplicate_is_rejected(self):
        self.assertIsNotNone(case_from_record(record()))
        self.assertIsNone(case_from_record(record(duplicate=True)))

    def test_decoy_matches_count_and_excludes_self(self):
        cases = [case("a", [2, 3]), case("b", [2, 4]), case("c", [8, 8])]
        mapping = build_decoy_mapping(cases)
        for source in cases:
            self.assertNotEqual(mapping[source.id].id, source.id)
            self.assertEqual(len(mapping[source.id].alias_groups), len(source.alias_groups))
        self.assertEqual(mapping["a"].id, "b")

    def test_true_prompt_shows_questions_but_not_aliases(self):
        source = case("a", [2, 3])
        prompt = render_user(source, "true_facets")
        self.assertIn(source.facet_questions[0].strip(), prompt)
        self.assertNotIn(source.alias_groups[0][0], prompt)

    def test_exact_mcnemar(self):
        p_value, true_only, decoy_only = exact_mcnemar_p(
            [True] * 10 + [False] * 2,
            [False] * 10 + [True] * 2,
        )
        self.assertEqual((true_only, decoy_only), (10, 2))
        self.assertLess(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
