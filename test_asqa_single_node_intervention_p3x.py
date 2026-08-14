import unittest

from asqa_clean_fixed_support_p1x import Case
from asqa_single_node_intervention_p3x import (
    build_single_decoys,
    render_single_user,
    score_generation,
    select_fresh_cases,
    summarize,
)


def case(record_id: str, facets: int = 2, first_words: int = 3) -> Case:
    questions = tuple(
        " ".join([f"facet{index}"] * (first_words + index)) + "?"
        for index in range(facets)
    )
    return Case(
        id=record_id,
        question=f"Ambiguous question {record_id}?",
        facet_questions=questions,
        alias_groups=tuple((f"value{record_id}{index}",) for index in range(facets)),
        documents=tuple(
            (f"Title {index}", "Evidence " + " ".join(f"value{record_id}{j}" for j in range(facets)))
            for index in range(5)
        ),
    )


def row(source: Case, arm: str, answer: str, index=None):
    return score_generation(source, answer, arm, index)


class ASQASingleNodeInterventionP3XTest(unittest.TestCase):
    def test_fresh_selection_is_disjoint_and_deterministic(self):
        eligible = [case(f"id-{index:03d}") for index in range(427)]
        old, pool, selected = select_fresh_cases(eligible, 192)
        old_again, pool_again, selected_again = select_fresh_cases(eligible, 192)
        self.assertEqual(len(old), 192)
        self.assertEqual(len(pool), 235)
        self.assertEqual(len(selected), 192)
        self.assertFalse({item.id for item in old} & {item.id for item in selected})
        self.assertEqual([item.id for item in old], [item.id for item in old_again])
        self.assertEqual([item.id for item in pool], [item.id for item in pool_again])
        self.assertEqual([item.id for item in selected], [item.id for item in selected_again])

    def test_decoy_excludes_self_and_is_deterministic(self):
        cases = [case("a", first_words=2), case("b", first_words=4), case("c", first_words=7)]
        mapping = build_single_decoys(cases)
        self.assertEqual(mapping, build_single_decoys(cases))
        for source in cases:
            self.assertNotEqual(source.id, mapping[source.id].case_id)

    def test_single_prompt_contains_only_selected_question_and_no_aliases(self):
        source = case("a")
        prompt = render_single_user(source, source.facet_questions[0])
        self.assertIn(source.facet_questions[0], prompt)
        self.assertNotIn(source.facet_questions[1], prompt)
        checklist = prompt.split("Coverage checklist", 1)[1].split("\n\nAnswer:", 1)[0]
        for aliases in source.alias_groups:
            self.assertNotIn(aliases[0], checklist)

    def test_summary_distinguishes_single_and_all_oracles(self):
        cases = [case(label) for label in "abcd"]
        rows = []
        # a: both Oracles succeed; individual node effects are mixed.
        source = cases[0]
        rows.extend([
            row(source, "fixed_direct", "none"),
            row(source, "all_true", "valuea0 valuea1"),
            row(source, "single_true", "valuea0 valuea1", 1),
            row(source, "single_true", "none", 2),
            row(source, "single_decoy", "none"),
        ])
        # b: only the single-node Oracle succeeds.
        source = cases[1]
        rows.extend([
            row(source, "fixed_direct", "none"),
            row(source, "all_true", "none"),
            row(source, "single_true", "valueb0 valueb1", 1),
            row(source, "single_true", "none", 2),
            row(source, "single_decoy", "none"),
        ])
        # c: only the all-node Oracle succeeds.
        source = cases[2]
        rows.extend([
            row(source, "fixed_direct", "none"),
            row(source, "all_true", "valuec0 valuec1"),
            row(source, "single_true", "none", 1),
            row(source, "single_true", "none", 2),
            row(source, "single_decoy", "none"),
        ])
        # d: direct succeeds, so both KEEP Oracles succeed.
        source = cases[3]
        rows.extend([
            row(source, "fixed_direct", "valued0 valued1"),
            row(source, "all_true", "valued0 valued1"),
            row(source, "single_true", "valued0 valued1", 1),
            row(source, "single_true", "valued0 valued1", 2),
            row(source, "single_decoy", "valued0 valued1"),
        ])

        result = summarize(cases, [], cases, cases, rows)
        self.assertEqual(result["counts"]["mixed_intervention_problems"], 2)
        self.assertEqual(result["oracle_mcnemar"]["single_only_successes"], 1)
        self.assertEqual(result["oracle_mcnemar"]["all_only_successes"], 1)
        self.assertEqual(result["absolute"]["keep_or_single_oracle"]["str_hit"], 0.75)
        self.assertEqual(result["absolute"]["keep_or_all_oracle"]["str_hit"], 0.75)


if __name__ == "__main__":
    unittest.main()
