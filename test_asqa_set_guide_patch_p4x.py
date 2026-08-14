import unittest

from asqa_clean_fixed_support_p1x import Case
from asqa_set_guide_patch_p4x import (
    CALIBRATION_N,
    build_wrong_mapping,
    select_calibration_cell,
    split_cases,
)


def make_case(index: int, facets: int = 2) -> Case:
    return Case(
        id=f"case-{index:03d}",
        question=f"Question {index}?",
        facet_questions=tuple(f"Facet {index}-{j}?" for j in range(facets)),
        alias_groups=tuple((f"answer-{index}-{j}",) for j in range(facets)),
        documents=tuple((f"Title {j}", "Evidence") for j in range(5)),
    )


class ASQASetGuidePatchP4XTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_64_128(self):
        cases = [make_case(index, 2 + index % 3) for index in range(192)]
        calibration, heldout = split_cases(cases)
        second_calibration, second_heldout = split_cases(list(reversed(cases)))
        self.assertEqual(len(calibration), CALIBRATION_N)
        self.assertEqual(len(heldout), 128)
        self.assertFalse({case.id for case in calibration} & {case.id for case in heldout})
        self.assertEqual([case.id for case in calibration], [case.id for case in second_calibration])
        self.assertEqual([case.id for case in heldout], [case.id for case in second_heldout])

    def test_wrong_guide_mapping_matches_count_and_excludes_self(self):
        cases = [make_case(index, 2 + index % 3) for index in range(18)]
        mapping = build_wrong_mapping(cases)
        for case in cases:
            self.assertNotEqual(mapping[case.id].id, case.id)
            self.assertEqual(
                len(mapping[case.id].facet_questions), len(case.facet_questions)
            )

    def test_calibration_tie_break_prefers_em_then_lower_alpha_then_layer(self):
        cells = [
            {"layer": 20, "alpha": 1.0, "metrics": {"str_hit": 0.5, "str_em": 0.7}},
            {"layer": 27, "alpha": 0.5, "metrics": {"str_hit": 0.5, "str_em": 0.8}},
            {"layer": 20, "alpha": 0.5, "metrics": {"str_hit": 0.5, "str_em": 0.8}},
            {"layer": 13, "alpha": 0.5, "metrics": {"str_hit": 0.5, "str_em": 0.8}},
        ]
        selected = select_calibration_cell(cells)
        self.assertEqual((selected["layer"], selected["alpha"]), (13, 0.5))


if __name__ == "__main__":
    unittest.main()
