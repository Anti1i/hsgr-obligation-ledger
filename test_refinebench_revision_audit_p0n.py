import unittest

from refinebench_revision_audit_p0n import (
    ARMS,
    STRATA,
    build_review,
    field_stratum,
    parse_evaluation,
    select_stratified,
    target_ids,
    transition_row,
)


FIELDS = {
    "math_statistics": "Math",
    "stem": "Physics",
    "law": "Law",
    "humanities": "Humanities/Social Science",
    "other": "Other",
}


def instance(index, stratum):
    return {
        "index": index,
        "question": f"Question {index}",
        "checklist": ["Does the response include A?", "Does the response include B?"],
        "field": FIELDS[stratum],
        "passages": [],
        "materials": [],
        "reference_answer": ["A and B"],
    }


class RefineBenchRevisionAuditP0NTests(unittest.TestCase):
    def test_field_strata(self):
        self.assertEqual(field_stratum("Statistics"), "math_statistics")
        self.assertEqual(field_stratum("Computer Science/AI"), "stem")
        self.assertEqual(field_stratum("Law"), "law")
        self.assertEqual(field_stratum("Humanities/Social Science"), "humanities")
        self.assertEqual(field_stratum("Economics/Business"), "other")
        self.assertIsNone(field_stratum("Unknown"))

    def test_selection_is_balanced_deterministic_and_outcome_free(self):
        rows = [
            instance(f"{stratum}-{number}", stratum)
            for stratum in STRATA for number in range(6)
        ]
        first = select_stratified(rows, 3, 50000)
        second = select_stratified(list(reversed(rows)), 3, 50000)
        self.assertEqual([row["index"] for row in first], [row["index"] for row in second])
        self.assertEqual(len(first), 15)
        for stratum in STRATA:
            self.assertEqual(sum(row["p0n_stratum"] == stratum for row in first), 3)

    def test_partial_targets_only_failed_and_are_deterministic(self):
        failures = [1, 3, 4, 7, 8]
        selected = target_ids("case", failures, "targeted_partial_failed")
        self.assertEqual(selected, target_ids("case", failures, "targeted_partial_failed"))
        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected).issubset(failures))
        self.assertEqual(target_ids("case", failures, "guided_failed"), failures)
        self.assertEqual(target_ids("case", [], "targeted_partial_failed"), [])

    def test_evaluator_parser_never_converts_invalid_to_all_no(self):
        parsed, valid, mode = parse_evaluation("1: Yes\n2: No", 2)
        self.assertTrue(valid)
        self.assertEqual(parsed, {1: True, 2: False})
        parsed, valid, mode = parse_evaluation("1: Yes", 2)
        self.assertFalse(valid)
        self.assertEqual(parsed, {})
        self.assertEqual(mode, "missing_or_extra_ids")
        parsed, valid, mode = parse_evaluation("1: Yes\n1: No\n2: Yes", 2)
        self.assertFalse(valid)
        self.assertEqual(mode, "duplicate_id")

    def test_transition_counts_fix_and_regression(self):
        row = {**instance("law-1", "law"), "p0n_stratum": "law"}
        old = {
            "judge_valid": True, "evaluation": {1: True, 2: False},
            "judge_parse_mode": "valid",
        }
        new = {
            "judge_valid": True, "evaluation": {1: False, 2: True},
            "judge_parse_mode": "valid",
        }
        transition = transition_row(
            row, "generator", ARMS[0], "A is present. Old detail remains.",
            "B is present. New detail remains.", old, new, [2],
        )
        self.assertEqual(transition["transitions"], {"YN": 1, "NY": 1})
        self.assertTrue(transition["successful_fix_regression"])
        self.assertEqual(transition["prior_yes"], 1)

    def test_review_keeps_manual_labels_unfilled(self):
        row = {**instance("law-1", "law"), "p0n_stratum": "law"}
        old = {"judge_valid": True, "evaluation": {1: True, 2: False}, "judge_parse_mode": "valid"}
        new = {"judge_valid": True, "evaluation": {1: False, 2: True}, "judge_parse_mode": "valid"}
        transition = transition_row(row, "g", ARMS[0], "A sentence stays here.", "B sentence stays here.", old, new, [2])
        review = build_review([transition])
        candidate = next(item for item in review if item["review_type"] == "candidate_yes_to_no")
        self.assertIsNone(candidate["manual_valid_transition"])
        self.assertIsNone(candidate["manual_category"])


if __name__ == "__main__":
    unittest.main()
