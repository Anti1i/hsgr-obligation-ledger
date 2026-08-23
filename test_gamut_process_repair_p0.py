import unittest

from gamut_process_repair_p0 import (
    ProcessCase,
    Requirement,
    build_process_cases,
    edit_ratio,
    extract_ordered_steps,
    negative_process_answer,
    parse_patch,
    typed_graph,
)


INGREDIENT = (
    "Evaluate the relative order of the matching mandatory steps against this master list: "
    "1. Start at a high temperature, 2. Bake for 5 to 7 minutes, 3. Reduce the temperature. "
    "Missing steps or extra unlisted steps do not fail this element."
)


class GamutProcessRepairP0Test(unittest.TestCase):
    def test_extract_ordered_steps(self):
        self.assertEqual(
            extract_ordered_steps(INGREDIENT),
            ("Start at a high temperature", "Bake for 5 to 7 minutes", "Reduce the temperature"),
        )
        self.assertEqual(extract_ordered_steps("State the oven temperature."), ())

    def test_graph_has_directed_edges(self):
        case = ProcessCase(
            "id", "question", "evidence", Requirement("Sequence", INGREDIENT),
            extract_ordered_steps(INGREDIENT), (Requirement("Sequence", INGREDIENT),),
        )
        rendered = typed_graph(case)
        self.assertIn("Type: ORDERED_PROCESS", rendered)
        self.assertIn("P1 -> P2", rendered)
        self.assertIn("P2 -> P3", rendered)

    def test_patch_requires_unique_exact_span(self):
        answer = "Heat the oven. Bake the muffins. Lower the temperature."
        raw = '{"old_text":"Bake the muffins.","new_text":"Bake for five minutes."}'
        revised, valid, mode = parse_patch(raw, answer)
        self.assertTrue(valid)
        self.assertEqual(mode, "valid")
        self.assertIn("Bake for five minutes.", revised)
        duplicate = '{"old_text":"the","new_text":"a"}'
        self.assertFalse(parse_patch(duplicate, "the pan and the oven")[1])

    def test_edit_ratio(self):
        self.assertEqual(edit_ratio("a b c", "a b c"), 0.0)
        self.assertGreater(edit_ratio("a b c", "x y z"), 0.5)

    def test_negative_control_reverses_same_steps(self):
        steps = ("one", "two", "three")
        answer = negative_process_answer(steps)
        self.assertLess(answer.find("three"), answer.find("two"))
        self.assertLess(answer.find("two"), answer.find("one"))

    def test_build_cases_prefers_parseable_process(self):
        rows = [{
            "session_id": "x",
            "question": "How?",
            "rubrics": {
                "Answer_Critical": [{"Handle": "Sequence", "Ingredient": INGREDIENT, "Specifics": []}],
                "Snippets": [{"id": "s1", "Title": "Source", "Text": "Use high heat, then lower it."}],
            },
        }]
        cases, audit = build_process_cases(rows, 48)
        self.assertEqual(len(cases), 1)
        self.assertEqual(audit["parseable_process_elements"], 1)


if __name__ == "__main__":
    unittest.main()
