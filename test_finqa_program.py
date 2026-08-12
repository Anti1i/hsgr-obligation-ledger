import unittest

from finqa_program import (
    Step,
    canonical_program,
    execute_program,
    execution_matches,
    parse_number,
    parse_program,
    structure_metrics,
)


class FinQAProgramTest(unittest.TestCase):
    def test_parse_and_canonicalize(self):
        source = "divide(9413, 20.01), subtract(#0, #1)"
        self.assertEqual(canonical_program(parse_program(source)), source)

    def test_parse_token_list(self):
        tokens = ["add(", "2", "3", ")", "EOF"]
        self.assertEqual(parse_program(tokens), [Step("add", ("2", "3"))])

    def test_numeric_executor_and_refs(self):
        steps = parse_program("divide(10, 2), add(#0, 3)")
        result = execute_program(steps, [])
        self.assertTrue(result.valid)
        self.assertTrue(execution_matches(result.value, 8))

    def test_table_executor(self):
        table = [["year", "2020", "2021"], ["revenue", "$10", "20"]]
        result = execute_program([Step("table_average", ("revenue", "none"))], table)
        self.assertTrue(result.valid)
        self.assertEqual(result.value, 15.0)

    def test_invalid_forward_ref(self):
        result = execute_program([Step("add", ("#0", "1"))], [])
        self.assertFalse(result.valid)

    def test_number_formats(self):
        self.assertEqual(parse_number("12%"), 0.12)
        self.assertEqual(parse_number("const_m1"), -1.0)
        self.assertEqual(parse_number("(2.5)"), -2.5)

    def test_structure(self):
        steps = [
            Step("add", ("1", "2")),
            Step("subtract", ("8", "3")),
            Step("multiply", ("#0", "#1")),
        ]
        metrics = structure_metrics(steps)
        self.assertEqual(metrics["depth"], 2)
        self.assertTrue(metrics["join"])
        self.assertFalse(metrics["deep"])


if __name__ == "__main__":
    unittest.main()

