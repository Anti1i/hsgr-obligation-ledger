import math
import unittest

from mathqa_executor_audit import (
    audit_rows, constant_value, correct_option_value, execute_formula,
    problem_numbers,
)


class MathQaExecutorAuditTest(unittest.TestCase):
    def test_problem_numbers(self):
        self.assertEqual(problem_numbers("x is 4,000 and y is .5"), [4000.0, 0.5])

    def test_constants(self):
        self.assertAlmostEqual(constant_value("const_0_2778"), 0.2778)
        self.assertAlmostEqual(constant_value("const_pi"), math.pi)

    def test_execute_chain(self):
        value = execute_formula(
            "values 2, 3 and 4",
            "add(n0,n1)|multiply(#0,n2)|",
        )
        self.assertEqual(value, 20.0)

    def test_correct_option_and_fraction(self):
        self.assertEqual(correct_option_value("a ) 3, b ) 1 / 2, c ) none", "b"), 0.5)

    def test_correct_option_spaced_negative_and_ratio(self):
        self.assertEqual(correct_option_value("a ) 0 b ) - 49 c ) 2", "b"), -49.0)
        self.assertAlmostEqual(correct_option_value("a ) 1 : 729, b ) 2", "a"), 1/729)

    def test_connected_dag_target(self):
        row = {
            "Problem": "values are 2, 3, 4 and 1",
            "linear_formula": (
                "add(n0,n1)|multiply(#0,n2)|subtract(#0,n3)|add(#1,#2)|"
            ),
            "options": "a ) 24, b ) 20, c ) 4, d ) 5, e ) 0",
            "correct": "a",
        }
        result = audit_rows([row])
        self.assertEqual(result["connected_target"], 1)
        self.assertEqual(result["target_matched"], 1)
        self.assertEqual(result["target_execution_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
