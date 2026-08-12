import unittest

from executable_dag_audit import (
    graph_metrics,
    parse_call,
    parse_excel_formula,
    parse_straight_line,
    scan_flat_calls,
    split_top_level,
)


class ExecutableDagAuditTest(unittest.TestCase):
    def test_top_level_split(self):
        self.assertEqual(split_top_level("a,b(c,d),e"), ["a", "b(c,d)", "e"])

    def test_scan_mathqa_pipe(self):
        calls = scan_flat_calls("add(n0,n1)|multiply(#0,const_2)|")
        self.assertEqual(calls, ["add(n0,n1)", "multiply(#0,const_2)"])

    def test_parse_call(self):
        self.assertEqual(parse_call("divide(#0, const_2)"), ("divide", ["#0", "const_2"]))

    def test_chain(self):
        metrics = graph_metrics(parse_straight_line(
            "add(1,2), multiply(#0,3), subtract(#1,4)"
        ))
        self.assertTrue(metrics["deep"])
        self.assertFalse(metrics["join"])
        self.assertFalse(metrics["reuse"])

    def test_diamond(self):
        metrics = graph_metrics(parse_straight_line(
            "add(1,2), multiply(#0,3), subtract(#0,4), add(#1,#2)"
        ))
        self.assertTrue(metrics["deep_join_reuse"])
        self.assertTrue(metrics["diamond"])
        self.assertEqual(metrics["connected_internal_nodes"], 3)

    def test_forward_ref(self):
        metrics = graph_metrics(parse_straight_line("add(#1,2), add(3,4)"))
        self.assertFalse(metrics["valid_references"])

    def test_excel_expression_tree(self):
        metrics = graph_metrics(parse_excel_formula("=(A1+B1)/(C1-D1)"))
        self.assertEqual(metrics["nodes"], 3)
        self.assertTrue(metrics["join"])
        self.assertFalse(metrics["reuse"])

    def test_excel_functions(self):
        metrics = graph_metrics(parse_excel_formula("=SUM(A1:A3)+AVERAGE(B1:B3)"))
        self.assertEqual(metrics["nodes"], 3)
        self.assertTrue(metrics["join"])


if __name__ == "__main__":
    unittest.main()

