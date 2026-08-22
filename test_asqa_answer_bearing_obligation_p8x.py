import unittest

from asqa_answer_bearing_obligation_p8x import parse_claim_nodes, valid_claim_nodes


class AnswerBearingObligationP8XTests(unittest.TestCase):
    def test_parses_claim_nodes(self):
        raw = '[{"scope":"film", "claim":"The film was released in 1999."}, {"scope":"book", "claim":"The book was published in 1995."}]'
        nodes, mode = parse_claim_nodes(raw)
        self.assertEqual(mode, "json")
        self.assertEqual(len(nodes), 2)

    def test_rejects_question_only_schema(self):
        self.assertIsNone(valid_claim_nodes([{"question": "What happened?"}, {"question": "Who?"}]))

    def test_repairs_singleton_object_lines(self):
        raw = '[{"scope":"A","claim":"Claim A."}]\n[{"scope":"B","claim":"Claim B."}]'
        nodes, mode = parse_claim_nodes(raw)
        self.assertEqual(mode, "multi_object")
        self.assertEqual(len(nodes), 2)


if __name__ == "__main__":
    unittest.main()
