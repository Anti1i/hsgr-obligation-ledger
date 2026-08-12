import os
import tempfile
import unittest

from typed_node_action_oracle import (
    ARMS,
    analyze,
    extract_boxed_all,
    parse_intervention,
    selected_rows,
    stable_rank,
)


class TypedNodeActionOracleTest(unittest.TestCase):
    def test_frozen_arm_count_and_parser(self):
        self.assertEqual(len(ARMS), 13)
        text = "TARGET: \\boxed{5}\nROOT: \\boxed{11}"
        self.assertEqual(extract_boxed_all(text), ["5", "11"])
        self.assertEqual(parse_intervention(text), ("5", "11"))

    def test_selection_is_outcome_independent(self):
        rows_a = [
            {"problem": str(i), "answer": str(i), "graph": {
                "edges": [["parent_0", "root"], ["parent_1", "root"]]
            }} for i in range(8)
        ]
        rows_b = [dict(row, answer="changed") for row in rows_a]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, rows in (("a", rows_a), ("b", rows_b)):
                path = os.path.join(directory, name + ".jsonl")
                with open(path, "w", encoding="utf-8") as handle:
                    for row in rows:
                        import json
                        handle.write(json.dumps(row) + "\n")
                paths.append(path)
            ids_a = [row["id"] for row in selected_rows(paths[0], limit=4)]
            ids_b = [row["id"] for row in selected_rows(paths[1], limit=4)]
        self.assertEqual(ids_a, ids_b)
        self.assertEqual(stable_rank(3), stable_rank(3))

    def test_joint_oracle_requires_node_action_interaction(self):
        units = [
            {"id": pid, "gold": "1", "parent_gold": ["1", "1"]}
            for pid in range(4)
        ]
        base = {
            pid: {
                "answer": "1" if pid == 0 else "0",
                "prompt_tokens": 1, "generated_tokens": 1,
                "calls": 1, "max_new": 512,
            }
            for pid in range(4)
        }
        generic = {
            pid: {
                "answer": "1" if pid in (0, 1) else "0",
                "prompt_tokens": 1, "generated_tokens": 1,
                "calls": 1, "max_new": 512,
            }
            for pid in range(4)
        }
        repairs = {
            1: ("parent_0", "equation"),
            2: ("parent_1", "backward"),
            3: ("root", "redecompose"),
        }
        interventions = {}
        for pid in range(4):
            for node, action in ARMS:
                hit = pid == 0 or repairs.get(pid) == (node, action)
                interventions[(pid, node, action)] = {
                    "target_answer": "1", "root_answer": "1" if hit else "0",
                    "prompt_tokens": 1, "generated_tokens": 1,
                    "calls": 1, "max_new": 512,
                }
        report = analyze(units, base, generic, interventions)
        accuracy = report["accuracy"]
        self.assertEqual(accuracy["node_x_action_oracle"], 1.0)
        self.assertEqual(accuracy["action_oracle_accuracy"], 0.5)
        self.assertEqual(accuracy["node_oracle_accuracy"], 0.5)
        self.assertEqual(
            report["exclusive_node_repairs"],
            {"parent_0": 1, "parent_1": 1, "root": 1},
        )


if __name__ == "__main__":
    unittest.main()
