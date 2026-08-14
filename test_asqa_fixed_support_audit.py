import json
import tempfile
import unittest
from pathlib import Path

from asqa_fixed_support_audit import (
    audit_asqa,
    exact_presence,
    extract_row,
    normalize_answer,
)


def qa(question, *answers):
    return {"question": question, "short_answers": list(answers)}


def alce_record(record_id="id-1"):
    return {
        "sample_id": record_id,
        "question": "Who was Mercury?",
        "qa_pairs": [
            qa("Which planet was Mercury?", "the first planet", "Mercury"),
            qa("Which god was Mercury?", "Roman messenger god"),
            qa("Duplicate planet facet", "Mercury", "first planet"),
        ],
        "annotations": [
            {"long_answer": "Mercury was the first planet and the Roman messenger god."},
            {"long_answer": "Mercury is a planet."},
        ],
        "docs": [
            {"title": f"Doc {index}", "text": "Mercury was the first planet. Roman messenger god."}
            for index in range(5)
        ],
    }


class ASQAFixedSupportAuditTest(unittest.TestCase):
    def test_alce_normalization(self):
        self.assertEqual(normalize_answer("The Ali-Daei!"), "alidaei")
        self.assertTrue(exact_presence(("ali daei",), "Record: Ali Daei."))

    def test_extract_row_deduplicates_facets_and_scores(self):
        row = extract_row("id-1", alce_record())
        self.assertEqual(row["raw_facet_count"], 3)
        self.assertEqual(row["facet_count"], 2)
        self.assertEqual(row["duplicate_facet_groups"], 1)
        self.assertTrue(row["passage_strict"])
        self.assertTrue(row["best_human_strict"])
        self.assertFalse(row["verbatim_human_answer_leak"])
        self.assertTrue(row["eligible"])

    def test_audit_requires_consistent_original_join(self):
        alce = alce_record()
        original_record = {
            "ambiguous_question": alce["question"],
            "qa_pairs": alce["qa_pairs"],
            "annotations": alce["annotations"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alce_path = root / "alce.json"
            original_path = root / "original.json"
            alce_path.write_text(json.dumps([alce]), encoding="utf-8")
            original_path.write_text(
                json.dumps({"dev": {"id-1": original_record}}), encoding="utf-8"
            )
            report = audit_asqa(alce_path, original_path)
        self.assertEqual(report["counts"]["aligned_examples"], 1)
        self.assertEqual(report["counts"]["eligible_examples"], 1)
        self.assertFalse(report["gates"]["g5_duplicate_facet_examples_below_2pct"])

    def test_audit_rejects_question_mismatch(self):
        alce = alce_record()
        original_record = {
            "ambiguous_question": "Different question",
            "qa_pairs": alce["qa_pairs"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alce_path = root / "alce.json"
            original_path = root / "original.json"
            alce_path.write_text(json.dumps([alce]), encoding="utf-8")
            original_path.write_text(
                json.dumps({"dev": {"id-1": original_record}}), encoding="utf-8"
            )
            report = audit_asqa(alce_path, original_path)
        self.assertEqual(report["counts"]["aligned_examples"], 0)
        self.assertEqual(
            report["alignment_mismatch_reasons"]["question_or_qa_pair_mismatch"], 1
        )


if __name__ == "__main__":
    unittest.main()
