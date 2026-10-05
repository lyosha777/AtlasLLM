import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.evaluation.rag import DATASET, evaluate_cases, load_dataset, ranking_metrics, save, source_valid, summary


class RagEvaluationTests(unittest.TestCase):
    def test_gold_dataset_and_failure_case(self):
        data = load_dataset(DATASET)
        self.assertEqual(len(data["cases"]), 45)
        self.assertEqual(len(data["passages"]), 65)
        self.assertGreaterEqual(len({case["style"] for case in data["cases"]}), 5)
        failure = json.loads(DATASET.with_name("failures-v1.json").read_text())["cases"][0]
        self.assertEqual(failure["question"], "Who existed first, Armenia or Azerbaijan?")
        self.assertFalse(failure["included_in_core_metrics"])
        self.assertEqual(failure["expected_chunk_ids"], [])

    def test_original_cases_preserved_and_distractors_present(self):
        current = load_dataset(DATASET)
        original = load_dataset(DATASET.with_name("rag-v1.json"))
        self.assertEqual(current["passages"][:25], original["passages"])
        self.assertEqual([{key: value for key, value in case.items() if key not in ("cohort", "primary_gold_chunk_id", "relevant_chunk_ids", "relevance_notes")}
                          for case in current["cases"][:25]], original["cases"])
        self.assertEqual(sum(case["cohort"] == "challenge" for case in current["cases"]), 20)
        self.assertEqual(sum(item["document_id"].endswith("-related") for item in current["passages"]), 20)
        self.assertTrue(all(0 < len(item["text"]) < 800 for item in current["passages"]))
        for case in current["cases"][25:]:
            document = case["expected_document_ids"][0]
            self.assertTrue(any(item["document_id"] == document + "-related" for item in current["passages"]))

    def test_recall_and_reciprocal_rank(self):
        values = ranking_metrics(["wrong", "a", "a", "b"], ["a", "b"])
        self.assertEqual(values["recall_at_1"], 0)
        self.assertEqual(values["recall_at_3"], 1)
        self.assertEqual(values["recall_at_5"], 1)
        self.assertEqual(values["mrr"], .5)
        self.assertEqual(ranking_metrics([], ["a"])["mrr"], 0)

    def test_curated_alternatives_and_adversarial_labels(self):
        data = load_dataset(DATASET)
        cases = {item["id"]: item for item in data["cases"]}
        self.assertEqual(sum(len(item["relevant_chunk_ids"])-1 for item in cases.values()), 4)
        for identity in ("q04", "q19", "q22", "q24"):
            self.assertEqual(len(cases[identity]["relevant_chunk_ids"]), 2)
            self.assertIn(cases[identity]["primary_gold_chunk_id"], cases[identity]["relevant_chunk_ids"])
        for identity in ("q14", "q26", "q42"):
            self.assertEqual(cases[identity]["relevant_chunk_ids"], cases[identity]["expected_chunk_ids"])
        self.assertEqual(ranking_metrics(["alternative", "wrong"], ["canonical", "alternative"])["recall_at_1"], 1)

    def test_evaluation_records_both_stages_without_generation(self):
        data = load_dataset(DATASET)
        data["cases"] = data["cases"][:1]
        passage = data["passages"][0]
        correct = SimpleNamespace(chunk_id=passage["id"], document_id=passage["document_id"],
                                  text=passage["text"], page=passage["page"], source_type="document", title="fixture", relevance_score=.8)
        wrong = SimpleNamespace(chunk_id="unknown", document_id="unknown", text="wrong", page=1, source_type="document", title="fixture", relevance_score=.1)
        checkpoints = []
        records = evaluate_cases(data, lambda question: [wrong, correct], lambda question, sources: list(reversed(sources)),
                                 lambda items: checkpoints.append(list(items)))
        self.assertEqual(records[0]["retrieval"]["mrr"], .5)
        self.assertEqual(records[0]["reranked"]["mrr"], 1)
        self.assertEqual(summary(records)["source_id_validity"], .5)
        self.assertEqual(len(checkpoints), 1)
        self.assertGreaterEqual(records[0]["retrieval_ms"], 0)
        correct.text = "altered passage"
        self.assertFalse(source_valid(correct, {passage["id"]: passage}))

    def test_invalid_gold_and_partial_result_storage(self):
        data = load_dataset(DATASET)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save(path, {"status": "timeout", "records": []})
            self.assertEqual(json.loads(path.read_text())["status"], "timeout")
            self.assertFalse(path.with_suffix(".tmp").exists())
            data["cases"][0]["expected_chunk_ids"] = ["unknown"]
            save(path, data)
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
