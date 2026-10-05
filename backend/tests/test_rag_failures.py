import unittest
from unittest.mock import patch

from backend.core.source import Source
from backend.evaluation.rag import DATASET, evaluate_cases, failure_analysis, load_dataset, print_failures


class RankingFailureTests(unittest.TestCase):
    def test_scores_are_snapshotted_before_mutating_reranker(self):
        data = load_dataset(DATASET)
        data["cases"] = data["cases"][:1]
        gold, distractor = data["passages"][:2]
        def source(item, score):
            return Source(item["text"], "document", document_id=item["document_id"],
                          chunk_id=item["id"], title="network.pdf", page=item["page"], relevance_score=score)
        first, second = source(gold, .9), source(distractor, .8)
        def rerank(query, sources):
            first.relevance_score, second.relevance_score = 1.2, 2.4
            return [second, first]
        records = evaluate_cases(data, lambda query: [first, second], rerank)
        self.assertEqual(records[0]["initial_ranking"][0]["score"], .9)
        self.assertEqual(records[0]["cross_encoder_candidates"][0]["score"], 2.4)
        report = failure_analysis(records, data)
        failure = report["failures"][0]
        self.assertEqual(failure["expected"][0]["final_rank"], 2)
        self.assertEqual(failure["expected"][0]["initial_rank"], 1)
        self.assertTrue(failure["expected_in_candidates"])
        self.assertIn("initial retrieval ranked correctly; final ranking reversed it", failure["patterns"])
        self.assertIn("winner and gold are different chunks in the same document", failure["patterns"])
        with patch("builtins.print") as output:
            print_failures(report, "dense")
        self.assertTrue(any("Dense retrieval" in str(call) for call in output.call_args_list))

    def test_absent_gold_and_legacy_results(self):
        data = load_dataset(DATASET)
        case = data["cases"][0]
        record = dict(case_id=case["id"], question=case["question"], expected_chunk_ids=case["expected_chunk_ids"],
                      retrieved_ids=[data["passages"][1]["id"]], ranked_ids=[data["passages"][1]["id"]])
        report = failure_analysis([record], data)
        self.assertFalse(report["failures"][0]["expected_in_candidates"])
        self.assertIsNone(report["failures"][0]["expected"][0]["final_rank"])
        self.assertEqual(report["patterns"]["expected evidence absent from initial candidates"], 1)
        record["ranked_ids"] = case["expected_chunk_ids"]
        self.assertEqual(failure_analysis([record], data)["failures"], [])

    def test_relevant_alternative_is_not_a_failure(self):
        data = load_dataset(DATASET)
        case = next(item for item in data["cases"] if item["id"] == "q22")
        alternative = case["relevant_chunk_ids"][1]
        record = dict(case_id=case["id"], question=case["question"], expected_chunk_ids=case["expected_chunk_ids"],
                      retrieved_ids=[alternative], ranked_ids=[alternative])
        report = failure_analysis([record], data)
        self.assertEqual(report["true_top1_failures"], 0)
        self.assertEqual(report["relevant_alternative_wins"], 1)
        self.assertEqual(report["patterns"], {})


if __name__ == "__main__":
    unittest.main()
