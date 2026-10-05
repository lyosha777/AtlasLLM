import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.core.source import Source
from backend.retrieval.hybrid import LexicalIndex, fuse_rankings, retrieve_hybrid


class Collection:
    def get(self, **options):
        rows = [("a:1", "alpha", "TCP reliable ordered delivery"),
                ("a:2", "alpha", "UDP datagrams unreliable delivery"),
                ("b:1", "beta", "TCP congestion control")]
        if options.get("where"):
            allowed = options["where"]["document_id"]["$in"]
            rows = [row for row in rows if row[1] in allowed]
        return dict(ids=[row[0] for row in rows], documents=[row[2] for row in rows],
                    metadatas=[dict(document_id=row[1], source=row[1]+".pdf", page=1) for row in rows])


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_ranking_and_metadata(self):
        index = LexicalIndex(Collection())
        sources = index.search("tcp RELIABLE")
        self.assertEqual(sources[0].chunk_id, "a:1")
        self.assertEqual(sources[0].document_id, "alpha")
        self.assertEqual(sources[0].page, 1)
        self.assertEqual(sources[0].text, "TCP reliable ordered delivery")
        self.assertEqual(index.search("unknownword"), [])
        sources[0].relevance_score = -100
        self.assertGreater(index.search("tcp reliable")[0].relevance_score, 0)

    def test_rrf_uses_ranks_deduplicates_and_preserves_identity(self):
        a, b, c = [Source(name, "document", document_id="doc", chunk_id=name) for name in ("a", "b", "c")]
        ranked = fuse_rankings([[a, b], [c, b, b]])
        self.assertEqual(ranked[0].chunk_id, "b")
        self.assertAlmostEqual(ranked[0].retrieval_score, 2/62)
        self.assertEqual(len(ranked), 3)
        self.assertIsNone(b.relevance_score)

    def test_scoping_empty_scope_and_hybrid_dense_call(self):
        collection = Collection()
        index = LexicalIndex(collection, ["alpha"])
        self.assertTrue(all(item.document_id == "alpha" for item in index.search("tcp")))
        self.assertEqual(LexicalIndex(collection, []).search("tcp"), [])
        dense_source = Source("TCP reliable ordered delivery", "document", document_id="alpha", chunk_id="a:1", page=1)
        dense = Mock(return_value=[dense_source])
        module = SimpleNamespace(get_document_collection=lambda: collection, retrieve=dense)
        with patch.dict("sys.modules", {"backend.retrieval.retriever": module}), \
             patch("backend.retrieval.retriever", module, create=True):
            result = retrieve_hybrid("TCP", document_ids=["alpha"], lexical_index=index)
            dense.assert_called_once_with("TCP", n_results=5, document_ids=["alpha"])
            self.assertEqual(result[0].chunk_id, "a:1")
            self.assertTrue(all(item.document_id == "alpha" for item in result))
            with self.assertRaises(ValueError):
                retrieve_hybrid("TCP", document_ids=["beta"], lexical_index=index)

    def test_empty_rankings_and_missing_ids(self):
        self.assertEqual(fuse_rankings([[], []]), [])
        with self.assertRaises(ValueError):
            fuse_rankings([[Source("text", "document")]])


if __name__ == "__main__":
    unittest.main()
