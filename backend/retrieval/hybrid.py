"""Optional BM25/RRF candidate retrieval over Atlas's existing Chroma chunks."""
from collections import Counter, defaultdict
import math
import re

from backend.core.source import Source


def tokenize(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class LexicalIndex:
    def __init__(self, collection, document_ids=None):
        self.collection = collection
        self.scope = None if document_ids is None else tuple(sorted(set(document_ids)))
        self.sources, self.frequencies, self.postings = [], [], defaultdict(list)
        if self.scope == ():
            self.average_length = 0
            return
        options = {"where": {"document_id": {"$in": list(self.scope)}}} if self.scope else {}
        records = collection.get(include=["documents", "metadatas"], **options)
        for chunk_id, text, metadata in zip(records["ids"], records["documents"], records["metadatas"]):
            self.sources.append(Source(text=text, source_type="document", title=metadata["source"],
                                       page=metadata["page"], document_id=metadata.get("document_id"), chunk_id=chunk_id))
            frequencies = Counter(tokenize(text))
            position = len(self.frequencies)
            self.frequencies.append(frequencies)
            for term in frequencies:
                self.postings[term].append(position)
        self.lengths = [sum(item.values()) for item in self.frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0

    def search(self, query, limit=5):
        if not self.sources or not self.average_length or limit < 1:
            return []
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            matches = self.postings.get(term, [])
            idf = math.log(1 + (len(self.sources) - len(matches) + .5) / (len(matches) + .5))
            for position in matches:
                frequency = self.frequencies[position][term]
                denominator = frequency + 1.5 * (1 - .75 + .75 * self.lengths[position] / self.average_length)
                scores[position] += idf * frequency * 2.5 / denominator
        # Return fresh instances: the existing reranker mutates source scores.
        return [Source(**{**vars(self.sources[position]), "relevance_score": scores[position], "retrieval_score": scores[position]})
                for position in sorted(scores, key=lambda item: (-scores[item], self.sources[item].chunk_id))[:limit]]


def fuse_rankings(rankings, limit=10, rank_constant=60):
    scores, sources = defaultdict(float), {}
    for ranking in rankings:
        seen = set()
        for rank, source in enumerate(ranking, 1):
            identity = source.chunk_id
            if identity is None:
                raise ValueError("Hybrid retrieval requires retained chunk IDs")
            if identity in seen:
                continue
            seen.add(identity)
            sources.setdefault(identity, source)
            scores[identity] += 1 / (rank_constant + rank)
    return [Source(**{**vars(sources[identity]), "relevance_score": scores[identity], "retrieval_score": scores[identity]})
            for identity in sorted(scores, key=lambda item: (-scores[item], item))[:limit]]


def retrieve_hybrid(query, document_ids=None, lexical_index=None, candidate_k=5, top_k=10):
    from backend.retrieval import retriever
    collection = retriever.get_document_collection()
    scope = None if document_ids is None else tuple(sorted(set(document_ids)))
    if lexical_index is None:
        lexical_index = LexicalIndex(collection, document_ids)
    if lexical_index.collection is not collection or lexical_index.scope != scope:
        raise ValueError("Lexical index must belong to the same collection and document scope")
    dense = retriever.retrieve(query, n_results=candidate_k, document_ids=document_ids)
    lexical = lexical_index.search(query, limit=candidate_k)
    return fuse_rankings([dense, lexical], limit=top_k)
