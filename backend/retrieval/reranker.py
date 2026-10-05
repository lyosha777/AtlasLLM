from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.core.config import RERANKER_MODEL
from backend.core.source import Source


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    sources: list[Source],
    top_k: int = 6,
    diversify: bool = True,
) -> list[Source]:

    if not sources:
        return []

    pairs = [
        (query, source.text)
        for source in sources
    ]

    scores = get_reranker().predict(pairs)

    for source, score in zip(sources, scores):
        source.relevance_score = float(score)

    ranked = sorted(
        sources,
        key=lambda source: source.relevance_score,
        reverse=True
    )

    return diversify_sources(ranked, top_k) if diversify else ranked[:top_k]


def diversify_sources(
    sources: list[Source],
    top_k: int
) -> list[Source]:

    selected = []
    seen = set()

    for source in sources:
        key = source.document_id or source.url or source.title

        if key in seen:
            continue

        seen.add(key)
        selected.append(source)

        if len(selected) >= top_k:
            break

    return selected
