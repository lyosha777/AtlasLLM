import time
import hashlib

import numpy as np
from backend.core.document_store import get_embedding_model
from backend.core.source import Source
from backend.tools.web_search import search_web
from backend.tools.web_reader import read_webpages


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[str]:

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def retrieve_web(
    query: str,
    search_results: int = 5,
    candidate_k: int = 20,
    fresh: bool = False,
    cache=None,
) -> list[Source]:

    start = time.perf_counter()

    if cache is not None:
        cache.last_operation = dict(urls=[], fetched=[], failed=[], budget_skipped=[])

    results = search_web(
        query,
        max_results=search_results,
        fresh=fresh
    )

    search_time = time.perf_counter() - start
    print(f"[Web search: {search_time:.2f}s]")

    if not results:
        return []

    urls = [result["url"] for result in results]

    start = time.perf_counter()
    pages = cache.read(urls) if cache is not None else read_webpages(urls)
    reading_time = time.perf_counter() - start

    print(f"[Web reading: {reading_time:.2f}s]")

    if not pages:
        return []

    title_lookup = {
        result["url"]: result["title"]
        for result in results
    }

    start = time.perf_counter()

    candidates = []

    for page in pages:
        chunks = chunk_text(page["text"])
        if cache is not None:
            from backend.research.schemas import MAX_PAGE_CHUNKS
            chunks = chunks[:MAX_PAGE_CHUNKS]
        for chunk in chunks:
            candidates.append({
                "text": chunk,
                "url": page["url"],
                "publication_date": page.get("publication_date"),
                "title": title_lookup.get(
                    page["url"],
                    page["url"]
                )
            })

    chunking_time = time.perf_counter() - start

    print(
        f"[Chunking: {chunking_time:.2f}s | "
        f"{len(candidates)} chunks]"
    )

    if not candidates:
        return []

    if cache is not None:
        cache.chunks_considered += len(candidates)
        cache.last_operation["chunks_considered"] = len(candidates)

    start = time.perf_counter()

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    chunk_embeddings = embedding_model.encode(
        [candidate["text"] for candidate in candidates],
        normalize_embeddings=True
    )

    embedding_time = time.perf_counter() - start
    print(f"[Embedding: {embedding_time:.2f}s]")

    start = time.perf_counter()

    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    ranked_indices = np.argsort(scores)[::-1][:candidate_k]

    ranking_time = time.perf_counter() - start
    print(f"[Vector ranking: {ranking_time:.3f}s]")

    sources = []

    for index in ranked_indices:
        candidate = candidates[index]

        sources.append(
            Source(
                text=candidate["text"],
                source_type="web",
                title=candidate["title"],
                url=candidate["url"],
                relevance_score=float(scores[index]),
                retrieval_score=float(scores[index]),
                chunk_id="web:" + hashlib.sha256((candidate["url"] + "\0" + candidate["text"]).encode()).hexdigest(),
                publication_date=candidate.get("publication_date")
            )
        )

    return sources
