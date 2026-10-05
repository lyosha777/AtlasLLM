import time
from collections.abc import Iterator

from backend.core.planner import plan
from backend.generation.chat import chat_events
from backend.generation.generator import answer_events
from backend.retrieval.reranker import rerank
from backend.retrieval.retriever import retrieve
from backend.retrieval.web_retriever import retrieve_web


def serialize_source(source, index: int) -> dict:
    return {
        "index": index,
        "type": source.source_type,
        "title": source.title,
        "url": source.url,
        "page": source.page,
        "score": source.relevance_score,
        "document_id": source.document_id,
    }


def stream_assistant(
    message: str,
    history: list[dict],
    force_web: bool = False,
    document_ids: list[str] | None = None,
    inference: dict | None = None,
    research_mode: bool = False,
    research_scope: str = "auto",
    research_run_id: str | None = None,
) -> Iterator[dict]:

    if research_mode:
        from backend.research.pipeline import research_events
        yield from research_events(message, history, inference, research_run_id, research_scope, document_ids, force_web)
        return

    total_start = time.perf_counter()
    timings = {}

    yield {
        "type": "status",
        "stage": "planning",
        "label": "Planning"
    }

    start = time.perf_counter()
    decision = plan(message, history, has_attachments=bool(document_ids))
    timings["planner"] = time.perf_counter() - start

    if force_web:
        decision["use_web"] = True

        if not decision.get("search_query"):
            decision["search_query"] = message

    if document_ids:
        decision["use_documents"] = True

    use_documents = decision["use_documents"]
    use_web = decision["use_web"]
    fresh = decision["fresh"]
    search_query = decision["search_query"] or message

    yield {
        "type": "plan",
        "documents": use_documents,
        "web": use_web,
        "fresh": fresh
    }

    if not use_documents and not use_web:
        yield {
            "type": "status",
            "stage": "generating",
            "label": "Writing"
        }

        start = time.perf_counter()

        yield from chat_events(message, history, inference)

        timings["generation"] = time.perf_counter() - start

    else:
        sources = []

        if use_documents:
            yield {
                "type": "status",
                "stage": "documents",
                "label": "Searching documents"
            }

            start = time.perf_counter()
            sources.extend(retrieve(search_query, document_ids=document_ids))
            timings["documents"] = time.perf_counter() - start

        if use_web:
            yield {
                "type": "status",
                "stage": "web",
                "label": (
                    "Searching recent sources"
                    if fresh
                    else "Searching the web"
                )
            }

            start = time.perf_counter()

            sources.extend(
                retrieve_web(
                    search_query,
                    fresh=fresh
                )
            )

            timings["web"] = time.perf_counter() - start

        if sources:
            yield {
                "type": "status",
                "stage": "reranking",
                "label": "Ranking sources"
            }

            start = time.perf_counter()
            sources = rerank(search_query, sources, top_k=6)
            timings["reranking"] = time.perf_counter() - start

        yield {
            "type": "sources",
            "sources": [
                serialize_source(source, index)
                for index, source in enumerate(sources, start=1)
            ]
        }

        yield {
            "type": "status",
            "stage": "generating",
            "label": "Writing"
        }

        start = time.perf_counter()

        yield from answer_events(message, sources, inference)

        timings["generation"] = time.perf_counter() - start

    timings["total"] = time.perf_counter() - total_start

    yield {
        "type": "done",
        "timings": {
            key: round(value, 3)
            for key, value in timings.items()
        }
    }
