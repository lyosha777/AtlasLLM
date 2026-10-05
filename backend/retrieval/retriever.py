from backend.core.document_store import get_document_collection, get_embedding_model
from backend.core.source import Source


def retrieve(
    query: str,
    n_results: int = 5,
    document_ids: list[str] | None = None,
) -> list[Source]:
    collection = get_document_collection()
    count = collection.count()
    if not count or n_results < 1 or document_ids == []:
        return []

    options = {}
    if document_ids:
        options["where"] = {"document_id": {"$in": document_ids}}
    query_embedding = get_embedding_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
        **options,
    )
    return [
        Source(
            text=document,
            source_type="document",
            title=metadata["source"],
            page=metadata["page"],
            relevance_score=1 / (1 + distance),
            document_id=metadata.get("document_id"),
            chunk_id=chunk_id, retrieval_score=1 / (1 + distance),
        )
        for document, metadata, distance, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results.get("ids", [[None] * len(results["documents"][0])])[0],
        )
    ]
