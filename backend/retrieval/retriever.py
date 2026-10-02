import chromadb
from sentence_transformers import SentenceTransformer
from backend.core.config import EMBEDDING_MODEL

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path="./data/chroma")

collection = client.get_collection(
    name="atlas_documents"
)


def retrieve(query: str, n_results: int = 5):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_chunks.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved_chunks