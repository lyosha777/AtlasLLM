from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from backend.core.config import CHROMA_PATH, DOCUMENT_COLLECTION, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_document_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=DOCUMENT_COLLECTION)
