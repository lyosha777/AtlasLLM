from dataclasses import dataclass
from typing import Optional


@dataclass
class Source:
    text: str
    source_type: str

    title: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    relevance_score: Optional[float] = None
    document_id: Optional[str] = None

    chunk_id: Optional[str] = None
    retrieval_score: Optional[float] = None
    publication_date: Optional[str] = None

    def citation(self) -> str:
        if self.source_type == "document":
            if self.page is not None:
                return f"{self.title}, page {self.page}"
            return self.title or "Document"

        if self.source_type == "web":
            return self.title or self.url or "Web source"

        return self.title or "Unknown source"
