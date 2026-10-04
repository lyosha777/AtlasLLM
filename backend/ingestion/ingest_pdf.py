import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader

from backend.core.config import DOCUMENTS_DIR
from backend.core.document_store import get_document_collection, get_embedding_model


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(text: str) -> list[str]:
    return [
        chunk
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)
        if (chunk := text[start:start + CHUNK_SIZE].strip())
    ]


def ingest_pdf(
    pdf_path: Path,
    document_id: str | None = None,
    filename: str | None = None,
) -> dict:
    legacy_import = document_id is None
    document_id = document_id or str(uuid5(NAMESPACE_URL, pdf_path.resolve().as_uri()))
    filename = filename or pdf_path.name
    reader = PdfReader(pdf_path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("Password-protected PDFs are not supported.")

    texts = []
    metadatas = []
    ids = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk_number, chunk in enumerate(chunk_text(page.extract_text() or "")):
            texts.append(chunk)
            ids.append(f"{document_id}:page:{page_number}:chunk:{chunk_number}")
            metadatas.append({
                "document_id": document_id,
                "source": filename,
                "page": page_number,
                "chunk": chunk_number,
            })
            if len(texts) > 10000:
                raise ValueError("This PDF contains too much text to index in one upload.")

    if not texts:
        raise ValueError("No readable text found. Scanned PDFs need OCR before uploading.")

    embeddings = get_embedding_model().encode(texts).tolist()
    collection = get_document_collection()
    previous_ids = collection.get(where={"document_id": document_id}, include=[])["ids"]
    if legacy_import:
        legacy_ids = collection.get(where={"source": filename}, include=[])["ids"]
        pattern = rf"{re.escape(pdf_path.stem)}_page_\d+_chunk_\d+"
        previous_ids.extend(chunk_id for chunk_id in legacy_ids if re.fullmatch(pattern, chunk_id))
    for start in range(0, len(texts), 128):
        end = start + 128
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    stale_ids = sorted(set(previous_ids) - set(ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "id": document_id,
        "name": filename,
        "size": pdf_path.stat().st_size,
        "page_count": len(reader.pages),
        "chunk_count": len(texts),
    }


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in documents/")
        return
    for pdf in pdf_files:
        result = ingest_pdf(pdf)
        print(f"{pdf.name}: stored {result['chunk_count']} chunks.")
    print(f"Atlas document database contains {get_document_collection().count()} chunks.")


if __name__ == "__main__":
    main()
