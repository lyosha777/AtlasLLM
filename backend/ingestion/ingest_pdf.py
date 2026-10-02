from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = Path("documents")
CHROMA_PATH = "./data/chroma"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="atlas_documents"
)


def chunk_text(text: str):
    """Split text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_pdf(pdf_path: Path):

    print(f"\nReading {pdf_path.name}...")

    reader = PdfReader(pdf_path)

    total_chunks = 0

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            embedding = embedding_model.encode(chunk).tolist()

            chunk_id = (
                f"{pdf_path.stem}_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number
                }]
            )

            total_chunks += 1

    print(f"Stored {total_chunks} chunks.")


def main():

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in documents/")
        return

    for pdf in pdf_files:
        ingest_pdf(pdf)

    print(
        f"\nAtlas document database now contains "
        f"{collection.count()} chunks."
    )


if __name__ == "__main__":
    main()