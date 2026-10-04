from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "atlas.db"
CHROMA_PATH = DATA_DIR / "chroma"
UPLOADS_DIR = DATA_DIR / "documents"
DOCUMENTS_DIR = PROJECT_DIR / "documents"
DOCUMENT_COLLECTION = "atlas_documents"
MAX_PDF_BYTES = 25 * 1024 * 1024

CHAT_MODEL = "qwen3:4b-instruct"
PLANNER_MODEL = "qwen3:4b-instruct"
GENERATION_MODEL = "qwen3:4b-instruct"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
