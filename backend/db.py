import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from backend.core.config import DB_PATH


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")

    try:
        with db:
            yield db
    finally:
        db.close()


def init_db() -> None:
    with connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat
            ON messages(chat_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_chats_updated
            ON chats(updated_at);

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                size INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generation_jobs (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                user_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                activity TEXT,
                error TEXT,
                request_json TEXT NOT NULL,
                timings_json TEXT NOT NULL DEFAULT '{}',
                acknowledged INTEGER NOT NULL DEFAULT 0,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_active_generation_chat
            ON generation_jobs(chat_id)
            WHERE status IN ('queued', 'running', 'cancelling');
            CREATE TABLE IF NOT EXISTS generation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                chat_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_generation_events_job
            ON generation_events(job_id, id);
            """
        )
        from backend.research.store import SCHEMA
        db.executescript(SCHEMA)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
        if "attachments_json" not in columns:
            db.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'")
        job_columns = {row["name"] for row in db.execute("PRAGMA table_info(generation_jobs)")}
        if "inference_json" not in job_columns:
            db.execute("ALTER TABLE generation_jobs ADD COLUMN inference_json TEXT")


def get_document(document_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def save_document(document: dict) -> dict:
    with connection() as db:
        db.execute(
            """
            INSERT INTO documents (id, name, size, page_count, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name, size = excluded.size,
                page_count = excluded.page_count, chunk_count = excluded.chunk_count
            """,
            (document["id"], document["name"], document["size"],
             document["page_count"], document["chunk_count"], now()),
        )
    return get_document(document["id"])


def create_chat(title: str) -> dict:
    chat_id = str(uuid4())
    timestamp = now()

    with connection() as db:
        db.execute(
            """
            INSERT INTO chats (
                id,
                title,
                archived,
                created_at,
                updated_at
            )
            VALUES (?, ?, 0, ?, ?)
            """,
            (
                chat_id,
                title,
                timestamp,
                timestamp
            )
        )

    return get_chat(chat_id)


def get_chat(chat_id: str) -> dict | None:
    with connection() as db:
        row = db.execute(
            """
            SELECT *
            FROM chats
            WHERE id = ?
            """,
            (chat_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "title": row["title"],
        "archived": bool(row["archived"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def list_chats(archived: bool = False) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT
                chats.*,
                COUNT(messages.id) AS message_count
            FROM chats
            LEFT JOIN messages
                ON messages.chat_id = chats.id
            WHERE chats.archived = ?
            GROUP BY chats.id
            ORDER BY chats.updated_at DESC
            """,
            (int(archived),)
        ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "archived": bool(row["archived"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"]
        }
        for row in rows
    ]


def update_chat(
    chat_id: str,
    title: str | None = None,
    archived: bool | None = None
) -> dict | None:

    current = get_chat(chat_id)

    if current is None:
        return None

    new_title = title if title is not None else current["title"]
    new_archived = (
        archived
        if archived is not None
        else current["archived"]
    )

    with connection() as db:
        db.execute(
            """
            UPDATE chats
            SET title = ?,
                archived = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                new_title,
                int(new_archived),
                now(),
                chat_id
            )
        )

    return get_chat(chat_id)


def delete_chat(chat_id: str) -> bool:
    with connection() as db:
        cursor = db.execute(
            """
            DELETE FROM chats
            WHERE id = ?
            """,
            (chat_id,)
        )

    return cursor.rowcount > 0


def list_messages(chat_id: str) -> list[dict]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT messages.*, generation_jobs.id AS generation_id,
                generation_jobs.status AS generation_status,
                generation_jobs.last_event_id AS generation_revision,
                generation_jobs.inference_json AS inference_json,
                generation_jobs.timings_json AS timings_json,
                research_runs.id AS research_run_id, research_runs.revision AS research_revision
            FROM messages LEFT JOIN generation_jobs
                ON generation_jobs.assistant_message_id = messages.id
            LEFT JOIN research_runs ON research_runs.job_id = generation_jobs.id
            WHERE messages.chat_id = ?
            ORDER BY messages.created_at ASC, messages.rowid ASC
            """,
            (chat_id,)
        ).fetchall()

    return [serialize_message(row) for row in rows]


def serialize_message(row) -> dict:
    try:
        sources = json.loads(row["sources_json"] or "[]")
    except json.JSONDecodeError:
        sources = []
    message = {
        "id": row["id"], "chat_id": row["chat_id"], "role": row["role"],
        "content": row["content"], "sources": sources,
        "attachments": json.loads(row["attachments_json"] or "[]"),
        "created_at": row["created_at"],
    }
    if "generation_id" in row.keys():
        message.update({
            "generation_id": row["generation_id"],
            "generation_status": row["generation_status"],
            "generation_revision": row["generation_revision"] or 0,
            "inference": json.loads(row["inference_json"]) if row["inference_json"] else None,
        })
    if "research_run_id" in row.keys():
        message.update(research_run_id=row["research_run_id"], research_revision=row["research_revision"])
    if "timings_json" in row.keys():
        message["timings"] = json.loads(row["timings_json"] or "{}")
    return message


def add_message(
    chat_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    attachments: list[dict] | None = None,
) -> dict:

    message_id = str(uuid4())
    timestamp = now()

    with connection() as db:
        db.execute(
            """
            INSERT INTO messages (
                id,
                chat_id,
                role,
                content,
                sources_json,
                attachments_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                chat_id,
                role,
                content,
                json.dumps(sources or []),
                json.dumps(attachments or []),
                timestamp
            )
        )

        db.execute(
            """
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                timestamp,
                chat_id
            )
        )

    return {
        "id": message_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "attachments": attachments or [],
        "created_at": timestamp
    }
