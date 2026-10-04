import json
from uuid import uuid4

from backend import db


ACTIVE = ("queued", "running", "cancelling")
TERMINAL = ("completed", "cancelled", "error", "interrupted")
EVENT_FIELDS = {
    "job": (), "acknowledged": (), "status": ("stage", "label"),
    "plan": ("documents", "web", "fresh"), "token": ("content",), "sources": ("sources",),
    "inference": ("settings",), "done": ("timings",), "error": ("message",),
    "cancelled": (), "interrupted": ("message",), "cancelling": (),
}


class JobConflict(ValueError):
    pass


def _snapshot(connection, job_id):
    row = connection.execute(
        "SELECT generation_jobs.*, generation_jobs.rowid AS ordinal, chats.title AS chat_title FROM generation_jobs "
        "JOIN chats ON chats.id = generation_jobs.chat_id WHERE generation_jobs.id = ?", (job_id,),
    ).fetchone()
    if row is None:
        return None
    job = {key: row[key] for key in (
        "id", "ordinal", "chat_id", "chat_title", "status", "activity", "error", "last_event_id", "created_at", "updated_at",
    )}
    job["acknowledged"] = bool(row["acknowledged"])
    job["timings"] = json.loads(row["timings_json"])
    job["inference"] = json.loads(row["inference_json"]) if row["inference_json"] else None
    for role in ("user", "assistant"):
        message = connection.execute("SELECT * FROM messages WHERE id = ?", (row[f"{role}_message_id"],)).fetchone()
        job[f"{role}_message"] = db.serialize_message(message)
    job["assistant_message"].update({
        "generation_id": job_id, "generation_status": row["status"], "generation_revision": row["last_event_id"],
        "inference": job["inference"],
    })
    return job


def _event(connection, job_id, event):
    event = {"type": event["type"], **{key: event[key] for key in EVENT_FIELDS[event["type"]] if key in event}}
    row = connection.execute("SELECT chat_id FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    cursor = connection.execute(
        "INSERT INTO generation_events (job_id, chat_id, payload_json, created_at) VALUES (?, ?, '{}', ?)",
        (job_id, row["chat_id"], db.now()),
    )
    sequence = cursor.lastrowid
    connection.execute("UPDATE generation_jobs SET last_event_id = ? WHERE id = ?", (sequence, job_id))
    snapshot = _snapshot(connection, job_id)
    payload = {"id": sequence, "chat_id": row["chat_id"], "job": snapshot, "event": event}
    connection.execute("UPDATE generation_events SET payload_json = ? WHERE id = ?", (json.dumps(payload), sequence))
    return snapshot


def validate_retry(original, chat_id, message, force_web, document_ids, model, reasoning):
    if original["chat_id"] != chat_id:
        raise JobConflict("Request ID belongs to another chat.")
    request = json.loads(original["request_json"])
    if (request["message"] != message or request["force_web"] != force_web
            or (request["document_ids"] or []) != (document_ids or [])
            or request.get("selection", {"model": None, "reasoning": None}) != {"model": model, "reasoning": reasoning}):
        raise JobConflict("Request ID already used for a different message or model settings.")


def replay_request(request_id, **selection):
    if not request_id:
        return None
    with db.connection() as connection:
        connection.execute("BEGIN")
        existing = connection.execute("SELECT * FROM generation_jobs WHERE request_id = ?", (request_id,)).fetchone()
        if existing is None:
            return None
        validate_retry(existing, **selection)
        return _snapshot(connection, existing["id"])


def create_job(chat_id: str, message: str, force_web=False, document_ids=None, request_id=None,
               inference=None, model=None, reasoning=None):
    request_id = request_id or str(uuid4())
    with db.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute("SELECT * FROM generation_jobs WHERE request_id = ?", (request_id,)).fetchone()
        if existing:
            validate_retry(existing, chat_id, message, force_web, document_ids, model, reasoning)
            return _snapshot(connection, existing["id"]), False
        chat = connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if chat is None:
            raise LookupError("Chat not found")
        if chat["archived"]:
            raise JobConflict("Restore this thread before sending a message.")
        if connection.execute(
            "SELECT 1 FROM generation_jobs WHERE chat_id = ? AND status IN ('queued', 'running', 'cancelling')", (chat_id,),
        ).fetchone():
            raise JobConflict("This thread already has an active response.")
        attachments = []
        for document_id in dict.fromkeys(document_ids or []):
            document = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if document is None:
                raise LookupError("An attached document was not found.")
            attachments.append(dict(document))
        history = [dict(row) for row in connection.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? AND content != '' ORDER BY created_at, rowid", (chat_id,),
        )]
        timestamp = db.now()
        user_id, assistant_id, job_id = str(uuid4()), str(uuid4()), str(uuid4())
        for message_id, role, content, files in (
            (user_id, "user", message, attachments), (assistant_id, "assistant", "", []),
        ):
            connection.execute(
                "INSERT INTO messages (id, chat_id, role, content, sources_json, attachments_json, created_at) "
                "VALUES (?, ?, ?, ?, '[]', ?, ?)",
                (message_id, chat_id, role, content, json.dumps(files), timestamp),
            )
        request = {"message": message, "history": history, "force_web": force_web, "document_ids": document_ids or None}
        request["selection"] = {"model": model, "reasoning": reasoning}
        if inference is not None:
            request["inference"] = inference
        connection.execute(
            "INSERT INTO generation_jobs (id, request_id, chat_id, user_message_id, assistant_message_id, "
            "status, activity, request_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'queued', 'Queued', ?, ?, ?)",
            (job_id, request_id, chat_id, user_id, assistant_id, json.dumps(request), timestamp, timestamp),
        )
        connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, chat_id))
        connection.execute("UPDATE generation_jobs SET inference_json = ? WHERE id = ?",
                           (json.dumps(inference) if inference is not None else None, job_id))
        return _event(connection, job_id, {"type": "job"}), True


def get_job(job_id):
    with db.connection() as connection:
        connection.execute("BEGIN")
        return _snapshot(connection, job_id)


def get_request(job_id):
    with db.connection() as connection:
        row = connection.execute("SELECT request_json FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    request = json.loads(row["request_json"])
    request.pop("selection", None)
    return request


def snapshot():
    with db.connection() as connection:
        connection.execute("BEGIN")
        cursor = connection.execute("SELECT COALESCE(MAX(id), 0) FROM generation_events").fetchone()[0]
        rows = connection.execute(
            "SELECT id FROM generation_jobs AS job WHERE NOT EXISTS (SELECT 1 FROM generation_jobs AS newer "
            "WHERE newer.chat_id = job.chat_id AND newer.rowid > job.rowid)"
        ).fetchall()
        return {"cursor": cursor, "jobs": [_snapshot(connection, row["id"]) for row in rows]}


def events(after=0, job_id=None, limit=100):
    with db.connection() as connection:
        if job_id:
            rows = connection.execute(
                "SELECT payload_json FROM generation_events WHERE id > ? AND job_id = ? ORDER BY id LIMIT ?",
                (after, job_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT payload_json FROM generation_events WHERE id > ? ORDER BY id LIMIT ?", (after, limit),
            ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def update(job_id, event):
    if event["type"] not in EVENT_FIELDS:
        return None
    with db.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] in TERMINAL:
            return None
        kind = event["type"]
        if row["status"] == "cancelling" and kind not in ("cancelled", "interrupted"):
            return None
        timestamp = db.now()
        status, activity, error = row["status"], row["activity"], row["error"]
        if kind == "status":
            status, activity = "running", event["label"]
        elif kind == "token":
            connection.execute("UPDATE messages SET content = content || ? WHERE id = ?",
                               (event["content"], row["assistant_message_id"]))
            status, activity = "running", "Writing"
        elif kind == "sources":
            connection.execute("UPDATE messages SET sources_json = ? WHERE id = ?",
                               (json.dumps(event["sources"]), row["assistant_message_id"]))
        elif kind == "inference":
            inference = json.loads(row["inference_json"]) if row["inference_json"] else None
            if inference is not None:
                inference["actual_settings"] = event["settings"]
                connection.execute("UPDATE generation_jobs SET inference_json = ? WHERE id = ?",
                                   (json.dumps(inference), job_id))
        elif kind == "done":
            status, activity = "completed", None
            connection.execute("UPDATE generation_jobs SET timings_json = ? WHERE id = ?",
                               (json.dumps(event.get("timings", {})), job_id))
        elif kind in ("cancelled", "interrupted", "error"):
            status, activity, error = kind, None, event.get("message")
        elif kind == "cancelling":
            status, activity = "cancelling", "Stopping"
        connection.execute(
            "UPDATE generation_jobs SET status = ?, activity = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, activity, error, timestamp, job_id),
        )
        connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, row["chat_id"]))
        return _event(connection, job_id, event)


def acknowledge(job_id):
    with db.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT status, acknowledged FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        if row["status"] in TERMINAL and not row["acknowledged"]:
            connection.execute("UPDATE generation_jobs SET acknowledged = 1 WHERE id = ?", (job_id,))
            return _event(connection, job_id, {"type": "acknowledged"})
        return _snapshot(connection, job_id)


def interrupt_unfinished():
    with db.connection() as connection:
        ids = [row[0] for row in connection.execute(
            "SELECT id FROM generation_jobs WHERE status IN ('queued', 'running', 'cancelling')"
        )]
    for job_id in ids:
        update(job_id, {"type": "interrupted", "message": "Backend restarted before this response finished."})


def delete_chat(chat_id):
    with db.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if not connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount:
            return False
        connection.execute("DELETE FROM generation_events WHERE chat_id = ?", (chat_id,))
        cursor = connection.execute(
            "INSERT INTO generation_events (chat_id, payload_json, created_at) VALUES (?, '{}', ?)", (chat_id, db.now()),
        )
        connection.execute("UPDATE generation_events SET payload_json = ? WHERE id = ?", (
            json.dumps({"id": cursor.lastrowid, "chat_id": chat_id, "job": None, "event": {"type": "deleted"}}),
            cursor.lastrowid,
        ))
        return True
