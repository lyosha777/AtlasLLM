import json

from backend import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES generation_jobs(id) ON DELETE CASCADE,
 status TEXT NOT NULL DEFAULT 'queued', scope TEXT NOT NULL, objective TEXT,
 budgets_json TEXT NOT NULL, metrics_json TEXT NOT NULL DEFAULT '{}', analysis_json TEXT,
 revision INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_subquestions (
 run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE, id TEXT NOT NULL,
 payload_json TEXT NOT NULL, PRIMARY KEY(run_id, id)
);
CREATE TABLE IF NOT EXISTS research_searches (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
 subquestion_id TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_evidence (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
 chunk_id TEXT NOT NULL, passage TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 0,
 payload_json TEXT NOT NULL, UNIQUE(run_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS research_claims (
 run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE, id TEXT NOT NULL,
 statement TEXT NOT NULL, PRIMARY KEY(run_id, id)
);
CREATE TABLE IF NOT EXISTS research_claim_evidence (
 run_id TEXT NOT NULL, claim_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
 relation TEXT NOT NULL CHECK(relation IN ('support', 'conflict')),
 PRIMARY KEY(run_id, claim_id, evidence_id, relation),
 FOREIGN KEY(run_id, claim_id) REFERENCES research_claims(run_id, id) ON DELETE CASCADE,
 FOREIGN KEY(evidence_id) REFERENCES research_evidence(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_research_search_run ON research_searches(run_id);
CREATE INDEX IF NOT EXISTS idx_research_evidence_run ON research_evidence(run_id);
"""

EVENTS = {name: ("run_id", "subquestion_id", "search_id", "count", "metrics") for name in (
    "research_started", "research_plan_ready", "subquestion_started", "search_started", "search_completed",
    "evidence_selected", "coverage_check", "followup_search", "synthesis_started", "research_completed", "research_depth_changed",
)}


def summary(connection, job_id):
    row = connection.execute("SELECT * FROM research_runs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return dict(id=row["id"], status=row["status"], scope=row["scope"], revision=row["revision"],
                metrics=json.loads(row["metrics_json"]), error=row["error"])


def apply(connection, job_id, event):
    row = connection.execute("SELECT * FROM research_runs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None or event.get("run_id") != row["id"]:
        raise ValueError("Research event does not belong to this job")
    run_id, kind = row["id"], event["type"]
    artifact = event.get("artifact", {})
    if kind == "research_plan_ready":
        connection.execute("UPDATE research_runs SET objective = ? WHERE id = ?", (artifact["objective"], run_id))
        for item in artifact["subquestions"]:
            connection.execute("INSERT INTO research_subquestions VALUES (?, ?, ?)",
                               (run_id, item["id"], json.dumps(item)))
    elif kind in ("search_started", "search_completed"):
        search = artifact["search"]
        connection.execute("INSERT INTO research_searches VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE "
                           "SET payload_json = excluded.payload_json",
                           (search["id"], run_id, search["subquestion_id"], json.dumps(search)))
        for evidence in artifact.get("evidence", []):
            connection.execute("INSERT INTO research_evidence VALUES (?, ?, ?, ?, 0, ?) "
                               "ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json",
                               (evidence["id"], run_id, evidence["chunk_id"], evidence["passage"], json.dumps(evidence)))
    elif kind == "evidence_selected":
        connection.execute("UPDATE research_evidence SET selected = 0 WHERE run_id = ?", (run_id,))
        for evidence_id in artifact["ids"]:
            connection.execute("UPDATE research_evidence SET selected = 1 WHERE id = ? AND run_id = ?",
                               (evidence_id, run_id))
    elif kind == "coverage_check":
        connection.execute("UPDATE research_runs SET analysis_json = ? WHERE id = ?", (json.dumps(artifact), run_id))
        connection.execute("DELETE FROM research_claims WHERE run_id = ?", (run_id,))
        for claim in artifact["claims"]:
            connection.execute("INSERT INTO research_claims VALUES (?, ?, ?)", (run_id, claim["id"], claim["statement"]))
            for field, relation in (("supporting_evidence_ids", "support"), ("conflicting_evidence_ids", "conflict")):
                for evidence_id in dict.fromkeys(claim[field]):
                    connection.execute("INSERT INTO research_claim_evidence VALUES (?, ?, ?, ?)",
                                       (run_id, claim["id"], evidence_id, relation))
    connection.execute("UPDATE research_runs SET status = 'running', revision = revision + 1, updated_at = ? WHERE id = ?",
                       (db.now(), run_id))
    if "metrics" in event:
        connection.execute("UPDATE research_runs SET metrics_json = ? WHERE id = ?", (json.dumps(event["metrics"]), run_id))
        if event["metrics"].get("depth"):
            from backend.research.schemas import budgets
            connection.execute("UPDATE research_runs SET budgets_json = ? WHERE id = ?",
                               (json.dumps(budgets(event["metrics"]["depth"])), run_id))


def finish(connection, job_id, status, error=None):
    row = connection.execute("SELECT * FROM research_runs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return
    from datetime import datetime
    timestamp = db.now()
    metrics = json.loads(row["metrics_json"])
    metrics["total_seconds"] = round((datetime.fromisoformat(timestamp) - datetime.fromisoformat(row["created_at"])).total_seconds(), 3)
    connection.execute("UPDATE research_runs SET metrics_json = ? WHERE id = ?", (json.dumps(metrics), row["id"]))
    for search in connection.execute("SELECT id, payload_json FROM research_searches WHERE run_id = ?", (row["id"],)).fetchall():
        payload = json.loads(search["payload_json"])
        if payload["status"] == "running":
            payload.update(status=status, completed_at=timestamp)
            connection.execute("UPDATE research_searches SET payload_json = ? WHERE id = ?", (json.dumps(payload), search["id"]))
    connection.execute("UPDATE research_runs SET status = ?, error = ?, revision = revision + 1, updated_at = ? "
                       "WHERE job_id = ?", ("failed" if status == "error" else status, error, timestamp, job_id))


def get_run(run_id):
    with db.connection() as connection:
        connection.execute("BEGIN")
        row = connection.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        for field in ("budgets", "metrics", "analysis"):
            result[field] = json.loads(result.pop(field + "_json") or "null")
        for field, table in (("subquestions", "research_subquestions"), ("searches", "research_searches"),
                             ("evidence", "research_evidence")):
            rows = connection.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
            result[field] = []
            for item in rows:
                payload = json.loads(item["payload_json"])
                if field == "evidence":
                    payload.update(passage=item["passage"], selected_for_synthesis=bool(item["selected"]))
                result[field].append(payload)
        return result
