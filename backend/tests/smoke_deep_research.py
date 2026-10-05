"""Real Ollama/web/PDF research against temporary Atlas stores."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend import api, db
from backend.core import document_store
from backend.models.ollama_provider import provider
from backend.tests.smoke_local import local_client
from backend.tests.test_research import pdf_bytes
from backend.tests.integration_timeout import run_with_deadline


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        document_store.get_document_collection.cache_clear()
        with (patch.object(db, "DB_PATH", root / "atlas.db"),
              patch.object(api, "UPLOADS_DIR", root / "documents"),
              patch.object(document_store, "CHROMA_PATH", root / "chroma"), local_client() as client):
            catalog = provider.discover()
            assert catalog["online"], catalog["error"]
            model = next(item["name"] for item in catalog["models"] if item["available"] and item["can_chat"]
                         and item["reasoning"]["kind"] != "fixed")
            chat = client.post("/api/chats", json={"title": "Isolated Deep Research smoke"}).json()["id"]
            uploaded = client.post("/api/documents", files={"file": ("transport.pdf", pdf_bytes(
                "TCP provides reliable ordered delivery using sequence numbers, acknowledgements and retransmissions. "
                "UDP sends independent datagrams without guaranteeing delivery or order. "
                "Applications can implement their own reliability above UDP. TCP uses a connection and flow control. "
                "UDP can be useful for latency-sensitive real-time traffic where occasional losses are tolerable."
            ), "application/pdf")})
            uploaded.raise_for_status()
            document_id = uploaded.json()["id"]
            submitted = client.post(f"/api/chats/{chat}/generations", json=dict(
                message="Compare how TCP and UDP handle delivery reliability, ordering, and suitable applications. Use this PDF and primary web references. Keep the answer concise.",
                research_mode=True, research_scope="both", document_ids=[document_id], model=model))
            submitted.raise_for_status()
            job_id = submitted.json()["id"]
            events = []
            with client.stream("GET", f"/api/generations/{job_id}/events", timeout=90) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if line.startswith("data: "):
                        item = json.loads(line[6:])
                        events.append(item)
                        if item["event"]["type"] in ("research_plan_ready", "search_completed", "coverage_check", "synthesis_started", "done", "error"):
                            print(json.dumps(item["event"]), flush=True)
            final = client.get(f"/api/generations/{job_id}").json()
            run = client.get(f"/api/research/{final['research']['id']}").json()
            assert final["status"] == "completed", final["error"]
            assert {item["source_type"] for item in run["evidence"]} == {"web", "document"}, "Smoke requires real extracted web and PDF evidence"
            assert all(item["document_id"] == document_id for item in run["evidence"] if item["source_type"] == "document")
            ids = {item["id"] for item in run["evidence"] if item["selected_for_synthesis"]}
            assert all(item["evidence_id"] in ids for item in final["assistant_message"]["sources"])
            assert "[SOURCE " in final["assistant_message"]["content"]
            assert all("thinking" not in item["event"] for item in events)
            assert "<think>" not in json.dumps(events)
            print(json.dumps(dict(model=model, status=run["status"], metrics=run["metrics"],
                                  sources=len(final["assistant_message"]["sources"]), answer=final["assistant_message"]["content"])), flush=True)
            print("Real Deep Research smoke passed; production chats, PDFs, and Chroma were untouched.", flush=True)
        document_store.get_document_collection.cache_clear()


if __name__ == "__main__":
    run_with_deadline(main, 300)
