import json
import sys
import tempfile
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn

from backend import api, db
from backend.core import document_store
from backend.tests.test_research import pdf_bytes
from backend.tests.integration_timeout import run_with_deadline


@contextmanager
def local_client():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Temporary API server did not start.")
            time.sleep(0.05)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


def send(client, chat_id, message, **options):
    events = []
    first_token = None
    start = time.perf_counter()
    with client.stream("POST", f"/api/chats/{chat_id}/messages/stream",
                       json={"message": message, **options}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if event["type"] == "token" and first_token is None:
                first_token = time.perf_counter() - start
                print(f"First token received over HTTP after {first_token:.2f}s", flush=True)
            events.append(event)
    errors = [event["message"] for event in events if event["type"] == "error"]
    if errors:
        raise RuntimeError(errors)
    assert events[-1]["type"] == "done", events
    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert tokens
    print(f"{message}: {len(tokens)} tokens; {events[-1]['timings']}", flush=True)
    print("".join(tokens)[:350], flush=True)
    return events


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        with (
            patch.object(db, "DB_PATH", root / "atlas.db"),
            patch.object(api, "UPLOADS_DIR", root / "documents"),
            patch.object(document_store, "CHROMA_PATH", root / "chroma"),
            local_client() as client,
        ):
            assert client.get("/api/health").json()["status"] == "ok"
            chat_id = client.post("/api/chats", json={"title": "Local smoke test"}).json()["id"]
            send(client, chat_id, "Say hello in four words. Do not search.")
            uploaded = client.post("/api/documents", files={
                "file": ("networking.pdf", pdf_bytes(
                    "TCP provides reliable ordered delivery using acknowledgements and retransmissions. "
                    "UDP sends datagrams without guaranteeing delivery or order."
                ), "application/pdf"),
            })
            uploaded.raise_for_status()
            document = uploaded.json()
            events = send(client, chat_id, "According to this PDF, how does TCP ensure delivery? One sentence.",
                          document_ids=[document["id"]])
            sources = next(event["sources"] for event in events if event["type"] == "sources")
            assert sources and all(source["document_id"] == document["id"] for source in sources)
            messages = client.get(f"/api/chats/{chat_id}/messages").json()
            assert messages[-2]["attachments"][0]["id"] == document["id"]
            if "--web" in sys.argv:
                events = send(client, chat_id, "Who invented the telephone? Answer in one sentence.", force_web=True)
                plan = next(event for event in events if event["type"] == "plan")
                assert plan["web"] and not plan["fresh"]
                sources = next(event["sources"] for event in events if event["type"] == "sources")
                assert any(source["type"] == "web" for source in sources)
            print("Local smoke checks passed; existing Atlas data was not used or changed.", flush=True)


if __name__ == "__main__":
    run_with_deadline(main, 300)
