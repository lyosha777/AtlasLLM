"""Check installed models through real jobs without retaining private thinking."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import ollama

from backend import db
from backend.tests.smoke_local import local_client


def main():
    original_chat = ollama.chat
    hidden_chunks = Counter()

    def observed_chat(**settings):
        result = original_chat(**settings)
        if not settings.get("stream"):
            return result

        def observe():
            try:
                for chunk in result:
                    if chunk.get("message", {}).get("thinking"):
                        hidden_chunks[settings["model"]] += 1
                    yield chunk
            finally:
                result.close()

        return observe()

    with tempfile.TemporaryDirectory() as directory, patch.object(db, "DB_PATH", Path(directory) / "models.db"), patch.object(ollama, "chat", side_effect=observed_chat), local_client() as client:
        catalog = client.get("/api/models").json()
        assert catalog["online"], catalog["error"]
        for model in catalog["models"]:
            if not model["available"] or not model["can_chat"]:
                continue
            print(json.dumps({key: model[key] for key in ("name", "family", "context_length", "reasoning")}), flush=True)
            chat = client.post("/api/chats", json={"title": model["name"]}).json()["id"]
            start = time.perf_counter()
            response = client.post(f"/api/chats/{chat}/generations", json={
                "message": "Compute 2+2. Reply only with the answer. Do not search.", "model": model["name"],
            })
            response.raise_for_status()
            submitted = response.json()
            events = []
            with client.stream("GET", f"/api/generations/{submitted['id']}/events") as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
            final = events[-1]["job"]
            assert final["status"] == "completed", final["error"]
            assert final["inference"]["actual_settings"]["model"] == model["name"]
            assert final["assistant_message"]["content"].strip()
            assert all("thinking" not in event["event"] for event in events)
            assert "<think>" not in json.dumps(events)
            messages = client.get(f"/api/chats/{chat}/messages").json()
            assert len(messages) == 2
            assert messages[1]["inference"] == final["inference"]
            print(json.dumps({"model": model["name"], "elapsed_seconds": round(time.perf_counter() - start, 2),
                              "actual_settings": final["inference"]["actual_settings"],
                              "hidden_chunks_discarded": hidden_chunks[model["name"]],
                              "answer": final["assistant_message"]["content"]}), flush=True)
            if "high" not in model["reasoning"]["settings"]:
                unsupported = client.post(f"/api/chats/{chat}/generations", json={
                    "message": "No fake effort", "model": model["name"], "reasoning": "high",
                })
                assert unsupported.status_code == 422
                assert len(client.get(f"/api/chats/{chat}/messages").json()) == 2
        print("Installed-model smoke checks passed; existing Atlas data was not changed.", flush=True)


if __name__ == "__main__":
    main()
