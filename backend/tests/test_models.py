import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import ollama

from backend import api, db
from backend.core import assistant
from backend.core.source import Source
from backend.generation import store
from backend.models.capabilities import detect_reasoning
from backend.models.ollama_provider import AnswerFilter, InferenceError, OllamaProvider, ProviderUnavailable, provider


SECRET = "PRIVATE_TRACE_MUST_NEVER_LEAVE_PROVIDER"


class FakeProvider(OllamaProvider):
    def __init__(self):
        self.metadata = {
            "plain:latest": {"capabilities": ["completion"]},
            "binary:latest": {"thinking": {"values": [False, True], "default": False}},
            "effort:latest": {"thinking": {"values": ["low", "medium", "high"], "default": "medium"}},
            "fixed:latest": {"thinking": {"values": [True], "default": True}},
            "unverified:latest": {"capabilities": ["completion", "thinking"]},
        }
        for name, metadata in self.metadata.items():
            metadata.setdefault("capabilities", ["completion", "thinking"])
            metadata.update(details={"family": "test-family", "parameter_size": "1B", "quantization_level": "Q4"},
                            model_info={"general.architecture": "test-family", "test-family.context_length": 8192},
                            template=SECRET, messages=[{"thinking": SECRET}])

    def request(self, path, payload=None):
        if path == "/api/version":
            return {"version": "test-version"}
        if path == "/api/tags":
            return {"models": [{"name": name, "digest": name + "-digest", "size": 123} for name in self.metadata]}
        return self.metadata[payload["model"]]


class CapabilityTests(unittest.TestCase):
    def test_none_fixed_binary_effort_and_unknown_are_distinct(self):
        fake = FakeProvider()
        expected = {"plain": "none", "binary": "binary", "effort": "effort", "fixed": "fixed", "unverified": "unverified"}
        for name, kind in expected.items():
            self.assertEqual(fake.inspect(name)["reasoning"]["kind"], kind)
        self.assertEqual(fake.resolve("effort")["settings"], {"think": "medium"})
        self.assertEqual(fake.resolve("fixed")["settings"], {"think": True})
        self.assertEqual(fake.resolve("plain")["settings"], {})
        self.assertEqual(fake.resolve("unverified")["settings"], {})

    def test_all_unsupported_controls_are_rejected(self):
        fake = FakeProvider()
        for model, requested in (("plain", "high"), ("binary", "high"), ("fixed", "off"), ("unverified", "on"), ("effort", "extreme")):
            with self.assertRaises(ValueError):
                fake.resolve(model, requested)
        self.assertEqual(fake.resolve("binary", "off")["settings"], {"think": False})
        self.assertEqual(fake.resolve("binary", "on")["settings"], {"think": True})

    def test_malformed_and_mixed_metadata_never_prove_controls(self):
        for values in ([True, "high"], [1, 2], [], [SECRET], None):
            capability = detect_reasoning({"capabilities": ["completion", "thinking"], "thinking": {"values": values}})
            self.assertEqual(capability.kind, "unverified")
            self.assertFalse(capability.settings)
        unsupported = detect_reasoning({"thinking": {"values": [False], "default": False}})
        self.assertEqual(unsupported.kind, "none")
        self.assertEqual(unsupported.resolve(), ("off", {"think": False}))

    def test_partial_named_levels_are_not_invented(self):
        detected = detect_reasoning({"thinking": {"values": ["low", "high"], "default": "high"}})
        self.assertEqual(detected.settings, ("low", "high"))
        with self.assertRaises(ValueError):
            detected.resolve("medium")

    def test_named_controls_keep_exact_provider_types_and_medium_default(self):
        named = detect_reasoning({"thinking": {"values": ["on", "off"], "default": "off"}})
        self.assertEqual(named.resolve("on"), ("on", {"think": "on"}))
        effort = detect_reasoning({"thinking": {"values": ["low", "medium", "high"], "default": "high"}})
        self.assertEqual(effort.resolve(), ("medium", {"think": "medium"}))
        self.assertEqual(effort.provider_default, "high")

    def test_inline_reasoning_filter_handles_every_chunk_boundary(self):
        content = f"Before<think>{SECRET}</think>After"
        for boundary in range(len(content) + 1):
            filter_ = AnswerFilter()
            visible = filter_.feed(content[:boundary]) + filter_.feed(content[boundary:]) + filter_.feed("", final=True)
            self.assertEqual(visible, "BeforeAfter")
        filter_ = AnswerFilter()
        self.assertEqual(filter_.feed("<think>" + SECRET) + filter_.feed("", final=True), "")

    def test_inference_errors_do_not_echo_provider_response_bodies(self):
        fake = FakeProvider()
        inference = fake.resolve("effort", "high")
        with patch.object(ollama, "chat", side_effect=ollama.ResponseError(SECRET, 500)):
            with self.assertRaises(InferenceError) as error:
                list(fake.stream([], inference))
        self.assertNotIn(SECRET, str(error.exception))

    def test_sdk_validation_errors_cannot_echo_raw_private_payloads(self):
        fake = FakeProvider()
        with patch.object(ollama, "chat", side_effect=ValueError(SECRET)):
            with self.assertRaises(InferenceError) as error:
                list(fake.stream([], fake.resolve("effort", "high")))
        self.assertNotIn(SECRET, str(error.exception))

    def test_model_changes_after_submission_are_rejected(self):
        fake = FakeProvider()
        inference = fake.resolve("binary", "on")
        fake.metadata["binary:latest"]["thinking"] = {"values": [True], "default": True}
        with self.assertRaises(InferenceError):
            list(fake.stream([], inference))


class ModelApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(db, "DB_PATH", Path(self.directory) / "models.db"))
        self.fake = FakeProvider()
        self.enterContext(patch.object(provider, "request", side_effect=self.fake.request))
        self.calls = []
        self.gates = {}
        self.enterContext(patch.object(ollama, "chat", side_effect=self.chat))
        self.client = self.enterContext(TestClient(api.app))
        self.addCleanup(lambda: [gate.set() for gate in self.gates.values()])

    def chat(self, **settings):
        if not settings.get("stream"):
            return {"message": {"content": json.dumps({"use_documents": False, "use_web": False, "fresh": False}), "thinking": SECRET}}
        self.calls.append(settings)

        def stream():
            yield {"message": {"thinking": SECRET, "content": ""}}
            gate = self.gates.get(settings["model"])
            while gate and not gate.is_set():
                yield {"message": {"thinking": SECRET, "content": ""}}
                time.sleep(0.01)
            yield {"message": {"content": "<thi"}}
            yield {"message": {"content": "nk>" + SECRET + "</think>Safe answer"}}

        return stream()

    def new_chat(self):
        return self.client.post("/api/chats", json={"title": "Capability test"}).json()["id"]

    def submit(self, chat, **options):
        response = self.client.post(f"/api/chats/{chat}/generations", json={"message": "Question", **options})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def wait_for(self, job, status="completed"):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            current = store.get_job(job["id"])
            if current["status"] == status:
                return current
            if current["status"] == "error":
                self.fail(current["error"])
            time.sleep(0.01)
        self.fail(f"Job did not reach {status}: {store.get_job(job['id'])}")

    def test_discovery_exposes_metadata_without_templates_or_traces(self):
        catalog = self.client.get("/api/models").json()
        self.assertTrue(catalog["online"])
        self.assertEqual(len(catalog["models"]), 5)
        model = catalog["models"][2]
        self.assertEqual(model["family"], "test-family")
        self.assertEqual(model["context_length"], 8192)
        self.assertEqual(model["reasoning"]["settings"], ["low", "medium", "high"])
        self.assertNotIn(SECRET, json.dumps(catalog))

    def test_offline_discovery_and_submission_are_graceful(self):
        with patch.object(provider, "request", side_effect=ProviderUnavailable("Ollama is unavailable.")):
            catalog = self.client.get("/api/models")
            self.assertEqual(catalog.status_code, 200)
            self.assertFalse(catalog.json()["online"])
            chat = self.new_chat()
            response = self.client.post(f"/api/chats/{chat}/generations", json={"message": "Question", "model": "plain"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(db.list_messages(chat), [])

    def test_unsupported_selection_creates_no_messages(self):
        chat = self.new_chat()
        for model, reasoning in (("plain", "high"), ("binary", "medium"), ("fixed", "off"), ("unverified", "high")):
            response = self.client.post(f"/api/chats/{chat}/generations", json={"message": "Question", "model": model, "reasoning": reasoning})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(db.list_messages(chat), [])
        self.assertEqual(self.client.post(f"/api/chats/{chat}/generations", json={"message": "Question", "model": "missing"}).status_code, 404)

    def test_binary_and_named_controls_reach_real_inference_arguments(self):
        for model, requested, actual in (("binary", "off", False), ("binary", "on", True), ("effort", "low", "low"), ("effort", "medium", "medium"), ("effort", "high", "high")):
            chat = self.new_chat()
            final = self.wait_for(self.submit(chat, model=model, reasoning=requested))
            call = self.calls[-1]
            self.assertEqual(call["model"], model + ":latest")
            self.assertEqual(call["think"], actual)
            self.assertEqual(call["options"], {"temperature": 0.3, "num_predict": 600})
            self.assertEqual(final["inference"]["actual_settings"]["think"], actual)
            self.assertEqual(final["inference"]["requested_reasoning"], requested)

    def test_medium_is_default_only_when_supported(self):
        final = self.wait_for(self.submit(self.new_chat(), model="effort"))
        self.assertIsNone(final["inference"]["requested_reasoning"])
        self.assertEqual(final["inference"]["reasoning_mode"], "medium")
        fixed = self.wait_for(self.submit(self.new_chat(), model="fixed"))
        self.assertEqual(fixed["inference"]["reasoning_mode"], "on")
        self.assertEqual(fixed["inference"]["actual_settings"]["think"], True)

    def test_rag_and_forced_web_reuse_provider_settings_and_citations(self):
        source = Source("Grounded evidence", "web", title="Reference", url="https://example.org")
        with patch.object(assistant, "retrieve_web", return_value=[source]), patch.object(assistant, "retrieve", return_value=[source]), patch.object(assistant, "rerank", side_effect=lambda query, sources, top_k: sources):
            chat = self.new_chat()
            db.save_document({"id": "document", "name": "notes.pdf", "size": 12, "page_count": 1, "chunk_count": 1})
            final = self.wait_for(self.submit(chat, model="effort", reasoning="high", force_web=True, document_ids=["document"]))
        self.assertEqual(self.calls[-1]["think"], "high")
        self.assertEqual(self.calls[-1]["options"]["temperature"], 0)
        self.assertTrue(final["assistant_message"]["sources"])
        self.assertEqual(final["user_message"]["attachments"][0]["id"], "document")

    def test_concurrent_jobs_keep_distinct_settings_and_reconnect_metadata(self):
        for model in ("effort:latest", "binary:latest"):
            self.gates[model] = threading.Event()
        a, b = self.new_chat(), self.new_chat()
        job_a = self.submit(a, model="effort", reasoning="high")
        job_b = self.submit(b, model="binary", reasoning="off")
        deadline = time.monotonic() + 4
        while len(self.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(job["status"] in store.ACTIVE for job in store.snapshot()["jobs"]))
        self.assertEqual(self.client.get(f"/api/chats/{b}/messages").json()[1]["inference"]["requested_reasoning"], "off")
        self.gates["effort:latest"].set()
        self.gates["binary:latest"].set()
        self.wait_for(job_a)
        self.wait_for(job_b)
        snapshot = self.client.get("/api/generations").json()
        settings = {job["chat_id"]: job["inference"]["actual_settings"]["think"] for job in snapshot["jobs"]}
        self.assertEqual(settings, {a: "high", b: False})
        replay = self.client.get(f"/api/generations/{job_a['id']}/events").text
        self.assertIn('"requested_reasoning": "high"', replay)
        self.assertNotIn(SECRET, replay)

    def test_stop_during_hidden_thinking_never_exposes_trace(self):
        self.gates["effort:latest"] = threading.Event()
        job = self.submit(self.new_chat(), model="effort", reasoning="high")
        deadline = time.monotonic() + 4
        while store.get_job(job["id"])["activity"] != "Thinking" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.client.post(f"/api/generations/{job['id']}/cancel")
        stopped = self.wait_for(job, "cancelled")
        self.assertEqual(stopped["assistant_message"]["content"], "")
        self.assertNotIn(SECRET, json.dumps(store.events()))

    def test_private_trace_never_reaches_sqlite_sse_or_history(self):
        chat = self.new_chat()
        final = self.wait_for(self.submit(chat, model="effort", reasoning="high"))
        self.assertEqual(final["assistant_message"]["content"], "Safe answer")
        messages = self.client.get(f"/api/chats/{chat}/messages").json()
        self.assertEqual(messages[1]["inference"], final["inference"])
        self.assertNotIn(SECRET, json.dumps(messages))
        self.assertNotIn(SECRET, self.client.get(f"/api/generations/{final['id']}/events").text)
        self.assertNotIn(SECRET.encode(), db.DB_PATH.read_bytes())
        self.assertIsNone(store.update(final["id"], {"type": "thinking", "content": SECRET}))

    def test_event_allowlist_and_error_logs_cannot_contain_private_payloads(self):
        job, _ = store.create_job(self.new_chat(), "Sanitized event")
        store.update(job["id"], {"type": "status", "label": "Thinking", "thinking": SECRET})
        store.update(job["id"], {"type": "done", "thinking": SECRET})
        with patch.object(ollama, "chat", side_effect=ValueError(SECRET)), self.assertLogs("backend.generation.jobs", level="ERROR") as logs:
            failed = self.submit(self.new_chat(), model="effort", reasoning="high")
            deadline = time.monotonic() + 4
            while store.get_job(failed["id"])["status"] != "error" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(store.get_job(failed["id"])["status"], "error")
        self.assertNotIn(SECRET, "".join(logs.output))
        self.assertNotIn(SECRET, json.dumps(store.events()))
        self.assertNotIn(SECRET.encode(), db.DB_PATH.read_bytes())

    def test_retry_preserves_metadata_even_when_ollama_is_offline(self):
        chat = self.new_chat()
        original = self.wait_for(self.submit(chat, model="effort", reasoning="high", request_id="retry"))
        with patch.object(provider, "request", side_effect=ProviderUnavailable("Offline")):
            retry = self.submit(chat, model="effort", reasoning="high", request_id="retry")
            self.assertEqual(retry["inference"], original["inference"])
            self.assertEqual(retry["id"], original["id"])
            conflict = self.client.post(f"/api/chats/{chat}/generations", json={"message": "Question", "model": "binary", "reasoning": "on", "request_id": "retry"})
            self.assertEqual(conflict.status_code, 409)
        self.assertEqual(len(db.list_messages(chat)), 2)

    def test_additive_migration_preserves_pre_phase_b_jobs_and_messages(self):
        chat = self.new_chat()
        job, _ = store.create_job(chat, "Legacy job")
        store.update(job["id"], {"type": "token", "content": "Existing answer"})
        store.update(job["id"], {"type": "done"})
        with db.connection() as connection:
            connection.execute("ALTER TABLE generation_jobs DROP COLUMN inference_json")
        db.init_db()
        db.init_db()
        messages = db.list_messages(chat)
        self.assertEqual(messages[1]["content"], "Existing answer")
        self.assertIsNone(messages[1]["inference"])
        self.assertIsNone(store.get_job(job["id"])["inference"])


if __name__ == "__main__":
    unittest.main()
