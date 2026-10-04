import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import api, db
from backend.generation import store
from backend.generation.jobs import JobManager
from backend.tests.smoke_local import local_client
from backend.tests.model_fixtures import default_inference


class Subscriber:
    async def is_disconnected(self):
        return False


class JobTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(db, "DB_PATH", Path(directory) / "jobs.db"))
        self.gates = {}
        self.started = {}
        self.enterContext(patch.object(api, "stream_assistant", side_effect=self.generate))
        self.enterContext(patch.object(api.provider, "resolve", side_effect=default_inference))
        self.client = self.enterContext(TestClient(api.app))
        self.addCleanup(lambda: [gate.set() for gate in self.gates.values()])
        self.a = self.client.post("/api/chats", json={"title": "A"}).json()["id"]
        self.b = self.client.post("/api/chats", json={"title": "B"}).json()["id"]

    def generate(self, message, **options):
        self.started[message].set()
        yield {"type": "status", "label": "Planning"}
        yield {"type": "token", "content": "First "}
        yield {"type": "status", "label": "Writing"}
        if not self.gates[message].wait(10):
            raise TimeoutError("Test gate was not released")
        yield {"type": "token", "content": "last"}
        yield {"type": "done", "timings": {"total": 1}}

    def submit(self, chat, message, request_id=None):
        self.gates.setdefault(message, threading.Event())
        self.started.setdefault(message, threading.Event())
        response = self.client.post(f"/api/chats/{chat}/generations", json={
            "message": message, "request_id": request_id,
        })
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertTrue(self.started[message].wait(3))
        self.wait_for(job["id"], lambda current: current["assistant_message"]["content"] == "First ")
        return job

    def wait_for(self, job_id, predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            job = store.get_job(job_id)
            if predicate(job):
                return job
            time.sleep(0.01)
        self.fail(f"Job condition not reached: {store.get_job(job_id)}")

    def finish(self, job, message):
        self.gates[message].set()
        return self.wait_for(job["id"], lambda current: current["status"] == "completed")

    def test_concurrent_jobs_switching_and_incremental_persistence(self):
        a = self.submit(self.a, "a")
        b = self.submit(self.b, "b")
        for chat in (self.b, self.a, self.b):
            messages = self.client.get(f"/api/chats/{chat}/messages").json()
            self.assertEqual(messages[1]["content"], "First ")
        self.assertEqual(len([j for j in store.snapshot()["jobs"] if j["status"] == "running"]), 2)
        final_a = self.finish(a, "a")
        self.assertEqual(store.get_job(b["id"])["status"], "running")
        self.assertEqual(final_a["assistant_message"]["id"], a["assistant_message"]["id"])
        self.assertEqual(final_a["assistant_message"]["content"], "First last")
        self.finish(b, "b")

    def test_disconnect_and_reconnect_replays_without_duplicates(self):
        job = self.submit(self.a, "a")

        async def disconnect():
            stream = api.generation_events(Subscriber(), 0, job["id"])
            first = await anext(stream)
            await stream.aclose()
            return json.loads(next(line[6:] for line in first.splitlines() if line.startswith("data: ")))

        first = asyncio.run(disconnect())
        self.assertEqual(store.get_job(job["id"])["status"], "running")
        self.finish(job, "a")

        async def reconnect():
            return [block async for block in api.generation_events(Subscriber(), first["id"], job["id"])]

        blocks = asyncio.run(reconnect())
        replay = [json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: "))) for block in blocks]
        self.assertTrue(all(event["id"] > first["id"] for event in replay))
        self.assertEqual(replay[-1]["job"]["assistant_message"]["content"], "First last")
        self.assertEqual(replay[-1]["event"]["type"], "done")
        self.assertEqual(len(db.list_messages(self.a)), 2)

    def test_idempotent_submission_and_one_active_per_chat(self):
        job = self.submit(self.a, "a", "same-request")
        retry = self.client.post(f"/api/chats/{self.a}/generations", json={"message": "a", "request_id": "same-request"})
        self.assertEqual(retry.json()["id"], job["id"])
        for payload in ({"message": "another"}, {"message": "changed", "request_id": "same-request"}):
            self.assertEqual(self.client.post(f"/api/chats/{self.a}/generations", json=payload).status_code, 409)
        self.finish(job, "a")
        self.assertEqual(self.client.post(f"/api/chats/{self.a}/generations", json={"message": "a", "request_id": "same-request"}).json()["id"], job["id"])
        self.assertEqual(len(db.list_messages(self.a)), 2)

    def test_cancel_only_relevant_job_preserves_partial_answer(self):
        a = self.submit(self.a, "a")
        b = self.submit(self.b, "b")
        stopping = self.client.post(f"/api/generations/{a['id']}/cancel").json()
        self.assertEqual(stopping["status"], "cancelling")
        self.assertEqual(self.client.post(f"/api/chats/{self.a}/generations", json={"message": "new"}).status_code, 409)
        self.gates["a"].set()
        stopped = self.wait_for(a["id"], lambda job: job["status"] == "cancelled")
        self.assertEqual(stopped["assistant_message"]["content"], "First ")
        self.assertEqual(store.get_job(b["id"])["status"], "running")
        self.finish(b, "b")

    def test_archive_allows_existing_job_to_finish(self):
        job = self.submit(self.a, "a")
        self.assertEqual(self.client.patch(f"/api/chats/{self.a}", json={"archived": True}).status_code, 200)
        self.assertEqual(self.client.post(f"/api/chats/{self.a}/generations", json={"message": "new"}).status_code, 409)
        self.finish(job, "a")
        self.assertTrue(db.get_chat(self.a)["archived"])
        self.assertEqual(db.list_messages(self.a)[1]["content"], "First last")

    def test_delete_prevents_resurrection_and_removes_event_content(self):
        job = self.submit(self.a, "a")
        self.assertEqual(self.client.delete(f"/api/chats/{self.a}").status_code, 200)
        self.gates["a"].set()
        deadline = time.monotonic() + 3
        while job["id"] in api.app.state.jobs.controls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertNotIn(job["id"], api.app.state.jobs.controls)
        self.assertIsNone(db.get_chat(self.a))
        self.assertIsNone(store.get_job(job["id"]))
        self.assertEqual(db.list_messages(self.a), [])
        events = [event for event in store.events() if event["chat_id"] == self.a]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"]["type"], "deleted")
        self.assertIsNone(events[0]["job"])

    def test_restart_marks_unfinished_interrupted_and_retains_answer(self):
        job, _ = store.create_job(self.a, "unfinished")
        store.update(job["id"], {"type": "token", "content": "Saved"})
        store.interrupt_unfinished()
        interrupted = store.get_job(job["id"])
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["assistant_message"]["content"], "Saved")
        self.assertEqual(store.events(job_id=job["id"])[-1]["event"]["type"], "interrupted")

    def test_queued_cancellation_never_runs_generator(self):
        blocker = threading.Event()
        started = threading.Event()
        calls = []

        def generate(message, **options):
            calls.append(message)
            started.set()
            blocker.wait(5)
            yield {"type": "done"}

        manager = JobManager(generate, workers=1)
        self.addCleanup(manager.close)
        self.addCleanup(blocker.set)
        manager.submit(chat_id=self.a, message="first")
        self.assertTrue(started.wait(3))
        queued = manager.submit(chat_id=self.b, message="second")
        self.assertEqual(manager.cancel(queued["id"])["status"], "cancelled")
        blocker.set()
        self.assertEqual(calls, ["first"])


class TransportTests(unittest.TestCase):
    def test_http_disconnect_reload_and_restart(self):
        gate = threading.Event()

        def generate(**request):
            yield {"type": "status", "label": "Writing"}
            yield {"type": "token", "content": "Saved "}
            yield {"type": "status", "label": "Writing"}
            gate.wait(5)
            yield {"type": "token", "content": "answer"}
            yield {"type": "done"}

        with tempfile.TemporaryDirectory() as directory, patch.object(db, "DB_PATH", Path(directory) / "transport.db"), patch.object(api, "stream_assistant", side_effect=generate), patch.object(api.provider, "resolve", side_effect=default_inference):
            with local_client() as client:
                try:
                    chat = client.post("/api/chats", json={"title": "Disconnect"}).json()["id"]
                    job = client.post(f"/api/chats/{chat}/generations", json={"message": "Question", "request_id": "transport"}).json()
                    with client.stream("GET", f"/api/generations/{job['id']}/events") as response:
                        response.raise_for_status()
                        first = next(line for line in response.iter_lines() if line.startswith("data: "))
                    cursor = json.loads(first[6:])["id"]
                    self.assertIn(client.get(f"/api/generations/{job['id']}").json()["status"], store.ACTIVE)
                    gate.set()
                    with client.stream("GET", f"/api/generations/{job['id']}/events?after={cursor}") as response:
                        replay = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
                    self.assertEqual(replay[-1]["job"]["status"], "completed")
                    self.assertEqual(replay[-1]["job"]["assistant_message"]["content"], "Saved answer")
                    self.assertEqual(len(client.get(f"/api/chats/{chat}/messages").json()), 2)
                finally:
                    gate.set()
            unfinished, _ = store.create_job(chat, "Restart")
            store.update(unfinished["id"], {"type": "token", "content": "Retained"})
            with local_client() as client:
                restored = client.get("/api/generations").json()["jobs"][0]
                self.assertEqual(restored["status"], "interrupted")
                self.assertEqual(restored["assistant_message"]["content"], "Retained")
                self.assertEqual(len(client.get(f"/api/chats/{chat}/messages").json()), 4)


if __name__ == "__main__":
    unittest.main()
