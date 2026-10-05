import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import db
from backend.generation.jobs import JobManager
from backend.generation import store


class ResponsePerformanceTests(unittest.TestCase):
    def test_job_metrics_persist_on_message_and_reconnect(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(db, "DB_PATH", Path(directory) / "test.db"):
            db.init_db()
            chat = db.create_chat("Performance")
            def generate(**request):
                yield {"type": "token", "content": "Answer"}
                yield {"type": "performance", "timings": {"tps": 42, "inference_ttft": .09}}
                yield {"type": "done", "timings": {"planner": .1}}
            manager = JobManager(generate)
            try:
                job = manager.submit(chat_id=chat["id"], message="Question")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    job = store.get_job(job["id"])
                    if job["status"] == "completed":
                        break
                    time.sleep(.01)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["assistant_message"]["timings"]["tps"], 42)
                messages = db.list_messages(chat["id"])
                self.assertEqual(messages[1]["timings"], job["timings"])
                self.assertGreaterEqual(job["timings"]["ttft"], 0)
                self.assertEqual(messages[0]["timings"], {})
            finally:
                manager.close()
