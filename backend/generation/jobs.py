import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backend.generation import store


logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, generate, workers=2):
        self.generate = generate
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atlas-generation")
        self.lock = threading.RLock()
        self.controls = {}
        self.closed = False

    def submit(self, **request):
        with self.lock:
            if self.closed:
                raise store.JobConflict("Atlas is shutting down.")
            job, created = store.create_job(**request)
            if created:
                cancelled = threading.Event()
                self.controls[job["id"]] = (cancelled, None)
                try:
                    future = self.executor.submit(self._run, job["id"], cancelled)
                    self.controls[job["id"]] = (cancelled, future)
                except Exception:
                    self.controls.pop(job["id"], None)
                    store.update(job["id"], {"type": "error", "message": "Could not start response worker."})
                    raise
            return job

    def _run(self, job_id, cancelled):
        stream = None
        pending = []
        last_flush = time.monotonic()
        started = time.perf_counter()
        first_token = None

        def flush():
            nonlocal last_flush
            if pending:
                store.update(job_id, {"type": "token", "content": "".join(pending)})
                pending.clear()
                last_flush = time.monotonic()

        try:
            request = store.get_request(job_id)
            if request is None:
                return
            if cancelled.is_set():
                store.update(job_id, {"type": "cancelled"})
                return
            stream = self.generate(**request)
            completed = False
            for event in stream:
                if cancelled.is_set():
                    break
                if event["type"] == "checkpoint":
                    continue
                if event["type"] == "token":
                    if first_token is None and event["content"]:
                        first_token = time.perf_counter()
                    pending.append(event["content"])
                    if time.monotonic() - last_flush >= 0.04:
                        flush()
                else:
                    if event["type"] == "done":
                        event = {**event, "timings": {**event.get("timings", {}), "total": time.perf_counter() - started}}
                        if first_token is not None:
                            event["timings"]["ttft"] = first_token - started
                    flush()
                    result = store.update(job_id, event)
                    if result is None:
                        break
                    if event["type"] == "done":
                        completed = True
                        break
            if cancelled.is_set():
                pending.clear()
                store.update(job_id, {"type": "cancelled"})
            elif not completed:
                flush()
                store.update(job_id, {"type": "error", "message": "Response ended before completion."})
        except Exception as error:
            flush()
            if cancelled.is_set():
                store.update(job_id, {"type": "cancelled"})
            else:
                logger.exception("Generation %s failed", job_id)
                store.update(job_id, {"type": "error", "message": str(error)})
        finally:
            try:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()
            finally:
                with self.lock:
                    self.controls.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            job = store.get_job(job_id)
            if job is None:
                return None
            if job["status"] not in store.ACTIVE:
                return job
            store.update(job_id, {"type": "cancelling"})
            control = self.controls.get(job_id)
            if control:
                cancelled, future = control
                cancelled.set()
                if future is not None and future.cancel():
                    self.controls.pop(job_id, None)
                    return store.update(job_id, {"type": "cancelled"})
            return store.get_job(job_id)

    def delete_chat(self, chat_id):
        with self.lock:
            for job_id, (cancelled, future) in list(self.controls.items()):
                job = store.get_job(job_id)
                if job and job["chat_id"] == chat_id:
                    cancelled.set()
                    if future is not None and future.cancel():
                        self.controls.pop(job_id, None)
            return store.delete_chat(chat_id)

    def close(self):
        with self.lock:
            self.closed = True
            for job_id, (cancelled, _) in self.controls.items():
                cancelled.set()
                store.update(job_id, {"type": "interrupted", "message": "Backend stopped before this response finished."})
        self.executor.shutdown(wait=True, cancel_futures=True)
