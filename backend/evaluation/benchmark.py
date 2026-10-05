"""Opt-in local HTTP benchmarks with persisted results and hard deadlines."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from backend.evaluation.metrics import InferenceMetrics, summarize

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    ("greeting", "Reply with exactly: Hello. Do not search."),
    ("arithmetic", "Compute 7 + 5. Reply only with the number. Do not search."),
    ("rewrite", "Rewrite 'please send the report' formally in one short sentence. Do not search."),
)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finish_report(report):
    report["summary"] = summarize(report["runs"])
    groups = defaultdict(list)
    for run in report["runs"]:
        groups[f"{run['model']} / {run['reasoning']} / {run['mode']}"].append(run)
    report["groups"] = {key: summarize(runs) for key, runs in groups.items()}


def print_report(report, path):
    print(f"\nATLAS {'FULL' if report['full'] else 'QUICK'} BENCHMARK")
    print("-" * 44)
    completed = sum(run["status"] == "completed" for run in report["runs"])
    print(f"{'Model':<20} {', '.join(dict.fromkeys(run['model'] for run in report['runs'])) or 'not started'}")
    print(f"{'Runs':<20} {completed}/{report['planned_runs']} completed ({report['status']})")
    labels = dict(ttft_ms=("TTFT", "ms"), inference_ttft_ms=("Inference TTFT", "ms"), tps=("TPS", "tok/s"),
                  tpot_ms=("TPOT (decode avg)", "ms/token"), e2e_seconds=("E2E", "s"),
                  planner_ms=("Planner", "ms"), retrieval_ms=("Retrieval", "ms"), reranker_ms=("Reranker", "ms"))
    for metric, (label, unit) in labels.items():
        stats = report.get("summary", {}).get(metric)
        print(f"{label:<20} {f'{stats["mean"]:.1f} {unit}' if stats else 'N/A'}")
    print(f"{'Total benchmark':<20} {report.get('total_seconds', 0):.1f} s")
    if report["full"]:
        for group, values in report.get("groups", {}).items():
            print(f"\n{group}\n{'Metric':<20} {'Mean':>9} {'Median':>9} {'p95':>9} {'Variance':>12} {'N':>4}")
            for metric, stats in values.items():
                if stats:
                    variance = f"{stats['variance']:.2f}" if stats["variance"] is not None else "N/A"
                    print(f"{labels[metric][0]:<20} {stats['mean']:>9.2f} {stats['median']:>9.2f} {stats['p95']:>9.2f} {variance:>12} {stats['count']:>4}")
    print("TTFT: local HTTP submission to first visible SSE text; browser paint excluded.")
    print("TPS/TPOT: provider eval_count/eval_duration, including hidden thinking if counted.")
    print("True token ITL: unavailable. Stream-chunk intervals are separate in saved JSON.")
    print(f"Saved: {path}")
    if report.get("error"):
        print(report["error"])


@contextmanager
def isolated_server(app, listener):
    import threading
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.perf_counter() + 20
        while not server.started:
            if not thread.is_alive() or time.perf_counter() >= deadline:
                raise TimeoutError("Isolated benchmark API did not start")
            time.sleep(.02)
        yield listener.getsockname()[1]
    finally:
        server.should_exit = True
        # Keep the temporary DB/vector paths installed until all jobs have stopped.
        # The supervising process imposes the hard deadline even during cleanup.
        thread.join()


def worker(args):
    # Heavy application imports and model initialization stay in the bounded child.
    import socket
    from unittest.mock import patch
    import httpx
    from backend import api, db
    from backend.core import assistant, document_store
    from backend.models.ollama_provider import provider

    path = Path(args.result_file)
    report = json.loads(path.read_text(encoding="utf-8"))
    original_stream, original_plan = provider.stream, assistant.plan
    current = None
    monitor = None

    def observed_stream(messages, inference, **settings):
        return original_stream(messages, inference, **settings, output_tokens=args.output_tokens, telemetry=monitor.observe)

    def controlled_plan(*values, **settings):
        start = time.perf_counter()
        decision = original_plan(*values, **settings)
        current["planner_ms"] = (time.perf_counter() - start) * 1000
        current["planner_route"] = {key: decision[key] for key in ("use_documents", "use_web", "fresh")}
        # Controlled benchmark scopes prohibit accidental live web/production retrieval.
        return {**decision, "use_web": False, "fresh": False, "use_documents": bool(settings.get("has_attachments"))}

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            with (patch.object(db, "DB_PATH", root / "atlas.db"), patch.object(api, "UPLOADS_DIR", root / "documents"),
                  patch.object(document_store, "CHROMA_PATH", root / "chroma"),
                  patch.object(provider, "stream", observed_stream), patch.object(assistant, "plan", controlled_plan),
                  isolated_server(api.app, listener) as port):
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                    document_id = None
                    if args.pdf:
                        with Path(args.pdf).open("rb") as uploaded:
                            response = client.post("/api/documents", files={"file": (Path(args.pdf).name, uploaded, "application/pdf")}, timeout=45)
                        response.raise_for_status()
                        document_id = response.json()["id"]
                    models = args.models or [args.model]
                    for model in models:
                        for requested_reasoning in args.reasoning_modes or [args.reasoning]:
                            for repetition in range(args.repeats):
                                for index, (label, prompt) in enumerate(PROMPTS):
                                    mode = "documents" if document_id and index == 2 else "chat"
                                    if mode == "documents":
                                        prompt = "According to the attached PDF, summarize its central topic in one short sentence. Do not search the web."
                                    monitor = InferenceMetrics()
                                    current = dict(model=model or "configured default", reasoning=requested_reasoning or "default",
                                                   mode=mode, prompt_id=label, repetition=repetition + 1, status="running")
                                    report["runs"].append(current)
                                    save(path, report)
                                    started = time.perf_counter()
                                    deadline = started + args.run_timeout
                                    job_id = None
                                    try:
                                        chat = client.post("/api/chats", json={"title": "Isolated performance benchmark"}).json()["id"]
                                        response = client.post(f"/api/chats/{chat}/generations", json=dict(message=prompt,
                                            model=model, reasoning=requested_reasoning, document_ids=[document_id] if mode == "documents" else []),
                                            timeout=min(15, max(.1, deadline - time.perf_counter())))
                                        response.raise_for_status()
                                        job = response.json()
                                        job_id = job["id"]
                                        current.update(model=job["inference"]["model"], reasoning=job["inference"]["reasoning_mode"],
                                                       inference=job["inference"])
                                        with client.stream("GET", f"/api/generations/{job_id}/events", timeout=args.run_timeout) as stream:
                                            stream.raise_for_status()
                                            for line in stream.iter_lines():
                                                now = time.perf_counter()
                                                if now >= deadline:
                                                    raise TimeoutError("Per-run deadline exceeded")
                                                if not line.startswith("data: "):
                                                    continue
                                                payload = json.loads(line[6:])
                                                job = payload["job"]
                                                if job["assistant_message"]["content"] and current.get("ttft_ms") is None:
                                                    current["ttft_ms"] = (now - started) * 1000
                                                    current.update(monitor.result())
                                                    save(path, report)
                                                if job["status"] in ("completed", "error", "cancelled", "interrupted"):
                                                    current.update(status=job["status"], e2e_seconds=now - started,
                                                                   inference=job["inference"], **monitor.result())
                                                    if job.get("error"):
                                                        current["error"] = job["error"]
                                                    timings = job["timings"]
                                                    current.update(planner_ms=timings.get("planner", current.get("planner_ms", 0) / 1000) * 1000,
                                                        retrieval_ms=timings["documents"] * 1000 if "documents" in timings else None,
                                                        reranker_ms=timings["reranking"] * 1000 if "reranking" in timings else None)
                                                    break
                                        if current["status"] == "running":
                                            current.update(status="error", error="SSE ended before completion")
                                    except (TimeoutError, httpx.TimeoutException):
                                        current.update(status="timed_out", e2e_seconds=time.perf_counter() - started, **monitor.result())
                                        report.update(status="timed_out", error="Per-run timeout reached; remaining runs were not started.")
                                    except Exception as error:
                                        current.update(status="error", error=f"Run failed ({type(error).__name__}).", **monitor.result())
                                    finish_report(report)
                                    save(path, report)
                                    if current["status"] == "timed_out":
                                        if job_id:
                                            try:
                                                client.post(f"/api/generations/{job_id}/cancel", timeout=2)
                                            except httpx.HTTPError:
                                                pass
                                        return
                    report["status"] = "completed" if all(run["status"] == "completed" for run in report["runs"]) else "partial"
                    save(path, report)
    except Exception as error:
        report.update(status="error", error=f"Benchmark setup failed ({type(error).__name__}). Check local dependencies, Ollama, and model availability.")
        save(path, report)
    finally:
        listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="+", help="Full-mode model comparison")
    parser.add_argument("--reasoning", help="Actual model-supported setting")
    parser.add_argument("--reasoning-modes", nargs="+", help="Full-mode comparison of proven settings; 'default' uses the model default")
    parser.add_argument("--pdf", help="Index this PDF into an isolated store and use RAG for prompt 3")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--timeout", type=float, help="Overall wall-clock seconds: Quick 120, Full 900")
    parser.add_argument("--run-timeout", type=float, help="Per-run seconds: Quick 25, Full 60")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "benchmarks"))
    parser.add_argument("--show", help="Print a saved JSON result without running inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.show:
        print_report(json.loads(Path(args.show).read_text(encoding="utf-8")), args.show)
        return 0
    args.repeats = args.repeats if args.repeats is not None else (5 if args.full else 1)
    args.timeout = args.timeout if args.timeout is not None else (900 if args.full else 120)
    args.run_timeout = args.run_timeout if args.run_timeout is not None else (60 if args.full else 25)
    if (args.repeats < 1 or args.output_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.run_timeout) or args.run_timeout <= 0):
        parser.error("Counts and timeouts must be positive and finite.")
    if not args.full and (args.models or args.reasoning_modes or args.repeats != 1 or args.timeout > 120):
        parser.error("Quick always uses 3 prompts and a deadline of at most 120s; use --full for comparisons or repetitions.")
    if args.model and args.models:
        parser.error("Choose --model or --models.")
    if args.reasoning and args.reasoning_modes:
        parser.error("Choose --reasoning or --reasoning-modes.")
    if args.pdf and not Path(args.pdf).is_file():
        parser.error("PDF not found.")
    if args.reasoning_modes:
        args.reasoning_modes = [None if value == "default" else value for value in args.reasoning_modes]
    if args.reasoning == "default":
        args.reasoning = None
    if args.worker:
        worker(args)
        return 0
    path = Path(args.output_dir).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json")
    report = dict(version=1, full=args.full, status="starting", runs=[], summary={}, groups={},
                  planned_runs=3 * args.repeats * len(args.models or [args.model]) * len(args.reasoning_modes or [args.reasoning]),
                  settings=dict(output_tokens=args.output_tokens, timeout_seconds=args.timeout, per_run_seconds=args.run_timeout,
                                repeats=args.repeats, web=False, deep_research=False, pdf=bool(args.pdf)),
                  created_at=datetime.now(timezone.utc).isoformat())
    save(path, report)
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [str(interpreter if interpreter.is_file() else sys.executable), "-m", "backend.evaluation.benchmark", *sys.argv[1:],
               "--worker", "--result-file", str(path)]
    print(f"Starting {'Full' if args.full else 'Quick'} benchmark ({report['planned_runs']} prompts, {args.timeout:g}s deadline)...", flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT)
    timed_out = False
    try:
        process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        timed_out = True
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["total_seconds"] = time.perf_counter() - started
    if timed_out:
        report.update(status="timed_out", error="Overall deadline or interruption reached; child stopped and partial measurements saved.")
        for run in report["runs"]:
            if run["status"] == "running":
                run["status"] = "timed_out"
    elif process.returncode or report["status"] == "starting":
        report.update(status="error", error="Benchmark worker did not complete. Check project .venv dependencies.")
    finish_report(report)
    save(path, report)
    print_report(report, path)
    return 124 if report["status"] == "timed_out" else 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
