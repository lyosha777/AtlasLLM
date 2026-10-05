import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from backend.evaluation import benchmark
from backend.evaluation.metrics import InferenceMetrics, summarize
from backend.models.ollama_provider import OllamaProvider


class BenchmarkTests(unittest.TestCase):
    def test_provider_counts_are_used_and_chunk_intervals_are_not_token_itl(self):
        metrics = InferenceMetrics()
        metrics.observe(dict(type="inference_started", at=2.0))
        for timestamp in (2.5, 2.6, 2.8):
            metrics.observe(dict(type="visible_chunk", at=timestamp))
        metrics.observe(dict(type="usage", at=3.0, eval_count=10, eval_duration=2500000000))
        result = metrics.result()
        self.assertEqual(result["inference_ttft_ms"], 500)
        self.assertEqual(result["tps"], 4)
        self.assertEqual(result["tpot_ms"], 250)
        self.assertAlmostEqual(result["stream_chunk_interval_ms"], 150)
        self.assertIsNone(result["true_token_itl_ms"])

    def test_missing_token_counts_never_fall_back_to_words_or_chunks(self):
        metrics = InferenceMetrics()
        metrics.observe(dict(type="inference_started", at=1.0))
        metrics.observe(dict(type="visible_chunk", at=1.1))
        result = metrics.result()
        self.assertIsNone(result["provider_output_tokens"])
        self.assertIsNone(result["tps"])
        self.assertIsNone(result["tpot_ms"])

    def test_full_statistics_and_failed_samples(self):
        runs = [dict(status="completed", ttft_ms=index) for index in range(1, 21)]
        runs.append(dict(status="error", ttft_ms=10000))
        stats = summarize(runs)["ttft_ms"]
        self.assertEqual(stats["count"], 20)
        self.assertEqual(stats["mean"], 10.5)
        self.assertEqual(stats["median"], 10.5)
        self.assertEqual(stats["p95"], 19)
        self.assertEqual(stats["variance"], 35)

    def test_singleton_variance_and_unavailable_metrics(self):
        stats = summarize([dict(status="completed", ttft_ms=100)])
        self.assertIsNone(stats["ttft_ms"]["variance"])
        self.assertIsNone(stats["retrieval_ms"])

    def fake_process(self, command, cwd):
        path = Path(command[-1])
        report = json.loads(path.read_text(encoding="utf-8"))
        report.update(status="completed", runs=[dict(model="local:test", reasoning="off", mode="chat",
            status="completed", ttft_ms=500, inference_ttft_ms=200, tps=30, tpot_ms=1000/30,
            e2e_seconds=2, planner_ms=100) for _ in range(3)])
        benchmark.save(path, report)
        class Process:
            returncode = 0
            def wait(self, timeout): return 0
        return Process()

    def test_quick_is_exactly_three_prompts_with_no_web_or_research(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.sys, "argv", ["benchmark", "--output-dir", directory]), \
             patch.object(benchmark.subprocess, "Popen", side_effect=self.fake_process), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(benchmark.main(), 0)
            result = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["planned_runs"], 3)
        self.assertFalse(result["full"])
        self.assertFalse(result["settings"]["web"])
        self.assertFalse(result["settings"]["deep_research"])
        self.assertEqual(result["settings"]["timeout_seconds"], 120)
        self.assertIn("ATLAS QUICK BENCHMARK", output.getvalue())

    def test_quick_rejects_repetitions_comparisons_and_long_deadlines(self):
        for options in (["--repeats", "2"], ["--models", "a", "b"], ["--timeout", "121"], ["--reasoning-modes", "low", "high"]):
            with patch.object(benchmark.sys, "argv", ["benchmark", *options]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                benchmark.main()
            self.assertEqual(caught.exception.code, 2)

    def test_full_requires_an_explicit_flag_and_reports_comparison_statistics(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.sys, "argv", ["benchmark", "--full", "--models", "a", "b", "--output-dir", directory]), \
             patch.object(benchmark.subprocess, "Popen", side_effect=self.fake_process), contextlib.redirect_stdout(io.StringIO()) as output:
            benchmark.main()
            result = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["planned_runs"], 30)
        self.assertIn("p95", output.getvalue())
        self.assertIn("Variance", output.getvalue())

    def test_timeout_stops_worker_and_retains_partial_results(self):
        def process(command, cwd):
            path = Path(command[-1])
            report = json.loads(path.read_text(encoding="utf-8"))
            report["runs"] = [dict(model="local", reasoning="default", mode="chat", status="completed", ttft_ms=600),
                              dict(model="local", reasoning="default", mode="chat", status="running", ttft_ms=900)]
            benchmark.save(path, report)
            class Process:
                returncode = -1
                stopped = False
                def wait(self, timeout):
                    if not self.stopped: raise subprocess.TimeoutExpired(command, timeout)
                    return -1
                def terminate(self): self.stopped = True
            return Process()
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.sys, "argv", ["benchmark", "--output-dir", directory]), \
             patch.object(benchmark.subprocess, "Popen", side_effect=process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark.main(), 124)
            result = json.loads(next(Path(directory).glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["summary"]["ttft_ms"]["mean"], 600)
        self.assertEqual(result["runs"][1]["status"], "timed_out")

    def test_saved_results_print_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            benchmark.save(path, dict(full=False, status="completed", planned_runs=3, runs=[], summary={}, total_seconds=1))
            with patch.object(benchmark.sys, "argv", ["benchmark", "--show", str(path)]), patch.object(benchmark.subprocess, "Popen") as spawn, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(), 0)
            spawn.assert_not_called()

    def test_telemetry_uses_real_provider_metadata_without_private_content(self):
        provider = OllamaProvider()
        inference = dict(model="local:test", model_digest="digest", capability={}, settings={})
        chunks = [dict(message=dict(thinking="PRIVATE_REASONING", content="")),
                  dict(message=dict(content="Hello")),
                  dict(done=True, eval_count=5, eval_duration=250000000, prompt_eval_count=2,
                       message=dict(thinking="PRIVATE_REASONING", content=""))]
        events = []
        with patch.object(provider, "inspect", return_value=dict(digest="digest", reasoning={})), \
             patch("backend.models.ollama_provider.ollama.chat", return_value=iter(chunks)):
            list(provider.stream([dict(role="user", content="Hello")], inference, telemetry=events.append))
        self.assertNotIn("PRIVATE_REASONING", json.dumps(events))
        self.assertNotIn("Hello", json.dumps(events))
        usage = next(event for event in events if event["type"] == "usage")
        self.assertEqual(usage["eval_count"], 5)
        self.assertEqual(usage["eval_duration"], 250000000)
