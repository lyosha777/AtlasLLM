import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import api, db
from backend.core.assistant import stream_assistant
from backend.core.source import Source
from backend.generation import store
from backend.generation.jobs import JobManager
from backend.research import pipeline
from backend.research.schemas import ResearchPlan, EvidenceAnalysis, MAX_SEARCHES, MAX_CANDIDATES, MAX_EVIDENCE
from backend.research.store import get_run
from backend.research.web_cache import WebCache
from backend.tests.model_fixtures import default_inference
from backend.models.ollama_provider import OllamaProvider
from backend.tools.web_reader import publication_date
import ollama


def public_plan(queries=1):
    return dict(objective="Compare transport protocols", depth="deep", depth_reason="Multi-part investigation", subquestions=[dict(id=f"sq{i}",
        question=f"Question {i}", reason="Establish relevant evidence", search_queries=[f"query {i}-{j}" for j in range(queries)]) for i in range(1, 4)])


class DeepResearchTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(db, "DB_PATH", Path(directory) / "research.db"))
        db.init_db()
        self.inference = default_inference()
        self.inference.update(model="answer:local", settings={"think": "high"}, requested_reasoning="high")
        self.enterContext(patch.object(api.provider, "resolve", return_value=self.inference))
        self.model_calls = []
        self.plan_responses = []
        self.coverage = "sufficient"
        self.conflicts = False
        self.ambiguous = False
        self.enterContext(patch.object(pipeline.provider, "stream", side_effect=self.model))
        self.documents = self.enterContext(patch.object(pipeline, "retrieve", side_effect=self.document_sources))
        self.web = self.enterContext(patch.object(pipeline, "retrieve_web", side_effect=self.web_sources))
        self.enterContext(patch.object(pipeline, "rerank", side_effect=lambda query, sources, **kwargs: sources))
        self.client = self.enterContext(TestClient(api.app))
        self.chat = self.client.post("/api/chats", json={"title": "Research"}).json()["id"]
        self.saved = db.save_document(dict(id="doc1", name="one.pdf", size=100, page_count=1, chunk_count=3))

    def document_sources(self, query, **kwargs):
        return [Source(text=f"Exact PDF passage\n{query}  ", source_type="document", title="one.pdf", page=1,
                       document_id="doc1", chunk_id=f"doc1:{query}", relevance_score=0.8, retrieval_score=0.7)]

    def web_sources(self, query, **kwargs):
        return [Source(text=f"Exact web passage\n{query}", source_type="web", title="Web source",
                       url="https://example.org/" + query, relevance_score=0.9, retrieval_score=0.6)]

    def model(self, messages, inference, **options):
        self.model_calls.append((inference, options))
        yield {"type": "inference", "settings": {"model": inference["model"], **inference["settings"],
               "options": {"num_predict": options.get("output_tokens", 600)}}}
        yield {"type": "status", "stage": "reasoning", "label": "Thinking"}
        if (options.get("schema") or {}).get("title") == "ResearchPlan":
            text = self.plan_responses.pop(0) if self.plan_responses else json.dumps(public_plan())
        elif (options.get("schema") or {}).get("title") == "EvidenceAnalysis":
            context = json.loads(messages[1]["content"])
            evidence = context["evidence"]
            text = json.dumps(dict(claims=[dict(id="c1", statement="Public conclusion",
                supporting_evidence_ids=[evidence[0]["id"]], conflicting_evidence_ids=[evidence[0]["id"]] if self.ambiguous else [evidence[1]["id"]] if self.conflicts and len(evidence) > 1 else [])],
                coverage=[dict(subquestion_id=item["id"], status=self.coverage, explanation="Public coverage conclusion", refined_query="refined " + item["question"])
                          for item in context["plan"]["subquestions"]]))
        else:
            text = "Grounded result [SOURCE 1]. Sources disagree when conflicting evidence is retained."
        yield {"type": "token", "content": text[:12]}
        yield {"type": "token", "content": text[12:]}

    def submit(self, **options):
        response = self.client.post(f"/api/chats/{self.chat}/generations", json=dict(
            message="Compare protocols", research_mode=True, research_scope="both", model="answer:local", reasoning="high", **options))
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def wait(self, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = store.get_job(job["id"])
            if current and current["status"] in store.TERMINAL:
                return current
            time.sleep(.01)
        self.fail("Research job did not terminate")

    def run_research(self, scope="both", document_ids=None):
        response = self.client.post(f"/api/chats/{self.chat}/generations", json=dict(message="Compare protocols",
            research_mode=True, research_scope=scope, document_ids=document_ids or [], model="answer:local", reasoning="high"))
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait(response.json())
        self.assertEqual(job["status"], "completed", job["error"])
        return job, get_run(job["research"]["id"])

    def test_plan_validation(self):
        self.assertEqual(len(ResearchPlan.model_validate(public_plan()).subquestions), 3)
        for modification in (dict(subquestions=[]), dict(objective="")):
            with self.assertRaises(ValidationError):
                ResearchPlan.model_validate({**public_plan(), **modification})

    def adaptive_plan(self, depth):
        value = public_plan(2)
        value.update(depth=depth, depth_reason="Controlled complexity classification")
        value["subquestions"] = value["subquestions"][:2 if depth == "quick" else 3]
        self.plan_responses = [json.dumps(value)]
        return value

    def test_quick_budget_and_metadata_reconnect(self):
        self.adaptive_plan("quick")
        self.web.side_effect = lambda query, **kwargs: [Source(f"{query}:{i}", "web", url=f"https://example.org/{i}", relevance_score=i) for i in range(20)]
        job, run = self.run_research("web")
        self.assertEqual(run["metrics"]["depth"], "quick")
        self.assertFalse(run["metrics"]["escalated"])
        self.assertEqual(run["metrics"]["searches"], 3)
        self.assertEqual(run["metrics"]["candidates"], 15)
        self.assertEqual(run["metrics"]["evidence_selected"], 5)
        self.assertEqual(run["metrics"]["followup_searches"], 0)
        self.assertEqual(len(self.model_calls), 3)
        replay = self.client.get(f"/api/research/{run['id']}").json()
        self.assertEqual(replay["metrics"], run["metrics"])
        self.assertEqual(job["research"]["metrics"]["depth"], "quick")
        self.assertEqual(run["budgets"]["searches"], 3)
        for stage in ("planning", "retrieval", "reranking", "selection", "analysis", "synthesis"):
            self.assertGreaterEqual(run["metrics"]["stage_timings"][stage], 0)

    def test_standard_budget(self):
        self.adaptive_plan("standard")
        _, run = self.run_research()
        self.assertEqual(run["metrics"]["depth"], "standard")
        self.assertLessEqual(run["metrics"]["searches"], 6)
        self.assertLessEqual(run["metrics"]["candidates"], 30)
        self.assertLessEqual(run["metrics"]["evidence_selected"], 8)

    def test_quick_gaps_escalate_monotonically_and_remain_bounded(self):
        self.adaptive_plan("quick")
        self.coverage = "weak"
        _, run = self.run_research("web")
        self.assertEqual(run["metrics"]["initial_depth"], "quick")
        self.assertEqual(run["metrics"]["depth"], "deep")
        self.assertTrue(run["metrics"]["escalated"])
        self.assertEqual([item["to_depth"] for item in run["metrics"]["escalations"]], ["standard", "deep"])
        self.assertEqual(run["metrics"]["followup_searches"], 3)
        self.assertLessEqual(run["metrics"]["searches"], MAX_SEARCHES)

    def test_conflict_and_ambiguity_escalate_without_coverage_gaps(self):
        self.adaptive_plan("quick")
        self.ambiguous = True
        _, run = self.run_research("web")
        self.assertTrue(run["metrics"]["escalated"])
        self.assertIn("contradictory or ambiguous", run["metrics"]["escalations"][0]["reason"])

    def test_resolved_gap_stops_at_standard(self):
        self.adaptive_plan("quick")
        original = self.model
        analyses = 0
        def resolving_model(messages, inference, **options):
            nonlocal analyses
            if (options.get("schema") or {}).get("title") == "EvidenceAnalysis":
                analyses += 1
                self.coverage = "weak" if analyses == 1 else "sufficient"
            yield from original(messages, inference, **options)
        with patch.object(pipeline.provider, "stream", side_effect=resolving_model):
            _, run = self.run_research("web")
        self.assertEqual(run["metrics"]["depth"], "standard")
        self.assertEqual(run["metrics"]["followup_searches"], 1)

    def test_depth_subquestion_bounds_rejected(self):
        value = public_plan()
        value["depth"] = "quick"
        with self.assertRaises(ValidationError):
            ResearchPlan.model_validate(value)

    def test_duplicate_plan_ids_rejected(self):
        value = public_plan()
        value["subquestions"][1]["id"] = "sq1"
        with self.assertRaises(ValidationError):
            ResearchPlan.model_validate(value)

    def test_malformed_plan_retries(self):
        self.plan_responses = ["broken", json.dumps(public_plan())]
        self.run_research()
        self.assertEqual(sum(call[1].get("schema", {}).get("title") == "ResearchPlan" for call in self.model_calls), 2)

    def test_malformed_plan_fails_gracefully_without_raw_output(self):
        self.plan_responses = ["SECRET_INVALID_OUTPUT", "SECRET_INVALID_OUTPUT"]
        job = self.wait(self.submit())
        self.assertEqual(job["status"], "error")
        run = get_run(job["research"]["id"])
        self.assertEqual(run["status"], "failed")
        self.assertNotIn("SECRET_INVALID_OUTPUT", json.dumps(store.events()) + json.dumps(run))

    def test_independent_subquestions_execute(self):
        _, run = self.run_research()
        self.assertEqual({item["subquestion_id"] for item in run["searches"]}, {"sq1", "sq2", "sq3"})
        self.assertEqual(len(run["searches"]), 6)

    def test_web_only(self):
        _, run = self.run_research("web")
        self.documents.assert_not_called()
        self.assertEqual({item["source_type"] for item in run["evidence"]}, {"web"})

    def test_documents_only(self):
        _, run = self.run_research("documents")
        self.web.assert_not_called()
        self.assertEqual({item["source_type"] for item in run["evidence"]}, {"document"})

    def test_combined(self):
        _, run = self.run_research()
        self.assertEqual({item["source_type"] for item in run["evidence"]}, {"web", "document"})

    def test_attachment_scoping(self):
        self.run_research("documents", ["doc1"])
        self.assertTrue(all(call.kwargs["document_ids"] == ["doc1"] for call in self.documents.call_args_list))

    def test_url_cache_dedup_and_failure_cache(self):
        cache = WebCache()
        with patch("backend.research.web_cache.read_webpages", return_value=[dict(url="https://example.org/a", text="body")]) as read:
            cache.read(["https://example.org/a#one", "https://example.org/a#two", "https://example.org/fail"])
            cache.read(["https://example.org/a", "https://example.org/fail"])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(cache.metrics()["urls_fetched"], 2)
            self.assertEqual(cache.metrics()["fetch_failures"], 1)

    def test_exact_passage_persisted(self):
        job, run = self.run_research("documents")
        self.assertEqual(run["evidence"][0]["passage"], "Exact PDF passage\nquery 1-0  ")
        with db.connection() as connection:
            passage = connection.execute("SELECT passage FROM research_evidence WHERE id = ?", (run["evidence"][0]["id"],)).fetchone()[0]
        self.assertEqual(passage, run["evidence"][0]["passage"])
        self.assertEqual(job["assistant_message"]["research_run_id"], run["id"])

    def test_passage_dedup_preserves_discovery_contexts(self):
        self.documents.side_effect = lambda *args, **kwargs: [Source("identical passage", "document", document_id="doc1", chunk_id="doc1:one", relevance_score=1)]
        _, run = self.run_research("documents")
        self.assertEqual(len(run["evidence"]), 1)
        self.assertEqual(len(run["evidence"][0]["occurrences"]), 3)
        self.assertEqual(run["metrics"]["duplicates"], 2)

    def test_claim_mapping_persisted(self):
        _, run = self.run_research()
        with db.connection() as connection:
            links = connection.execute("SELECT * FROM research_claim_evidence WHERE run_id = ?", (run["id"],)).fetchall()
        self.assertEqual(links[0]["evidence_id"], run["analysis"]["claims"][0]["supporting_evidence_ids"][0])
        self.assertEqual(links[0]["relation"], "support")

    def test_conflicting_evidence(self):
        self.conflicts = True
        job, run = self.run_research()
        self.assertEqual(len(run["analysis"]["claims"][0]["conflicting_evidence_ids"]), 1)
        self.assertNotIn("The disagreement remains unresolved", job["assistant_message"]["content"])
        with db.connection() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM research_claim_evidence WHERE relation='conflict'").fetchone()[0], 1)

    def test_weak_coverage_one_bounded_followup(self):
        self.coverage = "weak"
        job, run = self.run_research()
        self.assertEqual(run["metrics"]["followup_passes"], 1)
        self.assertEqual(run["metrics"]["followup_searches"], 2)
        self.assertTrue(all(item["status"] == "weak" for item in run["analysis"]["coverage"]))
        self.assertNotIn("**Evidence limitations**", job["assistant_message"]["content"])

    def test_missing_evidence_is_retained_without_hallucination(self):
        self.web.return_value = []
        self.web.side_effect = None
        job, run = self.run_research("web")
        self.assertEqual(run["analysis"]["claims"], [])
        self.assertTrue(all(item["status"] == "missing" for item in run["analysis"]["coverage"]))
        self.assertIn("no supported conclusion", job["assistant_message"]["content"])

    def test_failed_search_continues(self):
        self.web.side_effect = RuntimeError("PRIVATE_ERROR_BODY")
        job, run = self.run_research()
        self.assertTrue(any(item["status"] == "failed" for item in run["searches"]))
        self.assertNotIn("PRIVATE_ERROR_BODY", json.dumps(run) + json.dumps(store.events()))
        self.assertTrue(job["assistant_message"]["content"])

    def test_candidate_search_selection_budgets(self):
        self.documents.side_effect = lambda query, **kwargs: [Source(f"{query}:{i}", "document", document_id="doc1", relevance_score=i) for i in range(200)]
        _, run = self.run_research()
        self.assertLessEqual(run["metrics"]["candidates"], MAX_CANDIDATES)
        self.assertLessEqual(run["metrics"]["searches"], MAX_SEARCHES)
        self.assertLessEqual(run["metrics"]["evidence_selected"], MAX_EVIDENCE)

    def test_url_fetch_budget(self):
        cache = WebCache()
        with patch("backend.research.web_cache.read_webpages", return_value=[]):
            cache.read([f"https://example.org/{i}" for i in range(100)])
        self.assertEqual(cache.metrics()["urls_fetched"], 18)
        self.assertEqual(len(cache.last_operation["budget_skipped"]), 82)

    def test_citations_map_to_persisted_selected_evidence(self):
        job, run = self.run_research()
        selected_ids = {item["id"] for item in run["evidence"] if item["selected_for_synthesis"]}
        self.assertEqual({item["evidence_id"] for item in job["assistant_message"]["sources"]}, selected_ids)

    def test_invalid_and_split_citations(self):
        filter_ = pipeline.CitationFilter(2)
        text = filter_.feed("Answer [SOU") + filter_.feed("RCE 1], [SOURCE 99].") + filter_.feed("", final=True)
        self.assertEqual(text, "Answer [SOURCE 1], [citation unavailable].")

    def test_grouped_citations_preserve_each_valid_evidence_reference(self):
        filter_ = pipeline.CitationFilter(12)
        text = filter_.feed("Fact [SOURCE 1, SOUR") + filter_.feed("CE 2, 99]. [SOURCE 3; 6]")
        self.assertEqual(text, "Fact [SOURCE 1] [SOURCE 2] [citation unavailable]. [SOURCE 3] [SOURCE 6]")

    def test_reconnect_and_history_reconstruct(self):
        job, run = self.run_research()
        history = self.client.get(f"/api/chats/{self.chat}/messages").json()
        self.assertEqual(history[1]["research_run_id"], run["id"])
        self.assertEqual(history[1]["research_revision"], run["revision"])
        replay = self.client.get(f"/api/research/{run['id']}").json()
        self.assertEqual(replay["evidence"], run["evidence"])
        self.assertEqual(store.snapshot()["jobs"][0]["research"]["id"], run["id"])

    def test_model_and_reasoning_propagate_to_every_stage(self):
        self.run_research()
        self.assertEqual(len(self.model_calls), 3)
        for inference, options in self.model_calls:
            self.assertEqual(inference["model"], "answer:local")
            self.assertEqual(inference["settings"], {"think": "high"})
            self.assertTrue(options["research"])

    def test_artifacts_not_sent_in_sse(self):
        self.run_research()
        events = store.events()
        self.assertTrue(any(item["event"]["type"] == "research_plan_ready" for item in events))
        for event in events:
            self.assertNotIn("artifact", event["event"])
            self.assertNotIn("passage", event["event"])

    def test_analysis_rejects_unknown_evidence(self):
        analysis = EvidenceAnalysis(claims=[dict(id="c1", statement="claim", supporting_evidence_ids=["unknown"], conflicting_evidence_ids=[])],
            coverage=[dict(subquestion_id=f"sq{i}", status="sufficient", explanation="covered") for i in range(1, 4)])
        with self.assertRaises(ValueError):
            pipeline.validate_analysis(analysis, [], ResearchPlan.model_validate(public_plan()).subquestions)

    def test_same_passage_can_preserve_ambiguous_support_and_conflict(self):
        analysis = EvidenceAnalysis(claims=[dict(id="c1", statement="Qualified claim", supporting_evidence_ids=["retained"], conflicting_evidence_ids=["retained"])],
            coverage=[dict(subquestion_id=f"sq{i}", status="weak", explanation="Ambiguous support") for i in range(1, 4)])
        pipeline.validate_analysis(analysis, [dict(id="retained", subquestion_ids=["sq1", "sq2", "sq3"])], ResearchPlan.model_validate(public_plan()).subquestions)
        self.assertEqual(analysis.claims[0].conflicting_evidence_ids, ["retained"])

    def test_ambiguous_mapping_persisted_and_flagged(self):
        self.ambiguous = True
        job, run = self.run_research()
        claim = run["analysis"]["claims"][0]
        self.assertNotIn("the same passage appears in support and conflict", job["assistant_message"]["content"])
        self.assertEqual(claim["ambiguous_evidence_ids"], claim["supporting_evidence_ids"])
        with db.connection() as connection:
            relations = {row[0] for row in connection.execute("SELECT relation FROM research_claim_evidence WHERE run_id = ?", (run["id"],))}
        self.assertEqual(relations, {"support", "conflict"})

    def test_retry_cannot_change_research_mode(self):
        job = self.submit(request_id="same")
        self.wait(job)
        retry = self.client.post(f"/api/chats/{self.chat}/generations", json=dict(message="Compare protocols",
            request_id="same", research_mode=False, research_scope="both", model="answer:local", reasoning="high"))
        self.assertEqual(retry.status_code, 409)
        self.assertEqual(len(db.list_messages(self.chat)), 2)

    def test_restart_marks_run_interrupted(self):
        chat = db.create_chat("Unstarted")
        job, _ = store.create_job(chat["id"], "question", research_mode=True, inference=self.inference)
        store.interrupt_unfinished()
        self.assertEqual(get_run(job["research"]["id"])["status"], "interrupted")

    def test_delete_cascades_and_rejects_late_writes(self):
        job, run = self.run_research()
        self.client.delete(f"/api/chats/{self.chat}")
        self.assertIsNone(get_run(run["id"]))
        self.assertIsNone(store.update(job["id"], dict(type="research_started", run_id=run["id"])))
        self.assertEqual(self.client.get(f"/api/research/{run['id']}").status_code, 404)

    def test_concurrent_normal_research_navigation_disconnect_stop_archive_delete(self):
        entered, release = threading.Event(), threading.Event()
        original = self.documents.side_effect
        def gated(*args, **kwargs):
            entered.set()
            release.wait(4)
            return original(*args, **kwargs)
        self.documents.side_effect = gated
        research = self.submit()
        self.assertTrue(entered.wait(2))
        other = self.client.post("/api/chats", json={"title": "Normal"}).json()["id"]
        with patch("backend.core.assistant.plan", return_value=dict(use_documents=False, use_web=False, fresh=False, search_query="")):
            normal = self.client.post(f"/api/chats/{other}/generations", json=dict(message="Hello", model="answer:local")).json()
            self.assertEqual(self.wait(normal)["status"], "completed")
        for chat in (other, self.chat, other):
            self.assertEqual(self.client.get(f"/api/chats/{chat}/messages").status_code, 200)
        class Disconnected:
            async def is_disconnected(self): return True
        async def disconnected():
            async for _ in api.generation_events(Disconnected(), 0, research["id"]):
                self.fail("Disconnected subscriber received data")
        asyncio.run(disconnected())
        self.assertEqual(store.get_job(research["id"])["status"], "running")
        self.client.patch(f"/api/chats/{self.chat}", json={"archived": True})
        release.set()
        final = self.wait(research)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(get_run(final["research"]["id"])["status"], "completed")

    def test_stop_during_retrieval_retains_artifacts(self):
        entered, release = threading.Event(), threading.Event()
        def gated(*args, **kwargs):
            entered.set(); release.wait(4)
            return []
        self.documents.side_effect = gated
        job = self.submit()
        self.assertTrue(entered.wait(2))
        self.client.post(f"/api/generations/{job['id']}/cancel")
        release.set()
        final = self.wait(job)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(get_run(final["research"]["id"])["status"], "cancelled")
        self.assertFalse(final["assistant_message"]["content"])

    def test_delete_during_retrieval_never_resurrects(self):
        entered, release = threading.Event(), threading.Event()
        def gated(*args, **kwargs):
            entered.set(); release.wait(4)
            return []
        self.documents.side_effect = gated
        job = self.submit()
        self.assertTrue(entered.wait(2))
        self.client.delete(f"/api/chats/{self.chat}")
        release.set()
        self.assertIsNone(get_run(job["research"]["id"]))
        self.assertIsNone(store.get_job(job["id"]))

    def test_hidden_reasoning_never_reaches_research_sqlite_sse_api(self):
        secret = "PRIVATE_THINKING_SENTINEL"
        def sdk_chat(messages, **settings):
            yield {"message": {"thinking": secret, "content": ""}}
            yield {"message": {"content": "<thi"}}
            yield {"message": {"content": "nk>" + secret + "</think>"}}
            for event in self.model(messages, self.inference, schema=settings.get("format"),
                                    output_tokens=settings["options"]["num_predict"]):
                if event["type"] == "token":
                    yield {"message": {"content": event["content"]}}
        current = dict(digest=self.inference["model_digest"], reasoning=self.inference["capability"], context_length=32768)
        with patch.object(pipeline.provider, "stream", OllamaProvider.stream.__get__(pipeline.provider)), \
             patch.object(pipeline.provider, "inspect", return_value=current), patch.object(ollama, "chat", side_effect=sdk_chat):
            job, run = self.run_research()
        self.assertNotIn(secret, json.dumps(run) + json.dumps(store.events()) + json.dumps(db.list_messages(self.chat)))
        self.assertNotIn(secret.encode(), db.DB_PATH.read_bytes())
        self.assertEqual(job["inference"]["actual_settings"]["think"], "high")
        self.assertEqual(job["inference"]["actual_settings"]["options"]["num_ctx"], 16384)

    def test_private_sdk_error_never_reaches_logs_or_persistence(self):
        secret = "PRIVATE_SDK_ERROR_SENTINEL"
        current = dict(digest=self.inference["model_digest"], reasoning=self.inference["capability"], context_length=32768)
        with patch.object(pipeline.provider, "stream", OllamaProvider.stream.__get__(pipeline.provider)), \
             patch.object(pipeline.provider, "inspect", return_value=current), patch.object(ollama, "chat", side_effect=ValueError(secret)), \
             self.assertLogs("backend.generation.jobs", level="ERROR") as logs:
            job = self.wait(self.submit())
        self.assertEqual(job["status"], "error")
        self.assertNotIn(secret, "".join(logs.output) + json.dumps(store.events()) + json.dumps(get_run(job["research"]["id"])))

    def test_publication_dates_require_explicit_source_metadata(self):
        self.assertIsNone(publication_date("<article>Updated today, 2026.</article>"))
        self.assertEqual(publication_date('<meta property="article:published_time" content="2025-01-02">'), "2025-01-02")

    def test_plan_rejects_excess_queries(self):
        value = public_plan(3)
        with self.assertRaises(ValidationError):
            ResearchPlan.model_validate(value)

    def test_search_and_candidate_budget_terminates_retrieval(self):
        with patch.object(pipeline, "MAX_SEARCHES", 2), patch.object(pipeline, "MAX_CANDIDATES", 3):
            _, run = self.run_research()
        self.assertLessEqual(run["metrics"]["searches"], 2)
        self.assertLessEqual(run["metrics"]["candidates"], 3)

    def test_automatic_research_reuses_planner_and_attachment_scope(self):
        with patch.object(pipeline, "plan", return_value=dict(use_web=False, use_documents=True, fresh=False)) as planner:
            self.run_research("auto", ["doc1"])
        planner.assert_called_once()
        self.web.assert_not_called()

    def test_preexisting_messages_survive_additive_schema_initialization(self):
        message = db.add_message(self.chat, "assistant", "Existing reply")
        db.init_db()
        history = db.list_messages(self.chat)
        self.assertEqual(history[0]["id"], message["id"])
        self.assertEqual(history[0]["content"], "Existing reply")
        self.assertIsNone(history[0]["research_run_id"])
