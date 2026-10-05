# Atlas

An explainable AI research engine built to explore RAG, embeddings,
semantic search, reranking, web retrieval, and LLM systems.

## Evaluation

Atlas's frozen, manually curated 45-query document benchmark measures final source ranking:

| Metric | Result |
| --- | ---: |
| Recall@1 | 93.3% |
| Recall@3 | 100% |
| Recall@5 | 100% |
| MRR | 0.967 |

Atlas also benchmarks inference and end-to-end latency. See [Evaluation](docs/evaluation.md)
for methodology, latency results, the hybrid experiment, limitations, and raw artifacts.

## Run locally (PowerShell)

After the one-time dependency setup, start both services from the repository root:

```powershell
python dev.py
python dev.py --open   # Also open Atlas in your default browser once ready
python dev.py --check  # Check readiness without starting services
```

The launcher uses `.venv` for FastAPI and the locally installed Vite frontend.
It checks Python/backend imports, Node/npm/frontend dependencies, Ollama, every
configured chat/planner/answer model, and ports 8000/5173. It never installs
dependencies, starts Ollama, or downloads models; missing prerequisites produce
actionable errors. `OLLAMA_HOST`, when set, selects the same Ollama host used by
the backend client.

Backend and frontend logs have separate prefixes. Once both services are ready,
the launcher prints http://127.0.0.1:5173. Ctrl+C stops both services and their
child processes. If either service exits unexpectedly, the other is stopped too.
The backend reloads when files in `backend/` change; Vite retains its normal hot
reload. Backend reloads interrupt unfinished generation jobs as documented below.
If `python` is not on PATH, use `.\.venv\Scripts\python.exe dev.py` instead.

For first-time setup, create `.venv` if needed and install dependencies explicitly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Push-Location frontend
npm.cmd ci
Pop-Location
```

Ollama must be running with `qwen3:4b-instruct` available. If necessary, run
`ollama serve` in another terminal and `ollama pull qwen3:4b-instruct` once.
Embedding and reranking models download on first use; subsequent inference runs locally.

The separate commands remain available when debugging a single service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open http://127.0.0.1:5173. The API documentation is at http://127.0.0.1:8000/docs.

## Research tools

The planner chooses tools automatically. Enabling the globe forces web search for
one submitted message, without forcing a recent-results filter.

The document button accepts up to 20 PDFs per message, each at most 25 MB.
Files upload and index before sending; attachments scope document retrieval to
those files for that message. Web research can still run alongside document retrieval.
Removing an unsent file detaches it without deleting indexed knowledge.
Scanned PDFs need OCR beforehand; password-protected PDFs are unsupported.

Chats and attachment metadata persist in `data/atlas.db`. Uploaded PDFs live in
`data/documents/`; both manual and uploaded documents use `data/chroma` and the
existing `atlas_documents` collection. Schema updates preserve existing chats.

Manual PDF ingestion from `documents/` remains available:

```powershell
.\.venv\Scripts\python.exe -m backend.ingestion.ingest_pdf
```

## Models and reasoning

The composer model control discovers installed models through Ollama; refresh it
after adding or removing models. Atlas never downloads a model automatically.
`GET /api/models` reports model family, maximum context length, quantization,
capabilities, availability, and verified reasoning controls. Maximum context is
the model's advertised limit, not a promise that the current inference context
or available memory reaches that limit.

Controls come from `/api/show`'s `thinking.values` metadata. Boolean values expose
Off/On; named values expose only those exact levels. Medium is Atlas's default
when supported. A single supported value is fixed, and missing metadata means
controls are unverified. An advertised `thinking` capability alone never creates
Low/Medium/High controls. Unsupported requests are rejected before message creation.
See [Ollama's thinking API documentation](https://docs.ollama.com/capabilities/thinking).

Model/reasoning preferences apply to the next submitted job. Each job retains its
own model digest, requested setting, capability snapshot, resolved reasoning mode,
and actual answer inference arguments. Changing these preferences or switching
threads cannot change an existing job. Assistant metadata survives history loads,
SSE reconnects, and reloads. Older responses retain unknown inference metadata;
Atlas does not invent how they were generated.

Normal chat, PDF RAG, and web-grounded answers share the Ollama inference adapter.
The tool planner remains the separately configured `PLANNER_MODEL`; answer model
selection does not change retrieval routing. Retrieval and reasoning stay independent.
Final-answer text streams normally. Hidden thinking is discarded, including legacy
inline thinking markers; only safe Thinking/Writing activity reaches the UI.
Provider error bodies are not logged or persisted. Stop remains available during
hidden thinking. If a model or its capabilities change while a job is queued,
Atlas reports an error rather than silently changing the submitted controls.

The ordinary response cap of 600 tokens remains identical across reasoning settings.
This cap includes thinking tokens where Ollama counts them; a model that spends
the entire budget reasoning can return no final answer. Atlas reports that failure
instead of displaying private reasoning or pretending generation succeeded.

## Deep Research

Choose **Research → Deep Research** in the composer. Sources can be Automatic,
Web, Documents, or Web + documents. Automatic reuses the existing planner;
attached PDFs constrain document retrieval to those document IDs. Without
attachments, document research searches the existing indexed library. Model and
reasoning selection remains independent and is captured when the job is submitted.

Each run follows a validated public plan → independent subquestion retrieval →
passage deduplication and cross-encoder ranking → balanced evidence selection →
claim/support/conflict and coverage analysis → at most one follow-up pass →
streamed cited synthesis. Structured artifacts use Ollama's
[JSON-schema output support](https://docs.ollama.com/capabilities/structured-outputs)
and Pydantic validation. Invalid plans or analysis retry once and then fail with
safe diagnostics. Search failures can coexist with usable evidence from other
operations. Empty evidence produces an explicit unsupported-result message.

The existing structured planner chooses an adaptive budget without an extra model call:
Quick uses 1–2 subquestions, at most 3 searches, 15 candidate passages and 5 selected;
Standard uses up to 3 subquestions, 6 searches, 30 candidates and 8 selected.
Coverage gaps or conflicting/ambiguous evidence can escalate Quick to Standard,
then Standard to Deep. Budgets never downgrade during a run. The research panel
shows the current depth, counts, reason, escalation, and stage timings; these are persisted.

Deep limits live in `backend/research/schemas.py`: up to 5 subquestions, up to 2 queries per
subquestion, 22 retrieval operations total, 18 distinct URL fetch attempts, 60
chunks per webpage, 120 retrieved/reranked candidate passages, and 12 selected
passages, with up to 5 candidates per search. Analysis retains at most 6 major
claims and 4 supporting / 4 conflicting passages per claim. One follow-up pass permits at most 2 additional retrieval operations
within the same overall budgets. Structured output is capped at 4096 tokens and
synthesis at 2048, independently of reasoning mode. Research requests an explicit
16384-token context capped by the model's advertised maximum; a conservative
input-size check rejects oversized contexts rather than silently dropping evidence.
These are stage capacity limits, not fabricated Low/Medium/High effort levels.

Additive SQLite tables retain `research_runs`, `research_subquestions`,
`research_searches`, `research_evidence`, `research_claims`, and
`research_claim_evidence`. Runs belong to generation jobs through cascading foreign
keys. Artifact writes and safe SSE events share the existing job transaction,
so Stop/delete prevent late writes. Failed, cancelled, or interrupted runs retain
their completed artifacts; deleting a chat removes them. Reload and reconnect do
not repeat searches. `GET /api/research/{run_id}` returns saved artifacts; assistant
messages and job snapshots retain the run ID and artifact revision.

Expand a response's **Deep Research** panel to inspect actual searches, plan,
coverage, evidence, claims, counts, and stage timings. Click `[SOURCE n]` to open
the exact saved passage. Citation indices map to stable evidence IDs on that
message's saved source list; unknown generated indices are marked unavailable.
Evidence includes stable chunk IDs, document/page or URL provenance, retrieval
and reranking scores, original query, timestamp, and repeated discovery contexts.
Publication dates are retained only from explicit webpage publication metadata.
Unknown dates stay unknown. Domain/document counts are provenance signals;
they do not verify editorial independence or truth. Claim and coverage analysis
is model-generated and can be wrong even when citation IDs are valid. A passage
mapped as both support and conflict is retained with an explicit ambiguity flag;
the inspector exposes both links and synthesis must qualify that claim.
Detailed conflicts, ambiguous mappings, and weak/missing coverage remain in the
expandable research panel. The final answer briefly qualifies material uncertainty
without appending the internal evidence-analysis prose.

### Read aloud and response performance

**Settings** opens the preferences panel: Light/Dark/System appearance, installed
model and supported reasoning selection, Deep Research default for new chats,
local Read Aloud voice/rate, dictation language, and default Performance expansion.
Preferences persist in browser `localStorage`; model selection retains its existing
storage key. Reset restores browser preferences and model defaults without deleting
chats, documents, or backend configuration. Voice changes apply on the next playback.

The composer microphone supports **Dictate** in browsers exposing
`SpeechRecognition` or `webkitSpeechRecognition`. Click to start, click again to
stop, or press Escape. Final transcripts append to the current draft; review/edit
them and send explicitly. Switching chats stops recognition. Dictation and Read
Aloud stop each other, so only one speech session runs at a time. Permission or
microphone errors appear beside the composer; unsupported browsers disable dictation.
Browser-native recognition may send audio to a browser/platform service and may
require connectivity. Atlas does not guarantee local processing for dictation;
privacy and availability depend on that browser/platform implementation. No paid
API or additional speech service is configured by Atlas. A separate provider
interface allows a future local STT implementation.

Assistant message actions include a speaker control: click to read aloud, click
again to stop. Starting another message or switching chats stops previous playback.
The initial provider uses browser speech synthesis with local OS voices only;
install/enable a system voice if the browser reports none. Citations and link URLs
are omitted from spoken text. The speech provider interface allows a future replacement.

Expand **Performance** below an assistant response to inspect available timings.
New jobs persist backend TTFT/E2E and stage durations; actual Ollama usage supplies
TPS and decode-average TPOT when available. Inference TTFT measures time to the
first filtered answer chunk. Backend TTFT begins when the worker starts, so neither
it nor backend E2E includes queue wait or browser/network delivery. Old messages
show only measurements already retained; missing metrics are not estimated.

To try it in the UI:

1. From the AtlasLLM root run `python dev.py --open` (or open
   `http://127.0.0.1:5173` after `python dev.py` reports ready).
2. Create a thread. Use the model control to select an installed model;
   `qwen3:4b-instruct` completed the isolated research smoke. Reasoning controls
   still show only modes genuinely supported by that model.
3. Click **Research**, choose **Deep Research**, then choose **Web** for an
   initial trial. Ask: “Compare how TCP and UDP handle reliability, ordering,
   and suitable applications.” Click outside the popover, then Send.
4. Expand **Deep Research** below the response to inspect the plan, actual
   searches, coverage, retained passages, claims, and counts. After synthesis,
   click a `[SOURCE n]` citation to inspect its exact saved passage.
5. For PDF research, use **Add files**, select a text-based PDF, wait for indexing,
   then choose **Documents** or **Web + documents** and ask about its contents.
6. While research runs, switch to another thread and submit a Standard response.
   Both jobs continue. Reload the page to verify reconnection; use the sidebar
   or “Response ready” notification to return to the completed research.
7. Try **Stop** on a separate run. Archiving lets an existing run finish;
   deleting a thread removes its research artifacts too. Use a disposable
   thread for these checks. Choose **Standard** to return to ordinary generation.

Research uses the same executor, Stop, notifications, archive/delete rules,
restart interruption, incremental persistence, and reconnectable SSE as ordinary
responses. Cancellation waits for an in-progress synchronous search/fetch/ranking
operation to return. Private model thinking remains filtered by the provider and
is never stored as a research artifact or SSE payload. Run one backend process.

## Background responses

Responses run as backend jobs, independently of the selected thread or SSE
subscriber. Each thread allows one active job; two workers let different threads
run concurrently, with additional threads queued. Ollama may queue inference
according to its own hardware and parallelism settings.

Switching threads, closing the tab, or reloading does not cancel work. The sidebar
shows working, ready, and error states; a clickable notification opens a response
completed in the background. Stop cancels only that thread's job. Cancellation is
cooperative: an in-progress retrieval operation finishes before the worker observes
the request. The UI shows Stopping until cancellation completes.

SQLite retains job state, sequenced SSE events, and incremental assistant text.
Reconnect starts from an authoritative snapshot and resumes events after its
cursor. Submission request IDs prevent duplicate messages on retried requests.
Archiving permits an existing response to finish and blocks new messages until
restored. Deleting a thread cancels its work and atomically removes its messages,
jobs, and event content; late worker writes cannot recreate it. Unfinished jobs
become interrupted when the backend restarts, preserving saved text.

Run one Uvicorn application worker. The local job executor is process-owned;
multiple API worker processes are not supported. `--reload` restarts the backend
when Python files change and therefore interrupts unfinished jobs.

API: `POST /api/chats/{id}/generations` submits a job;
`GET /api/generations` returns the snapshot and cursor;
`GET /api/generations/events?after={cursor}` subscribes to all threads.
Individual jobs provide GET state/events and POST cancel/acknowledge endpoints
under `/api/generations/{id}`. The existing message-stream endpoint remains
compatible and subscribes to a backend-owned job.

## Performance benchmark (explicit, optional)

For the isolated, local-document RAG baseline (45 gold questions, no LLM or web calls),
run `python -m backend.evaluation.rag`. See [RAG baseline evaluation](docs/benchmarks/rag-baseline.md)
for metrics, storage isolation, timeouts, and dataset details.

From the repository root:

```powershell
python -m backend.evaluation.benchmark
python -m backend.evaluation.benchmark --model qwen3:4b-instruct
```

Quick runs exactly three short controlled prompts: a greeting, arithmetic, and a
one-sentence rewrite. It runs three answer inferences and the normal planner for
each request. It never runs Deep Research or live web searches. Answer output is
capped at 128 tokens, independent of reasoning mode. The CLI uses the project's
`.venv` for its isolated worker. Default limits are 25 seconds per request and a
hard 120-second overall deadline, including application startup and cleanup.
Short non-thinking models should be much faster; cold starts, hardware, or fixed
thinking can cause partial results. No benchmark runs automatically on import,
build, or unit-test discovery.

TTFT measures local HTTP submission → first visible assistant text received over
SSE, including planning, job/SQLite overhead, queueing, and applicable retrieval;
it excludes browser rendering. Inference TTFT measures the answer provider request
→ first filtered visible content chunk, including Ollama queue/load/prefill/thinking.
TPS is `eval_count / (eval_duration / 1e9)` and TPOT is mean provider decode time
`eval_duration / eval_count`; both include hidden thinking if Ollama counts it.
TPOT is **not** a measured per-token ITL distribution. Provider counts are never
estimated from words or chunks. Missing metadata yields N/A. Stream-chunk intervals
are saved separately, and true token ITL remains unavailable without token-level
timestamps. Timings use `time.perf_counter()`.

Optional `--pdf "path/to/file.pdf"` indexes a PDF through the existing upload/RAG
pipeline into temporary stores and uses document retrieval for prompt three.
Retrieval/reranking metrics are N/A in chat-only Quick mode. PDF preparation counts
toward total benchmark time, not request latency. Controlled benchmark routing
disables web and scopes document use to that PDF; real planner latency and its
original route are retained. Production chats, documents, and vectors are untouched.

Full mode is explicitly opt-in:

```powershell
python -m backend.evaluation.benchmark --full
python -m backend.evaluation.benchmark --full --models qwen3:4b-instruct qwen3:4b
python -m backend.evaluation.benchmark --full --model MODEL --reasoning-modes default off on
```

Full defaults to five repetitions of the three prompts per model/reasoning
combination, with 60-second request and 900-second overall limits. It reports mean,
median, nearest-rank p95, sample variance, and sample count, grouped by model,
actual reasoning mode, and chat/documents mode. Unsupported reasoning settings
fail through the existing capability checks; only supported comparisons produce
measurements. Small sample p95 values should not be treated as established tail
latency. No automatic warm-up or hidden extra answer runs occur.

Use `--timeout`, `--run-timeout`, and `--output-tokens` to set explicit bounds;
Quick's overall deadline cannot exceed 120 seconds. A timeout stops the owned
worker and saves completed and partial measurements (exit code 124). Results are
stored as timestamped JSON under `data/benchmarks/`, including inference settings
and token metadata, without answer text or private thinking. Reuse a saved result:

```powershell
python -m backend.evaluation.benchmark --show "data/benchmarks/RESULT.json"
```

Mocked benchmark tests: `.\.venv\Scripts\python.exe -m unittest backend.tests.test_benchmark -q`.
Never run Full automatically during implementation or after unrelated changes.

## Development checks

For normal development, run deterministic tests directly relevant to the change,
Python compilation/import checks for backend edits, and TypeScript/build for
frontend edits. Do not repeatedly run full regressions or live smoke tests after
small fixes. Run full regression testing once before a major commit/milestone.

For Phase C, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_deep_research backend.tests.test_integration_timeout -q
.\.venv\Scripts\python.exe -m compileall -q backend dev.py
.\.venv\Scripts\python.exe -c "import backend.api, backend.research.pipeline"
cd frontend
node --test tests/research.test.mjs
npm.cmd run build
```

Full regression / optional live checks, run explicitly at milestones:

```powershell
.\.venv\Scripts\python.exe -m compileall -q backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_dev.py
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
.\.venv\Scripts\python.exe -m backend.tests.smoke_local --web
.\.venv\Scripts\python.exe -m backend.tests.smoke_models
.\.venv\Scripts\python.exe -m backend.tests.smoke_deep_research
```

Regression tests use isolated SQLite/Chroma stores and deterministic model substitutes.
The launcher smoke test checks actual Windows Ctrl+C and forced-exit cleanup with
temporary ports and backend data, leaving any existing Atlas servers running.
The smoke test starts a temporary HTTP server and exercises real local Ollama,
PDF embeddings, reranking, streamed answers, and (with `--web`) live web research.
Both leave existing Atlas data untouched. The smoke test requires cached or
downloadable embedding/reranking models; omit `--web` to avoid live web requests.
Standalone Ollama/PDF/web smoke processes have a hard five-minute wall-clock
deadline (exit code 124). This also bounds continuous SSE heartbeats and hung
backend worker cleanup; a deadline exit can leave its temporary test directory
behind but closes its in-process server and worker threads. Deep Research SSE has
a 90-second read-inactivity limit. Windows launcher checks already bound startup,
shutdown, and process waits; their HTTP probes now time out after ten seconds.

From `frontend/`:

```powershell
npm.cmd run build
npm.cmd run lint
npm.cmd test
```
