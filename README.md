# Atlas

An explainable AI research engine built to explore RAG, embeddings,
semantic search, reranking, web retrieval, and LLM systems.

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

The existing 600-token response cap remains identical across reasoning settings.
This cap includes thinking tokens where Ollama counts them; a model that spends
the entire budget reasoning can return no final answer. Atlas reports that failure
instead of displaying private reasoning or pretending generation succeeded.

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

## Checks

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_dev.py
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
.\.venv\Scripts\python.exe -m backend.tests.smoke_local --web
.\.venv\Scripts\python.exe -m backend.tests.smoke_models
```

Regression tests use isolated SQLite/Chroma stores and deterministic model substitutes.
The launcher smoke test checks actual Windows Ctrl+C and forced-exit cleanup with
temporary ports and backend data, leaving any existing Atlas servers running.
The smoke test starts a temporary HTTP server and exercises real local Ollama,
PDF embeddings, reranking, streamed answers, and (with `--web`) live web research.
Both leave existing Atlas data untouched. The smoke test requires cached or
downloadable embedding/reranking models; omit `--web` to avoid live web requests.

From `frontend/`:

```powershell
npm.cmd run build
npm.cmd run lint
npm.cmd test
```
