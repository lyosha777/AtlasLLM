import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from typing import Literal
from backend.research.store import get_run

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf.errors import PyPdfError

from backend.core.assistant import stream_assistant
from backend.core.config import MAX_PDF_BYTES, UPLOADS_DIR
from backend.core.document_store import get_document_collection
from backend.ingestion.ingest_pdf import ingest_pdf
from backend.generation.jobs import JobManager
from backend.generation import store as job_store
from backend.models.ollama_provider import provider, ProviderUnavailable
from backend.db import (
    create_chat,
    get_chat,
    init_db,
    list_chats,
    list_messages,
    save_document,
    update_chat,
)


logger = logging.getLogger(__name__)


class ChatCreate(BaseModel):
    title: str = "New thread"


class ChatUpdate(BaseModel):
    title: str | None = None
    archived: bool | None = None


class MessageCreate(BaseModel):
    message: str
    research_mode: bool = False
    research_scope: Literal["auto", "web", "documents", "both"] = "auto"
    force_web: bool = False
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    request_id: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    reasoning: str | None = Field(default=None, min_length=1, max_length=40)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    job_store.interrupt_unfinished()
    manager = JobManager(lambda **request: stream_assistant(**request))
    app.state.jobs = manager
    try:
        yield
    finally:
        await asyncio.to_thread(manager.close)


app = FastAPI(
    title="Atlas API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse(event: dict, sequence: int | None = None) -> str:
    prefix = f"id: {sequence}\n" if sequence is not None else ""
    return f"{prefix}data: {json.dumps(event)}\n\n"


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/api/models")
def models():
    return provider.discover()


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile):
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if Path(filename).suffix.lower() != ".pdf":
        file.file.close()
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    temporary_path = UPLOADS_DIR / f"{document_id}.upload"
    pdf_path = UPLOADS_DIR / f"{document_id}.pdf"
    indexing_started = False
    try:
        size = 0
        with temporary_path.open("xb") as destination:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise HTTPException(status_code=413, detail="PDFs must be 25 MB or smaller.")
                destination.write(chunk)
        with temporary_path.open("rb") as uploaded:
            if b"%PDF-" not in uploaded.read(1024):
                raise HTTPException(status_code=422, detail="This file is not a valid PDF.")
        try:
            indexing_started = True
            result = ingest_pdf(temporary_path, document_id=document_id, filename=filename)
        except (ValueError, PyPdfError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        temporary_path.replace(pdf_path)
        return save_document(result)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        pdf_path.unlink(missing_ok=True)
        if indexing_started:
            try:
                get_document_collection().delete(where={"document_id": document_id})
            except Exception:
                logger.exception("Could not clean up failed upload %s", document_id)
        raise
    finally:
        file.file.close()


@app.get("/api/chats")
def chats(archived: bool = False):
    return list_chats(archived)


@app.post("/api/chats")
def new_chat(payload: ChatCreate):
    title = payload.title.strip() or "New thread"

    return create_chat(title)


@app.patch("/api/chats/{chat_id}")
def patch_chat(
    chat_id: str,
    payload: ChatUpdate,
):
    chat = update_chat(
        chat_id,
        title=payload.title,
        archived=payload.archived,
    )

    if chat is None:
        raise HTTPException(
            status_code=404,
            detail="Chat not found",
        )

    return chat


@app.delete("/api/chats/{chat_id}")
def remove_chat(chat_id: str, request: Request):
    if not request.app.state.jobs.delete_chat(chat_id):
        raise HTTPException(
            status_code=404,
            detail="Chat not found",
        )

    return {
        "deleted": True
    }


@app.get("/api/chats/{chat_id}/messages")
def messages(chat_id: str):
    if get_chat(chat_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Chat not found",
        )

    return list_messages(chat_id)



def submit_job(request: Request, chat_id: str, payload: MessageCreate):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        selection = dict(chat_id=chat_id, message=message, force_web=payload.force_web,
                         document_ids=list(dict.fromkeys(payload.document_ids)),
                         model=payload.model, reasoning=payload.reasoning,
                         research_mode=payload.research_mode, research_scope=payload.research_scope)
        existing = job_store.replay_request(payload.request_id, **selection)
        if existing is not None:
            return existing
        chat = get_chat(chat_id)
        if chat is None:
            raise LookupError("Chat not found")
        if chat["archived"]:
            raise job_store.JobConflict("Restore this thread before sending a message.")
        inference = provider.resolve(payload.model, payload.reasoning)
        return request.app.state.jobs.submit(
            **selection, request_id=payload.request_id, inference=inference,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except job_store.JobConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ProviderUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/chats/{chat_id}/generations", status_code=202)
def create_generation(chat_id: str, payload: MessageCreate, request: Request):
    return submit_job(request, chat_id, payload)


@app.get("/api/generations")
def generation_snapshot():
    return job_store.snapshot()


async def generation_events(request: Request, after: int, job_id=None, legacy=False):
    cursor = after
    heartbeat = asyncio.get_running_loop().time()
    while not await request.is_disconnected():
        batch = await asyncio.to_thread(job_store.events, cursor, job_id)
        for payload in batch:
            cursor = payload["id"]
            yield sse(payload["event"] if legacy else payload, cursor)
        if job_id:
            job = await asyncio.to_thread(job_store.get_job, job_id)
            if job is None:
                return
            if job["status"] in job_store.TERMINAL and cursor >= job["last_event_id"]:
                return
        if asyncio.get_running_loop().time() - heartbeat >= 15:
            yield ": keepalive\n\n"
            heartbeat = asyncio.get_running_loop().time()
        if not batch:
            await asyncio.sleep(0.05)


def stream_headers():
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/api/generations/events")
def global_events(request: Request, after: int = Query(default=0, ge=0)):
    return StreamingResponse(generation_events(request, after), media_type="text/event-stream", headers=stream_headers())


@app.get("/api/generations/{job_id}")
def get_generation(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return job


@app.get("/api/generations/{job_id}/events")
def subscribe_generation(job_id: str, request: Request, after: int = Query(default=0, ge=0)):
    get_generation(job_id)
    return StreamingResponse(generation_events(request, after, job_id), media_type="text/event-stream", headers=stream_headers())


@app.post("/api/generations/{job_id}/cancel")
def cancel_generation(job_id: str, request: Request):
    job = request.app.state.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return job


@app.post("/api/generations/{job_id}/acknowledge")
def acknowledge_generation(job_id: str):
    job = job_store.acknowledge(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return job


@app.post("/api/chats/{chat_id}/messages/stream")
def send_message(chat_id: str, payload: MessageCreate, request: Request):
    job = submit_job(request, chat_id, payload)
    return StreamingResponse(
        generation_events(request, 0, job["id"], legacy=True),
        media_type="text/event-stream", headers=stream_headers(),
    )


@app.get("/api/research/{run_id}")
def research_run(run_id: str):
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    return result
