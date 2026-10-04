import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from backend import api, db
from backend.core import assistant, planner
from backend.core.source import Source
from backend.ingestion import ingest_pdf as ingestion
from backend.retrieval import retriever
from backend.tests.model_fixtures import default_inference


def pdf_bytes(text: str | None) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text is not None:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class TestEmbeddings:
    def encode(self, texts):
        def vector(text):
            return [1.0, 0.0] if "tcp" in text.lower() else [0.0, 1.0]
        return np.array([vector(text) for text in texts] if isinstance(texts, list) else vector(texts))


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.directory.name)
        cls.collection = chromadb.PersistentClient(path=str(cls.root / "chroma")).get_or_create_collection(
            "atlas_documents"
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.patches = [
            patch.object(api.provider, "resolve", side_effect=default_inference),
            patch.object(db, "DB_PATH", self.root / f"{self._testMethodName}.db"),
            patch.object(api, "UPLOADS_DIR", self.root / self._testMethodName),
            patch.object(api, "get_document_collection", return_value=self.collection),
            patch.object(ingestion, "get_document_collection", return_value=self.collection),
            patch.object(retriever, "get_document_collection", return_value=self.collection),
            patch.object(ingestion, "get_embedding_model", return_value=TestEmbeddings()),
            patch.object(retriever, "get_embedding_model", return_value=TestEmbeddings()),
        ]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        self.client = self.enterContext(TestClient(api.app))

    def upload(self, content, name="notes.pdf"):
        return self.client.post("/api/documents", files={"file": (name, content, "application/pdf")})

    def test_upload_same_filename_and_scoped_retrieval(self):
        tcp = self.upload(pdf_bytes("TCP provides reliable delivery."))
        volcano = self.upload(pdf_bytes("Volcanoes erupt lava."))
        self.assertEqual(tcp.status_code, 201, tcp.text)
        self.assertEqual(volcano.status_code, 201, volcano.text)
        first, second = tcp.json(), volcano.json()
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue((api.UPLOADS_DIR / f"{first['id']}.pdf").exists())
        sources = retriever.retrieve("TCP", document_ids=[second["id"]])
        self.assertTrue(sources)
        self.assertEqual({source.document_id for source in sources}, {second["id"]})
        self.assertIn("Volcanoes", sources[0].text)
        self.assertEqual(db.get_document(first["id"])["name"], "notes.pdf")

    def test_reingestion_removes_stale_chunks(self):
        path = self.root / "reingest.pdf"
        path.write_bytes(pdf_bytes("TCP " * 700))
        ingestion.ingest_pdf(path, "stable-id")
        self.assertGreater(len(self.collection.get(where={"document_id": "stable-id"})["ids"]), 1)
        path.write_bytes(pdf_bytes("TCP revised."))
        result = ingestion.ingest_pdf(path, "stable-id")
        stored = self.collection.get(where={"document_id": "stable-id"})
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(len(stored["ids"]), 1)
        self.assertEqual(stored["documents"], ["TCP revised."])

    def test_cli_reingestion_migrates_only_its_legacy_chunks(self):
        path = self.root / "manual.pdf"
        path.write_bytes(pdf_bytes("TCP revised."))
        self.collection.upsert(
            ids=["manual_page_1_chunk_0", "another-document"],
            documents=["TCP old.", "Unrelated document."],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{"source": "manual.pdf", "page": 1}, {"source": "manual.pdf", "page": 1}],
        )
        result = ingestion.ingest_pdf(path)
        ids = self.collection.get(include=[])["ids"]
        self.assertNotIn("manual_page_1_chunk_0", ids)
        self.assertIn("another-document", ids)
        self.assertTrue(any(chunk_id.startswith(result["id"]) for chunk_id in ids))

    def test_invalid_empty_encrypted_and_oversized_uploads(self):
        self.assertEqual(self.upload(b"plain text", "notes.txt").status_code, 415)
        self.assertEqual(self.upload(b"not a PDF").status_code, 422)
        self.assertEqual(self.upload(pdf_bytes(None)).status_code, 422)
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.encrypt("secret")
        encrypted = io.BytesIO()
        writer.write(encrypted)
        self.assertEqual(self.upload(encrypted.getvalue()).status_code, 422)
        with patch.object(api, "MAX_PDF_BYTES", 32):
            self.assertEqual(self.upload(pdf_bytes("TCP")).status_code, 413)
        self.assertEqual(self.collection.count(), 0)
        self.assertEqual(list(api.UPLOADS_DIR.iterdir()), [])

    def test_empty_collection_does_not_load_embeddings(self):
        with patch.object(retriever, "get_embedding_model") as embedding:
            self.assertEqual(retriever.retrieve("hello"), [])
            embedding.assert_not_called()

    def test_combined_research_stream_and_attachment_persistence(self):
        document = self.upload(pdf_bytes("TCP provides reliable delivery.")).json()
        chat = self.client.post("/api/chats", json={"title": "Research"}).json()
        web = Source("External TCP reference.", "web", title="Reference", url="https://example.org")
        decision = {"use_documents": False, "use_web": False, "fresh": False, "search_query": None}
        with (
            patch.object(assistant, "plan", return_value=decision) as plan,
            patch.object(assistant, "retrieve_web", return_value=[web]) as search,
            patch.object(assistant, "rerank", side_effect=lambda query, sources, top_k: sources),
            patch.object(assistant, "answer_events", return_value=iter([
                {"type": "token", "content": "TCP "}, {"type": "token", "content": "answer [SOURCE 1]."},
            ])),
        ):
            response = self.client.post(f"/api/chats/{chat['id']}/messages/stream", json={
                "message": "Explain TCP", "force_web": True, "document_ids": [document["id"]],
            })
            plan.assert_called_once_with("Explain TCP", [], has_attachments=True)
            search.assert_called_once_with("Explain TCP", fresh=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([event["label"] for event in events if event["type"] == "status"], [
            "Planning", "Searching documents", "Searching the web", "Ranking sources", "Writing",
        ])
        self.assertEqual("".join(event["content"] for event in events if event["type"] == "token"),
                         "TCP answer [SOURCE 1].")
        self.assertEqual(events[-1]["type"], "done")
        messages = self.client.get(f"/api/chats/{chat['id']}/messages").json()
        self.assertEqual(messages[0]["attachments"][0]["id"], document["id"])
        self.assertEqual(messages[1]["content"], "TCP answer [SOURCE 1].")
        self.assertEqual(len(messages[1]["sources"]), 2)
        self.assertEqual(messages[1]["sources"][0]["document_id"], document["id"])

    def test_default_chat_and_message_validation(self):
        chat = self.client.post("/api/chats", json={}).json()
        endpoint = f"/api/chats/{chat['id']}/messages/stream"
        self.assertEqual(self.client.post(endpoint, json={"message": " "}).status_code, 400)
        self.assertEqual(self.client.post(endpoint, json={"message": "Hi", "document_ids": ["missing"]}).status_code, 404)
        self.assertEqual(db.list_messages(chat["id"]), [])
        decision = {"use_documents": False, "use_web": False, "fresh": False, "search_query": None}
        with patch.object(assistant, "plan", return_value=decision), patch.object(
            assistant, "chat_events", return_value=iter([
                {"type": "token", "content": "Hello"}, {"type": "token", "content": " there"},
            ])
        ):
            response = self.client.post(endpoint, json={"message": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(db.list_messages(chat["id"])[1]["content"], "Hello there")
        self.client.patch(f"/api/chats/{chat['id']}", json={"archived": True})
        self.assertEqual(self.client.post(endpoint, json={"message": "Hi"}).status_code, 409)
        self.client.patch(f"/api/chats/{chat['id']}", json={"archived": False, "title": "Renamed"})
        self.assertEqual(db.get_chat(chat["id"])["title"], "Renamed")
        self.assertEqual(self.client.delete(f"/api/chats/{chat['id']}").status_code, 200)
        self.assertEqual(db.list_messages(chat["id"]), [])

    def test_stream_error_preserves_partial_answer(self):
        def failing_stream(*args, **kwargs):
            yield {"type": "token", "content": "Partial answer"}
            raise RuntimeError("Model unavailable")
        chat = self.client.post("/api/chats", json={}).json()
        with patch.object(api, "stream_assistant", side_effect=failing_stream):
            response = self.client.post(f"/api/chats/{chat['id']}/messages/stream", json={"message": "Hello"})
        self.assertIn('"type": "error"', response.text)
        self.assertEqual(db.list_messages(chat["id"])[1]["content"], "Partial answer")

    def test_planner_attachments_and_freshness(self):
        response = {"message": {"content": json.dumps({
            "use_documents": False, "use_web": False, "fresh": True, "search_query": "today",
        })}}
        with patch.object(planner.ollama, "chat", return_value=response):
            decision = planner.plan("today", [], has_attachments=True)
        self.assertTrue(decision["use_documents"])
        self.assertTrue(decision["use_web"])
        self.assertTrue(decision["fresh"])
        with patch.object(planner.ollama, "chat", return_value={"message": {"content": "[]"}}):
            self.assertTrue(planner.plan("read this", [], has_attachments=True)["use_documents"])

    def test_migration_preserves_legacy_messages(self):
        path = self.root / "legacy.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript("""
                CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL, archived INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE messages (id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, sources_json TEXT, created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE);
                INSERT INTO chats VALUES ('chat', 'Existing thread', 0, '2026', '2026');
                INSERT INTO messages VALUES ('message', 'chat', 'assistant', 'Keep this answer', '[]', '2026');
            """)
            connection.commit()
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            db.init_db()
            self.assertEqual(db.get_chat("chat")["title"], "Existing thread")
            message = db.list_messages("chat")[0]
            self.assertEqual(message["content"], "Keep this answer")
            self.assertEqual(message["attachments"], [])


if __name__ == "__main__":
    unittest.main()
