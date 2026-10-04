import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText;
const { streamMessage, submitGeneration, cancelGeneration, getModels } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

test("SSE delivers split UTF-8 tokens before completion and submits message options", async (t) => {
  let stream;
  let payload;
  const received = [];
  let accepted = false;
  let complete = false;
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response(new ReadableStream({ start(controller) { stream = controller; } }));
  });
  const pending = streamMessage("chat", "Research", (event) => received.push(event), undefined, {
    forceWeb: true,
    documentIds: ["pdf-id"],
    onAccepted: () => { accepted = true; },
  }).then(() => { complete = true; });
  await new Promise(setImmediate);
  assert.equal(accepted, true);
  assert.deepEqual(payload, { message: "Research", force_web: true, document_ids: ["pdf-id"] });
  const encoded = new TextEncoder().encode('data: {"type":"token","content":"café"}\n\n');
  const split = encoded.indexOf(0xc3) + 1;
  stream.enqueue(encoded.slice(0, split));
  await new Promise(setImmediate);
  assert.equal(received.length, 0);
  stream.enqueue(encoded.slice(split));
  await new Promise(setImmediate);
  assert.deepEqual(received, [{ type: "token", content: "café" }]);
  assert.equal(complete, false);
  stream.enqueue(new TextEncoder().encode('data: {"type":"done","timings":{}}\n\n'));
  stream.close();
  await pending;
  assert.equal(received[1].type, "done");
});

test("default submissions preserve planner control", async (t) => {
  let payload;
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    payload = JSON.parse(options.body);
    return new Response('data: {"type":"done","timings":{}}\n\n');
  });
  await streamMessage("chat", "Hello", () => {});
  assert.deepEqual(payload, { message: "Hello", force_web: false, document_ids: [] });
});

test("rejected requests retain unsubmitted composer options", async (t) => {
  let accepted = false;
  t.mock.method(globalThis, "fetch", async () => new Response(
    JSON.stringify({ detail: "Document not found" }), { status: 404 },
  ));
  await assert.rejects(streamMessage("chat", "Hello", () => {}, undefined, {
    onAccepted: () => { accepted = true; },
  }), /Document not found/);
  assert.equal(accepted, false);
});

test("stream errors are surfaced with the backend message", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(
    'data: {"type":"error","message":"Ollama unavailable"}\n\n',
  ));
  await assert.rejects(streamMessage("chat", "Hello", () => {}), /Ollama unavailable/);
});

test("interrupted streams report failure while retaining received tokens", async (t) => {
  const received = [];
  t.mock.method(globalThis, "fetch", async () => new Response(
    'data: {"type":"token","content":"Partial"}\n\n',
  ));
  await assert.rejects(streamMessage("chat", "Hello", (event) => received.push(event)), /ended unexpectedly/);
  assert.equal(received[0].content, "Partial");
});

test("aborting an SSE subscription closes its streaming request", async (t) => {
  const abort = new AbortController();
  const received = [];
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    assert.equal(options.signal, abort.signal);
    return new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"type":"token","content":"First"}\n\n'));
        options.signal.addEventListener("abort", () => controller.error(
          new DOMException("Generation stopped", "AbortError"),
        ), { once: true });
      },
    }));
  });
  const pending = streamMessage("chat", "Hello", (event) => received.push(event), abort.signal);
  await new Promise(setImmediate);
  assert.equal(received[0].content, "First");
  abort.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("job submission carries an idempotency key and Stop explicitly cancels that job", async (t) => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, ...options });
    return Response.json({ id: "job-A" });
  });
  await submitGeneration("A", "Question", { forceWeb: true, documentIds: ["pdf"], requestId: "stable-request" });
  await cancelGeneration("job-A");
  assert.deepEqual(JSON.parse(calls[0].body), {
    message: "Question", force_web: true, document_ids: ["pdf"], request_id: "stable-request",
  });
  assert.equal(calls[0].signal, undefined);
  assert.ok(calls[1].url.endsWith("/generations/job-A/cancel"));
  assert.equal(calls[1].method, "POST");
});

test("different chats submit independent model and reasoning settings", async (t) => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return Response.json({ id: url });
  });
  const selection = { forceWeb: true, documentIds: ["document"], requestId: "A-request", model: "effort", reasoning: "high" };
  const a = submitGeneration("A", "Question A", selection);
  selection.model = "binary";
  selection.reasoning = "off";
  selection.requestId = "B-request";
  const b = submitGeneration("B", "Question B", selection);
  await Promise.all([a, b]);
  assert.equal(calls[0].body.model, "effort");
  assert.equal(calls[0].body.reasoning, "high");
  assert.equal(calls[1].body.model, "binary");
  assert.equal(calls[1].body.reasoning, "off");
  assert.equal(calls[0].body.force_web, true);
  assert.deepEqual(calls[0].body.document_ids, ["document"]);
});

test("model discovery represents Ollama offline without inventing available models", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({ online: false, models: [], error: "Ollama unavailable" }));
  const catalog = await getModels();
  assert.equal(catalog.online, false);
  assert.deepEqual(catalog.models, []);
});
