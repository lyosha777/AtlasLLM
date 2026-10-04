import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

async function moduleUrl(path) {
  const source = await readFile(new URL(path, import.meta.url), "utf8");
  const code = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
  }).outputText;
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}
const apiUrl = await moduleUrl("../src/api.ts");
const source = await readFile(new URL("../src/generationStore.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText.replace('from "./api"', `from "${apiUrl}"`);
const { GenerationStore } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

function job(chat, revision, status = "running", ordinal = 1) {
  const id = `${chat}-${ordinal}`;
  const message = { chat_id: chat, sources: [], attachments: [], created_at: "2026-10-04" };
  return {
    id, ordinal, chat_id: chat, chat_title: chat, last_event_id: revision, status,
    activity: status === "running" ? "Writing" : null, acknowledged: false,
    user_message: { ...message, id: `${id}-user`, role: "user", content: "Question" },
    assistant_message: { ...message, id: `${id}-assistant`, role: "assistant",
      content: `Answer ${revision}`, generation_revision: revision, generation_status: status },
  };
}

test("switching conversation views retains independent active and completed jobs", () => {
  const store = new GenerationStore();
  store.applyJob(job("A", 1));
  store.applyJob(job("B", 2));
  let activeChat = "A";
  assert.equal(store.getSnapshot().jobs[activeChat].status, "running");
  activeChat = "B";
  store.applyJob(job("A", 3, "completed"));
  assert.equal(store.getSnapshot().jobs[activeChat].status, "running");
  activeChat = "A";
  assert.equal(store.getSnapshot().messages[activeChat][1].content, "Answer 3");
  assert.equal(store.getSnapshot().jobs[activeChat].acknowledged, false);
});

test("replayed events and stale history never duplicate or roll back messages", () => {
  const store = new GenerationStore();
  const event = { id: 5, chat_id: "A", job: job("A", 5), event: { type: "token" } };
  store.applyEvent(event);
  store.applyEvent(event);
  store.mergeHistory("A", [job("A", 2).assistant_message]);
  store.applyJob(job("A", 3));
  assert.equal(store.getSnapshot().messages.A.length, 2);
  assert.equal(store.getSnapshot().messages.A[1].content, "Answer 5");
});

test("late acknowledgement of an older job cannot replace the current generation", () => {
  const store = new GenerationStore();
  store.applyJob(job("A", 4, "completed", 1));
  store.applyJob(job("A", 5, "running", 2));
  store.applyJob({ ...job("A", 6, "completed", 1), acknowledged: true });
  assert.equal(store.getSnapshot().jobs.A.id, "A-2");
  assert.equal(store.getSnapshot().messages.A.length, 4);
});

test("delete tombstones prevent delayed submission responses from reviving a chat", () => {
  const store = new GenerationStore();
  store.applyJob(job("A", 1));
  store.applyEvent({ id: 3, chat_id: "A", job: null, event: { type: "deleted" } });
  store.applyJob(job("A", 2));
  assert.equal(store.getSnapshot().jobs.A, undefined);
  assert.equal(store.getSnapshot().messages.A, undefined);
});

test("replayed and refreshed messages retain immutable reasoning metadata per job", () => {
  const store = new GenerationStore();
  const inferenceA = { model: "effort", requested_reasoning: "high", actual_settings: { think: "high" } };
  const inferenceB = { model: "binary", requested_reasoning: "off", actual_settings: { think: false } };
  const a = { ...job("A", 3), inference: inferenceA };
  a.assistant_message.inference = inferenceA;
  const b = { ...job("B", 4), inference: inferenceB };
  b.assistant_message.inference = inferenceB;
  store.applyEvent({ id: 3, chat_id: "A", job: a, event: { type: "inference" } });
  store.applyEvent({ id: 4, chat_id: "B", job: b, event: { type: "inference" } });
  store.mergeHistory("A", [job("A", 1).assistant_message]);
  assert.deepEqual(store.getSnapshot().messages.A[1].inference, inferenceA);
  assert.deepEqual(store.getSnapshot().jobs.B.inference, inferenceB);
});

test("leaving a view unsubscribes its observer without cancelling or deleting jobs", () => {
  const store = new GenerationStore();
  let updates = 0;
  const unsubscribe = store.subscribe(() => updates++);
  store.applyJob(job("A", 1));
  unsubscribe();
  store.applyJob(job("A", 2, "completed"));
  assert.equal(updates, 1);
  assert.equal(store.getSnapshot().jobs.A.status, "completed");
});

test("subscription reconnect restores durable state and abort never calls cancellation", async (t) => {
  const calls = [];
  let snapshots = 0;
  let secondConnection;
  const reconnected = new Promise((resolve) => { secondConnection = resolve; });
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, method: options?.method ?? "GET" });
    if (url.endsWith("/generations")) {
      snapshots++;
      return Response.json({ cursor: snapshots, jobs: [job("A", snapshots, snapshots === 1 ? "running" : "completed")] });
    }
    if (snapshots === 1) return new Response("");
    return new Response(new ReadableStream({ start(controller) {
      options.signal.addEventListener("abort", () => controller.error(new DOMException("Disconnected", "AbortError")), { once: true });
      secondConnection();
    } }));
  });
  const store = new GenerationStore();
  const controller = new AbortController();
  const pending = store.connect(controller.signal);
  await reconnected;
  assert.equal(store.getSnapshot().jobs.A.status, "completed");
  assert.equal(store.getSnapshot().messages.A.length, 2);
  assert.ok(calls.some(({ url }) => url.endsWith("/events?after=2")));
  controller.abort();
  await pending;
  assert.ok(calls.every(({ method }) => method === "GET"));
  assert.equal(store.getSnapshot().jobs.A.status, "completed");
});
