import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

async function compile(path, replacements = {}) {
  const source = await readFile(new URL(path, import.meta.url), "utf8");
  let code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX,
  } }).outputText;
  for (const [name, url] of Object.entries(replacements)) code = code.replaceAll(`from "${name}"`, `from "${url}"`);
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}
const api = await compile("../src/api.ts");
const dependencies = { react: import.meta.resolve("react"), "react/jsx-runtime": import.meta.resolve("react/jsx-runtime"),
  "lucide-react": import.meta.resolve("lucide-react"), "./api": api };
const { researchCitations } = await import(await compile("../src/researchCitations.ts"));
const { ResearchControls } = await import(await compile("../src/ResearchControls.tsx", dependencies));
const { EvidencePassage, ResearchPanel } = await import(await compile("../src/ResearchPanel.tsx", dependencies));

test("research control is independent from model reasoning and has a custom trigger", () => {
  const html = renderToStaticMarkup(createElement(ResearchControls, { enabled: true, scope: "both", onChange() {}, disabled: false }));
  assert.ok(html.includes("Deep Research"));
  assert.ok(html.includes('aria-haspopup="dialog"'));
  assert.ok(!html.includes("<select"));
});

test("research citation transform links only known retained indices", () => {
  const tree = { type: "root", children: [{ type: "paragraph", children: [{ type: "text", value: "Fact [SOURCE 1]. [SOURCE 99]" }] }] };
  researchCitations([1])()(tree);
  const children = tree.children[0].children;
  assert.equal(children[1].url, "#atlas-evidence-1");
  assert.equal(children[2].value, ". [SOURCE 99]");
});

test("citation transform preserves code, ordinary links, and adjacent citations", () => {
  const tree = { type: "root", children: [
    { type: "code", value: "[SOURCE 1]" },
    { type: "link", url: "https://example.org", children: [{ type: "text", value: "Reference" }] },
    { type: "paragraph", children: [{ type: "text", value: "[SOURCE 1][SOURCE 2]" }] },
  ] };
  researchCitations([1, 2])()(tree);
  assert.equal(tree.children[0].value, "[SOURCE 1]");
  assert.equal(tree.children[1].children[0].type, "text");
  assert.equal(tree.children[1].url, "https://example.org");
  assert.deepEqual(tree.children[2].children.map((item) => item.url), ["#atlas-evidence-1", "#atlas-evidence-2"]);
});

test("a model-formatted citation link opens retained evidence", () => {
  const tree = { type: "root", children: [{ type: "link", url: "https://example.org",
    children: [{ type: "text", value: "[SOURCE 1]" }] }] };
  researchCitations([1])()(tree);
  assert.equal(tree.children[0].url, "#atlas-evidence-1");
});

test("evidence inspector renders exact retained text and stable identifiers", () => {
  const html = renderToStaticMarkup(createElement(EvidencePassage, { evidence: {
    title: "Local PDF", passage: "Exact passage\nwith spacing  preserved", id: "evidence-id", chunk_id: "doc:page:1:chunk:2",
    page: 1, document_id: "document-id", query: "original query", publication_date: null, retrieved_at: "2026-10-05",
    retrieval_score: .4, reranker_score: 2.1, occurrences: [],
  } }));
  for (const expected of ["Exact passage\nwith spacing  preserved", "doc:page:1:chunk:2", "document-id", "evidence-id", "Unknown", "original query"])
    assert.ok(html.includes(expected));
  assert.ok(!html.includes("<think>"));
});

test("research panel is collapsible and inspector is a custom dialog", () => {
  const html = renderToStaticMarkup(createElement(ResearchPanel, { runId: "run", revision: 2, onCloseInspector() {} }));
  assert.ok(html.includes('<details class="research-panel">'));
  assert.ok(html.includes('<dialog class="evidence-inspector">'));
  assert.ok(html.includes("Close evidence inspector"));
});

test("submitted research settings travel with the job payload", async () => {
  const { submitGeneration } = await import(api);
  const oldFetch = globalThis.fetch;
  let body;
  globalThis.fetch = async (_url, options) => { body = JSON.parse(options.body); return { ok: true, json: async () => ({ id: "job" }) }; };
  try {
    await submitGeneration("chat", "question", { forceWeb: false, documentIds: ["pdf"], requestId: "request",
      model: "local:model", reasoning: "high", researchMode: true, researchScope: "both" });
    assert.equal(body.research_mode, true);
    assert.equal(body.research_scope, "both");
    assert.equal(body.reasoning, "high");
    assert.equal(body.model, "local:model");
    assert.deepEqual(body.document_ids, ["pdf"]);
  } finally { globalThis.fetch = oldFetch; }
});

test("research reload uses the persisted endpoint and propagates failures", async () => {
  const { getResearchRun } = await import(api);
  const oldFetch = globalThis.fetch;
  globalThis.fetch = async (url) => { assert.ok(url.endsWith("/research/run-id")); return { ok: true, json: async () => ({ evidence: [{ passage: "saved" }] }) }; };
  try {
    assert.equal((await getResearchRun("run-id")).evidence[0].passage, "saved");
    globalThis.fetch = async () => ({ ok: false, status: 404 });
    await assert.rejects(getResearchRun("run-id"), /no longer available/);
  } finally { globalThis.fetch = oldFetch; }
});
