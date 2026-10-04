import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

async function compile(path, replacements = {}) {
  const source = await readFile(new URL(path, import.meta.url), "utf8");
  let code = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  for (const [name, url] of Object.entries(replacements)) code = code.replaceAll(`from "${name}"`, `from "${url}"`);
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}
const selectionUrl = await compile("../src/modelSelection.ts");
const { loadModelSelection, resolveSelection, reasoningLabel } = await import(selectionUrl);
const controlsUrl = await compile("../src/ModelControls.tsx", {
  "./modelSelection": selectionUrl, "react": import.meta.resolve("react"),
  "react/jsx-runtime": import.meta.resolve("react/jsx-runtime"), "lucide-react": import.meta.resolve("lucide-react"),
});
const { ReasoningOptions, ModelControls } = await import(controlsUrl);

function capability(kind, settings = [], defaultSetting = null) {
  return { kind, settings, default: defaultSetting, explanation: "Verified metadata", verified: kind !== "unverified" };
}

test("reasoning UI exposes only levels proven by model metadata", () => {
  for (const [kind, settings, expected] of [
    ["binary", ["off", "on"], ["Off", "On"]],
    ["effort", ["low", "medium", "high"], ["Low", "Medium", "High"]],
    ["effort", ["low", "high"], ["Low", "High"]],
  ]) {
    const html = renderToStaticMarkup(createElement(ReasoningOptions, {
      capability: capability(kind, settings, settings[0]), selected: null, onSelect() {},
    }));
    assert.equal((html.match(/role="radio"/g) ?? []).length, expected.length);
    for (const label of expected) assert.ok(html.includes(`>${label}</button>`));
    if (!settings.includes("medium")) assert.ok(!html.includes(">Medium</button>"));
    assert.ok(!html.includes("<select"));
  }
});

test("fixed, unsupported, and unverified models never get fake reasoning controls", () => {
  for (const kind of ["fixed", "none", "unverified"]) {
    const html = renderToStaticMarkup(createElement(ReasoningOptions, {
      capability: capability(kind, [], kind === "fixed" ? "on" : null), selected: null, onSelect() {},
    }));
    assert.ok(!html.includes('role="radio"'));
    assert.ok(html.includes(reasoningLabel(capability(kind, [], kind === "fixed" ? "on" : null))));
  }
});

test("model control displays the selected local model and accurate fixed reasoning state", () => {
  const html = renderToStaticMarkup(createElement(ModelControls, {
    catalog: { online: true, default_model: "local:latest", models: [{ name: "local:latest", available: true,
      can_chat: true, reasoning: capability("fixed", [], "on") }] },
    selection: { model: "local:latest", reasoning: null }, onChange() {}, refresh() {}, loading: false, error: null, disabled: false,
  }));
  assert.ok(html.includes("local:latest"));
  assert.ok(html.includes("Reasoning fixed on"));
  assert.ok(html.includes('aria-haspopup="dialog"'));
  assert.ok(!html.includes("<select"));
});

test("offline, removed models and stale reasoning preferences block submission", () => {
  const model = { name: "local", available: true, can_chat: true, reasoning: capability("binary", ["off", "on"], "on") };
  const catalog = { online: true, default_model: "local", models: [model] };
  assert.equal(resolveSelection(catalog, { model: null, reasoning: null }).ready, true);
  assert.equal(resolveSelection({ ...catalog, online: false }, { model: "local", reasoning: "on" }).ready, false);
  assert.equal(resolveSelection(catalog, { model: "removed", reasoning: null }).ready, false);
  assert.equal(resolveSelection(catalog, { model: "local", reasoning: "high" }).ready, false);
  assert.equal(resolveSelection(catalog, { model: "local", reasoning: "off" }).ready, true);
});

test("saved model selection survives reload and malformed preferences fall back safely", (t) => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  const storage = { getItem: () => '{"model":"local","reasoning":"high"}' };
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: storage });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "localStorage", previous);
    else delete globalThis.localStorage;
  });
  assert.deepEqual(loadModelSelection(), { model: "local", reasoning: "high" });
  storage.getItem = () => "invalid json";
  assert.deepEqual(loadModelSelection(), { model: null, reasoning: null });
});
