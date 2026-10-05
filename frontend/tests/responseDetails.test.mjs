import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import ts from "typescript";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

async function compile(path, replacements = {}) {
  let code = ts.transpileModule(await readFile(new URL(path, import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  for (const [key, value] of Object.entries(replacements)) code = code.replaceAll(`from "${key}"`, `from "${value}"`);
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}
const coordinationUrl = await compile("../src/speechCoordination.ts");
const preferencesUrl = await compile("../src/preferences.ts");
const speechUrl = await compile("../src/responseSpeech.ts", { "./speechCoordination": coordinationUrl, "./preferences": preferencesUrl });
const { spokenText, browserSpeechProvider } = await import(speechUrl);
const { PerformanceDetails } = await import(await compile("../src/ResponseDetails.tsx", {
  react: import.meta.resolve("react"), "react/jsx-runtime": import.meta.resolve("react/jsx-runtime"),
  "lucide-react": import.meta.resolve("lucide-react"), "./responseSpeech": speechUrl,
}));

test("speech removes citation markers and link metadata", () => {
  assert.equal(spokenText("**Answer** [SOURCE 1]. [Reference](https://example.org)"), "Answer . Reference");
});
test("browser provider uses local voices and cancels existing playback", () => {
  const original = globalThis.window;
  const originalUtterance = globalThis.SpeechSynthesisUtterance;
  let cancelled = 0, spoken;
  globalThis.window = { speechSynthesis: { cancel() { cancelled++; },
    getVoices() { return [{ localService: true, lang: "en", name: "Local" }]; },
    speak(value) { spoken = value; }, removeEventListener() {} } };
  globalThis.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
  try {
    const provider = browserSpeechProvider();
    provider.speak("Hello [SOURCE 1]", () => {}, () => {}, () => {});
    assert.equal(spoken.text, "Hello");
    assert.equal(spoken.voice.localService, true);
    provider.stop(); assert.equal(cancelled, 2);
  } finally { globalThis.window = original; globalThis.SpeechSynthesisUtterance = originalUtterance; }
});
test("performance omits unmeasured metrics", () => {
  const html = renderToStaticMarkup(createElement(PerformanceDetails, { timings: { tps: 42, planner: .1 } }));
  assert.match(html, /42.0 tok\/s/); assert.match(html, /100 ms/);
  assert.doesNotMatch(html, /Inference TTFT/);
  assert.equal(renderToStaticMarkup(createElement(PerformanceDetails, {})), "");
});
