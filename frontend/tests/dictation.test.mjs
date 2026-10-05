import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import ts from "typescript";

async function compile(path, replacements = {}) {
  let code = ts.transpileModule(await readFile(new URL(path, import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
  }).outputText;
  for (const [key, value] of Object.entries(replacements)) code = code.replaceAll(`from "${key}"`, `from "${value}"`);
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}
const coordinator = await compile("../src/speechCoordination.ts");
const preferences = await compile("../src/preferences.ts");
const { appendTranscript, browserDictationProvider } = await import(await compile("../src/dictation.ts", { "./speechCoordination": coordinator, "./preferences": preferences }));
const { claimSpeech, releaseSpeech } = await import(coordinator);

test("transcripts append naturally without replacing drafts", () => {
  assert.equal(appendTranscript("Existing draft", " spoken text "), "Existing draft spoken text");
  assert.equal(appendTranscript("Draft\n", "more"), "Draft\nmore");
  assert.equal(appendTranscript("", "hello"), "hello");
  assert.equal(appendTranscript("keep", " "), "keep");
});
test("unsupported browser reports limitation", () => {
  const original = globalThis.window; globalThis.window = {};
  try {
    const provider = browserDictationProvider(); let error;
    assert.equal(provider.available, false);
    provider.start({ error: (message) => { error = message; } });
    assert.match(error, /unavailable/);
  } finally { globalThis.window = original; }
});
test("start stop transcript errors and TTS mutual exclusion", () => {
  const original = globalThis.window;
  let session, cancelled = 0, ended = 0, draft = "Draft", error;
  class Recognition {
    constructor() { session = this; }
    start() { this.onstart?.(); }
    stop() { this.onend?.(); }
    abort() { cancelled++; }
  }
  globalThis.window = { webkitSpeechRecognition: Recognition };
  const callbacks = { transcript: (text) => { draft = appendTranscript(draft, text); }, listening() {},
    ended() { ended++; }, error(message) { error = message; } };
  try {
    const provider = browserDictationProvider(); let ttsStopped = 0;
    const ttsOwner = {};
    claimSpeech(ttsOwner, () => { ttsStopped++; });
    provider.start(callbacks); assert.equal(ttsStopped, 1);
    session.onresult({ resultIndex: 0, results: [{ isFinal: true, 0: { transcript: "hello" } }] });
    assert.equal(draft, "Draft hello");
    provider.stop(); assert.equal(ended, 1);
    provider.start(callbacks); session.onerror({ error: "not-allowed" });
    assert.match(error, /permission was denied/);
    provider.start(callbacks); session.onerror({ error: "audio-capture" });
    assert.match(error, /No microphone/);
    provider.start(callbacks); session.onerror({ error: "aborted" });
    error = undefined;
    provider.start(callbacks); const previous = session;
    claimSpeech(ttsOwner, () => {});
    assert.equal(previous.onresult, null); assert.ok(cancelled >= 4);
    releaseSpeech(ttsOwner);
    provider.start(callbacks); provider.cancel(); assert.equal(session.onresult, null);
  } finally { globalThis.window = original; }
});
