import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const code = ts.transpileModule(await readFile(new URL("../src/preferences.ts", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText;
const { loadPreferences, savePreferences, defaultPreferences } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

test("settings defaults, persistence, migration and invalid storage", () => {
  const old = globalThis.localStorage;
  const values = new Map();
  globalThis.localStorage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  try {
    assert.deepEqual(loadPreferences(), defaultPreferences);
    values.set("atlas.theme", "dark"); assert.equal(loadPreferences().appearance, "dark");
    const preferences = { appearance: "system", research: true, rate: 1.4, voice: "local-voice", dictationLanguage: "de-DE", performanceOpen: true };
    savePreferences(preferences); assert.deepEqual(loadPreferences(), preferences);
    savePreferences(defaultPreferences); assert.deepEqual(loadPreferences(), defaultPreferences);
    values.set("atlas.preferences", JSON.stringify({ rate: 99, appearance: "invalid" }));
    assert.equal(loadPreferences().rate, 1);
    values.set("atlas.preferences", "invalid JSON"); assert.deepEqual(loadPreferences(), defaultPreferences);
  } finally { globalThis.localStorage = old; }
});
