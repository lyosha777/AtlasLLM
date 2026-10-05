import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { ModelControls } from "./ModelControls";
import type { Preferences } from "./preferences";

export function SettingsPanel({ preferences, onChange, onClose, reset, modelControls }: {
  preferences: Preferences; onChange: (value: Preferences) => void; onClose: () => void; reset: () => void;
  modelControls: React.ComponentProps<typeof ModelControls>;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const speechAvailable = "speechSynthesis" in window;
  const recognitionAvailable = "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    const refresh = () => setVoices(window.speechSynthesis.getVoices().filter((voice) => voice.localService));
    if (speechAvailable) { refresh(); window.speechSynthesis.addEventListener("voiceschanged", refresh); }
    return () => {
      if (speechAvailable) window.speechSynthesis.removeEventListener("voiceschanged", refresh);
      previouslyFocused?.focus();
    };
  }, [speechAvailable]);
  const update = <K extends keyof Preferences>(key: K, value: Preferences[K]) => onChange({ ...preferences, [key]: value });
  return <dialog className="atlas-settings" ref={dialog} aria-labelledby="settings-title"
    onCancel={(event) => { event.preventDefault(); onClose(); }}
    onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <button className="icon-button settings-close" aria-label="Close settings" onClick={onClose}><X size={18} /></button>
    <h2 id="settings-title">Settings</h2>
    <section><h3>Appearance</h3><div className="reasoning-choices" role="radiogroup" aria-label="Appearance">
      {(["light", "dark", "system"] as const).map((value) => <button key={value} role="radio"
        aria-checked={preferences.appearance === value} onClick={() => update("appearance", value)}>{value}</button>)}
    </div><p>System follows your operating system’s appearance.</p></section>
    <section><h3>Default model and reasoning</h3><ModelControls {...modelControls} />
      <p>Uses the installed model’s verified capabilities. Changes apply to future submissions; active jobs keep their settings.</p></section>
    <section><h3>Research</h3><label><input type="checkbox" checked={preferences.research}
      onChange={(event) => update("research", event.target.checked)} /> Start new chats in Deep Research</label>
      <p>Automatic source selection and adaptive research budgets remain in use.</p></section>
    <section><h3>Voice</h3><div className="settings-voices" role="radiogroup" aria-label="Read aloud voice">
      <button role="radio" aria-checked={!preferences.voice} disabled={!speechAvailable}
        onClick={() => update("voice", "")}>Automatic local voice</button>
      {voices.map((voice) => <button key={voice.voiceURI} role="radio" aria-checked={preferences.voice === voice.voiceURI}
        onClick={() => update("voice", voice.voiceURI)}>{voice.name} · {voice.lang}</button>)}
    </div>{!voices.length && <p>No local voices exposed. Enable a system voice to use Read Aloud.</p>}
      {preferences.voice && !voices.some((voice) => voice.voiceURI === preferences.voice) && <p>The saved voice is unavailable; Read Aloud falls back to an available local voice.</p>}
      <label>Speech rate · {preferences.rate.toFixed(1)}×<input type="range" min="0.5" max="2" step="0.1"
        disabled={!speechAvailable || !voices.length} value={preferences.rate} onChange={(event) => update("rate", Number(event.target.value))} /></label>
      <label>Dictation language<input type="text" placeholder="Browser default (e.g. en-US)" disabled={!recognitionAvailable}
        value={preferences.dictationLanguage} onChange={(event) => update("dictationLanguage", event.target.value)} /></label>
      <p>Use a language tag such as en-US or de-DE; leave blank for the browser language. Support and speech recognition privacy depend on your browser/platform.</p>
      {!recognitionAvailable && <p>Dictation is not supported by this browser.</p>}</section>
    <section><h3>Developer</h3><label><input type="checkbox" checked={preferences.performanceOpen}
      onChange={(event) => update("performanceOpen", event.target.checked)} /> Expand response Performance details by default</label></section>
    <button className="secondary-button" onClick={reset}>Reset settings to defaults</button>
  </dialog>;
}
