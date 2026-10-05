import { claimSpeech, releaseSpeech } from "./speechCoordination";
import { loadPreferences } from "./preferences";

export function spokenText(text: string): string {
  return text.replace(/\[SOURCE\s+\d+(?:[^\]]*)\]/gi, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/[*_`#>|]/g, "").replace(/\s+/g, " ").trim();
}

export interface SpeechProvider {
  available: boolean;
  speak(text: string, onStart: () => void, onEnd: () => void, onError: () => void): void;
  stop(): void;
}

export function browserSpeechProvider(): SpeechProvider {
  const available = typeof window !== "undefined" && "speechSynthesis" in window;
  let pending: (() => void) | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const owner = {};
  const stop = () => {
    if (timer) clearTimeout(timer);
    if (pending && available) window.speechSynthesis.removeEventListener("voiceschanged", pending);
    pending = undefined;
    releaseSpeech(owner);
    if (available) window.speechSynthesis.cancel();
  };
  return { available, stop, speak(text, onStart, onEnd, onError) {
    stop();
    if (!available) { onError(); return; }
    claimSpeech(owner, () => { stop(); onEnd(); });
    const start = () => {
      const voices = window.speechSynthesis.getVoices().filter((voice) => voice.localService);
      if (!voices.length) return false;
      if (timer) clearTimeout(timer);
      if (pending) window.speechSynthesis.removeEventListener("voiceschanged", pending);
      pending = undefined;
      const utterance = new SpeechSynthesisUtterance(spokenText(text));
      const preferences = loadPreferences();
      utterance.voice = voices.find((voice) => voice.voiceURI === preferences.voice) ?? voices.find((voice) => voice.lang.startsWith(navigator.language.split("-")[0])) ?? voices[0];
      utterance.rate = preferences.rate;
      utterance.onstart = onStart;
      utterance.onend = () => { releaseSpeech(owner); onEnd(); };
      utterance.onerror = () => { releaseSpeech(owner); onError(); };
      window.speechSynthesis.speak(utterance);
      return true;
    };
    if (!start()) {
      pending = () => { start(); };
      window.speechSynthesis.addEventListener("voiceschanged", pending);
      timer = setTimeout(() => { stop(); onError(); }, 3000);
    }
  } };
}
