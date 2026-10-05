import { claimSpeech, releaseSpeech } from "./speechCoordination";
import { loadPreferences } from "./preferences";

type Recognition = {
  continuous: boolean; interimResults: boolean; lang: string;
  onstart: (() => void) | null; onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onresult: ((event: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  start(): void; stop(): void; abort(): void;
};
type RecognitionConstructor = new () => Recognition;

export function appendTranscript(draft: string, transcript: string) {
  const text = transcript.trim();
  return text ? draft + (draft && !/\s$/.test(draft) ? " " : "") + text : draft;
}

const errors: Record<string, string> = {
  "not-allowed": "Microphone permission was denied. Allow microphone access in your browser settings.",
  "service-not-allowed": "Speech recognition is not allowed by this browser or platform.",
  "audio-capture": "No microphone is available. Check your microphone connection and settings.",
  network: "Speech recognition could not connect. Check your connection or try again.",
  "no-speech": "No speech was detected. Try again when ready.",
};

export interface DictationProvider {
  available: boolean;
  start(callbacks: { transcript(text: string): void; listening(): void; ended(): void; error(message: string): void }): void;
  stop(): void;
  cancel(): void;
}

export function browserDictationProvider(): DictationProvider {
  const host = typeof window === "undefined" ? undefined : window as unknown as {
    SpeechRecognition?: RecognitionConstructor; webkitSpeechRecognition?: RecognitionConstructor;
  };
  const Constructor = host?.SpeechRecognition ?? host?.webkitSpeechRecognition;
  let recognition: Recognition | undefined;
  const owner = {};
  const cancel = () => {
    const current = recognition; recognition = undefined;
    if (current) { current.onstart = current.onend = current.onerror = current.onresult = null; current.abort(); }
    releaseSpeech(owner);
  };
  return { available: Boolean(Constructor), cancel,
    stop() { recognition?.stop(); },
    start(callbacks) {
      cancel();
      if (!Constructor) { callbacks.error("Dictation is unavailable in this browser. Try a browser with speech recognition support."); return; }
      claimSpeech(owner, () => { cancel(); callbacks.ended(); });
      const session = new Constructor(); recognition = session;
      session.continuous = true; session.interimResults = false;
      const language = loadPreferences().dictationLanguage.trim();
      try { session.lang = language ? Intl.getCanonicalLocales(language)[0] : navigator.language; }
      catch { cancel(); callbacks.ended(); callbacks.error("Invalid dictation language. Use a language tag such as en-US in Settings."); return; }
      session.onstart = () => callbacks.listening();
      session.onresult = (event) => {
        if (recognition !== session) return;
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) callbacks.transcript(event.results[i][0].transcript);
        }
      };
      session.onend = () => { if (recognition === session) { recognition = undefined; releaseSpeech(owner); callbacks.ended(); } };
      session.onerror = (event) => {
        if (recognition !== session) return;
        cancel(); callbacks.ended();
        if (event.error !== "aborted") callbacks.error(errors[event.error] ?? "Speech recognition failed. Please try again.");
      };
      try { session.start(); } catch { cancel(); callbacks.ended(); callbacks.error("Could not start dictation. Check browser microphone permissions."); }
    },
  };
}
