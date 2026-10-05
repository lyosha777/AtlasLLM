import { useEffect, useState } from "react";
import { Mic } from "lucide-react";
import { appendTranscript, browserDictationProvider } from "./dictation";

export function DictationControl({ chatId, disabled, append, focus }: {
  chatId: string | null; disabled: boolean; append: (update: (draft: string) => string) => void; focus: () => void;
}) {
  const [provider] = useState(browserDictationProvider);
  const [listening, setListening] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string>();
  useEffect(() => {
    provider.cancel(); setListening(false); setStarting(false); setError(undefined);
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { provider.cancel(); setListening(false); setStarting(false); focus(); }
    };
    document.addEventListener("keydown", escape);
    return () => { provider.cancel(); document.removeEventListener("keydown", escape); };
  }, [chatId, disabled, provider, focus]);
  const active = listening || starting;
  return <span className="dictation-control">
    <span className="tool-control">
      <button aria-label={active ? "Stop dictation" : "Dictate"} aria-describedby="dictation-tooltip" aria-pressed={active}
        className={active ? "tool-active dictation-active" : ""} disabled={disabled || !provider.available}
        onClick={() => {
          if (active) { provider.stop(); focus(); return; }
          setError(undefined); setStarting(true);
          provider.start({ transcript: (text) => { append((draft) => appendTranscript(draft, text)); focus(); },
            listening: () => { setStarting(false); setListening(true); },
            ended: () => { setStarting(false); setListening(false); focus(); }, error: setError });
        }}><Mic size={17} /></button>
      <span className="tool-tooltip" id="dictation-tooltip" role="tooltip">{provider.available ? "Dictate" : "Dictation unsupported in this browser"}</span>
    </span>
    {active && <span className="dictation-label" role="status">{listening ? "Listening…" : "Starting…"}</span>}
    {error && <span className="dictation-error" role="status">{error}</span>}
  </span>;
}
