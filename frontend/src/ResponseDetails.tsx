import { useEffect, useRef, useState } from "react";
import { Volume2, Square } from "lucide-react";
import { browserSpeechProvider } from "./responseSpeech";

export function useResponseSpeech(chatId: string | null) {
  const [provider] = useState(browserSpeechProvider);
  const [state, setState] = useState<{ id?: string; loading?: boolean; error?: string }>({});
  const revision = useRef(0);
  useEffect(() => {
    revision.current++; provider.stop(); setState({});
    return () => { revision.current++; provider.stop(); };
  }, [chatId, provider]);
  return { state, available: provider.available, toggle(id: string, text: string) {
    const current = ++revision.current;
    provider.stop();
    if (state.id === id) { setState({}); return; }
    setState({ id, loading: true });
    const update = (next: typeof state) => { if (revision.current === current) setState(next); };
    provider.speak(text, () => update({ id }), () => update({}),
      () => update({ error: "Read aloud unavailable. Enable a local system voice in your browser or operating system." }));
  } };
}

export function SpeechButton({ id, text, speech }: { id: string; text: string; speech: ReturnType<typeof useResponseSpeech> }) {
  const active = speech.state.id === id;
  const label = active ? (speech.state.loading ? "Loading audio" : "Stop reading") : "Read aloud";
  return <span className="response-speech-tool">
    <button disabled={!speech.available} aria-label={label} aria-pressed={active} onClick={() => speech.toggle(id, text)}>
      {active ? <Square size={15} /> : <Volume2 size={15} />}{active && <span>{speech.state.loading ? "Loading…" : "Reading"}</span>}
    </button><span className="response-tool-tip" role="tooltip">{speech.available ? label : "Read aloud unavailable in this browser"}</span>
  </span>;
}

export function PerformanceDetails({ timings, defaultOpen = false }: { timings?: Record<string, number>; defaultOpen?: boolean }) {
  if (!timings || !Object.keys(timings).length) return null;
  const seconds = (value: number) => value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(2)} s`;
  const rows: [string, number | undefined, boolean?][] = [
    ["TTFT (backend)", timings.ttft], ["Inference TTFT", timings.inference_ttft],
    ["Generation", timings.tps, true], ["TPOT", timings.tpot], ["E2E (backend)", timings.total],
    ["Planner", timings.planner ?? timings.planning], ["Documents", timings.documents],
    ["Web retrieval", timings.web], ["Retrieval", timings.retrieval],
    ["Reranker", timings.reranking ?? timings.reranker], ["Analysis", timings.analysis],
  ];
  return <details className="response-performance" open={defaultOpen || undefined}><summary>Performance</summary><dl>
    {rows.filter(([, value]) => value != null && Number.isFinite(value)).map(([label, value, rate]) =>
      <div key={label}><dt>{label}</dt><dd>{rate ? `${value!.toFixed(1)} tok/s` : seconds(value!)}</dd></div>)}
  </dl><p>Backend timings exclude browser/network delivery. TPOT uses provider decode duration and token count, not stream-chunk intervals.</p></details>;
}
