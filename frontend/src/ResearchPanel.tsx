import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { getResearchRun } from "./api";
import type { ResearchEvidence, ResearchRun } from "./api";

export function EvidencePassage({ evidence }: { evidence: ResearchEvidence }) {
  return <>
    <h3>{evidence.title ?? "Retained passage"}</h3>
    {evidence.url && <a href={evidence.url} target="_blank" rel="noreferrer">{evidence.url}</a>}
    <dl className="evidence-metadata">
      {evidence.provenance?.domain && <><dt>Domain</dt><dd>{evidence.provenance.domain}</dd></>}
      {evidence.page != null && <><dt>Page</dt><dd>{evidence.page}</dd></>}
      <dt>Publication date</dt><dd>{evidence.publication_date ?? "Unknown"}</dd>
      <dt>Retrieved</dt><dd>{evidence.retrieved_at}</dd>
      <dt>Query</dt><dd>{evidence.query}</dd>
      {evidence.document_id && <><dt>Document</dt><dd>{evidence.document_id}</dd></>}
      <dt>Chunk</dt><dd>{evidence.chunk_id}</dd>
      <dt>Evidence</dt><dd>{evidence.id}</dd>
      <dt>Retrieval / ranking</dt><dd>{evidence.retrieval_score?.toFixed(3) ?? "Unknown"} / {evidence.reranker_score?.toFixed(3) ?? "Unknown"}</dd>
    </dl>
    <blockquote className="evidence-passage">{evidence.passage}</blockquote>
    {evidence.occurrences.length > 1 && <details><summary>Found in {evidence.occurrences.length} searches</summary>
      {evidence.occurrences.map((item, index) => <p key={index}>{item.subquestion_id} · {item.query}</p>)}
    </details>}
    <p className="research-note">Scores describe relevance. Distinct domains or documents do not prove reliability or independence.</p>
  </>;
}

export function ResearchPanel({ runId, revision, inspectId, onCloseInspector }: {
  runId: string; revision: number; inspectId?: string; onCloseInspector: () => void;
}) {
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localId, setLocalId] = useState<string>();
  const [retry, setRetry] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    // Debounce bursts of persisted operations; reconnect uses the authoritative endpoint.
    const timer = window.setTimeout(() => {
      void getResearchRun(runId, controller.signal).then((result) => { setRun(result); setError(null); })
        .catch((failure: unknown) => { if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : "Couldn't load research."); });
    }, 150);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [runId, revision, retry]);
  const evidenceId = inspectId ?? localId;
  const evidence = run?.evidence.find((item) => item.id === evidenceId);
  useEffect(() => {
    if (evidenceId && !dialog.current?.open) dialog.current?.showModal();
    if (!evidenceId && dialog.current?.open) dialog.current?.close();
  }, [evidenceId]);
  const close = () => { setLocalId(undefined); onCloseInspector(); };
  return <>
    <details className="research-panel">
      <summary>DEEP RESEARCH{run?.metrics.depth && ` · ${run.metrics.depth.toUpperCase()}`} <span>{run?.status ?? "Loading"}</span>
        {run && <small>{run.metrics.searches ?? 0} searches · {run.metrics.candidates ?? 0} candidate passages · {run.metrics.evidence_selected ?? 0} selected</small>}
      </summary>
      {error && <p role="status">{error} <button onClick={() => setRetry(retry + 1)}>Retry</button></p>}
      {run && <div className="research-details">
        <p>{run.objective}</p>
        {run.metrics.depth_reason && <p className="research-note">{run.metrics.depth_reason}
          {run.metrics.escalated && ` Escalated from ${run.metrics.initial_depth}.`}</p>}
        {run.error && <p role="status">{run.error}</p>}
        <div className="model-section-label">PLAN</div>
        {run.subquestions.map((question) => <div className="research-row" key={question.id}>
          <strong>{question.id} · {question.question}</strong><p>{question.reason}</p>
          {run.analysis?.coverage.filter((item) => item.subquestion_id === question.id).map((item) =>
            <p key={item.subquestion_id} className={`coverage-${item.status}`}>{item.status} · {item.explanation}</p>)}
        </div>)}
        <div className="model-section-label">SEARCHES</div>
        {run.searches.map((search) => <div className="research-row" key={search.id}>
          <strong>{search.query}</strong><small>{search.subquestion_id} · {search.source_type} · {search.status}
            {search.followup ? " · follow-up" : ""} · {search.candidates ?? 0} passages · {search.duration_seconds ?? 0}s</small>
          {search.error && <p>{search.error}</p>}
          {!!search.fetch?.failed.length && <p>{search.fetch.failed.length} pages could not be extracted.</p>}
        </div>)}
        <div className="model-section-label">EVIDENCE</div>
        {run.evidence.map((item) => <button className="research-evidence" key={item.id} onClick={() => setLocalId(item.id)}>
          <strong>{item.title ?? item.url ?? "Document"}</strong>
          <small>{item.selected_for_synthesis ? "Selected" : "Retained"} · {item.subquestion_ids.join(", ")} · {item.source_type}</small>
          <span>{item.passage.slice(0, 170)}{item.passage.length > 170 ? "…" : ""}</span>
        </button>)}
        {!!run.analysis?.claims.length && <><div className="model-section-label">CLAIMS / SUPPORT</div>
          {run.analysis.claims.map((claim) => <div className="research-row" key={claim.id}><p>{claim.statement}</p>
            {!!claim.ambiguous_evidence_ids?.length && <p className="coverage-weak">The same passage appears in support and conflict. Interpret this claim cautiously.</p>}
            <small>{claim.supporting_origins} distinct origins · independence unverified</small>
            {(["supporting_evidence_ids", "conflicting_evidence_ids"] as const).map((field) =>
              claim[field].length > 0 && <div key={field}>{field.startsWith("supporting") ? "Support" : "Conflict"}: {claim[field].map((id, index) =>
                <button className="citation-button" key={id} onClick={() => setLocalId(id)}>Passage {index + 1}</button>)}</div>)}
          </div>)}</>}
        <p className="research-note">{run.metrics.urls_fetched ?? 0} URL fetch attempts · {run.metrics.pages_fetched ?? 0} pages extracted · {run.metrics.duplicates ?? 0} duplicate passages · {run.metrics.followup_searches ?? 0} follow-up searches · {run.metrics.total_seconds ?? 0}s</p>
        <details><summary>Stage timings and limits</summary><pre>{JSON.stringify({ timings: run.metrics.stage_timings, budgets: run.budgets }, null, 2)}</pre></details>
      </div>}
    </details>
    <dialog ref={dialog} className="evidence-inspector" onCancel={(event) => { event.preventDefault(); close(); }}
      onClick={(event) => { if (event.target === event.currentTarget) close(); }}>
      <button className="evidence-close" aria-label="Close evidence inspector" onClick={close}><X size={18} /></button>
      {evidence ? <EvidencePassage evidence={evidence} /> : <p role="status">{error ?? "Loading retained evidence…"}</p>}
    </dialog>
  </>;
}
