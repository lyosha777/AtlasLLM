import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { ResearchScope } from "./api";

export function ResearchControls({ enabled, scope, onChange, disabled }: {
  enabled: boolean; scope: ResearchScope; onChange: (enabled: boolean, scope: ResearchScope) => void; disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus();
    const close = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  return <div className="research-control" ref={container}>
    <button className={`research-trigger ${enabled ? "tool-active" : ""}`} ref={trigger}
      aria-haspopup="dialog" aria-expanded={open} disabled={disabled} onClick={() => setOpen(!open)}>
      {enabled ? "Deep Research" : "Research"}<ChevronDown size={12} />
    </button>
    {open && <div className="research-popover" role="dialog" aria-label="Research mode" ref={menu}
      onKeyDown={(event) => { if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); } }}>
      <div className="model-section-label">RESEARCH MODE</div>
      <div role="radiogroup" aria-label="Generation mode" className="reasoning-choices">
        <button role="radio" aria-checked={!enabled} onClick={() => onChange(false, scope)}>Standard</button>
        <button role="radio" aria-checked={enabled} onClick={() => onChange(true, scope)}>Deep Research</button>
      </div>
      <p className="model-explanation">A bounded research plan, evidence review, and cited synthesis. Model reasoning is selected separately.</p>
      {enabled && <><div className="model-section-label">SOURCES</div>
        <div className="research-scopes" role="radiogroup" aria-label="Research sources">
          {(["auto", "web", "documents", "both"] as ResearchScope[]).map((value) =>
            <button key={value} role="radio" aria-checked={scope === value} onClick={() => onChange(true, value)}>
              {{ auto: "Automatic", web: "Web", documents: "Documents", both: "Web + documents" }[value]}
            </button>)}
        </div>
        <p className="model-explanation">Attached files limit research to those documents. With no attachments, Atlas can search your indexed library.</p></>}
    </div>}
  </div>;
}
