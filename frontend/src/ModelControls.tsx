import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, RefreshCw } from "lucide-react";

import type { ModelCatalog, ModelSelection, ReasoningCapability } from "./api";
import { reasoningLabel, resolveSelection } from "./modelSelection";


export function ReasoningOptions({ capability, selected, onSelect }: {
  capability: ReasoningCapability; selected: string | null; onSelect: (value: string) => void;
}) {
  return (
    <section className="reasoning-options">
      <div className="model-section-label">REASONING</div>
      {capability.settings.length > 0 ? (
        <div className="reasoning-choices" role="radiogroup" aria-label="Reasoning setting">
          {capability.settings.map((value) => (
            <button key={value} role="radio" aria-checked={(selected ?? capability.default) === value}
              onClick={() => onSelect(value)}>
              {value.charAt(0).toUpperCase() + value.slice(1)}
            </button>
          ))}
        </div>
      ) : <p className="model-fixed-mode">{reasoningLabel(capability)}</p>}
      <p className="model-explanation">{capability.explanation}</p>
    </section>
  );
}


export function ModelControls({ catalog, selection, onChange, loading, error, refresh, disabled }: {
  catalog: ModelCatalog | null; selection: ModelSelection; onChange: (selection: ModelSelection) => void;
  loading: boolean; error: string | null; refresh: () => void; disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const popover = useRef<HTMLDivElement>(null);
  const resolved = resolveSelection(catalog, selection);

  useEffect(() => {
    if (!open) return;
    popover.current?.querySelector<HTMLButtonElement>('[role="radio"][aria-checked="true"], button')?.focus();
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);

  return (
    <div className="model-control" ref={container}>
      <button className="model-trigger" ref={trigger} aria-haspopup="dialog" aria-expanded={open}
        aria-controls="model-popover" disabled={disabled} onClick={() => setOpen((value) => !value)}>
        <span>{resolved.name ?? "Local model"}</span>
        <small>{loading ? "Checking" : reasoningLabel(resolved.model?.reasoning, selection.reasoning)}</small>
        <ChevronDown size={12} />
      </button>
      {open && (
        <div id="model-popover" ref={popover} className="model-popover" role="dialog" aria-label="Model and reasoning"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.stopPropagation();
              setOpen(false);
              trigger.current?.focus();
            }
            if (["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft"].includes(event.key)) {
              const group = (event.target as HTMLElement).closest('[role="radiogroup"]');
              const choices = [...(group?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])];
              const index = choices.indexOf(event.target as HTMLButtonElement);
              if (index >= 0) {
                event.preventDefault();
                choices[(index + (event.key === "ArrowDown" || event.key === "ArrowRight" ? 1 : -1) + choices.length) % choices.length]?.focus();
              }
            }
          }}>
          <div className="model-popover-header">
            <span className="model-section-label">LOCAL MODELS</span>
            <button className="model-refresh" disabled={loading} onClick={refresh} aria-label="Refresh local models">
              <RefreshCw size={13} />
            </button>
          </div>
          {(error || catalog?.error) && <p className="model-error" role="status">{error ?? catalog?.error}</p>}
          {!loading && catalog?.online && !catalog.models.length && <p className="model-explanation">No models are installed. Add a model with Ollama, then refresh.</p>}
          <div className="model-list" role="radiogroup" aria-label="Local model">
            {catalog?.models.map((model) => (
              <button key={model.name} role="radio" aria-checked={resolved.name === model.name}
                disabled={!catalog.online || !model.available || !model.can_chat || disabled}
                onClick={() => onChange({ model: model.name, reasoning: null })}>
                <span><strong>{model.name}</strong><small>{model.available
                  ? [model.parameter_size, model.quantization, model.context_length ? `Max context ${model.context_length.toLocaleString()}` : null].filter(Boolean).join(" · ")
                  : "Metadata unavailable"}{model.available && !model.can_chat ? " · Not a chat model" : ""}</small></span>
                {resolved.name === model.name && <Check size={14} />}
              </button>
            ))}
          </div>
          {resolved.model?.reasoning && <ReasoningOptions capability={resolved.model.reasoning}
            selected={selection.reasoning} onSelect={(reasoning) => onChange({ model: resolved.name, reasoning })} />}
          {selection.reasoning && !resolved.model?.reasoning?.settings.includes(selection.reasoning) && (
            <p className="model-error">The saved reasoning setting is no longer supported.
              <button onClick={() => onChange({ model: resolved.name, reasoning: null })}>Use model default</button>
            </p>
          )}
          {!resolved.ready && !loading && <p className="model-explanation">Choose an available chat model with valid settings before sending.</p>}
        </div>
      )}
    </div>
  );
}
