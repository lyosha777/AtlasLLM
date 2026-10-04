import type { ModelCatalog, ModelSelection, ReasoningCapability } from "./api";


export function loadModelSelection(): ModelSelection {
  try {
    const value = JSON.parse(localStorage.getItem("atlas.modelSelection") ?? "null");
    return { model: typeof value?.model === "string" ? value.model : null,
      reasoning: typeof value?.reasoning === "string" ? value.reasoning : null };
  } catch {
    return { model: null, reasoning: null };
  }
}

export function resolveSelection(catalog: ModelCatalog | null, selection: ModelSelection) {
  const name = selection.model ?? catalog?.default_model ?? null;
  const model = catalog?.models.find((candidate) => candidate.name === name);
  const reasoning = selection.reasoning;
  const validReasoning = reasoning === null || !!model?.reasoning?.settings.includes(reasoning);
  return { name, model, reasoning, ready: !!catalog?.online && !!model?.available &&
    !!model.can_chat && !!model.reasoning && validReasoning };
}

export function reasoningLabel(capability?: ReasoningCapability, selected: string | null = null) {
  if (!capability) return "Unavailable";
  if (capability.kind === "fixed") return `Reasoning fixed ${capability.default ?? "by model"}`;
  if (capability.kind === "none") return "No thinking support";
  if (capability.kind === "unverified") return "Controls unverified";
  const mode = selected ?? capability.default;
  return mode ? mode.charAt(0).toUpperCase() + mode.slice(1) : "Model default";
}
