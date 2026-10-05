export type AtlasDocument = {
  id: string;
  name: string;
  size: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
};

export type Source = {
  index: number;
  type: "document" | "web";
  title: string | null;
  url: string | null;
  page: number | null;
  score: number | null;
  document_id?: string | null;
  evidence_id?: string;
  research_run_id?: string;
  chunk_id?: string;
};

export type Message = {
  id: string;
  chat_id?: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  attachments?: AtlasDocument[];
  created_at: string;
  generation_id?: string | null;
  generation_status?: GenerationStatus | null;
  generation_revision?: number;
  inference?: InferenceMetadata | null;
  research_run_id?: string | null;
  research_revision?: number;
  timings?: Record<string, number>;
};

export type Chat = {
  id: string;
  title: string;
  archived: boolean;
  created_at: string;
  updated_at: string;
  message_count?: number;
};

export type StreamEvent =
  | {
      type: "status";
      stage: string;
      label: string;
    }
  | {
      type: "plan";
      documents: boolean;
      web: boolean;
      fresh: boolean;
    }
  | {
      type: "sources";
      sources: Source[];
    }
  | {
      type: "token";
      content: string;
    }
  | {
      type: "done";
      timings: Record<string, number>;
    }
  | {
      type: "error";
      message: string;
    }
  | {
      type: "inference";
      settings: NonNullable<InferenceMetadata["actual_settings"]>;
    };

const API = import.meta.env.VITE_API_URL ?? "/api";

export type ReasoningCapability = {
  kind: "none" | "fixed" | "binary" | "effort" | "unverified";
  thinking_supported: boolean | null;
  settings: string[];
  default: string | null;
  values: (boolean | string)[];
  verified: boolean;
  explanation: string;
  provider_default?: boolean | string | null;
};

export type LocalModel = {
  name: string;
  available: boolean;
  can_chat: boolean;
  family?: string | null;
  context_length?: number | null;
  parameter_size?: string | null;
  quantization?: string | null;
  capabilities?: string[];
  reasoning?: ReasoningCapability;
  error?: string;
};

export type ModelCatalog = {
  online: boolean;
  default_model: string;
  models: LocalModel[];
  error: string | null;
  server_version?: string | null;
};

export type ModelSelection = { model: string | null; reasoning: string | null };

export type InferenceMetadata = {
  provider: string;
  model: string;
  model_digest: string | null;
  requested_reasoning: string | null;
  reasoning_mode: string;
  capability: ReasoningCapability;
  settings: { think?: boolean | string };
  actual_settings: { model: string; think?: boolean | string; options: Record<string, number> } | null;
};

export async function getModels(signal?: AbortSignal): Promise<ModelCatalog> {
  const response = await fetch(`${API}/models`, { signal });
  if (!response.ok) throw new Error("Couldn't discover local models. Check the Atlas backend.");
  return response.json();
}

export type GenerationStatus = "queued" | "running" | "cancelling" | "completed" | "cancelled" | "error" | "interrupted";

export type GenerationJob = {
  id: string;
  ordinal: number;
  chat_id: string;
  chat_title: string;
  status: GenerationStatus;
  activity: string | null;
  error: string | null;
  last_event_id: number;
  acknowledged: boolean;
  user_message: Message;
  assistant_message: Message;
  timings: Record<string, number>;
  created_at: string;
  updated_at: string;
  inference?: InferenceMetadata | null;
};

export type GenerationEvent = {
  id: number;
  chat_id: string;
  job: GenerationJob | null;
  event: { type: string };
};

export const isActiveGeneration = (job?: GenerationJob) =>
  !!job && ["queued", "running", "cancelling"].includes(job.status);

async function jobResponse(response: Response): Promise<GenerationJob> {
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === "string" ? error.detail : "Couldn't update this response.");
  }
  return response.json();
}

export async function submitGeneration(
  chatId: string, message: string,
  options: { forceWeb: boolean; documentIds: string[]; requestId: string; model?: string; reasoning?: string | null; researchMode?: boolean; researchScope?: ResearchScope },
): Promise<GenerationJob> {
  return jobResponse(await fetch(`${API}/chats/${chatId}/generations`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, force_web: options.forceWeb, document_ids: options.documentIds,
      ...(options.researchMode ? { research_mode: true, research_scope: options.researchScope ?? "auto" } : {}),
      request_id: options.requestId, ...(options.model ? { model: options.model, reasoning: options.reasoning ?? null } : {}) }),
  }));
}

export async function getGenerationSnapshot(): Promise<{ cursor: number; jobs: GenerationJob[] }> {
  const response = await fetch(`${API}/generations`);
  if (!response.ok) throw new Error("Couldn't reconnect to Atlas responses.");
  return response.json();
}

export async function cancelGeneration(jobId: string): Promise<GenerationJob> {
  return jobResponse(await fetch(`${API}/generations/${jobId}/cancel`, { method: "POST" }));
}

export async function acknowledgeGeneration(jobId: string): Promise<GenerationJob> {
  return jobResponse(await fetch(`${API}/generations/${jobId}/acknowledge`, { method: "POST" }));
}

export async function subscribeGenerationEvents(
  after: number, onEvent: (event: GenerationEvent) => void, signal: AbortSignal,
) {
  const response = await fetch(`${API}/generations/events?after=${after}`, { signal });
  if (!response.ok || !response.body) throw new Error("Lost connection to Atlas responses.");
  await readSse<GenerationEvent>(response.body, onEvent);
}

async function readSse<T>(body: ReadableStream<Uint8Array>, onEvent: (event: T) => void) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        const line = block.split("\n").find((item) => item.startsWith("data: "));
        if (line) onEvent(JSON.parse(line.slice(6)) as T);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function getChats(
  archived = false
): Promise<Chat[]> {
  const response = await fetch(
    `${API}/chats?archived=${archived}`
  );

  if (!response.ok) {
    throw new Error("Failed to load chats");
  }

  return response.json();
}

export async function createChat(
  title: string,
  signal?: AbortSignal,
): Promise<Chat> {
  const response = await fetch(`${API}/chats`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
    signal,
  });

  if (!response.ok) {
    throw new Error("Failed to create chat");
  }

  return response.json();
}

export async function updateChat(
  id: string,
  data: {
    title?: string;
    archived?: boolean;
  }
): Promise<Chat> {
  const response = await fetch(`${API}/chats/${id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error("Failed to update chat");
  }

  return response.json();
}

export async function deleteChat(
  id: string
): Promise<void> {
  const response = await fetch(`${API}/chats/${id}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to delete chat");
  }
}

export async function getMessages(
  chatId: string
): Promise<Message[]> {
  const response = await fetch(
    `${API}/chats/${chatId}/messages`
  );

  if (!response.ok) {
    throw new Error("Failed to load messages");
  }

  return response.json();
}

export async function streamMessage(
  chatId: string,
  message: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
  options: {
    forceWeb?: boolean;
    documentIds?: string[];
    onAccepted?: () => void;
  } = {},
): Promise<void> {
  const response = await fetch(
    `${API}/chats/${chatId}/messages/stream`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        force_web: options.forceWeb ?? false,
        document_ids: options.documentIds ?? [],
      }),
      signal,
    }
  );

  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === "string" ? data.detail : "Failed to send message");
  }

  options.onAccepted?.();

  let completed = false;
  await readSse<StreamEvent>(response.body, (event) => {
    onEvent(event);
    if (event.type === "done") completed = true;
    if (event.type === "error") throw new Error(event.message);
  });
  if (!completed) throw new Error("Atlas's response ended unexpectedly.");
}


export function uploadDocument(
  file: File,
  onProgress: (progress: number | null) => void,
  signal: AbortSignal,
): Promise<AtlasDocument> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const abort = () => request.abort();
    const cleanup = () => signal.removeEventListener("abort", abort);
    request.open("POST", `${API}/documents`);
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      onProgress(event.lengthComputable ? Math.round(event.loaded / event.total * 100) : 0);
    };
    request.upload.onload = () => onProgress(null);
    request.onload = () => {
      cleanup();
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response as AtlasDocument);
      } else {
        reject(new Error(request.response?.detail ?? "Couldn't upload this PDF."));
      }
    };
    request.onerror = () => {
      cleanup();
      reject(new Error("Couldn't reach Atlas to upload this PDF."));
    };
    request.onabort = () => {
      cleanup();
      reject(new DOMException("Upload cancelled", "AbortError"));
    };
    if (signal.aborted) {
      reject(new DOMException("Upload cancelled", "AbortError"));
      return;
    }
    signal.addEventListener("abort", abort, { once: true });
    const body = new FormData();
    body.append("file", file);
    request.send(body);
  });
}


export type ResearchScope = "auto" | "web" | "documents" | "both";
export type ResearchEvidence = {
  id: string; run_id: string; subquestion_id: string; subquestion_ids: string[];
  source_type: string; title: string | null; url: string | null; document_id: string | null;
  page: number | null; chunk_id: string; passage: string; query: string; publication_date: string | null;
  retrieved_at: string; retrieval_score: number | null; reranker_score: number | null;
  selected_for_synthesis: boolean; occurrences: { subquestion_id: string; query: string }[];
  provenance?: { domain: string | null; retrieval: string; ranking: string };
};
export type ResearchRun = {
  id: string; status: string; scope: ResearchScope; objective: string | null; revision: number; error: string | null;
  metrics: { searches?: number; candidates?: number; evidence_selected?: number; urls_fetched?: number;
    pages_fetched?: number; web_chunks_considered?: number; duplicates?: number; followup_searches?: number;
    total_seconds?: number; stage_timings?: Record<string, number>; depth?: "quick" | "standard" | "deep";
    initial_depth?: string; depth_reason?: string; escalated?: boolean;
    escalations?: { from_depth: string; to_depth: string; reason: string }[] };
  budgets: Record<string, number>;
  subquestions: { id: string; question: string; reason: string; search_queries: string[] }[];
  searches: { id: string; subquestion_id: string; query: string; source_type: string; status: string;
    followup: boolean; candidates?: number; duration_seconds?: number; error?: string; fetch?: { failed: string[] } }[];
  evidence: ResearchEvidence[];
  analysis: { coverage: { subquestion_id: string; status: string; explanation: string }[];
    claims: { id: string; statement: string; supporting_evidence_ids: string[]; conflicting_evidence_ids: string[];
      supporting_origins: number; ambiguous_evidence_ids?: string[] }[] } | null;
};
export async function getResearchRun(runId: string, signal?: AbortSignal): Promise<ResearchRun> {
  const response = await fetch(`${API}/research/${encodeURIComponent(runId)}`, { signal });
  if (!response.ok) throw new Error(response.status === 404 ? "This research run is no longer available." : "Couldn't load research artifacts.");
  return response.json();
}
