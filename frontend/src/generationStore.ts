import {
  acknowledgeGeneration, getGenerationSnapshot, subscribeGenerationEvents,
  type GenerationEvent, type GenerationJob, type Message,
} from "./api";

export type GenerationState = {
  jobs: Record<string, GenerationJob>;
  messages: Record<string, Message[]>;
  connectionError: string | null;
  connected: boolean;
};

export function mergeMessages(current: Message[], incoming: Message[]): Message[] {
  const byId = new Map(current.map((message) => [message.id, message]));
  for (const message of incoming) {
    const existing = byId.get(message.id);
    if (!existing || (message.generation_revision ?? 0) >= (existing.generation_revision ?? 0)) {
      byId.set(message.id, message);
    }
  }
  return [...byId.values()].sort((a, b) => a.created_at.localeCompare(b.created_at));
}

export class GenerationStore {
  private state: GenerationState = { jobs: {}, messages: {}, connectionError: null, connected: false };
  private listeners = new Set<() => void>();
  private acknowledgements = new Set<string>();
  private lastEvent = 0;
  private deletedChats = new Set<string>();

  getSnapshot = () => this.state;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private publish(state: GenerationState) {
    this.state = state;
    this.listeners.forEach((listener) => listener());
  }

  mergeHistory(chatId: string, messages: Message[]) {
    if (this.deletedChats.has(chatId)) return;
    this.publish({ ...this.state, messages: {
      ...this.state.messages, [chatId]: mergeMessages(this.state.messages[chatId] ?? [], messages),
    } });
  }

  removeChat(chatId: string) {
    this.deletedChats.add(chatId);
    const jobs = { ...this.state.jobs };
    const messages = { ...this.state.messages };
    delete jobs[chatId];
    delete messages[chatId];
    this.publish({ ...this.state, jobs, messages });
  }

  applyJob(job: GenerationJob) {
    if (this.deletedChats.has(job.chat_id)) return;
    const previous = this.state.jobs[job.chat_id];
    if (previous?.id === job.id && previous.last_event_id > job.last_event_id) return;
    this.publish({ ...this.state,
      jobs: { ...this.state.jobs, [job.chat_id]: previous && previous.ordinal > job.ordinal ? previous : job },
      messages: { ...this.state.messages, [job.chat_id]: mergeMessages(
        this.state.messages[job.chat_id] ?? [], [job.user_message, job.assistant_message],
      ) },
    });
  }

  applyEvent = (event: GenerationEvent) => {
    if (event.id <= this.lastEvent) return;
    this.lastEvent = event.id;
    if (event.event.type === "deleted") this.removeChat(event.chat_id);
    else if (event.job) this.applyJob(event.job);
  };

  async acknowledge(job: GenerationJob) {
    if (job.acknowledged || this.acknowledgements.has(job.id)) return;
    this.acknowledgements.add(job.id);
    try {
      this.applyJob(await acknowledgeGeneration(job.id));
    } finally {
      this.acknowledgements.delete(job.id);
    }
  }

  async connect(signal: AbortSignal) {
    let delay = 300;
    while (!signal.aborted) {
      try {
        const snapshot = await getGenerationSnapshot();
        if (signal.aborted) return;
        const jobs = Object.fromEntries(snapshot.jobs
          .filter((job) => !this.deletedChats.has(job.chat_id)).map((job) => [job.chat_id, job]));
        for (const job of Object.values(this.state.jobs)) {
          if (job.last_event_id > snapshot.cursor) jobs[job.chat_id] = job;
        }
        const messages = { ...this.state.messages };
        for (const chatId of Object.keys(this.state.jobs)) {
          if (!jobs[chatId]) delete messages[chatId];
        }
        for (const job of snapshot.jobs) {
          if (this.deletedChats.has(job.chat_id)) continue;
          messages[job.chat_id] = mergeMessages(messages[job.chat_id] ?? [], [job.user_message, job.assistant_message]);
        }
        this.lastEvent = snapshot.cursor;
        this.publish({ jobs, messages, connected: true, connectionError: null });
        delay = 300;
        await subscribeGenerationEvents(this.lastEvent, this.applyEvent, signal);
        if (!signal.aborted) throw new Error("Lost connection to Atlas responses. Reconnecting…");
      } catch (error) {
        if (signal.aborted) return;
        this.publish({ ...this.state, connected: false,
          connectionError: error instanceof Error ? error.message : "Reconnecting to Atlas responses…",
        });
        await new Promise<void>((resolve) => {
          const abort = () => { clearTimeout(timer); resolve(); };
          const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, delay);
          signal.addEventListener("abort", abort, { once: true });
        });
        delay = Math.min(delay * 2, 5000);
      }
    }
  }
}
