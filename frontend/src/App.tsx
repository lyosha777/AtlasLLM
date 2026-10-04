import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import {
  Archive,
  ArchiveRestore,
  ArrowUp,
  Check,
  Copy,
  FileText,
  Globe2,
  Moon,
  MoreHorizontal,
  Pencil,
  Plus,
  Search,
  Settings,
  Square,
  Sun,
  Trash2,
  X,
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  createChat,
  deleteChat,
  getChats,
  getMessages,
  submitGeneration,
  cancelGeneration,
  isActiveGeneration,
  updateChat,
  uploadDocument,
  type AtlasDocument,
  type Chat,
  type Message,
  type Source,
  type GenerationJob,
} from "./api";

import "./App.css";
import { useGenerations } from "./useGenerations";
import { useModels } from "./useModels";
import { ModelControls } from "./ModelControls";
import { loadModelSelection, reasoningLabel, resolveSelection } from "./modelSelection";

const EMPTY_MESSAGES: Message[] = [];


type Theme = "light" | "dark";

type ComposerFile = {
  id: string;
  file: File;
  status: "queued" | "uploading" | "ready" | "error";
  progress: number | null;
  document?: AtlasDocument;
  error?: string;
};

type ConfirmState = {
  chat: Chat;
} | null;


function makeLocalId() {
  return crypto.randomUUID();
}


function makeTitle(text: string) {
  const clean = text.replace(/\s+/g, " ").trim();

  if (clean.length <= 42) {
    return clean;
  }

  return `${clean.slice(0, 42)}…`;
}


function loadTheme(): Theme {
  const stored = localStorage.getItem("atlas.theme");

  if (stored === "light" || stored === "dark") {
    return stored;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}


function formatGroup(dateString: string) {
  const date = new Date(dateString);
  const today = new Date();

  const startToday = new Date(
    today.getFullYear(),
    today.getMonth(),
    today.getDate(),
  );

  const startDate = new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate(),
  );

  const difference = Math.round(
    (startToday.getTime() - startDate.getTime()) / 86_400_000,
  );

  if (difference === 0) return "TODAY";
  if (difference === 1) return "YESTERDAY";
  if (difference < 7) return "THIS WEEK";

  return date
    .toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
    })
    .toUpperCase();
}


function sourceLabel(source: Source) {
  if (source.type === "document" && source.page != null) {
    return `PAGE ${source.page}`;
  }

  if (source.type === "web") {
    return "WEB";
  }

  return source.type.toUpperCase();
}


function App() {
  const [theme, setTheme] = useState<Theme>(loadTheme);

  const [chats, setChats] = useState<Chat[]>([]);

  const [activeChatId, setActiveChatId] =
    useState<string | null>(() => localStorage.getItem("atlas.activeChat"));

  const generations = useGenerations();
  const models = useModels();
  const [modelSelection, setModelSelection] = useState(loadModelSelection);
  const selectedModel = resolveSelection(models.catalog, modelSelection);
  const messages = activeChatId ? generations.messages[activeChatId] ?? EMPTY_MESSAGES : EMPTY_MESSAGES;
  const activeJob = activeChatId ? generations.jobs[activeChatId] : undefined;
  const generating = isActiveGeneration(activeJob);

  const [showArchived, setShowArchived] = useState(false);
  const [search, setSearch] = useState("");

  const [input, setInput] = useState("");
  const [forceWeb, setForceWeb] = useState(false);
  const [files, setFiles] = useState<ComposerFile[]>([]);
  const [submissions, setSubmissions] = useState<Set<string>>(new Set());
  const [draftId, setDraftId] = useState(makeLocalId);

  const [loadingChats, setLoadingChats] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);

  const activity = activeJob?.activity;

  const [openMenuId, setOpenMenuId] =
    useState<string | null>(null);

  const [renamingId, setRenamingId] =
    useState<string | null>(null);

  const [renameValue, setRenameValue] = useState("");

  const [confirmDelete, setConfirmDelete] =
    useState<ConfirmState>(null);

  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const composerRef = useRef<HTMLDivElement | null>(null);
  const uploadControllers = useRef(new Map<string, AbortController>());
  const uploadQueue = useRef<Promise<void>>(Promise.resolve());
  const submissionLocks = useRef(new Set<string>());
  const viewVersion = useRef(0);
  const historyRequest = useRef(0);
  const submissionKey = activeChatId ?? draftId;
  const submitting = submissions.has(submissionKey);
  const terminalJobs = Object.values(generations.jobs)
    .filter((job) => !isActiveGeneration(job))
    .map((job) => `${job.id}:${job.status}`).sort().join(",");
  const notifications = Object.values(generations.jobs)
    .filter((job) => job.chat_id !== activeChatId && job.status === "completed" && !job.acknowledged);


  const activeChat = useMemo(
    () =>
      chats.find((chat) => chat.id === activeChatId) ??
      null,
    [chats, activeChatId],
  );


  const filteredChats = useMemo(() => {
    const query = search.trim().toLowerCase();

    return chats
      .filter((chat) => chat.archived === showArchived)
      .filter(
        (chat) =>
          !query ||
          chat.title.toLowerCase().includes(query),
      )
      .sort(
        (a, b) =>
          new Date(b.updated_at).getTime() -
          new Date(a.updated_at).getTime(),
      );
  }, [chats, search, showArchived]);


  const groupedChats = useMemo(() => {
    const groups: Record<string, Chat[]> = {};

    for (const chat of filteredChats) {
      const group = formatGroup(chat.updated_at);

      if (!groups[group]) {
        groups[group] = [];
      }

      groups[group].push(chat);
    }

    return groups;
  }, [filteredChats]);


  const archivedCount = chats.filter(
    (chat) => chat.archived,
  ).length;


  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("atlas.theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("atlas.modelSelection", JSON.stringify(modelSelection));
  }, [modelSelection]);


  useEffect(() => {
    void refreshChats();
  }, [terminalJobs]);

  useEffect(() => {
    if (activeChatId) {
      localStorage.setItem("atlas.activeChat", activeChatId);
      void loadConversation(activeChatId);
    } else {
      localStorage.removeItem("atlas.activeChat");
    }
    // The store merges history by message ID and persisted revision.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChatId]);

  useEffect(() => {
    if (!activeJob || isActiveGeneration(activeJob)) return;
    void generations.store.acknowledge(activeJob).catch(() => {});
    requestAnimationFrame(() => {
      const focused = document.activeElement;
      if (focused === document.body || focused?.closest(".composer")) textareaRef.current?.focus();
    });
  }, [activeJob, generations.store]);


  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: generating ? "auto" : "smooth",
    });
  }, [messages, activity, generating]);


  useEffect(() => {
    const textarea = textareaRef.current;

    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height =
      `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [input]);

  useEffect(() => {
    const composer = composerRef.current;
    if (!composer) return;
    const observer = new ResizeObserver(() => {
      composer.parentElement?.style.setProperty("--composer-height", `${composer.offsetHeight}px`);
    });
    observer.observe(composer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const controllers = uploadControllers.current;
    return () => {
      controllers.forEach((controller) => controller.abort());
    };
  }, []);


  async function refreshChats() {
    try {
      setLoadingChats(true);

      const [active, archived] = await Promise.all([
        getChats(false),
        getChats(true),
      ]);

      setChats([...active, ...archived]);
    } catch {
      setError(
        "Atlas couldn't reach the local backend.",
      );
    } finally {
      setLoadingChats(false);
    }
  }


  async function loadConversation(chatId: string) {
    const request = ++historyRequest.current;
    try {
      setLoadingMessages(true);

      const result = await getMessages(chatId);
      if (request !== historyRequest.current) return;
      generations.store.mergeHistory(chatId, result);
      setError(null);
    } catch {
      if (request !== historyRequest.current) return;
      setError("Couldn't load this conversation.");
    } finally {
      if (request === historyRequest.current) setLoadingMessages(false);
    }
  }


  function startNewChat() {

    viewVersion.current += 1;
    setDraftId(makeLocalId());
    historyRequest.current += 1;
    setLoadingMessages(false);
    clearFiles();
    setForceWeb(false);

    setActiveChatId(null);
    setInput("");
    setError(null);
    setOpenMenuId(null);

    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }


  function selectChat(id: string) {

    viewVersion.current += 1;
    clearFiles();
    setForceWeb(false);
    setInput("");

    setOpenMenuId(null);
    setError(null);
    setActiveChatId(id);
  }


  async function handleArchive(chat: Chat) {
    try {
      await updateChat(chat.id, {
        archived: true,
      });

      if (activeChatId === chat.id) {
        startNewChat();
      }

      setOpenMenuId(null);
      await refreshChats();
    } catch {
      setError("Couldn't archive the thread.");
    }
  }


  async function handleRestore(chat: Chat) {
    try {
      await updateChat(chat.id, {
        archived: false,
      });

      setOpenMenuId(null);
      await refreshChats();
    } catch {
      setError("Couldn't restore the thread.");
    }
  }


  function beginRename(chat: Chat) {
    setRenamingId(chat.id);
    setRenameValue(chat.title);
    setOpenMenuId(null);
  }


  async function commitRename() {
    if (!renamingId) return;

    const title = renameValue.trim();

    if (!title) {
      setRenamingId(null);
      return;
    }

    try {
      await updateChat(renamingId, {
        title,
      });

      setRenamingId(null);
      await refreshChats();
    } catch {
      setError("Couldn't rename the thread.");
    }
  }


  async function handleDelete() {
    if (!confirmDelete) return;

    const chat = confirmDelete.chat;

    try {
      await deleteChat(chat.id);
      generations.store.removeChat(chat.id);

      if (activeChatId === chat.id) {
        startNewChat();
      }

      setConfirmDelete(null);
      await refreshChats();
    } catch {
      setError("Couldn't delete the thread.");
    }
  }


  function stopGeneration() {
    if (!activeJob || !generating) return;
    void cancelGeneration(activeJob.id).then((job) => generations.store.applyJob(job))
      .catch((error: unknown) => setError(error instanceof Error ? error.message : "Couldn't stop this response."));
  }


  async function copyMessage(message: Message) {
    await navigator.clipboard.writeText(message.content);

    setCopiedId(message.id);

    window.setTimeout(() => {
      setCopiedId(null);
    }, 1300);
  }


  function clearFiles() {
    uploadControllers.current.forEach((controller) => controller.abort());
    uploadControllers.current.clear();
    setFiles([]);
  }

  function removeFile(id: string) {
    uploadControllers.current.get(id)?.abort();
    uploadControllers.current.delete(id);
    setFiles((current) => current.filter((file) => file.id !== id));
  }

  function queueUpload(item: ComposerFile) {
    const controller = new AbortController();
    uploadControllers.current.set(item.id, controller);
    uploadQueue.current = uploadQueue.current.then(async () => {
      if (controller.signal.aborted) return;
      setFiles((current) => current.map((file) => file.id === item.id
        ? { ...file, status: "uploading", progress: 0, error: undefined } : file));
      try {
        const document = await uploadDocument(item.file, (progress) => {
          setFiles((current) => current.map((file) => file.id === item.id
            ? { ...file, progress } : file));
        }, controller.signal);
        setFiles((current) => current.map((file) => file.id === item.id
          ? { ...file, status: "ready", document } : file));
      } catch (error) {
        if (controller.signal.aborted) return;
        setFiles((current) => current.map((file) => file.id === item.id
          ? { ...file, status: "error", error: error instanceof Error ? error.message : "Upload failed." }
          : file));
      } finally {
        uploadControllers.current.delete(item.id);
      }
    });
  }

  function addFiles(selected: FileList | null) {
    if (!selected) return;
    const candidates = Array.from(selected);
    if (files.length + candidates.length > 20) {
      setError("Attach up to 20 PDFs per message.");
      return;
    }
    const added = candidates.map((file): ComposerFile => {
      const error = !file.name.toLowerCase().endsWith(".pdf")
        ? "Only PDF files are supported."
        : file.size > 25 * 1024 * 1024
          ? "PDFs must be 25 MB or smaller."
          : undefined;
      return { id: makeLocalId(), file, status: error ? "error" : "queued", progress: 0, error };
    });
    setFiles((current) => [...current, ...added]);
    added.filter((file) => !file.error).forEach(queueUpload);
  }


  async function send() {
    const text = input.trim();
    const key = submissionKey;
    const job = activeChatId ? generations.store.getSnapshot().jobs[activeChatId] : undefined;
    if (!text || !generations.connected || !selectedModel.ready || !selectedModel.name || submissionLocks.current.has(key) || isActiveGeneration(job) || loadingMessages ||
      activeChat?.archived || files.some((file) => file.status !== "ready")) return;

    submissionLocks.current.add(key);
    setSubmissions((current) => new Set(current).add(key));
    setError(null);
    setInput("");
    textareaRef.current?.focus();
    const version = viewVersion.current;
    const selectedFiles = files;
    const documentIds = selectedFiles.flatMap((file) => file.document ? [file.document.id] : []);
    const useWeb = forceWeb;
    const requestId = makeLocalId();
    const submittedModel = selectedModel.name;
    const submittedReasoning = modelSelection.reasoning;
    let chatId = activeChatId;

    try {
      if (!chatId) {
        const chat = await createChat(makeTitle(text));
        chatId = chat.id;
        submissionLocks.current.add(chatId);
        setSubmissions((current) => new Set(current).add(chat.id));
        setChats((current) => [chat, ...current.filter((item) => item.id !== chat.id)]);
        if (version === viewVersion.current) setActiveChatId(chat.id);
      }
      const job = await submitGeneration(chatId, text, { forceWeb: useWeb, documentIds, requestId,
        model: submittedModel, reasoning: submittedReasoning });
      generations.store.applyJob(job);
      if (version === viewVersion.current) {
        setForceWeb(false);
        setFiles((current) => current.filter((file) => !selectedFiles.some((sent) => sent.id === file.id)));
      }
      await refreshChats();
    } catch (error) {
      if (version === viewVersion.current) {
        setInput((draft) => draft ? `${text}\n${draft}` : text);
        setError(error instanceof Error ? error.message : "Couldn't submit this message.");
      }
    } finally {
      submissionLocks.current.delete(key);
      if (chatId) submissionLocks.current.delete(chatId);
      setSubmissions((current) => {
        const next = new Set(current);
        next.delete(key);
        if (chatId) next.delete(chatId);
        return next;
      });
      if (version === viewVersion.current) {
        requestAnimationFrame(() => {
          if (document.activeElement === document.body || document.activeElement?.closest(".composer")) {
            textareaRef.current?.focus();
          }
        });
      }
    }
  }

  function handleComposerKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      void send();
    }
  }


  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand">
            ATLAS
          </div>

          <button
            className="icon-button"
            onClick={() =>
              setTheme((current) =>
                current === "light"
                  ? "dark"
                  : "light",
              )
            }
            title="Toggle appearance"
          >
            {theme === "light" ? (
              <Moon size={18} />
            ) : (
              <Sun size={18} />
            )}
          </button>
        </div>

        <button
          className="new-thread"
          onClick={startNewChat}
        >
          <Plus size={17} />
          New thread
        </button>

        <label className="sidebar-search">
          <Search size={16} />

          <input
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
            placeholder="Search threads"
          />
        </label>

        <div className="view-switch">
          <button
            className={
              !showArchived ? "selected" : ""
            }
            onClick={() =>
              setShowArchived(false)
            }
          >
            Threads
          </button>

          <button
            className={
              showArchived ? "selected" : ""
            }
            onClick={() =>
              setShowArchived(true)
            }
          >
            Archive

            {archivedCount > 0 && (
              <span>{archivedCount}</span>
            )}
          </button>
        </div>

        <div className="thread-list">
          {loadingChats && (
            <div className="sidebar-message">
              Loading…
            </div>
          )}

          {!loadingChats &&
            Object.keys(groupedChats).length ===
              0 && (
              <div className="sidebar-message">
                {showArchived
                  ? "No archived threads."
                  : search
                    ? "No matching threads."
                    : "No threads yet."}
              </div>
            )}

          {Object.entries(groupedChats).map(
            ([group, groupChats]) => (
              <section
                className="thread-group"
                key={group}
              >
                <div className="thread-date">
                  {group}
                </div>

                {groupChats.map((chat) => (
                  <div
                    className={`thread-row ${
                      activeChatId === chat.id
                        ? "thread-active"
                        : ""
                    }`}
                    key={chat.id}
                  >
                    {renamingId === chat.id ? (
                      <input
                        className="rename-input"
                        value={renameValue}
                        autoFocus
                        onChange={(event) =>
                          setRenameValue(
                            event.target.value,
                          )
                        }
                        onBlur={() =>
                          void commitRename()
                        }
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter"
                          ) {
                            void commitRename();
                          }

                          if (
                            event.key === "Escape"
                          ) {
                            setRenamingId(null);
                          }
                        }}
                      />
                    ) : (
                      <button
                        className="thread-title"
                        onClick={() =>
                          void selectChat(chat.id)
                        }
                      >
                        <span className="thread-title-text">{chat.title}</span>
                        {generations.jobs[chat.id] && (
                          <JobIndicator job={generations.jobs[chat.id]} />
                        )}
                      </button>
                    )}

                    <button
                      className="thread-menu-trigger"
                      onClick={() =>
                        setOpenMenuId(
                          openMenuId === chat.id
                            ? null
                            : chat.id,
                        )
                      }
                    >
                      <MoreHorizontal size={17} />
                    </button>

                    {openMenuId === chat.id && (
                      <div className="thread-menu">
                        <button
                          onClick={() =>
                            beginRename(chat)
                          }
                        >
                          <Pencil size={15} />
                          Rename
                        </button>

                        {chat.archived ? (
                          <button
                            onClick={() =>
                              void handleRestore(
                                chat,
                              )
                            }
                          >
                            <ArchiveRestore
                              size={15}
                            />
                            Restore
                          </button>
                        ) : (
                          <button
                            onClick={() =>
                              void handleArchive(
                                chat,
                              )
                            }
                          >
                            <Archive size={15} />
                            Archive
                          </button>
                        )}

                        <div className="menu-rule" />

                        <button
                          className="danger"
                          onClick={() => {
                            setConfirmDelete({
                              chat,
                            });

                            setOpenMenuId(null);
                          }}
                        >
                          <Trash2 size={15} />
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </section>
            ),
          )}
        </div>

        <button className="settings-button">
          <Settings size={17} />
          Settings
        </button>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="thread-heading">
            <span>THREAD</span>

            <strong>
              {activeChat?.title ??
                "New conversation"}
            </strong>
          </div>

          <div className="local-indicator">
            <i />
            LOCAL
          </div>
        </header>

        <div className="conversation">
          {!activeChatId &&
          messages.length === 0 ? (
            <section className="welcome">
              <div className="welcome-number">
                A/01
              </div>

              <h1>
                What are we
                <br />
                working on?
              </h1>

              <p>
                Ask Atlas directly, research the
                web, or work from your documents.
              </p>

              <div className="welcome-hints">
                <span>
                  <Globe2 size={15} />
                  Live research
                </span>

                <span>
                  <FileText size={15} />
                  Your knowledge
                </span>
              </div>
            </section>
          ) : loadingMessages ? (
            <div className="conversation-loading">
              Opening thread…
            </div>
          ) : (
            <section className="messages">
              {messages.map(
                (message, index) => (
                  <article
                    className={`message ${
                      message.role ===
                      "assistant"
                        ? "assistant-message"
                        : "user-message"
                    }`}
                    key={message.id}
                  >
                    <aside className="message-meta">
                      {message.role ===
                      "assistant"
                        ? "ATLAS"
                        : "YOU"}
                    </aside>

                    <div className="message-main">
                      {message.role === "assistant" && message.inference && (
                        <div className="inference-caption">
                          {message.inference.model} · {reasoningLabel(message.inference.capability,
                            message.inference.requested_reasoning)}
                        </div>
                      )}
                      {!!message.attachments?.length && (
                        <div className="message-files">
                          {message.attachments.map((document) => (
                            <span key={document.id}><FileText size={14} />{document.name}</span>
                          ))}
                        </div>
                      )}
                      {message.role ===
                        "assistant" &&
                      !message.content &&
                      generating &&
                      index ===
                        messages.length - 1 ? (
                        <Thinking
                          label={
                            activity ??
                            "Working"
                          }
                        />
                      ) : (
                        <div className="markdown">
                          <ReactMarkdown
                            remarkPlugins={[
                              remarkGfm,
                            ]}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      )}
                      {message.generation_status && ["error", "interrupted", "cancelled"].includes(message.generation_status) && (
                        <div className={`generation-result ${message.generation_status}`} role="status">
                          {message.generation_status === "cancelled" ? "Response stopped"
                            : message.generation_status === "interrupted" ? "Response interrupted by backend restart"
                              : activeJob?.id === message.generation_id ? activeJob?.error : "Response failed"}
                        </div>
                      )}

                      {message.sources.length >
                        0 && (
                        <Sources
                          sources={
                            message.sources
                          }
                        />
                      )}

                      {message.role ===
                        "assistant" &&
                        message.content && (
                          <div className="message-actions">
                            <button
                              onClick={() =>
                                void copyMessage(
                                  message,
                                )
                              }
                            >
                              {copiedId ===
                              message.id ? (
                                <Check
                                  size={15}
                                />
                              ) : (
                                <Copy
                                  size={15}
                                />
                              )}

                              {copiedId ===
                              message.id
                                ? "Copied"
                                : "Copy"}
                            </button>
                          </div>
                        )}
                      {message.role === "assistant" && message.content && generating &&
                        index === messages.length - 1 && activity && (
                          <div className="writing-activity" role="status">{activity}</div>
                        )}
                    </div>
                  </article>
                ),
              )}

              <div ref={bottomRef} />
            </section>
          )}
        </div>

        <div className="composer-region" ref={composerRef}>
          {notifications.length > 0 && (
            <div className="response-notifications" aria-live="polite">
              {notifications.map((job) => (
                <div className="response-notification" key={job.id}>
                  <button onClick={() => selectChat(job.chat_id)}>
                    <Check size={14} /> Response ready — {chats.find((chat) => chat.id === job.chat_id)?.title ?? job.chat_title}
                  </button>
                  <button aria-label="Dismiss notification"
                    onClick={() => void generations.store.acknowledge(job).catch(() => setError("Couldn't dismiss this notification."))}>
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
          {generations.connectionError && (
            <div className="connection-status" role="status">{generations.connectionError}</div>
          )}
          {error && (
            <div className="error-banner">
              <span>{error}</span>

              <button
                onClick={() =>
                  setError(null)
                }
              >
                <X size={15} />
              </button>
            </div>
          )}

          {activeChat?.archived ? (
            <div className="archived-banner">
              <span>
                This thread is archived.
              </span>

              <button
                onClick={() =>
                  void handleRestore(
                    activeChat,
                  )
                }
              >
                <ArchiveRestore size={15} />
                Restore
              </button>
            </div>
          ) : (
            <>
              <div className="composer">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  multiple
                  hidden
                  onChange={(event) => {
                    addFiles(event.target.files);
                    event.target.value = "";
                    textareaRef.current?.focus();
                  }}
                />
                {files.length > 0 && (
                  <div className="composer-files" aria-label="Attached PDFs">
                    {files.map((item) => (
                      <div className={`composer-file ${item.status === "error" ? "file-error" : ""}`} key={item.id}>
                        <FileText size={16} />
                        <div className="composer-file-copy">
                          <strong>{item.file.name}</strong>
                          <span role="status">
                            {item.status === "ready" ? `Indexed · ${item.document?.page_count} pages`
                              : item.status === "error" ? item.error
                                : item.status === "queued" ? "Waiting to upload"
                                  : item.progress === null ? "Indexing PDF"
                                    : item.progress > 0 ? `Uploading · ${item.progress}%` : "Uploading"}
                          </span>
                        </div>
                        {item.status === "error" && item.file.name.toLowerCase().endsWith(".pdf") &&
                          item.file.size <= 25 * 1024 * 1024 && (
                            <button className="file-retry" disabled={submitting} onClick={() => {
                              setFiles((current) => current.map((file) => file.id === item.id
                                ? { ...file, status: "queued", error: undefined } : file));
                              queueUpload(item);
                            }}>Retry</button>
                          )}
                        <button className="file-remove" disabled={submitting}
                          aria-label={`Remove ${item.file.name}`} onClick={() => removeFile(item.id)}>
                          <X size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) =>
                    setInput(
                      event.target.value,
                    )
                  }
                  onKeyDown={
                    handleComposerKeyDown
                  }
                  placeholder="Message Atlas"
                  rows={1}
                  aria-label="Message Atlas"
                />

                <div className="composer-footer">
                  <div className="composer-left">
                    <span className="tool-control">
                      <button aria-label="Add files" aria-describedby="attachment-tooltip"
                        disabled={submitting} onClick={() => fileInputRef.current?.click()}>
                        <FileText size={17} />
                      </button>
                      <span className="tool-tooltip" id="attachment-tooltip" role="tooltip">Add files</span>
                    </span>
                    <span className="tool-control">
                      <button aria-label="Search the web" aria-describedby="web-tooltip"
                        aria-pressed={forceWeb} className={forceWeb ? "tool-active" : ""}
                        disabled={submitting} onClick={() => setForceWeb((current) => !current)}>
                        <Globe2 size={17} />
                      </button>
                      <span className="tool-tooltip" id="web-tooltip" role="tooltip">Search the web</span>
                    </span>
                    {forceWeb && <span className="web-mode-label">WEB · NEXT MESSAGE</span>}
                  </div>

                  <ModelControls catalog={models.catalog} selection={modelSelection} onChange={setModelSelection}
                    loading={models.loading} error={models.error} refresh={models.refresh} disabled={submitting} />

                  <span className="tool-control send-control">
                    {generating ? (
                      <button
                        className="send-button stop-button"
                        onClick={
                          stopGeneration
                        }
                        aria-label="Stop generation"
                        disabled={activeJob?.status === "cancelling"}
                        aria-describedby="send-tooltip"
                      >
                        <Square
                          size={13}
                          fill="currentColor"
                        />
                      </button>
                    ) : (
                      <button
                        className="send-button"
                        onClick={() =>
                          void send()
                        }
                        disabled={
                          !input.trim() || submitting || loadingMessages || !generations.connected || !selectedModel.ready || files.some((file) => file.status !== "ready")
                        }
                        aria-label="Send message"
                        aria-describedby="send-tooltip"
                      >
                        <ArrowUp
                          size={18}
                        />
                      </button>
                    )}
                    <span className="tool-tooltip" id="send-tooltip" role="tooltip">
                      {generating ? "Stop generation" : "Send message"}
                    </span>
                  </span>
                </div>
              </div>

              <div className="composer-caption">
                <span>
                  ENTER TO SEND · SHIFT +
                  ENTER FOR NEW LINE
                </span>

                <span>
                  ATLAS · LOCAL
                </span>
              </div>
            </>
          )}
        </div>
      </main>

      {confirmDelete && (
        <div
          className="modal-backdrop"
          onMouseDown={() =>
            setConfirmDelete(null)
          }
        >
          <div
            className="modal"
            onMouseDown={(event) =>
              event.stopPropagation()
            }
          >
            <div className="modal-index">
              DELETE THREAD
            </div>

            <h2>
              Remove this conversation?
            </h2>

            <p>
              “{confirmDelete.chat.title}”
              will be permanently removed from
              Atlas. This cannot be undone.
            </p>

            <div className="modal-actions">
              <button
                className="secondary-button"
                onClick={() =>
                  setConfirmDelete(null)
                }
              >
                Cancel
              </button>

              <button
                className="danger-button"
                onClick={() =>
                  void handleDelete()
                }
              >
                <Trash2 size={15} />
                Delete thread
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


function JobIndicator({ job }: { job: GenerationJob }) {
  const working = isActiveGeneration(job);
  const failed = job.status === "error" || job.status === "interrupted";
  if (!working && (job.acknowledged || job.status === "cancelled")) return null;
  return (
    <span className={`job-indicator ${working ? "working" : failed ? "failed" : "ready"}`}
      aria-label={working ? job.activity ?? "Working" : failed ? "Response failed" : "Response ready"}>
      {working ? <i /> : failed ? "!" : <Check size={12} />}
    </span>
  );
}


function Thinking({
  label,
}: {
  label: string;
}) {
  return (
    <div className="thinking">
      <div className="thinking-mark">
        <span />
        <span />
        <span />
      </div>

      <div className="thinking-copy">
        <strong>{label}</strong>
        <span>Atlas is working</span>
      </div>
    </div>
  );
}


function Sources({
  sources,
}: {
  sources: Source[];
}) {
  return (
    <div className="sources">
      <div className="sources-heading">
        SOURCES · {sources.length}
      </div>

      <div className="source-grid">
        {sources.map((source) => {
          const content = (
            <>
              <div className="source-top">
                <span>
                  {source.index
                    .toString()
                    .padStart(2, "0")}
                </span>

                <small>
                  {sourceLabel(source)}
                </small>
              </div>

              <strong>
                {source.title ??
                  "Untitled source"}
              </strong>
            </>
          );

          if (source.url) {
            return (
              <a
                className="source-card"
                href={source.url}
                target="_blank"
                rel="noreferrer"
                key={`${source.index}-${source.url}`}
              >
                {content}
              </a>
            );
          }

          return (
            <div
              className="source-card"
              key={`${source.index}-${source.title}`}
            >
              {content}
            </div>
          );
        })}
      </div>
    </div>
  );
}


export default App;
