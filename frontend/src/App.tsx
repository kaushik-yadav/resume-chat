import { useCallback, useEffect, useState } from "react";
import { Download, Plus, Redo2, Undo2 } from "lucide-react";
import { api, streamChat } from "./api";
import ChatPanel from "./components/ChatPanel";
import ResumePanel, { type Tab } from "./components/ResumePanel";
import UploadPane from "./components/UploadPane";
import type {
  ChatMessage,
  Flow,
  ConversationListItem,
  ConversationState,
  LiveTurn,
  StreamEvent,
  VersionSummary,
} from "./types";

const STORAGE_KEY = "resumeChat.conversationId";

function remember(id: string | null) {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable (private mode) - the app still works */
  }
}

function recall(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

const emptyTurn = (): LiveTurn => ({
  status: "Understanding your request",
  plan: null,
  activeAgent: null,
  progress: [],
  results: [],
  reply: "",
});

export default function App() {
  const [conversationId, setConversationId] = useState<string | null>(recall);
  const [state, setState] = useState<ConversationState | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [recent, setRecent] = useState<ConversationListItem[]>([]);
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [flow, setFlow] = useState<Flow | null>(null);
  const [tab, setTab] = useState<Tab>("resume");

  const refreshVersions = useCallback(async (id: string) => {
    setVersions(await api.listVersions(id));
  }, []);

  const loadRecent = useCallback(() => {
    api.listConversations().then(setRecent).catch(() => setRecent([]));
  }, []);

  // Load the remembered conversation (if any) on start-up.
  useEffect(() => {
    loadRecent();
    if (!conversationId) return;
    api
      .getConversation(conversationId)
      .then((s) => {
        setState(s);
        setMessages(s.messages ?? []);
        return refreshVersions(conversationId);
      })
      .catch(() => {
        remember(null);
        setConversationId(null);
      });
  }, [conversationId, refreshVersions, loadRecent]);

  async function upload(file: File) {
    const { id } = await api.createConversation();
    const result = await api.uploadResume(id, file);
    remember(id);
    setConversationId(id);
    setState(result);
    setMessages([result.message]);
    setError(null);
    await refreshVersions(id);
    loadRecent();
  }

  function newConversation() {
    remember(null);
    setConversationId(null);
    setState(null);
    setMessages([]);
    setVersions([]);
    setError(null);
    setFlow(null);
    setTab("resume");
  }

  function applyState(s: ConversationState) {
    setState((prev) => ({ ...prev, ...s }));
  }

  function handleEvent(event: StreamEvent) {
    switch (event.type) {
      case "user_message":
        setMessages((m) => [...m, event.message]);
        break;
      case "status":
        setLive((t) => t && { ...t, status: event.message });
        break;
      case "plan":
        setLive((t) => t && { ...t, plan: { intent: event.intent, reasoning: event.reasoning, steps: event.steps } });
        break;
      case "agent_start":
        setLive((t) => t && { ...t, activeAgent: event.agent, status: `${event.title} is working` });
        break;
      case "progress":
        setLive((t) => t && { ...t, progress: [...t.progress, { agent: event.agent, message: event.message }] });
        break;
      case "agent_result": {
        const { type: _type, ...result } = event;
        setLive((t) => t && { ...t, results: [...t.results, result] });
        break;
      }
      case "token":
        setLive((t) => t && { ...t, reply: t.reply + event.text });
        break;
      case "version":
        setState((s) => s && { ...s, current_version: event.version, can_undo: event.can_undo, can_redo: event.can_redo });
        break;
      case "done":
        setMessages((m) => [...m, event.message]);
        applyState(event);
        break;
      case "flow": {
        const { type: _flowType, ...snapshot } = event;
        setFlow(snapshot);
        break;
      }
      case "error":
        setError(event.message);
        break;
    }
  }

  async function send(message: string, jobDescription: string | null) {
    if (!conversationId) return;
    setError(null);
    setLive(emptyTurn());
    setFlow(null);
    setTab("flow"); // watch the agents execute
    try {
      await streamChat(conversationId, message, jobDescription, handleEvent);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLive(null);
      refreshVersions(conversationId).catch(() => undefined);
      loadRecent();
    }
  }

  async function versionAction(action: () => Promise<ConversationState>) {
    if (!conversationId) return;
    setBusy(true);
    setError(null);
    try {
      applyState(await action());
      await refreshVersions(conversationId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const version = state?.current_version ?? null;
  const working = busy || !!live;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" /> Resume Chat
        </div>
        {state && <span className="title">· {state.conversation.title}</span>}
        <span className="spacer" />
        {recent.length > 0 && (
          <select
            className="btn small"
            aria-label="Open a previous conversation"
            value={conversationId ?? ""}
            onChange={(e) => {
              const id = e.target.value;
              if (!id) return;
              remember(id);
              setConversationId(id);
            }}
          >
            <option value="">Recent…</option>
            {recent.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title}
              </option>
            ))}
          </select>
        )}
        {version && conversationId && (
          <>
            <button
              className="btn small"
              disabled={working || !state?.can_undo}
              onClick={() => versionAction(() => api.undo(conversationId))}
              title="Undo last change"
            >
              <Undo2 size={15} /> Undo
            </button>
            <button
              className="btn small"
              disabled={working || !state?.can_redo}
              onClick={() => versionAction(() => api.redo(conversationId))}
              title="Redo"
            >
              <Redo2 size={15} /> Redo
            </button>
            <a className="btn small" href={api.exportUrl(conversationId, "docx")} title="Download as Word">
              <Download size={15} /> DOCX
            </a>
            <a className="btn small" href={api.exportUrl(conversationId, "md")} title="Download as Markdown">
              MD
            </a>
          </>
        )}
        <button className="btn small" onClick={newConversation} disabled={!!live}>
          <Plus size={15} /> New
        </button>
      </header>

      {!version || !conversationId ? (
        <UploadPane onUpload={upload} />
      ) : (
        <main className="workspace">
          <ChatPanel
            messages={messages}
            live={live}
            versions={versions}
            error={error}
            hasJobDescription={!!state?.conversation.context.job_description}
            onSend={send}
            onViewFlow={(f) => {
              setFlow(f);
              setTab("flow");
            }}
          />
          <ResumePanel
            conversationId={conversationId}
            version={version}
            versions={versions}
            busy={working}
            tab={tab}
            onTab={setTab}
            flow={flow}
            flowLive={!!live}
            onRestore={(vid) => versionAction(() => api.restore(conversationId, vid))}
          />
        </main>
      )}
    </div>
  );
}
