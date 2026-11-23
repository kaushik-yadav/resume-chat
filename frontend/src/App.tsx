import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Loader2, Plus, Redo2, Undo2 } from "lucide-react";
import { ApiError, api, streamChat } from "./api";
import ChatPanel from "./components/ChatPanel";
import ResumePanel, { type Tab } from "./components/ResumePanel";
import PlanPicker from "./components/PlanPicker";
import UploadPane from "./components/UploadPane";
import type {
  ChatMessage,
  Flow,
  Plan,
  Preset,
  ConversationListItem,
  ConversationState,
  LiveTurn,
  StreamEvent,
  VersionSummary,
} from "./types";

// Session handling:
// - sessionStorage holds the conversation open in *this tab*: a refresh restores it.
// - localStorage holds the last conversation across visits: offered as "Continue", never
//   opened automatically (a new visit starts at the upload screen).
const SESSION_KEY = "resumeChat.session";
const LAST_KEY = "resumeChat.last";

function store(kind: "session" | "local"): Storage | null {
  try {
    return kind === "session" ? window.sessionStorage : window.localStorage;
  } catch {
    return null; // storage unavailable (private mode) - the app still works
  }
}

function remember(id: string | null) {
  try {
    if (id) {
      store("session")?.setItem(SESSION_KEY, id);
      store("local")?.setItem(LAST_KEY, id);
    } else {
      store("session")?.removeItem(SESSION_KEY);
    }
  } catch {
    /* ignore */
  }
}

function recall(kind: "session" | "local"): string | null {
  try {
    return store(kind)?.getItem(kind === "session" ? SESSION_KEY : LAST_KEY) ?? null;
  } catch {
    return null;
  }
}

/** Append a message unless it is already shown (stream retries replay events). */
function addOnce(list: ChatMessage[], message: ChatMessage): ChatMessage[] {
  return list.some((m) => m.id === message.id) ? list : [...list, message];
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
  const [conversationId, setConversationId] = useState<string | null>(() => recall("session"));
  // True while a conversation is being fetched: shows a restoring state, never the upload screen.
  const [loading, setLoading] = useState<boolean>(() => !!recall("session"));
  const loadedId = useRef<string | null>(null);
  const [state, setState] = useState<ConversationState | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [recent, setRecent] = useState<ConversationListItem[]>([]);
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [flow, setFlow] = useState<Flow | null>(null);
  const [tab, setTab] = useState<Tab>("resume");
  const [plans, setPlans] = useState<Plan[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);

  const refreshVersions = useCallback(async (id: string) => {
    setVersions(await api.listVersions(id));
  }, []);

  const loadRecent = useCallback(() => {
    api.listConversations().then(setRecent).catch(() => setRecent([]));
  }, []);

  // Product catalogue: plans (models per tier) and one-click suggested actions.
  useEffect(() => {
    api.plans().then((r) => setPlans(r.plans)).catch(() => setPlans([]));
    api.presets().then(setPresets).catch(() => setPresets([]));
  }, []);

  // Open the selected conversation: on refresh (this tab's session) or from "Recent".
  useEffect(() => {
    loadRecent();
    if (!conversationId || loadedId.current === conversationId) {
      setLoading(false); // nothing to open, or already in memory (e.g. right after an upload)
      return;
    }
    let cancelled = false; // a later selection wins over a slower, earlier response
    setLoading(true);
    // Conversation state and version list are independent: fetch them in parallel.
    Promise.all([api.getConversation(conversationId), api.listVersions(conversationId)])
      .then(([s, list]) => {
        if (cancelled) return;
        loadedId.current = conversationId;
        setState(s);
        setMessages(s.messages ?? []);
        setVersions(list);
      })
      .catch(() => {
        if (cancelled) return;
        remember(null);
        loadedId.current = null;
        setConversationId(null);
        setState(null);
        setError("That conversation could not be restored. Upload a resume to start again.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId, refreshVersions, loadRecent]);

  function openConversation(id: string) {
    if (id === conversationId) return;
    remember(id);
    loadedId.current = null;
    setState(null);
    setMessages([]);
    setVersions([]);
    setFlow(null);
    setError(null);
    setTab("resume");
    setConversationId(id);
  }

  async function upload(file: File) {
    const { id } = await api.createConversation();
    const result = await api.uploadResume(id, file);
    remember(id);
    loadedId.current = id; // already have its state: no second fetch
    setConversationId(id);
    setState(result);
    setMessages([result.message]);
    setError(null);
    await refreshVersions(id);
    loadRecent();
  }

  function newConversation() {
    remember(null);
    loadedId.current = null;
    setLoading(false);
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
        setMessages((m) => addOnce(m, event.message));
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
        setMessages((m) => addOnce(m, event.message));
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

  async function send(message: string, jobDescription: string | null, preset: string | null) {
    if (!conversationId) return;
    setError(null);
    setLive(emptyTurn());
    setFlow(null);
    setTab("flow"); // watch the agents execute
    // One idempotency key per message: if the connection drops, the retry reuses it and the
    // server replays the stored answer instead of running the agents a second time.
    const clientMessageId = crypto.randomUUID();
    const options = { jobDescription, clientMessageId, preset };
    try {
      try {
        await streamChat(conversationId, message, options, handleEvent);
      } catch (e) {
        if (e instanceof ApiError) throw e; // the server answered: retrying won't help
        await new Promise((r) => setTimeout(r, 800));
        await streamChat(conversationId, message, options, handleEvent);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLive(null);
      refreshVersions(conversationId).catch(() => undefined);
      loadRecent();
    }
  }

  /** refreshList=false for undo/redo: they move a pointer and never change the version list. */
  async function versionAction(action: () => Promise<ConversationState>, refreshList = true) {
    if (!conversationId) return;
    setBusy(true);
    setError(null);
    try {
      applyState(await action());
      if (refreshList) await refreshVersions(conversationId);
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
              if (id) openConversation(id);
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
              onClick={() => versionAction(() => api.undo(conversationId), false)}
              title="Undo last change"
            >
              <Undo2 size={15} /> Undo
            </button>
            <button
              className="btn small"
              disabled={working || !state?.can_redo}
              onClick={() => versionAction(() => api.redo(conversationId), false)}
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
        {state && conversationId && (
          <PlanPicker
            plans={plans}
            current={state.plan ?? "pro"}
            disabled={working}
            onChange={(plan) => versionAction(() => api.setPlan(conversationId, plan))}
          />
        )}
        <button className="btn small" onClick={newConversation} disabled={!!live}>
          <Plus size={15} /> New
        </button>
      </header>

      {loading ? (
        <div className="restoring" role="status">
          <Loader2 size={22} className="spin" />
          <p>Restoring your session…</p>
          <button className="btn small" onClick={newConversation}>
            Start a new one instead
          </button>
        </div>
      ) : !version || !conversationId ? (
        <UploadPane
          onUpload={upload}
          notice={error}
          last={!conversationId ? recent.find((c) => c.id === recall("local")) ?? null : null}
          onContinue={openConversation}
        />
      ) : (
        <main className="workspace">
          <ChatPanel
            messages={messages}
            live={live}
            versions={versions}
            error={error}
            hasJobDescription={!!state?.conversation.context.job_description}
            presets={presets}
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
