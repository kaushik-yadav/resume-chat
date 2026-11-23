import { useEffect, useState } from "react";
import { History, RotateCcw, Workflow } from "lucide-react";
import { api } from "../api";
import type { Flow, ResumeVersion, SectionChange, VersionSummary } from "../types";
import DiffView from "./DiffView";
import FlowView from "./FlowView";
import ResumeSectionBody from "./ResumeSectionBody";

export type Tab = "resume" | "changes" | "history" | "flow";

interface Props {
  conversationId: string;
  version: ResumeVersion;
  versions: VersionSummary[];
  onRestore: (versionId: string) => void;
  busy: boolean;
}

function ResumeView({ version }: { version: ResumeVersion }) {
  const changed = new Set(version.changes.map((c) => c.section_id));
  const { resume } = version;
  return (
    <article className="paper">
      {(resume.language !== "en" || resume.locale) && (
        <div className="lang-badge">
          {resume.locale ?? resume.language.toUpperCase()}
        </div>
      )}
      {resume.sections.map((s) =>
        s.kind === "header" ? (
          <header key={s.id} className={`r-header section ${changed.has(s.id) ? "changed" : ""}`}>
            {s.content.split("\n").filter(Boolean).map((line, i) => <p key={i}>{line}</p>)}
          </header>
        ) : (
          <div key={s.id} className={`section ${changed.has(s.id) ? "changed" : ""}`}>
            <h3>
              {s.title}
              {changed.has(s.id) && <span className="changed-tag">● updated</span>}
            </h3>
            <ResumeSectionBody content={s.content} />
          </div>
        ),
      )}
    </article>
  );
}

function Changes({ conversationId, version, versions }: Omit<Props, "onRestore" | "busy">) {
  const [from, setFrom] = useState(version.parent_id ?? "");
  const [to, setTo] = useState(version.id);
  const [changes, setChanges] = useState<SectionChange[] | null>(null);

  useEffect(() => {
    setFrom(version.parent_id ?? "");
    setTo(version.id);
  }, [version.id, version.parent_id]);

  useEffect(() => {
    if (!from || !to) return setChanges(null);
    if (from === version.parent_id && to === version.id) return setChanges(version.changes);
    api.compare(conversationId, from, to).then(setChanges).catch(() => setChanges([]));
  }, [conversationId, from, to, version]);

  if (versions.length < 2) return <div className="empty">Changes appear here after the first edit.</div>;
  const options = versions.map((v) => (
    <option key={v.id} value={v.id}>
      v{v.number} · {v.label}
    </option>
  ));
  return (
    <div>
      <div className="compare-bar">
        Compare
        <select className="btn small" value={from} onChange={(e) => setFrom(e.target.value)}>
          <option value="">—</option>
          {options}
        </select>
        →
        <select className="btn small" value={to} onChange={(e) => setTo(e.target.value)}>
          {options}
        </select>
      </div>
      {changes ? <DiffView changes={changes} /> : <div className="empty">Pick two versions.</div>}
    </div>
  );
}

function HistoryList({ version, versions, onRestore, busy }: Omit<Props, "conversationId">) {
  return (
    <ul className="history">
      {[...versions].reverse().map((v) => (
        <li key={v.id} className={v.id === version.id ? "current" : ""}>
          <span className="num">v{v.number}</span>
          <div>
            <div className="label">{v.label}</div>
            <div className="sub">
              {new Date(v.created_at).toLocaleString()} ·{" "}
              {v.changed_sections.length ? `${v.changed_sections.length} section(s) changed` : v.source}
            </div>
          </div>
          <div className="actions">
            {v.id === version.id ? (
              <span className="version-chip">current</span>
            ) : (
              <button className="btn small" disabled={busy} onClick={() => onRestore(v.id)}>
                <RotateCcw size={13} /> Restore
              </button>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}

interface PanelProps extends Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
  flow: Flow | null;
  flowLive: boolean;
}

export default function ResumePanel(props: PanelProps) {
  const { tab, onTab: setTab } = props;
  const changeCount = props.version.changes.length;
  return (
    <section className="resume-pane" aria-label="Resume">
      <nav className="tabs">
        <button className={tab === "resume" ? "on" : ""} onClick={() => setTab("resume")}>
          Resume
        </button>
        <button className={tab === "changes" ? "on" : ""} onClick={() => setTab("changes")}>
          Changes{changeCount > 0 && <span className="count">{changeCount}</span>}
        </button>
        <button className={tab === "history" ? "on" : ""} onClick={() => setTab("history")}>
          <History size={13} style={{ verticalAlign: -2 }} /> History
          <span className="count">{props.versions.length}</span>
        </button>
        <button className={tab === "flow" ? "on" : ""} onClick={() => setTab("flow")}>
          <Workflow size={13} style={{ verticalAlign: -2 }} /> Agent flow
          {props.flowLive && <span className="live-dot" aria-label="running" />}
        </button>
      </nav>
      <div className="pane-body">
        {tab === "resume" && <ResumeView version={props.version} />}
        {tab === "changes" && (
          <Changes conversationId={props.conversationId} version={props.version} versions={props.versions} />
        )}
        {tab === "history" && (
          <HistoryList version={props.version} versions={props.versions} onRestore={props.onRestore} busy={props.busy} />
        )}
        {tab === "flow" && <FlowView flow={props.flow} live={props.flowLive} />}
      </div>
    </section>
  );
}
