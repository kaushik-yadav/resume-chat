import { useEffect, useState } from "react";
import { Download, FileText, GitCompare, History, LayoutTemplate, RotateCcw, Workflow } from "lucide-react";
import { api } from "../api";
import ResumeDocument from "../resume/ResumeDocument";
import type { Flow, ResumeVersion, SectionChange, VersionSummary } from "../types";
import Carousel from "./Carousel";
import DiffView from "./DiffView";
import FlowView from "./FlowView";

export type Tab = "resume" | "changes" | "history" | "flow";

interface Props {
  conversationId: string;
  version: ResumeVersion;
  versions: VersionSummary[];
  onRestore: (versionId: string) => void;
  busy: boolean;
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

  if (versions.length < 2) return <div className="empty">Changes show up here after your first edit.</div>;
  const options = versions.map((v) => (
    <option key={v.id} value={v.id}>
      Version {v.number}: {v.label}
    </option>
  ));
  return (
    <div className="changes">
      <div className="compare-bar">
        <label>
          <span>From</span>
          <select value={from} onChange={(e) => setFrom(e.target.value)}>
            <option value="">Choose a version</option>
            {options}
          </select>
        </label>
        <label>
          <span>To</span>
          <select value={to} onChange={(e) => setTo(e.target.value)}>
            {options}
          </select>
        </label>
      </div>
      {changes ? <DiffView changes={changes} /> : <div className="empty">Pick two versions to compare.</div>}
    </div>
  );
}

// Full versions for the previews, fetched once each (the server serves them from memory).
const versionCache = new Map<string, Promise<ResumeVersion>>();
function loadVersion(conversationId: string, id: string): Promise<ResumeVersion> {
  const key = `${conversationId}/${id}`;
  if (!versionCache.has(key)) versionCache.set(key, api.getVersion(conversationId, id));
  return versionCache.get(key)!;
}

function VersionPreview({ conversationId, summary }: { conversationId: string; summary: VersionSummary }) {
  const [full, setFull] = useState<ResumeVersion | null>(null);
  useEffect(() => {
    let alive = true;
    loadVersion(conversationId, summary.id).then((v) => alive && setFull(v)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [conversationId, summary.id]);
  return (
    <div className="vp-frame" aria-hidden>
      {full ? (
        <div className="vp-scale">
          <ResumeDocument version={full} compact />
        </div>
      ) : (
        <div className="vp-skeleton" />
      )}
    </div>
  );
}

/** Whether this version's PDF keeps the uploaded file's layout (and why not, when it can't). */
function LayoutBadge({ conversationId, versionId }: { conversationId: string; versionId: string }) {
  const [info, setInfo] = useState<{ layout: string; note: string } | null>(null);
  useEffect(() => {
    let alive = true;
    api
      .exportInfo(conversationId, versionId)
      .then((r) => alive && setInfo(r.pdf))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [conversationId, versionId]);
  if (!info) return null;
  const original = info.layout === "original";
  return (
    <span
      className={`vc-layout${original ? " is-original" : ""}`}
      title={original ? "The PDF keeps your uploaded file's fonts and placement" : `Clean layout: ${info.note}`}
    >
      <LayoutTemplate size={12} aria-hidden /> {original ? "Original layout" : "Clean layout"}
    </span>
  );
}

function HistoryCarousel({ conversationId, version, versions, onRestore, busy }: Props) {
  const ordered = [...versions].reverse();
  return (
    <div className="history">
      <p className="history-intro">
        Every change is saved as a version. Restoring one adds it as a new version, so nothing is lost.
      </p>
      <Carousel label="Resume versions" itemWidth="248px">
        {ordered.map((v) => {
          const current = v.id === version.id;
          return (
            <article key={v.id} className={`version-card${current ? " is-current" : ""}`}>
              <VersionPreview conversationId={conversationId} summary={v} />
              <div className="vc-body">
                <div className="vc-head">
                  <h4>Version {v.number}</h4>
                  {current && <span className="vc-current">Current</span>}
                </div>
                <p className="vc-label">{v.label}</p>
                <p className="vc-time">{new Date(v.created_at).toLocaleString()}</p>
                <LayoutBadge conversationId={conversationId} versionId={v.id} />
                {v.changed_sections.length > 0 && (
                  <ul className="chip-row">
                    {v.changed_sections.slice(0, 3).map((s) => (
                      <li key={s} className="chip">
                        {s}
                      </li>
                    ))}
                  </ul>
                )}
                <div className="vc-actions">
                  <div className="btn-group" role="group" aria-label={`Download version ${v.number}`}>
                    <a className="btn small" href={api.exportUrl(conversationId, "pdf", v.id)} title="Download this version as PDF">
                      <Download size={13} /> PDF
                    </a>
                    <a className="btn small" href={api.exportUrl(conversationId, "docx", v.id)} title="Download this version as Word">
                      DOCX
                    </a>
                    <a className="btn small" href={api.exportUrl(conversationId, "md", v.id)} title="Download this version as Markdown">
                      MD
                    </a>
                  </div>
                  <a className="vc-clean" href={api.exportUrl(conversationId, "pdf", v.id, "clean")} title="Single-column PDF in our clean, ATS-friendly layout">
                    Clean PDF
                  </a>
                  {!current && (
                    <button className="btn small" disabled={busy} onClick={() => onRestore(v.id)}>
                      <RotateCcw size={13} /> Restore
                    </button>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </Carousel>
    </div>
  );
}

interface PanelProps extends Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
  flow: Flow | null;
  flowLive: boolean;
  replyChars: number;
  onSelectText: (lines: string[]) => void;
}

const TABS: { id: Tab; label: string; icon: typeof FileText }[] = [
  { id: "resume", label: "Resume", icon: FileText },
  { id: "changes", label: "Changes", icon: GitCompare },
  { id: "history", label: "History", icon: History },
  { id: "flow", label: "Agent flow", icon: Workflow },
];

export default function ResumePanel(props: PanelProps) {
  const { tab, onTab } = props;
  const counts: Partial<Record<Tab, number>> = {
    changes: props.version.changes.length || undefined,
    history: props.versions.length || undefined,
  };
  return (
    <section className="workspace-pane" aria-label="Resume">
      <nav className="segmented" role="tablist">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "is-on" : undefined} onClick={() => onTab(id)}>
            <Icon size={14} aria-hidden />
            {label}
            {counts[id] !== undefined && <span className="tab-count">{counts[id]}</span>}
            {id === "flow" && props.flowLive && <span className="live-dot" aria-label="running" />}
          </button>
        ))}
      </nav>
      <div className="pane-scroll">
        {tab === "resume" && <ResumeDocument version={props.version} onSelectText={props.onSelectText} />}
        {tab === "changes" && (
          <Changes conversationId={props.conversationId} version={props.version} versions={props.versions} />
        )}
        {tab === "history" && <HistoryCarousel {...props} />}
        {tab === "flow" && <FlowView flow={props.flow} live={props.flowLive} replyChars={props.replyChars} />}
      </div>
    </section>
  );
}
