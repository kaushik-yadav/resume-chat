import { AlertTriangle, Building2, FileSearch, Languages, Sparkles } from "lucide-react";
import type { AgentResult } from "../types";

export const AGENT_ICONS: Record<string, typeof Sparkles> = {
  company_research: Building2,
  jd_matching: FileSearch,
  translation: Languages,
  section_enhancement: Sparkles,
};

function Score({ before, after }: { before: number; after: number }) {
  return (
    <div className="score">
      <span>Match</span>
      <div className="bar" aria-hidden>
        <span style={{ width: `${after}%` }} />
      </div>
      <b>{after}</b>
      <span className="sub">{after !== before ? `was ${before}` : "/ 100"}</span>
    </div>
  );
}

/** One specialist agent's reasoning, edits and insights. */
export default function AgentCard({ result }: { result: AgentResult }) {
  const Icon = AGENT_ICONS[result.agent] ?? Sparkles;
  const ins = result.insights ?? {};
  return (
    <div className="agent-card">
      <h4>
        <Icon size={15} /> {result.title}
      </h4>
      <p className="reason">{result.reasoning}</p>

      {typeof ins.match_score_after === "number" && (
        <Score before={ins.match_score_before} after={ins.match_score_after} />
      )}
      {Array.isArray(ins.requirements) && ins.requirements.length > 0 && (
        <div className="kv" title="Requirement coverage">
          {ins.requirements.map((r: any) => (
            <span key={r.name} className={`chip ${r.status}`} title={r.evidence || "No evidence found"}>
              {r.name}
            </span>
          ))}
        </div>
      )}
      {Array.isArray(ins.keywords_missing) && ins.keywords_missing.length > 0 && (
        <div className="sub">Missing keywords: {ins.keywords_missing.join(", ")}</div>
      )}

      {ins.company_profile && (
        <>
          <div className="kv">
            {(ins.company_profile.values ?? []).map((v: string) => (
              <span key={v} className="chip">{v}</span>
            ))}
          </div>
          <div className="sub">Emphasis: {(ins.company_profile.emphasis ?? []).join(" · ")}</div>
        </>
      )}
      {Array.isArray(ins.localization_notes) && ins.localization_notes.length > 0 && (
        <ul>
          {ins.localization_notes.map((n: string) => <li key={n}>{n}</li>)}
        </ul>
      )}
      {Array.isArray(ins.issues_found) && ins.issues_found.length > 0 && (
        <div className="sub">Fixed: {ins.issues_found.join("; ")}</div>
      )}

      {result.edits.length > 0 && (
        <ul>
          {result.edits.map((e) => (
            <li key={e.section_id}>
              <b>{e.title}</b>: {e.rationale}
            </li>
          ))}
        </ul>
      )}
      {Array.isArray(ins.sources) && ins.sources.length > 0 && (
        <div className="sub">
          Sources:{" "}
          {ins.sources.map((s: any, i: number) => (
            <span key={s.url}>
              {i > 0 && " · "}
              <a href={s.url} target="_blank" rel="noreferrer">{s.title || new URL(s.url).hostname}</a>
            </span>
          ))}
        </div>
      )}
      {result.warnings.map((w) => (
        <div key={w} className="warning">
          <AlertTriangle size={14} style={{ flex: "none", marginTop: 2 }} /> {w}
        </div>
      ))}
    </div>
  );
}
