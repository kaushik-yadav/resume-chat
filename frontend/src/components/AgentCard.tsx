import type { ReactNode } from "react";
import { AlertTriangle, ArrowRight, Building2, FileSearch, Languages, Sparkles } from "lucide-react";
import type { AgentResult } from "../types";

export const AGENT_ICONS: Record<string, typeof Sparkles> = {
  company_research: Building2,
  jd_matching: FileSearch,
  translation: Languages,
  section_enhancement: Sparkles,
};

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="ac-block">
      <h5>{title}</h5>
      {children}
    </section>
  );
}

/** One specialist agent's result: why, what changed, evidence, and what needs review. */
export default function AgentCard({ result }: { result: AgentResult }) {
  const Icon = AGENT_ICONS[result.agent] ?? Sparkles;
  const ins = result.insights ?? {};
  const requirements: any[] = Array.isArray(ins.requirements) ? ins.requirements : [];
  const lineEdits: { before: string; after: string }[] = Array.isArray(ins.edited_lines) ? ins.edited_lines : [];

  return (
    <article className="agent-card">
      <header className="ac-head">
        <span className="ac-icon">
          <Icon size={16} />
        </span>
        <h4>{result.title}</h4>
      </header>
      <p className="ac-reason">{result.reasoning}</p>

      {typeof ins.match_score_after === "number" && (
        <Block title="Match score">
          <div className="score">
            <div className="bar" aria-hidden>
              <span style={{ width: `${ins.match_score_after}%` }} />
            </div>
            <b>{ins.match_score_after}</b>
            <span className="muted">
              {ins.match_score_after !== ins.match_score_before ? `was ${ins.match_score_before}` : "of 100"}
            </span>
          </div>
          {requirements.length > 0 && (
            <ul className="chip-row">
              {requirements.map((r) => (
                <li key={r.name} className={`chip is-${r.status}`} title={r.evidence || "No evidence in your resume"}>
                  {r.name}
                </li>
              ))}
            </ul>
          )}
        </Block>
      )}

      {lineEdits.length > 0 && (
        <Block title={lineEdits.length === 1 ? "Edited line" : `Edited lines (${lineEdits.length})`}>
          <ul className="ac-line-edits">
            {lineEdits.map((e, i) => (
              <li key={i}>
                <span className="before">{e.before.replace(/^- /, "")}</span>
                <ArrowRight size={13} aria-hidden />
                <span className="after">{e.after.replace(/^- /, "")}</span>
              </li>
            ))}
          </ul>
        </Block>
      )}

      {lineEdits.length === 0 && result.edits.length > 0 && (
        <Block title="Changes">
          <ul className="ac-list">
            {result.edits.map((e) => (
              <li key={e.section_id}>
                <strong>{e.title}</strong>
                <span>{e.rationale}</span>
              </li>
            ))}
          </ul>
        </Block>
      )}

      {ins.company_profile && (
        <Block title={`What ${ins.company_profile.company ?? "they"} value`}>
          <ul className="chip-row">
            {(ins.company_profile.values ?? []).map((v: string) => (
              <li key={v} className="chip">
                {v}
              </li>
            ))}
          </ul>
        </Block>
      )}

      {Array.isArray(ins.localization_notes) && ins.localization_notes.length > 0 && (
        <Block title="Localization notes">
          <ul className="ac-notes">
            {ins.localization_notes.map((n: string) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Block>
      )}

      {Array.isArray(ins.sources) && ins.sources.length > 0 && (
        <Block title="Sources">
          <ul className="ac-sources">
            {ins.sources.map((s: any) => (
              <li key={s.url}>
                <a href={s.url} target="_blank" rel="noreferrer">
                  {s.title || new URL(s.url).hostname}
                </a>
              </li>
            ))}
          </ul>
        </Block>
      )}

      {result.warnings.length > 0 && (
        <Block title="Needs your review">
          {result.warnings.map((w) => (
            <p key={w} className="warning">
              <AlertTriangle size={14} aria-hidden /> <span>{w}</span>
            </p>
          ))}
        </Block>
      )}
    </article>
  );
}
