import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  Circle,
  CircleSlash,
  GitBranch,
  GitCommitHorizontal,
  Loader2,
  MessageSquare,
  MessageSquareText,
  Workflow,
  XCircle,
} from "lucide-react";
import type { Flow, FlowNode, FlowStatus } from "../types";
import { AGENT_ICONS } from "./AgentCard";

const KIND_ICONS = {
  trigger: MessageSquare,
  router: GitBranch,
  respond: MessageSquareText,
  version: GitCommitHorizontal,
};

function fmt(ms: number | null | undefined): string {
  if (ms == null) return "";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function StatusIcon({ status, size = 15 }: { status: FlowStatus; size?: number }) {
  switch (status) {
    case "running":
      return <Loader2 size={size} className="spin st-running" />;
    case "done":
      return <CheckCircle2 size={size} className="st-done" />;
    case "error":
      return <XCircle size={size} className="st-error" />;
    case "skipped":
      return <CircleSlash size={size} className="st-skipped" />;
    default:
      return <Circle size={size} className="st-pending" />;
  }
}

function nodeIcon(node: FlowNode) {
  if (node.kind === "agent") return AGENT_ICONS[node.detail.agent ?? node.id.slice(6)] ?? Workflow;
  return KIND_ICONS[node.kind] ?? Workflow;
}

/** Elapsed time for running nodes, ticking while the turn is live. */
function useElapsed(flow: Flow | null, live: boolean) {
  const anchor = useRef<{ client: number; server: number } | null>(null);
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (flow) anchor.current = { client: performance.now(), server: flow.total_ms };
  }, [flow]);
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setNow(performance.now()), 200);
    return () => window.clearInterval(id);
  }, [live]);
  return (node: { status: FlowStatus; start_ms: number | null; end_ms: number | null }) => {
    if (node.start_ms == null) return null;
    if (node.end_ms != null) return node.end_ms - node.start_ms;
    if (node.status !== "running" || !anchor.current) return null;
    const serverNow = anchor.current.server + (now - anchor.current.client);
    return Math.max(0, Math.round(serverNow - node.start_ms));
  };
}

function NodeDetails({ node, elapsed }: { node: FlowNode; elapsed: ReturnType<typeof useElapsed> }) {
  const d = node.detail;
  const tokens = node.llm.reduce((sum, c) => sum + c.input_tokens + c.output_tokens, 0);
  return (
    <div className="flow-details">
      <div className="fd-head">
        <StatusIcon status={node.status} size={17} />
        <b>{node.label}</b>
        <span className="sub">
          {node.status}
          {elapsed(node) != null && ` · ${fmt(elapsed(node))}`}
          {node.start_ms != null && ` · started at +${fmt(node.start_ms)}`}
        </span>
      </div>

      <div className="fd-grid">
        <section>
          <h5>Input</h5>
          {node.kind === "trigger" && (
            <>
              <p className="quote">{d.message}</p>
              {d.job_description_attached && <span className="chip">job description attached</span>}
            </>
          )}
          {node.kind === "router" && <p>Classifies intent and plans which agents run, in order.</p>}
          {node.kind === "agent" && (
            <>
              <p>{d.instruction}</p>
              {d.target_sections?.length > 0 && (
                <div className="kv">
                  {d.target_sections.map((s: string) => <span key={s} className="chip">{s}</span>)}
                </div>
              )}
            </>
          )}
          {node.kind === "respond" && <p>Agent results → streamed chat reply.</p>}
          {node.kind === "version" && <p>Commit the edited resume as a new immutable version.</p>}
        </section>

        <section>
          <h5>Output</h5>
          {node.kind === "router" && d.intent && (
            <>
              <p>
                Intent <span className="chip">{d.intent}</span>
                {d.plan?.length > 0 && <> → {d.plan.join(" → ")}</>}
              </p>
              <p className="sub">{d.reasoning}</p>
            </>
          )}
          {node.kind === "agent" && (
            <>
              {d.reasoning && <p className="sub">{d.reasoning}</p>}
              {typeof d.match_score_after === "number" && (
                <p>Match score <b>{d.match_score_after}</b>{d.match_score_before !== d.match_score_after && ` (was ${d.match_score_before})`}</p>
              )}
              {d.edited?.length > 0 && <p>Edited: {d.edited.join(", ")}</p>}
              {d.warnings?.length > 0 && <p className="st-warn">{d.warnings.length} warning(s) for you to review</p>}
            </>
          )}
          {node.kind === "respond" && d.reply_preview && <p className="quote">{d.reply_preview}…</p>}
          {node.kind === "version" &&
            (d.number ? (
              <p>
                Version <b>{d.number}</b> · {d.label}
                {d.changed_sections?.length > 0 && <> · {d.changed_sections.join(", ")}</>}
              </p>
            ) : (
              <p className="sub">{d.reason ?? "Not run"}</p>
            ))}
          {node.status === "pending" && <p className="sub">Waiting…</p>}
          {node.status === "running" && <p className="sub">Running…</p>}
        </section>
      </div>

      {node.steps.length > 0 && (
        <section>
          <h5>Steps</h5>
          <ol className="fd-steps">
            {node.steps.map((s, i) => (
              <li key={i}>
                <StatusIcon status={s.status} size={13} />
                <span>{s.label}</span>
                <span className="sub">{fmt(elapsed(s))}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {node.llm.length > 0 && (
        <section>
          <h5>LLM calls · {tokens.toLocaleString()} tokens</h5>
          <table className="fd-llm">
            <tbody>
              {node.llm.map((c, i) => (
                <tr key={i}>
                  <td><code>{c.name}</code></td>
                  <td className="sub">{c.model}</td>
                  <td>{fmt(c.ms)}</td>
                  <td className="sub">{c.input_tokens.toLocaleString()} in · {c.output_tokens.toLocaleString()} out</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

/** n8n-style execution view of one chat turn: nodes, edges, live status and details. */
export default function FlowView({ flow, live }: { flow: Flow | null; live: boolean }) {
  const [selected, setSelected] = useState<string | null>(null);
  const canvas = useRef<HTMLDivElement>(null);
  const elapsed = useElapsed(flow, live);

  const running = flow?.nodes.find((n) => n.status === "running") ?? null;
  const active = useMemo(() => {
    if (!flow) return null;
    return (
      flow.nodes.find((n) => n.id === selected) ??
      running ??
      [...flow.nodes].reverse().find((n) => n.status !== "skipped" && n.status !== "pending") ??
      flow.nodes[0]
    );
  }, [flow, selected, running]);

  // Follow the running node while the turn is live.
  useEffect(() => {
    if (!running || !canvas.current) return;
    canvas.current
      .querySelector(`[data-node="${CSS.escape(running.id)}"]`)
      ?.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }, [running?.id]);

  // A new turn resets the manual selection.
  const firstMessage = flow?.nodes[0]?.detail.message;
  useEffect(() => setSelected(null), [firstMessage]);

  if (!flow) {
    return (
      <div className="empty">
        <Workflow size={28} />
        <p>Send a message to watch the router and agents execute here, step by step.</p>
      </div>
    );
  }

  const llmCalls = flow.nodes.reduce((sum, n) => sum + n.llm.length, 0);
  return (
    <div className="flow">
      <div className="flow-bar">
        <b>Execution</b>
        <span className="sub">
          {live ? "running" : "finished"} · {fmt(live ? elapsed({ status: "running", start_ms: 0, end_ms: null }) : flow.total_ms)} ·{" "}
          {flow.nodes.length} nodes · {llmCalls} LLM call{llmCalls === 1 ? "" : "s"}
        </span>
      </div>
      <div className="flow-canvas" ref={canvas}>
        <div className="flow-track">
          {flow.nodes.map((node, i) => {
            const Icon = nodeIcon(node);
            const next = flow.nodes[i + 1];
            const t = elapsed(node);
            return (
              <div key={node.id} className="flow-cell">
                <button
                  data-node={node.id}
                  className={`flow-node k-${node.kind} s-${node.status} ${active?.id === node.id ? "selected" : ""}`}
                  onClick={() => setSelected(node.id)}
                  aria-label={`${node.label}: ${node.status}`}
                >
                  <div className="fn-top">
                    <span className="fn-icon"><Icon size={16} /></span>
                    <StatusIcon status={node.status} />
                  </div>
                  <div className="fn-label">{node.label}</div>
                  <div className="fn-meta">
                    {t != null ? fmt(t) : node.status}
                    {node.llm.length > 0 && ` · ${node.llm.length} LLM`}
                    {node.steps.length > 0 && ` · ${node.steps.length} steps`}
                  </div>
                  {node.status === "running" && node.steps.length > 0 && (
                    <div className="fn-step">{node.steps[node.steps.length - 1].label}…</div>
                  )}
                </button>
                {next && (
                  <div
                    className={`flow-edge ${next.status === "running" ? "active" : ""} ${
                      next.status === "done" || next.status === "error" ? "done" : ""
                    } ${next.status === "skipped" ? "skipped" : ""}`}
                    aria-hidden
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
      {active && <NodeDetails node={active} elapsed={elapsed} />}
    </div>
  );
}
