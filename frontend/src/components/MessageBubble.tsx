import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, GitCommitHorizontal, Sparkles, Workflow } from "lucide-react";
import type { ChatMessage, Flow } from "../types";
import { flowTokens } from "../tokens";
import AgentCard from "./AgentCard";
import Carousel from "./Carousel";
import TokenMeter from "./TokenMeter";

const ROUTED_BY: Record<string, string> = {
  preset: "suggested action",
  rules: "rule",
  cache: "cached route",
  classifier: "fast classifier",
  llm: "router",
};

/** User text plus any resume lines they selected (stored as "> line" quotes). */
function UserContent({ content }: { content: string }) {
  const lines = content.split("\n");
  const quotes = lines.filter((l) => l.startsWith("> ")).map((l) => l.slice(2));
  const text = lines.filter((l) => !l.startsWith("> ")).join("\n").trim();
  return (
    <>
      {quotes.length > 0 && (
        <blockquote className="msg-quote">
          {quotes.map((q, i) => (
            <span key={i}>{q}</span>
          ))}
        </blockquote>
      )}
      <span className="msg-text">{text}</span>
    </>
  );
}

export default function MessageBubble({
  message,
  versionNumber,
  onViewFlow,
}: {
  message: ChatMessage;
  versionNumber?: number;
  onViewFlow?: (flow: Flow) => void;
}) {
  const [open, setOpen] = useState(false);
  const trace = message.trace;

  if (message.role === "user") {
    return (
      <div className="msg is-user">
        <div className="msg-bubble">
          <UserContent content={message.content} />
        </div>
      </div>
    );
  }

  const results = trace?.results ?? [];
  const tokens = trace?.flow ? flowTokens(trace.flow) : (trace?.llm.input_tokens ?? 0) + (trace?.llm.output_tokens ?? 0);
  return (
    <div className="msg is-assistant">
      <span className="msg-avatar" aria-hidden>
        <Sparkles size={15} />
      </span>
      <div className="msg-body">
        <div className="msg-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>

        {results.length > 0 && (
          <div className="msg-agents">
            <button className="msg-agents-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
              <ChevronDown size={14} className={open ? "is-open" : undefined} />
              {open ? "Hide" : "Show"} what the {results.length > 1 ? `${results.length} agents` : "agent"} did
            </button>
            {open && (
              <Carousel label="Agent results" itemWidth="min(320px, 82%)">
                {results.map((r) => (
                  <AgentCard key={r.agent} result={r} />
                ))}
              </Carousel>
            )}
          </div>
        )}

        {trace && (
          <footer className="msg-meta">
            <TokenMeter value={tokens} live={false} cost={trace.cost_usd} compact />
            <span>{(trace.ms / 1000).toFixed(1)} s</span>
            {trace.decided_by && <span>via {ROUTED_BY[trace.decided_by] ?? trace.decided_by}</span>}
            {trace.subscription_plan && <span className="meta-plan">{trace.subscription_plan}</span>}
            {versionNumber && (
              <span className="meta-version">
                <GitCommitHorizontal size={13} /> Version {versionNumber}
              </span>
            )}
            {trace.flow && onViewFlow && (
              <button className="meta-link" onClick={() => onViewFlow(trace.flow!)}>
                <Workflow size={13} /> View flow
              </button>
            )}
          </footer>
        )}
        {!trace && versionNumber && (
          <footer className="msg-meta">
            <span className="meta-version">
              <GitCommitHorizontal size={13} /> Version {versionNumber}
            </span>
          </footer>
        )}
      </div>
    </div>
  );
}
