import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { GitCommitHorizontal } from "lucide-react";
import type { ChatMessage } from "../types";
import AgentCard from "./AgentCard";

const INTENT_LABEL: Record<string, string> = {
  edit: "Edit",
  question: "Question",
  clarify: "Clarification",
  undo: "Undo",
  redo: "Redo",
  chitchat: "Chat",
};

export default function MessageBubble({
  message,
  versionNumber,
}: {
  message: ChatMessage;
  versionNumber?: number;
}) {
  const trace = message.trace;
  return (
    <div className={`msg ${message.role}`}>
      <div className="bubble">
        {message.role === "assistant" ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        ) : (
          message.content
        )}
        {trace && (
          <details className="trace">
            <summary>
              Agent reasoning · {INTENT_LABEL[trace.intent] ?? trace.intent}
              {trace.results.length > 0 && ` · ${trace.results.length} agent${trace.results.length > 1 ? "s" : ""}`}
              {` · ${(trace.ms / 1000).toFixed(1)}s · ${trace.llm.calls} LLM call${trace.llm.calls === 1 ? "" : "s"}`}
            </summary>
            <div className="trace-body">
              <div className="router-note">
                <strong>Router:</strong> {trace.reasoning}
              </div>
              {trace.results.map((r) => (
                <AgentCard key={r.agent} result={r} />
              ))}
            </div>
          </details>
        )}
      </div>
      {message.role === "assistant" && versionNumber && (
        <div className="meta">
          <span className="version-chip">
            <GitCommitHorizontal size={13} /> Saved as version {versionNumber}
          </span>
        </div>
      )}
    </div>
  );
}
