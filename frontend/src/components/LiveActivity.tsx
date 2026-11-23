import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import type { LiveTurn } from "../types";
import AgentCard from "./AgentCard";

function Typing() {
  return (
    <span className="typing" aria-label="Assistant is typing">
      <span />
      <span />
      <span />
    </span>
  );
}

/** Real-time view of an in-flight turn: routing, agent steps, progress and streamed reply. */
export default function LiveActivity({ turn }: { turn: LiveTurn }) {
  const steps = turn.plan?.steps ?? [];
  const doneAgents = new Set(turn.results.map((r) => r.agent));
  return (
    <div className="msg assistant live" aria-live="polite">
      <div className="bubble">
        {turn.plan && turn.plan.intent === "edit" && (
          <div className="router-note">
            <strong>Plan:</strong> {turn.plan.reasoning}
          </div>
        )}
        {steps.length > 0 && (
          <ul className="steps">
            {steps.map((s) => {
              const done = doneAgents.has(s.agent);
              const active = turn.activeAgent === s.agent && !done;
              const progress = turn.progress.filter((p) => p.agent === s.agent);
              return (
                <li key={s.agent} className={done ? "done" : active ? "active" : ""}>
                  {done ? (
                    <CheckCircle2 className="dot" />
                  ) : active ? (
                    <Loader2 className="dot spin" />
                  ) : (
                    <Circle className="dot" />
                  )}
                  <div>
                    <div>{s.title ?? s.agent}</div>
                    {active && progress.length > 0 && (
                      <div className="progress-line">{progress[progress.length - 1].message}…</div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        {turn.results.length > 0 && (
          <details className="trace">
            <summary>Agent reasoning so far</summary>
            <div className="trace-body">
              {turn.results.map((r) => (
                <AgentCard key={r.agent} result={r} />
              ))}
            </div>
          </details>
        )}
        {turn.reply ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.reply}</ReactMarkdown>
        ) : (
          <div className="router-note">
            {turn.status} <Typing />
          </div>
        )}
      </div>
    </div>
  );
}
