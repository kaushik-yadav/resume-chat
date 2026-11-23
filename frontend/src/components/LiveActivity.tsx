import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CheckCircle2, Circle, Loader2, Sparkles } from "lucide-react";
import type { Flow, LiveTurn } from "../types";
import { useLiveTokens } from "../tokens";
import TokenMeter from "./TokenMeter";

/** Real-time view of an in-flight turn: plan, step progress, streamed reply, moving token count. */
export default function LiveActivity({ turn, flow }: { turn: LiveTurn; flow: Flow | null }) {
  const steps = turn.plan?.steps ?? [];
  const done = new Set(turn.results.map((r) => r.agent));
  const tokens = useLiveTokens(flow, turn.reply.length, true);
  const latest = turn.progress[turn.progress.length - 1];

  return (
    <div className="msg is-assistant is-live" aria-live="polite">
      <span className="msg-avatar is-live" aria-hidden>
        <Sparkles size={15} />
      </span>
      <div className="msg-body">
        <div className="live-head">
          <span className="live-status">{turn.reply ? "Writing the reply" : turn.status}</span>
          <TokenMeter value={tokens} live compact />
        </div>

        {steps.length > 0 && (
          <ol className="live-steps" aria-label="Plan">
            {steps.map((s) => {
              const isDone = done.has(s.agent);
              const active = turn.activeAgent === s.agent && !isDone;
              return (
                <li key={s.agent} className={isDone ? "is-done" : active ? "is-active" : undefined}>
                  {isDone ? <CheckCircle2 size={14} /> : active ? <Loader2 size={14} className="spin" /> : <Circle size={14} />}
                  <span>{s.title ?? s.agent}</span>
                </li>
              );
            })}
          </ol>
        )}
        {latest && !turn.reply && <p className="live-progress">{latest.message}…</p>}

        {turn.reply && (
          <div className="msg-markdown is-streaming">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.reply}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
