import { useEffect, useRef } from "react";
import type { ChatMessage, Flow, LiveTurn, VersionSummary } from "../types";
import Composer from "./Composer";
import LiveActivity from "./LiveActivity";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  live: LiveTurn | null;
  versions: VersionSummary[];
  error: string | null;
  hasJobDescription: boolean;
  onSend: (message: string, jobDescription: string | null) => void;
  onViewFlow: (flow: Flow) => void;
}

export default function ChatPanel({
  messages,
  live,
  versions,
  error,
  hasJobDescription,
  onSend,
  onViewFlow,
}: Props) {
  const end = useRef<HTMLDivElement>(null);
  const numberOf = (id: string | null) => versions.find((v) => v.id === id)?.number;

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, live?.reply, live?.progress.length, live?.results.length]);

  return (
    <section className="chat" aria-label="Chat">
      <div className="messages">
        {messages.map((m) => (
          <MessageBubble
            key={m.id}
            message={m}
            versionNumber={m.version_id ? numberOf(m.version_id) : undefined}
            onViewFlow={onViewFlow}
          />
        ))}
        {live && <LiveActivity turn={live} />}
        <div ref={end} />
      </div>
      <div>
        {error && (
          <div className="composer" style={{ paddingBottom: 0 }}>
            <div className="error-banner">{error}</div>
          </div>
        )}
        <Composer disabled={!!live} hasJobDescription={hasJobDescription} onSend={onSend} />
      </div>
    </section>
  );
}
