import { useEffect, useRef } from "react";
import { AlertCircle } from "lucide-react";
import type { ChatMessage, Flow, LiveTurn, Preset, VersionSummary } from "../types";
import Composer from "./Composer";
import LiveActivity from "./LiveActivity";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  live: LiveTurn | null;
  liveFlow: Flow | null;
  versions: VersionSummary[];
  error: string | null;
  hasJobDescription: boolean;
  presets: Preset[];
  selection: string[];
  onClearSelection: () => void;
  onSend: (message: string, jobDescription: string | null, preset: string | null, selection: string[]) => void;
  onViewFlow: (flow: Flow) => void;
}

export default function ChatPanel(props: Props) {
  const { messages, live, liveFlow, versions, error } = props;
  const end = useRef<HTMLDivElement>(null);
  const numberOf = (id: string | null) => versions.find((v) => v.id === id)?.number;

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, live?.reply, live?.progress.length, live?.results.length]);

  return (
    <section className="chat" aria-label="Chat">
      <div className="chat-scroll">
        <div className="chat-thread">
          {messages.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              versionNumber={m.version_id ? numberOf(m.version_id) : undefined}
              onViewFlow={props.onViewFlow}
            />
          ))}
          {live && <LiveActivity turn={live} flow={liveFlow} />}
          <div ref={end} />
        </div>
      </div>
      <div className="chat-dock">
        {error && (
          <p className="error-banner" role="alert">
            <AlertCircle size={15} aria-hidden /> {error}
          </p>
        )}
        <Composer
          disabled={!!live}
          hasJobDescription={props.hasJobDescription}
          presets={props.presets}
          selection={props.selection}
          onClearSelection={props.onClearSelection}
          onSend={props.onSend}
        />
      </div>
    </section>
  );
}
