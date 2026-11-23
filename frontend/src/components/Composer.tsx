import { useState, type KeyboardEvent } from "react";
import { Paperclip, SendHorizontal, X, Zap } from "lucide-react";
import type { Preset } from "../types";

interface Props {
  disabled: boolean;
  hasJobDescription: boolean;
  presets: Preset[];
  onSend: (message: string, jobDescription: string | null, preset: string | null) => void;
}

export default function Composer({ disabled, hasJobDescription, presets, onSend }: Props) {
  const [text, setText] = useState("");
  const [jd, setJd] = useState("");
  const [showJd, setShowJd] = useState(false);
  // Chip that opened the JD box; dropped if the user rewrites the message.
  const [pending, setPending] = useState<Preset | null>(null);

  function send(message = text, preset: Preset | null = pending) {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    const presetId = preset && preset.label === trimmed ? preset.id : null;
    onSend(trimmed, showJd && jd.trim() ? jd.trim() : null, presetId);
    setText("");
    setJd("");
    setShowJd(false);
    setPending(null);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="composer">
      <div className="suggestions">
        {presets.map((p) => (
          <button
            key={p.id}
            disabled={disabled}
            title="Suggested action: runs instantly, no intent classification needed"
            onClick={() => {
              if (p.needs_job_description && !hasJobDescription) {
                setText(p.label);
                setPending(p);
                setShowJd(true);
              } else send(p.label, p);
            }}
          >
            <Zap size={11} /> {p.label}
          </button>
        ))}
      </div>
      {showJd && (
        <div className="jd-box">
          <label htmlFor="jd">Job description (pasted text is remembered for follow-ups)</label>
          <textarea
            id="jd"
            value={jd}
            placeholder="Paste the job posting here…"
            onChange={(e) => setJd(e.target.value)}
          />
        </div>
      )}
      <div className="input-row">
        <button
          className="btn ghost"
          title={showJd ? "Remove job description" : "Attach a job description"}
          aria-label="Attach job description"
          onClick={() => setShowJd((v) => !v)}
        >
          {showJd ? <X size={18} /> : <Paperclip size={18} />}
        </button>
        <textarea
          rows={1}
          value={text}
          placeholder={disabled ? "Working on it…" : "Ask for a change, or a question about your resume"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          aria-label="Message"
        />
        <button className="btn primary" disabled={disabled || !text.trim()} onClick={() => send()} aria-label="Send">
          <SendHorizontal size={16} />
        </button>
      </div>
    </div>
  );
}
