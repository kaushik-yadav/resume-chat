import { useState, type KeyboardEvent } from "react";
import { Paperclip, SendHorizontal, X } from "lucide-react";

const SUGGESTIONS = [
  "Optimize my resume for Google",
  "Translate this to Spanish for the Mexican market",
  "Update my experience section to better match this job description",
  "Make my skills section more relevant for a data scientist role",
];

interface Props {
  disabled: boolean;
  hasJobDescription: boolean;
  onSend: (message: string, jobDescription: string | null) => void;
}

export default function Composer({ disabled, hasJobDescription, onSend }: Props) {
  const [text, setText] = useState("");
  const [jd, setJd] = useState("");
  const [showJd, setShowJd] = useState(false);

  function send(message = text) {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, showJd && jd.trim() ? jd.trim() : null);
    setText("");
    setJd("");
    setShowJd(false);
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
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            disabled={disabled}
            onClick={() => {
              if (s.includes("job description") && !hasJobDescription) {
                setText(s);
                setShowJd(true);
              } else send(s);
            }}
          >
            {s}
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
