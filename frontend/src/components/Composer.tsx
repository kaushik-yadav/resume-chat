import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, FileText, Paperclip, TextQuote, X, Zap } from "lucide-react";
import type { Preset } from "../types";
import Carousel from "./Carousel";

interface Props {
  disabled: boolean;
  hasJobDescription: boolean;
  presets: Preset[];
  /** Lines selected in the resume: the next message edits only these. */
  selection: string[];
  onClearSelection: () => void;
  onSend: (message: string, jobDescription: string | null, preset: string | null, selection: string[]) => void;
}

export default function Composer({ disabled, hasJobDescription, presets, selection, onClearSelection, onSend }: Props) {
  const [text, setText] = useState("");
  const [jd, setJd] = useState("");
  const [showJd, setShowJd] = useState(false);
  // Suggestion that opened the job-description box; dropped if the user rewrites the message.
  const [pending, setPending] = useState<Preset | null>(null);
  const field = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a limit, like a modern chat input.
  useEffect(() => {
    const el = field.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  useEffect(() => {
    if (selection.length) field.current?.focus();
  }, [selection]);

  function send(message = text, preset: Preset | null = pending) {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    const presetId = preset && preset.label === trimmed && !selection.length ? preset.id : null;
    onSend(trimmed, showJd && jd.trim() ? jd.trim() : null, presetId, selection);
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
    <div className="composer-wrap">
      {!selection.length && presets.length > 0 && (
        <Carousel label="Suggested actions" itemWidth="auto" className="suggestions">
          {presets.map((p) => (
            <button
              key={p.id}
              className="suggestion"
              disabled={disabled}
              title="Runs right away - no need to work out what you meant"
              onClick={() => {
                if (p.needs_job_description && !hasJobDescription) {
                  setText(p.label);
                  setPending(p);
                  setShowJd(true);
                } else send(p.label, p);
              }}
            >
              <Zap size={12} aria-hidden /> {p.label}
            </button>
          ))}
        </Carousel>
      )}

      <div className={`composer${disabled ? " is-busy" : ""}`}>
        {selection.length > 0 && (
          <div className="composer-attachment">
            <TextQuote size={15} aria-hidden />
            <div>
              <strong>
                {selection.length === 1 ? "1 selected line" : `${selection.length} selected lines`} - only these will
                change
              </strong>
              <span>{selection.join(" / ")}</span>
            </div>
            <button aria-label="Remove selection" onClick={onClearSelection}>
              <X size={14} />
            </button>
          </div>
        )}
        {showJd && (
          <div className="composer-attachment is-jd">
            <FileText size={15} aria-hidden />
            <textarea
              value={jd}
              placeholder="Paste the job posting - it's remembered for follow-up requests"
              onChange={(e) => setJd(e.target.value)}
              aria-label="Job description"
            />
            <button aria-label="Remove job description" onClick={() => setShowJd(false)}>
              <X size={14} />
            </button>
          </div>
        )}
        <div className="composer-row">
          <button
            className="composer-icon"
            title={showJd ? "Remove job description" : "Attach a job description"}
            aria-label="Attach a job description"
            aria-pressed={showJd}
            onClick={() => setShowJd((v) => !v)}
          >
            <Paperclip size={17} />
          </button>
          <textarea
            ref={field}
            rows={1}
            value={text}
            placeholder={
              disabled
                ? "Working on it…"
                : selection.length
                  ? "What should change in the selected lines?"
                  : "Ask for a change, or ask about your resume"
            }
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            aria-label="Message"
          />
          <button className="composer-send" disabled={disabled || !text.trim()} onClick={() => send()} aria-label="Send">
            <ArrowUp size={17} />
          </button>
        </div>
      </div>
    </div>
  );
}
