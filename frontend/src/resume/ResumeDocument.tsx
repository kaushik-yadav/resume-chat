import { useRef, useState, type MouseEvent, type ReactNode } from "react";
import {
  Building2,
  FolderGit2,
  GraduationCap,
  Link2,
  Mail,
  MapPin,
  MessageSquareQuote,
  Phone,
  User,
} from "lucide-react";
import type { ResumeSection, ResumeVersion } from "../types";
import {
  ENTRY_KINDS,
  editedLines,
  isEdited,
  parseEntries,
  parseHeader,
  parseSkills,
  splitColumns,
  stripBullet,
  type ContactKind,
} from "./parse";

const CONTACT_ICON: Record<ContactKind, typeof Mail> = {
  email: Mail,
  phone: Phone,
  link: Link2,
  location: MapPin,
  text: User,
};
const ENTRY_ICON: Record<string, typeof Building2> = {
  experience: Building2,
  volunteer: Building2,
  education: GraduationCap,
  projects: FolderGit2,
};

/** Render **bold** spans; everything else as text. */
function inline(text: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*)/g)
    .map((part, i) =>
      part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
    );
}

function Heading({ section, count, updated }: { section: ResumeSection; count?: number; updated: boolean }) {
  return (
    <h3 className="rd-heading">
      <span>{section.title}</span>
      {updated && <span className="rd-updated">Updated</span>}
      {count !== undefined && count > 1 && <span className="rd-count">{count}</span>}
    </h3>
  );
}

function Prose({ content, edited }: { content: string; edited: Set<string> }) {
  const lines = content.split("\n").filter((l) => l.trim());
  const bullets = lines.filter((l) => /^\s*[-*•]\s+/.test(l));
  const paragraphs = lines.filter((l) => !/^\s*[-*•]\s+/.test(l));
  return (
    <div className="rd-prose">
      {paragraphs.length > 0 && (
        <p className={paragraphs.some((l) => isEdited(edited, l)) ? "rd-edited" : undefined}>
          {inline(paragraphs.join(" "))}
        </p>
      )}
      {bullets.length > 0 && (
        <ul className="rd-bullets">
          {bullets.map((b, i) => (
            <li key={i} className={isEdited(edited, b) ? "rd-edited" : undefined}>
              {inline(stripBullet(b))}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EntryCards({ section, edited }: { section: ResumeSection; edited: Set<string> }) {
  const entries = parseEntries(section.content);
  const Icon = ENTRY_ICON[section.kind] ?? Building2;
  return (
    <div className="rd-cards">
      {entries.map((entry, i) => (
        <article key={i} className="rd-card">
          {(entry.title || entry.date) && (
            <header className="rd-card-head">
              <h4 className={entry.lines.slice(0, 1).some((l) => isEdited(edited, l)) ? "rd-edited" : undefined}>
                {inline(entry.title)}
              </h4>
              {entry.date && <span className="rd-date">{entry.date}</span>}
            </header>
          )}
          {entry.subtitle && (
            <div className="rd-sub">
              <Icon size={15} aria-hidden />
              <span>{inline(entry.subtitle)}</span>
            </div>
          )}
          {entry.chips.length > 0 && (
            <ul className="rd-chips rd-chips-small" aria-label="Technologies">
              {entry.chips.map((chip) => (
                <li key={chip}>{chip}</li>
              ))}
            </ul>
          )}
          {entry.bullets.length > 0 && (
            <ul className="rd-bullets">
              {entry.bullets.map((b, j) => (
                <li key={j} className={isEdited(edited, b) ? "rd-edited" : undefined}>
                  {inline(b)}
                </li>
              ))}
            </ul>
          )}
        </article>
      ))}
    </div>
  );
}

function SideSection({ section, edited, updated }: { section: ResumeSection; edited: Set<string>; updated: boolean }) {
  if (section.kind === "skills") {
    return (
      <section className="rd-block">
        <Heading section={section} updated={updated} />
        {parseSkills(section.content).map((group, i) => (
          <div key={i} className="rd-skill-group">
            {group.label && <h4 className="rd-label">{group.label}</h4>}
            <ul className="rd-chips">
              {group.items.map((item) => (
                <li key={item} className={isEdited(edited, item) ? "rd-edited" : undefined}>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    );
  }
  const entries = parseEntries(section.content);
  return (
    <section className="rd-block">
      <Heading section={section} updated={updated} />
      <ul className="rd-side-list">
        {entries.map((entry, i) => (
          <li key={i}>
            {entry.title && <strong>{inline(entry.title)}</strong>}
            {entry.subtitle && <span>{inline(entry.subtitle)}</span>}
            {entry.date && <span className="rd-side-date">{entry.date}</span>}
            {entry.bullets.map((b, j) => (
              <span key={j} className={isEdited(edited, b) ? "rd-edited" : undefined}>
                {inline(b)}
              </span>
            ))}
          </li>
        ))}
      </ul>
    </section>
  );
}

interface Props {
  version: ResumeVersion;
  /** Miniature rendering for version previews (no interactions). */
  compact?: boolean;
  /** Called with the selected lines when the user chooses "Edit selection". */
  onSelectText?: (lines: string[]) => void;
}

export default function ResumeDocument({ version, compact = false, onSelectText }: Props) {
  const { resume } = version;
  const { header, sidebar, main } = splitColumns(resume);
  const identity = parseHeader(header);
  const edited = compact ? new Set<string>() : editedLines(version);
  const updated = new Set(compact ? [] : version.changes.map((c) => c.section_id));
  const root = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<{ lines: string[]; x: number; y: number } | null>(null);

  function onMouseUp(e: MouseEvent) {
    if (!onSelectText || !root.current) return;
    const sel = window.getSelection();
    const text = sel?.toString().trim() ?? "";
    if (!sel || text.length < 12 || !root.current.contains(sel.anchorNode)) {
      setSelection(null);
      return;
    }
    const lines = text
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length >= 12)
      .slice(0, 10);
    const box = root.current.getBoundingClientRect();
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    setSelection({
      lines,
      x: Math.min(Math.max(rect.left - box.left + rect.width / 2, 80), box.width - 80),
      y: rect.top - box.top + root.current.scrollTop,
    });
    e.stopPropagation();
  }

  return (
    <div className="rd-root" ref={root} onMouseUp={onMouseUp}>
      <article className={`resume-doc${compact ? " is-compact" : ""}`} lang={resume.language}>
        <aside className="rd-side">
          <header className="rd-identity">
            <h2>{identity.name || "Your resume"}</h2>
            {identity.headline && <p>{identity.headline}</p>}
            {resume.locale && <span className="rd-locale">{resume.locale}</span>}
          </header>
          {identity.contacts.length > 0 && (
            <section className="rd-block">
              <h3 className="rd-heading">
                <span>Contact</span>
              </h3>
              <ul className="rd-contact">
                {identity.contacts.map((c, i) => {
                  const Icon = CONTACT_ICON[c.kind];
                  return (
                    <li key={i}>
                      <Icon size={15} aria-hidden />
                      {c.href && !compact ? (
                        <a href={c.href} target="_blank" rel="noreferrer">
                          {c.text}
                        </a>
                      ) : (
                        <span>{c.text}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          )}
          {sidebar.map((s) => (
            <SideSection key={s.id} section={s} edited={edited} updated={updated.has(s.id)} />
          ))}
        </aside>
        <div className="rd-main">
          {main.map((s) => {
            const entries = ENTRY_KINDS.has(s.kind) ? parseEntries(s.content) : [];
            const cardLike = entries.some((e) => e.title || e.date);
            return (
              <section key={s.id} className="rd-block">
                <Heading section={s} count={cardLike ? entries.length : undefined} updated={updated.has(s.id)} />
                {cardLike ? <EntryCards section={s} edited={edited} /> : <Prose content={s.content} edited={edited} />}
              </section>
            );
          })}
        </div>
      </article>
      {selection && onSelectText && (
        <button
          className="rd-select-action"
          style={{ left: selection.x, top: selection.y }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            onSelectText(selection.lines);
            setSelection(null);
            window.getSelection()?.removeAllRanges();
          }}
        >
          <MessageSquareQuote size={14} /> Edit selection
        </button>
      )}
    </div>
  );
}
