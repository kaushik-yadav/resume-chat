import { diffWords } from "diff";
import type { SectionChange } from "../types";

function WordDiff({ before, after }: { before: string; after: string }) {
  const parts = diffWords(before, after);
  return (
    <div className="diff-text">
      {parts.map((p, i) =>
        p.added ? <ins key={i}>{p.value}</ins> : p.removed ? <del key={i}>{p.value}</del> : <span key={i}>{p.value}</span>,
      )}
    </div>
  );
}

export default function DiffView({ changes }: { changes: SectionChange[] }) {
  if (!changes.length) return <div className="empty">No differences.</div>;
  return (
    <div>
      {changes.map((c) => (
        <div className="diff-section" key={c.section_id}>
          <h4>
            {c.title}
            <span className={`chip ${c.change_type === "removed" ? "missing" : c.change_type === "added" ? "met" : ""}`}>
              {c.change_type}
            </span>
            <span className="sub">
              +{c.words_added} / −{c.words_removed} words
            </span>
          </h4>
          <WordDiff before={c.before} after={c.after} />
        </div>
      ))}
    </div>
  );
}
