import type { ReactNode } from "react";

// "Jan 2022 – Present", "2019 - 2021", "May 2025 – Jan 2026", "01/2022 - heute", "ene. 2022 – actualidad"
const DATE_TAIL =
  /\s*\|?\s*((?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}\/)?(?:19|20)\d{2}\s*(?:-|–|—|to|a|bis)\s*(?:present|current|now|today|actualidad|presente|heute|(?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}\/)?(?:19|20)\d{2}))\s*$/i;

type Block =
  | { kind: "bullets"; items: string[] }
  | { kind: "role"; title: string; date: string }
  | { kind: "sub"; text: string }
  | { kind: "line"; text: string };

/** Render **bold** spans; everything else as text. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
  );
}

export function toBlocks(content: string): Block[] {
  const blocks: Block[] = [];
  let afterRole = false;
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line) {
      afterRole = false;
      continue;
    }
    const bullet = /^[-*•]\s+/.exec(line);
    if (bullet) {
      const text = line.slice(bullet[0].length);
      const last = blocks[blocks.length - 1];
      if (last?.kind === "bullets") last.items.push(text);
      else blocks.push({ kind: "bullets", items: [text] });
      afterRole = false;
      continue;
    }
    const date = DATE_TAIL.exec(line);
    if (date && date.index > 0) {
      blocks.push({ kind: "role", title: line.slice(0, date.index).replace(/[|,\s]+$/, ""), date: date[1] });
      afterRole = true;
    } else if (afterRole) {
      blocks.push({ kind: "sub", text: line }); // company / location under a role line
      afterRole = false;
    } else {
      blocks.push({ kind: "line", text: line });
    }
  }
  return blocks;
}

/**
 * Line-faithful rendering of a resume section. (Markdown would merge a role line into the
 * bullet above it and join consecutive plain lines into one paragraph.)
 */
export default function ResumeSectionBody({ content }: { content: string }) {
  return (
    <div className="r-body">
      {toBlocks(content).map((b, i) => {
        switch (b.kind) {
          case "bullets":
            return (
              <ul key={i}>
                {b.items.map((item, j) => <li key={j}>{inline(item)}</li>)}
              </ul>
            );
          case "role":
            return (
              <div key={i} className="r-role">
                <span>{inline(b.title)}</span>
                <span className="r-date">{b.date}</span>
              </div>
            );
          case "sub":
            return <div key={i} className="r-sub">{inline(b.text)}</div>;
          default:
            return <p key={i}>{inline(b.text)}</p>;
        }
      })}
    </div>
  );
}
