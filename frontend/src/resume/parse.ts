// Turns resume section text into structure the UI can design with: contact items, entries
// (roles, projects, degrees) with dates, subtitles and bullets, and grouped skill chips.
import type { Resume, ResumeSection, ResumeVersion } from "../types";

// "Jan 2022 – Present", "2019 - 2021", "May 2025 – Jan 2026", "01/2022 - heute", "2024-10 to Present",
// and month spans inside one year: "Dec – Jan 2024", "June –July 2023". Same rules as the PDF export.
const DATE_RANGE =
  /((?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}\/)?(?:19|20)\d{2}(?:-\d{2})?\s*(?:-|–|—|to|a|bis)\s*(?:present|current|now|today|actualidad|presente|heute|aujourd'hui|(?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}\/)?(?:19|20)\d{2}(?:-\d{2})?)|[A-Za-zÀ-ÿ]{3,}\.?\s*[-–—]\s*[A-Za-zÀ-ÿ]{3,}\.?\s+(?:19|20)\d{2})\s*$/i;
const SINGLE_DATE = /((?:[A-Za-z]+\.?\s+)?(?:19|20)\d{2})\s*$/;
const BULLET = /^\s*[-*•▪◦]\s+/;

export type ContactKind = "email" | "phone" | "link" | "location" | "text";
export interface ContactItem {
  kind: ContactKind;
  text: string;
  href?: string;
}

export interface Entry {
  title: string;
  subtitle: string;
  date: string;
  chips: string[];
  bullets: string[];
  lines: string[]; // raw lines of this entry (for edit highlighting)
}

export interface SkillGroup {
  label: string;
  items: string[];
}

export const SIDEBAR_KINDS = new Set(["skills", "education", "certifications", "languages", "awards", "interests"]);
export const ENTRY_KINDS = new Set(["experience", "projects", "education", "volunteer", "publications", "certifications", "awards"]);

export const stripBullet = (line: string) => line.replace(BULLET, "").trim();
const normalize = (line: string) => stripBullet(line).toLowerCase().replace(/\s+/g, " ");

/** Split on commas that are not inside parentheses: "AWS (Lambda, S3), Docker" -> 2 items. */
export function splitList(text: string): string[] {
  const items: string[] = [];
  let depth = 0;
  let current = "";
  for (const ch of text) {
    if (ch === "(") depth++;
    if (ch === ")") depth = Math.max(0, depth - 1);
    if ((ch === "," || ch === ";" || ch === "•") && depth === 0) {
      if (current.trim()) items.push(current.trim());
      current = "";
    } else current += ch;
  }
  if (current.trim()) items.push(current.trim());
  return items;
}

function classify(item: string): ContactItem {
  const text = item.trim();
  if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)) return { kind: "email", text, href: `mailto:${text}` };
  if (/^\+?[\d\s().-]{7,}$/.test(text)) return { kind: "phone", text, href: `tel:${text.replace(/[^\d+]/g, "")}` };
  if (/(https?:\/\/|www\.|linkedin\.|github\.|\.(com|io|dev|me|ai|org|net)\b)/i.test(text)) {
    return { kind: "link", text: text.replace(/^https?:\/\/(www\.)?/i, ""), href: text.startsWith("http") ? text : `https://${text}` };
  }
  if (/,/.test(text) || /^[A-Z][a-zA-Z\s]+$/.test(text)) return { kind: "location", text };
  return { kind: "text", text };
}

export function parseHeader(section: ResumeSection | undefined): { name: string; headline: string; contacts: ContactItem[] } {
  const lines = (section?.content ?? "").split("\n").map((l) => l.trim()).filter(Boolean);
  const [name = "", ...rest] = lines;
  const contacts: ContactItem[] = [];
  const free: string[] = [];
  for (const line of rest) {
    const parts = line.split(/\s*[|·•]\s*/).filter(Boolean);
    const classified = parts.map(classify);
    // A contact line has an email, phone or link; "Senior Recruiter | IT Hiring" is a headline.
    if (classified.some((c) => c.kind === "email" || c.kind === "phone" || c.kind === "link")) contacts.push(...classified);
    else free.push(line);
  }
  return { name, headline: free.join("\n"), contacts };
}

function splitDate(line: string): { text: string; date: string } {
  const range = DATE_RANGE.exec(line);
  if (range && range.index > 0) return { text: line.slice(0, range.index).replace(/[|,\s–-]+$/, ""), date: range[1] };
  const single = SINGLE_DATE.exec(line);
  if (single && single.index > 8) return { text: line.slice(0, single.index).replace(/[|,\s–-]+$/, ""), date: single[1] };
  return { text: line, date: "" };
}

/** Group lines into entries: header lines (title, subtitle, date) followed by bullets. */
export function parseEntries(content: string): Entry[] {
  const entries: Entry[] = [];
  let current: Entry | null = null;
  let headers: string[] = [];

  const finish = () => {
    if (!current) return;
    const dated = headers.map(splitDate);
    const date = dated.find((d) => d.date)?.date ?? "";
    const texts = dated.map((d) => d.text).filter(Boolean);
    let title = texts[0] ?? "";
    let subtitle = texts.slice(1).join(" · ");
    let chips: string[] = [];
    if (title.includes(" | ")) {
      // "Name | Short description | Tech, Stack (Extra)"
      const parts = title.split(/\s+\|\s+/);
      title = parts[0];
      const last = parts.length > 1 ? parts[parts.length - 1] : "";
      if (last.includes(",")) {
        chips = splitList(last.replace(/\(([^)]*)\)\s*$/, "").trim());
        subtitle = [parts.slice(1, -1).join(" · "), subtitle].filter(Boolean).join(" · ");
      } else subtitle = [parts.slice(1).join(" · "), subtitle].filter(Boolean).join(" · ");
    }
    entries.push({ ...current, title, subtitle, date, chips });
  };

  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (BULLET.test(line)) {
      if (!current) {
        current = { title: "", subtitle: "", date: "", chips: [], bullets: [], lines: [] };
        headers = [];
      }
      current.bullets.push(stripBullet(line));
      current.lines.push(raw);
      continue;
    }
    // A non-bullet line after bullets starts a new entry; consecutive ones form its header.
    if (!current || current.bullets.length > 0) {
      finish();
      current = { title: "", subtitle: "", date: "", chips: [], bullets: [], lines: [] };
      headers = [];
    } else if (splitDate(line).date && headers.some((h) => splitDate(h).date)) {
      // A second dated line with no bullets in between is a new entry (two degrees, two short
      // roles). Lines after the previous date move with it when that entry put its date below
      // its first line ("School" / "Degree  2018 - 2021").
      const dated = headers.flatMap((h, i) => (splitDate(h).date ? [i] : []));
      const carry = dated[0] > 0 ? headers.slice(dated[dated.length - 1] + 1) : [];
      headers = headers.slice(0, headers.length - carry.length);
      current.lines = current.lines.slice(0, current.lines.length - carry.length);
      finish();
      current = { title: "", subtitle: "", date: "", chips: [], bullets: [], lines: [...carry] };
      headers = [...carry];
    }
    headers.push(line);
    current.lines.push(raw);
  }
  finish();
  return entries;
}

export function parseSkills(content: string): SkillGroup[] {
  const groups: SkillGroup[] = [];
  const loose: string[] = [];
  for (const raw of content.split("\n")) {
    const line = stripBullet(raw);
    if (!line) continue;
    const labelled = /^([^:]{2,40}):\s*(.+)$/.exec(line);
    if (labelled) groups.push({ label: labelled[1].trim(), items: splitList(labelled[2]) });
    else loose.push(...splitList(line));
  }
  if (loose.length) groups.push({ label: "", items: loose });
  return groups;
}

/** Lines that are new in the current version (to highlight exactly what an edit changed). */
export function editedLines(version: ResumeVersion | null): Set<string> {
  const edited = new Set<string>();
  for (const change of version?.changes ?? []) {
    if (change.change_type === "removed") continue;
    const before = new Set(change.before.split("\n").map(normalize));
    for (const line of change.after.split("\n")) {
      const key = normalize(line);
      if (key && !before.has(key)) edited.add(key);
    }
  }
  return edited;
}

export const isEdited = (edited: Set<string>, line: string) => edited.has(normalize(line));

export function splitColumns(resume: Resume) {
  const header = resume.sections.find((s) => s.kind === "header");
  const body = resume.sections.filter((s) => s.kind !== "header");
  return {
    header,
    sidebar: body.filter((s) => SIDEBAR_KINDS.has(s.kind)),
    main: body.filter((s) => !SIDEBAR_KINDS.has(s.kind)),
  };
}
