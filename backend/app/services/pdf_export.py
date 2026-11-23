"""PDF export: a clean single-column, ATS-friendly layout rendered with fpdf2.

Why fpdf2: it is pure Python (no system libraries such as GTK or a headless browser), permissively
licensed (LGPL), and lays out a resume page in tens of milliseconds. Fonts are embedded and
subset, so the file is small and looks the same everywhere.

Fonts: Noto Sans (bundled, SIL OFL) covers Latin, Greek and Cyrillic, which includes every
European locale the translation agent supports. Text in other scripts (Japanese, Chinese,
Korean, Devanagari, ...) is rendered with a fallback font: the first installed font that
covers the missing characters (``PDF_FALLBACK_FONTS`` first, then common system fonts). The
fallback is only loaded when the resume actually needs it, so Latin resumes stay fast.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.models.resume import Resume, ResumeSection

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FAMILY = "NotoSans"

# Installed fonts that cover scripts Noto Sans does not (path, collection index).
SYSTEM_FALLBACKS: list[tuple[str, int]] = [
    # Windows
    (r"C:\Windows\Fonts\YuGothR.ttc", 0),
    (r"C:\Windows\Fonts\msgothic.ttc", 0),
    (r"C:\Windows\Fonts\msyh.ttc", 0),
    (r"C:\Windows\Fonts\malgun.ttf", 0),
    (r"C:\Windows\Fonts\Nirmala.ttc", 0),
    # Linux (Debian/Ubuntu fonts-noto-cjk, fonts-noto)
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", 0),
    # macOS
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
]

# Page size: Letter where it is the norm, A4 everywhere else.
LETTER_LOCALES = {"en-US", "en-CA", "fr-CA", "es-MX", "es-US"}

INK = (28, 31, 36)
MUTED = (92, 99, 110)
ACCENT = (38, 56, 78)
RULE = (196, 201, 208)

BULLET = re.compile(r"^\s*[-*•▪◦]\s+")
BOLD = re.compile(r"\*\*(.+?)\*\*")
DATE_RANGE = re.compile(
    # "Jan 2022 - Present", "2019 - 2021", "01/2022 - heute", "September 2022-2026"
    r"((?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}/)?(?:19|20)\d{2}(?:-\d{2})?\s*(?:-|–|—|to|a|bis)\s*"
    r"(?:present|current|now|today|actualidad|presente|heute|aujourd'hui|"
    r"(?:[A-Za-zÀ-ÿ]+\.?\s*)?(?:\d{1,2}/)?(?:19|20)\d{2}(?:-\d{2})?)"
    # "Dec - Jan 2024", "June -July 2023": a month span inside one year
    r"|[A-Za-zÀ-ÿ]{3,}\.?\s*[-–—]\s*[A-Za-zÀ-ÿ]{3,}\.?\s+(?:19|20)\d{2})\s*$",
    re.IGNORECASE,
)
PHONE = re.compile(r"\+?[\d\s().-]{7,}")
SINGLE_DATE = re.compile(r"((?:[A-Za-z]+\.?\s+)?(?:19|20)\d{2})\s*$")
ENTRY_KINDS = {"experience", "projects", "education", "volunteer", "publications", "certifications"}


# ----------------------------------------------------------------------------- structure


@dataclass
class Entry:
    title: str = ""
    subtitle: str = ""
    date: str = ""
    tags: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)


def _strip_bullet(line: str) -> str:
    return BULLET.sub("", line).strip()


def _split_list(text: str) -> list[str]:
    """Split on commas outside parentheses: "AWS (S3, ECS), Docker" -> 2 items."""
    items, depth, current = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth = max(0, depth - (ch == ")"))
        if ch in ",;" and depth == 0:
            if current.strip():
                items.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        items.append(current.strip())
    return items


def _split_date(line: str) -> tuple[str, str]:
    m = DATE_RANGE.search(line)
    if m and m.start() > 0:
        return line[: m.start()].rstrip(" |,–-"), m.group(1).strip()
    m = SINGLE_DATE.search(line)
    if m and m.start() > 8:
        return line[: m.start()].rstrip(" |,–-"), m.group(1).strip()
    return line, ""


def parse_entries(content: str) -> list[Entry]:
    """Group lines into entries: header lines (title, subtitle, date) followed by bullets."""
    entries: list[Entry] = []
    current: Entry | None = None
    for raw in content.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if BULLET.match(line):
            if current is None:
                current = Entry()
                entries.append(current)
            current.bullets.append(_strip_bullet(line))
            continue
        if current is None or current.bullets:
            current = Entry()
            entries.append(current)
        elif _split_date(line)[1] and any(_split_date(h)[1] for h in current.headers):
            # A second dated line without bullets in between is a new entry (two degrees,
            # two short roles). Lines after the previous entry's date move with it when that
            # entry put its date below its first line ("School" / "Degree  2018 - 2021").
            dated = [i for i, h in enumerate(current.headers) if _split_date(h)[1]]
            carry = current.headers[dated[-1] + 1 :] if dated[0] > 0 else []
            del current.headers[len(current.headers) - len(carry) :]
            current = Entry(headers=carry)
            entries.append(current)
        current.headers.append(line)

    for entry in entries:
        dated = [_split_date(h) for h in entry.headers]
        entry.date = next((d for _, d in dated if d), "")
        texts = [t for t, _ in dated if t]
        title = texts[0] if texts else ""
        subtitle_parts = texts[1:]
        if " | " in title:
            parts = re.split(r"\s+\|\s+", title)
            title, rest = parts[0], parts[1:]
            if rest and "," in rest[-1]:
                # A comma list at the end is the tech stack. Kept verbatim: an export must
                # never drop text (the screen view may tidy it, the file may not).
                entry.tags = _split_list(rest[-1])
                rest = rest[:-1]
            subtitle_parts = rest + subtitle_parts
        entry.title = title
        entry.subtitle = " · ".join(p for p in subtitle_parts if p)
    return entries


def _header(section: ResumeSection | None) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Name, free-text lines (headline, location) and contact items (text, link)."""
    lines = [ln.strip() for ln in (section.content if section else "").split("\n") if ln.strip()]
    if not lines:
        return "", [], []
    name, free, contacts = lines[0], [], []
    for line in lines[1:]:
        parts = [p.strip() for p in re.split(r"\s*[|·•]\s*", line) if p.strip()]
        linked = [(p, _href(p)) for p in parts]
        # A contact line has at least one email, phone or link; "Senior Recruiter | IT Hiring"
        # is a headline even though it is pipe-separated.
        if any(link or PHONE.fullmatch(p) for p, link in linked):
            contacts.extend(linked)
        else:
            free.append(line)
    return name, free, contacts


def _href(text: str) -> str:
    if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", text):
        return f"mailto:{text}"
    if re.search(r"(https?://|www\.|linkedin\.|github\.|\.(com|io|dev|me|ai|org|net)\b)", text, re.I):
        return text if text.startswith("http") else f"https://{text}"
    return ""


# ----------------------------------------------------------------------------- fonts


@lru_cache(maxsize=16)
def _cmap(path: str, index: int = 0) -> frozenset[int]:
    font = TTFont(path, fontNumber=index, lazy=True)
    try:
        return frozenset(font.getBestCmap() or {})
    finally:
        font.close()


BULLET_LIKE = set("○◦▪▫■□►▸‣⁃∙➢➤➔✓✔✗❖◆◇●")


def _base_cmap() -> frozenset[int]:
    return _cmap(str(FONT_DIR / f"{FAMILY}-Regular-pdf.ttf"))


def _is_script(ch: str) -> bool:
    """Letters, marks and digits: text that must render, even if it needs a fallback font."""
    return unicodedata.category(ch)[0] in "LMN"


def _symbol_map(text: str) -> dict[int, str]:
    """Uncovered decorative symbols: bullet-like ones become "•", icons and emoji are dropped.

    Contact icons (✉ 📞 📍) carry no text, and loading a multi-megabyte symbol font just for
    them would cost about a second per document.
    """
    base = _base_cmap()
    return {
        ord(c): ("•" if c in BULLET_LIKE else "")
        for c in set(text)
        if not c.isspace() and ord(c) not in base and not _is_script(c)
    }


def _fallback_fonts(text: str) -> list[tuple[str, int]]:
    """Fonts needed for script characters the bundled font cannot draw (usually none)."""
    base = _base_cmap()
    missing = {ord(c) for c in text if _is_script(c) and ord(c) not in base}
    if not missing:
        return []
    configured = [(p, 0) for p in os.environ.get("PDF_FALLBACK_FONTS", "").split(os.pathsep) if p]
    chosen: list[tuple[str, int]] = []
    for path, index in configured + SYSTEM_FALLBACKS:
        if not missing:
            break
        if not os.path.exists(path):
            continue
        try:
            covered = missing & _cmap(path, index)
        except Exception:  # unreadable font file: skip it
            continue
        if covered:
            chosen.append((path, index))
            missing -= covered
    return chosen


# ----------------------------------------------------------------------------- rendering


class _ResumePDF(FPDF):
    body = 9.4  # pt
    lh = 4.5  # line height, mm

    def __init__(self, format: str, fallbacks: list[tuple[str, int]]):
        super().__init__(unit="mm", format=format)
        self.set_margins(16, 14, 16)
        self.set_auto_page_break(True, margin=14)
        self.c_margin = 0
        # Trimmed, unhinted builds (scripts/build_pdf_fonts.py): much faster to load and subset.
        for style, suffix in (("", "Regular"), ("B", "Bold")):
            self.add_font(FAMILY, style, FONT_DIR / f"{FAMILY}-{suffix}-pdf.ttf")
        names = []
        for i, (path, index) in enumerate(fallbacks):
            name = f"fallback{i}"
            # The same file serves both styles: CJK fonts rarely ship separate weights.
            for style in ("", "B"):
                self.add_font(name, style, path, collection_font_number=index)
            names.append(name)
        if names:
            self.set_fallback_fonts(names, exact_match=False)

    @property
    def width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def style(self, size: float, style: str = "", color: tuple[int, int, int] = INK) -> None:
        self.set_font(FAMILY, style, size)
        self.set_text_color(*color)

    def rich(self, text: str, size: float | None = None, color=INK) -> None:
        """Flowing text with **bold** spans; wraps at the current left margin."""
        size = size or self.body
        pos = 0
        for m in BOLD.finditer(text):
            if m.start() > pos:
                self.style(size, "", color)
                self.write(self.lh, text[pos : m.start()])
            self.style(size, "B", INK)
            self.write(self.lh, m.group(1))
            pos = m.end()
        if pos < len(text):
            self.style(size, "", color)
            self.write(self.lh, text[pos:])
        self.ln(self.lh)

    def heading(self, title: str) -> None:
        # Keep a heading with at least two lines of its content.
        if self.get_y() > self.h - self.b_margin - 16:
            self.add_page()
        self.ln(3.2)
        self.style(10.4, "B", ACCENT)
        self.set_char_spacing(0.7)
        self.cell(0, 5.2, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_char_spacing(0)
        y = self.get_y() + 0.6
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_y(y + 2.2)

    def bullet(self, text: str) -> None:
        indent = 4.2
        left = self.l_margin
        self.style(self.body, "", MUTED)
        self.set_x(left + 1)
        self.cell(indent - 1, self.lh, "•")
        self.set_left_margin(left + indent)
        self.rich(text)
        self.set_left_margin(left)
        self.set_x(left)
        self.ln(0.6)

    def entry(self, e: Entry) -> None:
        if self.get_y() > self.h - self.b_margin - 14:
            self.add_page()
        if e.title or e.date:
            # Title first, then the date on the same baseline: text extractors (and ATS
            # parsers) read in drawing order, so "Role ... Jan 2024 - Present" stays in order.
            y = self.get_y()
            self.style(8.8, "", MUTED)
            date_w = self.get_string_width(e.date) + 1 if e.date else 0.0
            self.style(10.2, "B")
            self.multi_cell(
                self.width - date_w - 4, 5, BOLD.sub(r"\1", e.title) or " ",
                align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            after = self.get_y()
            if e.date:
                self.style(8.8, "", MUTED)
                self.set_xy(self.w - self.r_margin - date_w, y)
                self.cell(date_w, 5, e.date, align="R")
                self.set_xy(self.l_margin, after)
        if e.subtitle:
            self.style(9.2, "", MUTED)
            self.multi_cell(0, 4.6, BOLD.sub(r"\1", e.subtitle), align="L",
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if e.tags:
            self.style(8.6, "", MUTED)
            self.multi_cell(0, 4.3, ", ".join(e.tags), align="L",
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if e.bullets:
            self.ln(0.8)
            for b in e.bullets:
                self.bullet(b)
        self.ln(1.8)


def _join_prose(lines: list[str]) -> list[str]:
    """Merge consecutive plain lines into paragraphs; bullet lines stay separate."""
    out: list[str] = []
    for raw in lines:
        if BULLET.match(raw.strip()) or not out or BULLET.match(out[-1].strip()):
            out.append(raw.strip())
        else:
            out[-1] = f"{out[-1]} {raw.strip()}"
    return out


def _render_skills(pdf: _ResumePDF, lines: list[str]) -> None:
    """"Label: a, b" lines stay as they are; runs of one-item lines flow as one comma list
    (under a label when a short capitalised line such as "INDUSTRIES" introduces them)."""
    groups: list[tuple[str, list[str]]] = []
    run: list[str] | None = None
    for raw in lines:
        line = _strip_bullet(raw)
        m = re.match(r"^([^:]{2,40}):\s*(.+)$", line)
        if m:
            groups.append((m.group(1).strip(), _split_list(m.group(2))))
            run = None
        elif not BULLET.match(raw.strip()) and line.isupper() and len(line.split()) <= 4:
            run = []
            groups.append((line.title(), run))
        elif BULLET.match(raw.strip()) and len(line.split()) <= 8:
            if run is None:
                run = []
                groups.append(("", run))
            run.append(line)
        else:
            groups.append(("", [line]))
            run = None
    for label, items in groups:
        if not items:
            continue
        if label:
            pdf.style(pdf.body, "B")
            pdf.write(pdf.lh, f"{label}: ")
        pdf.rich(", ".join(items))
        pdf.ln(0.8)


def _render_section(pdf: _ResumePDF, section: ResumeSection) -> None:
    lines = [ln for ln in section.content.split("\n") if ln.strip()]
    if not lines:
        return
    pdf.heading(section.title)
    if section.kind == "skills":
        _render_skills(pdf, lines)
        return
    entries = parse_entries(section.content) if section.kind in ENTRY_KINDS else []
    if any(e.title or e.date for e in entries):
        for e in entries:
            pdf.entry(e)
        return
    if section.kind == "summary":
        # Lines wrapped by the original PDF are one paragraph (as on screen).
        lines = _join_prose(lines)
    for raw in lines:
        if BULLET.match(raw.strip()):
            pdf.bullet(_strip_bullet(raw))
        else:
            pdf.rich(raw.strip())
            pdf.ln(0.8)


def to_pdf(resume: Resume) -> bytes:
    text = "\n".join(s.content for s in resume.sections) + "\n" + "\n".join(
        s.title for s in resume.sections
    )
    symbols = _symbol_map(text)
    if symbols:
        resume = resume.model_copy(deep=True)
        for section in resume.sections:
            section.title = section.title.translate(symbols).strip()
            section.content = "\n".join(
                re.sub(r" {2,}", " ", ln.translate(symbols)).strip()
                for ln in section.content.split("\n")
            )
        text = text.translate(symbols)
    header = next((s for s in resume.sections if s.kind == "header"), None)
    name, free, contacts = _header(header)
    page = "Letter" if (resume.locale or "") in LETTER_LOCALES else "A4"
    pdf = _ResumePDF(page, _fallback_fonts(text))
    pdf.set_title(f"{name} - Resume" if name else "Resume")
    if name:
        pdf.set_author(name)
    pdf.set_creator("Resume Chat")
    pdf.set_lang(resume.locale or resume.language or "en")
    pdf.add_page()

    if name:
        pdf.style(20, "B")
        pdf.set_char_spacing(1.2)
        pdf.multi_cell(0, 9, name.upper(), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_char_spacing(0)
    for line in free:
        pdf.style(10, "", MUTED)
        pdf.multi_cell(0, 5, line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if contacts:
        pdf.ln(0.8)
        for i, (item, link) in enumerate(contacts):
            if i:
                pdf.style(8.8, "", RULE)
                pdf.write(4.6, "   |   ")
            pdf.style(8.8, "", ACCENT if link else MUTED)
            pdf.write(4.6, item, link)
        pdf.ln(4.6)
    pdf.ln(1)

    for section in resume.sections:
        if section.kind != "header":
            _render_section(pdf, section)
    return bytes(pdf.output())
