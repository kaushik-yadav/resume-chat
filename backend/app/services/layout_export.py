"""Exports that keep the uploaded file's own layout: fonts, sizes, colours and positions.

Instead of re-typesetting the resume, the original file is edited in place:

PDF (PyMuPDF)
    1. Every glyph of the original PDF is read with its position, font, size and colour.
    2. Each line of the parsed original resume is located among those glyphs (matching on
       letters and digits only, so spacing, hyphenation and ligatures do not matter).
    3. The lines that differ in the exported version are diffed against the original. For a
       changed line, only its glyphs are removed (true redaction: the old text is gone from
       the file, so ATS parsers never read old and new text together). Words at the start of
       the line that did not change, and an unchanged trailing segment such as a right-aligned
       date, are left exactly where they are.
    4. The new text is drawn at the same baseline(s), with the original font when it is
       embedded and covers the new characters (checked glyph by glyph, including widths),
       otherwise with the closest available font (same PostScript name on this machine, then
       Times/Helvetica/Courier by style, then bundled Noto Sans). Text wraps into the space
       the old line used (plus free space below it); if needed the size shrinks by up to 18%.
    5. The output is verified by extracting its text again: every new line must be present,
       every replaced line gone and every untouched line intact.

    If any step cannot be done faithfully (a section or extra lines were added, text no longer
    fits, a line cannot be located), ``LayoutUnavailable`` explains why and the caller falls
    back to the clean single-column export.

DOCX (python-docx)
    Paragraphs are edited in place, keeping their style, numbering and run formatting; added
    lines become copies of the previous paragraph, so a DOCX can grow where a PDF cannot.

Licensing note: PyMuPDF is AGPL-3.0 (or a commercial licence from Artifex). A closed-source
SaaS needs the commercial licence; the clean export (fpdf2) has no such constraint.
"""

from __future__ import annotations

import copy
import difflib
import io
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf

from app.models.resume import Resume

BULLET_PREFIX = re.compile(r"^\s*[-*•▪◦]\s+")
BOLD = re.compile(r"\*\*(.+?)\*\*")
BULLET_GLYPHS = set("•●○◦▪▫■□►▸‣⁃∙➢➤✓✔-–—*\uf0b7\uf0a7")
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
MIN_SCALE = 0.82
# Written into the PDF producer field; the uploader reads such files in position order.
EDITED_MARK = "(edited in place by Resume Chat)"
# (font size scale, horizontal scale) tried in order until the new text fits: shrink a little
# first, then condense slightly. The last step holds ~1.35x the original text per row.
FIT_STEPS = [(1, 1), (0.98, 1), (0.96, 1), (0.94, 1), (0.92, 1), (0.92, 0.96), (0.9, 0.94),
             (0.88, 0.92), (0.86, 0.9), (0.84, 0.9), (0.82, 0.9)]


class LayoutUnavailable(Exception):
    """The original layout cannot be kept faithfully; the message says why."""


def plain(line: str) -> str:
    return BOLD.sub(r"\1", BULLET_PREFIX.sub("", line)).strip()


def norm(text: str) -> str:
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text) if c.isalnum())


# --------------------------------------------------------------------------- line diff


@dataclass
class Change:
    old: list[tuple]  # keys of original lines: ("t", section_id) or ("c", section_id, index)
    new: list[str]  # replacement lines (plain text); fewer lines = the rest are deleted
    after: tuple | None = None  # for pure insertions: the original line they follow


def original_lines(base: Resume) -> list[tuple[tuple, str]]:
    """Every line of the original resume in document order, with a stable key."""
    out: list[tuple[tuple, str]] = []
    for s in base.sections:
        if s.kind != "header":
            out.append((("t", s.id), s.title.strip()))
        lines = [plain(ln) for ln in s.content.split("\n") if plain(ln)]
        out.extend((("c", s.id, i), ln) for i, ln in enumerate(lines))
    return out


def diff_lines(base: Resume, target: Resume) -> list[Change]:
    targets = {s.id: s for s in target.sections}
    base_ids = {s.id for s in base.sections}
    for s in target.sections:
        if s.id not in base_ids and s.content.strip():
            raise LayoutUnavailable(f"the edit added a new section ({s.title})")
    changes: list[Change] = []
    for b in base.sections:
        t = targets.get(b.id)
        if b.kind != "header" and (t is None or t.title.strip() != b.title.strip()):
            changes.append(Change([("t", b.id)], [t.title.strip()] if t else []))
        old = [plain(ln) for ln in b.content.split("\n") if plain(ln)]
        raw = [BULLET_PREFIX.sub("", ln).strip() for ln in (t.content if t else "").split("\n") if plain(ln)]
        new = [plain(ln) for ln in raw]
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != "equal":  # replacements keep their **bold** markup
                after = ("c", b.id, i1 - 1) if i1 else ("t", b.id)
                changes.append(Change([("c", b.id, i) for i in range(i1, i2)], raw[j1:j2], after))
    return changes


# --------------------------------------------------------------------------- PDF: glyphs


@dataclass
class Glyph:
    c: str
    rect: pymupdf.Rect
    x: float  # origin
    y: float  # baseline
    size: float
    font: str
    color: int
    flags: int
    page: int
    line_rect: pymupdf.Rect  # MuPDF line (one column, one baseline)


def read_glyphs(doc: pymupdf.Document) -> list[Glyph]:
    glyphs: list[Glyph] = []
    for pno, page in enumerate(doc):
        data = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES)
        for block in data["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                lrect = pymupdf.Rect(line["bbox"])
                for span in line["spans"]:
                    for ch in span["chars"]:
                        glyphs.append(
                            Glyph(
                                ch["c"], pymupdf.Rect(ch["bbox"]), ch["origin"][0], ch["origin"][1],
                                span["size"], span["font"], span["color"], span["flags"], pno, lrect,
                            )
                        )
    return glyphs


class Locator:
    """Finds the glyphs of a text line, matching on letters and digits only."""

    def __init__(self, glyphs: list[Glyph]):
        self.glyphs = glyphs
        chars, owner = [], []
        for gi, g in enumerate(glyphs):
            for ch in norm(g.c):
                chars.append(ch)
                owner.append(gi)
        self.stream, self.owner, self.cursor = "".join(chars), owner, 0
        self.claimed: set[int] = set()

    def find(self, text: str) -> list[int] | None:
        key = norm(text)
        if not key:
            return None
        pos = self.stream.find(key, self.cursor)
        while pos != -1 and any(self.owner[p] in self.claimed for p in range(pos, pos + len(key))):
            pos = self.stream.find(key, pos + 1)
        if pos == -1:
            pos = self.stream.find(key)
            while pos != -1 and any(self.owner[p] in self.claimed for p in range(pos, pos + len(key))):
                pos = self.stream.find(key, pos + 1)
        if pos == -1:
            return None
        self.cursor = pos + len(key)
        matched = sorted(set(self.owner[pos : pos + len(key)]))
        self.claimed.update(matched)
        return self._expand(matched)

    def _same_row(self, a: Glyph, b: Glyph) -> bool:
        return a.page == b.page and abs(a.y - b.y) < 1.0

    def _expand(self, matched: list[int]) -> list[int]:
        """Add spaces/punctuation between matched glyphs and adjacent trailing punctuation."""
        g = self.glyphs
        out = set(matched)
        for a, b in zip(matched, matched[1:], strict=False):
            if b - a > 1 and self._same_row(g[a], g[b]):
                out.update(i for i in range(a + 1, b) if self._same_row(g[i], g[a]))
        last = matched[-1]
        i = last + 1
        while i < len(g) and self._same_row(g[i], g[last]) and not g[i].c.isalnum() and not g[i].c.isspace():
            if g[i].rect.x0 - g[i - 1].rect.x1 > g[i].size * 0.4:
                break
            out.add(i)
            i += 1
        first = matched[0]
        i = first - 1
        while i >= 0 and self._same_row(g[i], g[first]) and g[i].c in "([{\"'“‘":
            out.add(i)
            i -= 1
        self.claimed.update(out)
        return sorted(out, key=lambda k: (g[k].page, round(g[k].y, 1), g[k].x))


# --------------------------------------------------------------------------- PDF: fonts


@dataclass
class FontChoice:
    key: str  # name used on the page
    font: pymupdf.Font
    buffer: bytes | None  # None for built-in base-14 fonts


@lru_cache(maxsize=1)
def _system_fonts() -> dict[str, str]:
    """PostScript name (lower-case, no spaces) -> font file, for fonts installed here."""
    from fontTools.ttLib import TTFont

    dirs = [r"C:\Windows\Fonts", os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
            "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
            "/Library/Fonts", "/System/Library/Fonts"]
    index: dict[str, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith((".ttf", ".otf")):
                    continue
                path = os.path.join(root, f)
                try:
                    font = TTFont(path, lazy=True, fontNumber=0)
                    name = font["name"].getDebugName(6) or ""
                    font.close()
                except Exception:
                    continue
                if name:
                    index.setdefault(name.lower().replace(" ", ""), path)
    return index


class Fonts:
    """Chooses, per line, a font that can draw the new text and looks like the original."""

    def __init__(self, doc: pymupdf.Document, glyphs: list[Glyph]):
        self.doc = doc
        self.embedded: dict[str, tuple[bytes, pymupdf.Font] | None] = {}
        self.seen: dict[str, dict[str, float]] = {}  # font -> char -> advance width at size 1
        for g in glyphs:
            if g.c.strip() and g.size:
                self.seen.setdefault(g.font, {}).setdefault(g.c, (g.rect.x1 - g.rect.x0) / g.size)
        self.basefonts: dict[str, int] = {}
        for page in doc:
            for xref, _ext, _type, basefont, *_ in page.get_fonts(full=True):
                short = basefont.split("+", 1)[-1]
                self.basefonts.setdefault(short, xref)

    def _embedded(self, name: str) -> tuple[bytes, pymupdf.Font] | None:
        if name not in self.embedded:
            found = None
            xref = self.basefonts.get(name)
            if xref:
                try:
                    _, ext, _, buffer = self.doc.extract_font(xref)
                    if buffer and ext not in ("n/a", ""):
                        found = (buffer, pymupdf.Font(fontbuffer=buffer))
                except Exception:
                    found = None
            self.embedded[name] = found
        return self.embedded[name]

    def _faithful(self, name: str, font: pymupdf.Font, text: str) -> bool:
        """The embedded font draws every new character, with the widths seen in the original."""
        seen = self.seen.get(name, {})
        for ch in set(text):
            if ch.isspace():
                continue
            if not font.has_glyph(ord(ch)) or ch not in seen:
                return False
        checks = [c for c in seen if not c.isspace()][:24]
        for ch in checks:
            width = font.text_length(ch, 1)
            if seen[ch] and abs(width - seen[ch]) > 0.12 * max(seen[ch], 0.3):
                return False
        return True

    def choose(self, g: Glyph, text: str) -> FontChoice:
        found = self._embedded(g.font)
        if found and self._faithful(g.font, found[1], text):
            return FontChoice(f"orig-{abs(hash(g.font)) % 10**8}", found[1], found[0])
        bold = bool(g.flags & 16) or "bold" in g.font.lower()
        italic = bool(g.flags & 2) or "italic" in g.font.lower()
        path = _system_fonts().get(g.font.lower().replace(" ", ""))
        if path:
            font = pymupdf.Font(fontfile=path)
            if all(font.has_glyph(ord(c)) for c in set(text) if not c.isspace()):
                return FontChoice(f"sys-{abs(hash(path)) % 10**8}", font, Path(path).read_bytes())
        if g.flags & 8:
            base = "cobo" if bold else "cour"
        elif g.flags & 4 or any(k in g.font.lower() for k in ("times", "serif", "georgia", "garamond", "cmr", "cmbx", "minion", "cambria")):
            base = {(True, True): "tibi", (True, False): "tibo", (False, True): "tiit"}.get((bold, italic), "tiro")
        else:
            base = {(True, True): "hebi", (True, False): "hebo", (False, True): "heit"}.get((bold, italic), "helv")
        font = pymupdf.Font(base)
        if all(font.has_glyph(ord(c)) for c in set(text) if not c.isspace()):
            return FontChoice(base, font, None)
        noto = FONT_DIR / f"NotoSans-{'Bold' if bold else 'Regular'}-pdf.ttf"
        font = pymupdf.Font(fontfile=str(noto))
        if all(font.has_glyph(ord(c)) for c in set(text) if not c.isspace()):
            return FontChoice(f"noto{'b' if bold else ''}", font, noto.read_bytes())
        return FontChoice("cjk", pymupdf.Font("cjk"), None)


# --------------------------------------------------------------------------- PDF: editing


@dataclass
class Draw:
    page: int
    x: float
    y: float
    text: str
    size: float
    color: tuple[float, float, float]
    font: FontChoice
    x_scale: float = 1.0


@dataclass
class Plan:
    removals: dict[int, list[pymupdf.Rect]] = field(default_factory=dict)
    draws: list[Draw] = field(default_factory=list)


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _rows(glyphs: list[Glyph], ids: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for i in ids:
        g = glyphs[i]
        if rows and glyphs[rows[-1][0]].page == g.page and abs(glyphs[rows[-1][0]].y - g.y) < 1.0:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [sorted(r, key=lambda k: glyphs[k].x) for r in rows]


def _words(glyphs: list[Glyph], rows: list[list[int]]):
    """[(text, glyph ids, row index, segment index)]. Words split at spaces or visible gaps;
    segments split at wide gaps (a right-aligned date is its own segment)."""
    out = []
    for r, row in enumerate(rows):
        seg, cur, prev = 0, [], None
        for i in row:
            g = glyphs[i]
            if prev is not None:
                gap = g.rect.x0 - glyphs[prev].rect.x1
                if gap > 1.5 * g.size:
                    if cur:
                        out.append((r, seg, cur))
                    cur, seg = [], seg + 1
                elif gap > 0.22 * g.size and cur:
                    out.append((r, seg, cur))
                    cur = []
            if g.c.isspace():
                if cur:
                    out.append((r, seg, cur))
                cur = []
            else:
                cur.append(i)
            prev = i
        if cur:
            out.append((r, seg, cur))
    return [("".join(glyphs[i].c for i in ids), ids, r, seg) for r, seg, ids in out]


class BoldWord(str):
    """A word marked **bold** in the edited text."""


def _tokens(text: str) -> list[str]:
    words: list[str] = []
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            words += [BoldWord(w) for w in part[2:-2].split()]
        else:
            words += part.split()
    return words


def _wrap(pick, size: float, space: float, words: list[str], widths: list[float], x_scale: float) -> list[list[str]] | None:
    """Greedy word wrap into rows of the given widths (None if it does not fit)."""
    rows: list[list[str]] = [[]]
    used = 0.0
    for w in words:
        width = pick(w).font.text_length(w, size) * x_scale
        gap = space * x_scale if rows[-1] else 0.0
        if used + gap + width <= widths[len(rows) - 1] + 0.5:
            rows[-1].append(w)
            used += gap + width
            continue
        if not rows[-1] or len(rows) == len(widths):
            return None
        rows.append([w])
        used = width
    return rows


def _space(font: pymupdf.Font, size: float, glyphs: list[Glyph], words) -> float:
    """Width of a word gap: the font's space, or (fonts without one, e.g. TeX) the original's."""
    if font.has_glyph(32):
        return font.text_length(" ", size)
    gaps = [
        glyphs[b[1][0]].rect.x0 - glyphs[a[1][-1]].rect.x1
        for a, b in zip(words, words[1:], strict=False)
        if a[2] == b[2] and a[3] == b[3]
    ]
    gaps = sorted(g / glyphs[a[1][0]].size for g, a in zip(gaps, words, strict=False) if g > 0)
    return (gaps[len(gaps) // 2] if gaps else 0.33) * size


def _draw_row(plan: Plan, page: int, x: float, y: float, words: list[str], size: float, space: float,
              color, pick, x_scale: float) -> None:
    """One draw per word at a computed position: exact spacing even without a space glyph."""
    for w in words:
        font = pick(w)
        plan.draws.append(Draw(page, x, y, str(w), size, color, font, x_scale))
        x += (font.font.text_length(w, size) + space) * x_scale


def _row_width(pick, size: float, space: float, words: list[str]) -> float:
    return sum(pick(w).font.text_length(w, size) for w in words) + space * max(len(words) - 1, 0)


def _plan_line(glyphs, ids, new_text, fonts, plan, all_lines, heading=False, keep_prefix=True) -> None:
    rows = _rows(glyphs, ids)
    words = _words(glyphs, rows)
    first = glyphs[rows[0][0]]
    if heading and new_text:
        letters = [glyphs[i].c for i in ids if glyphs[i].c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            new_text = new_text.upper()
    new_words = _tokens(new_text)
    # Words that were bold in this line stay bold wherever they reappear in the new text.
    def is_bold(i: int) -> bool:
        return bool(glyphs[i].flags & 16) or "bold" in glyphs[i].font.lower()

    bold_before = {norm(w[0]) for w in words if norm(w[0]) and all(is_bold(i) for i in w[1])}
    if bold_before and len(bold_before) < len(words):
        new_words = [BoldWord(w) if norm(w) in bold_before else w for w in new_words]

    # Keep unchanged leading words, and an unchanged trailing segment (e.g. a right-aligned date).
    k = 0
    def same(a: str, b: str) -> bool:
        return unicodedata.normalize("NFKC", a) == unicodedata.normalize("NFKC", b)

    while keep_prefix and k < min(len(words), len(new_words)) and same(words[k][0], new_words[k]):
        k += 1
    tail_keep = 0
    last_row = len(rows) - 1
    tail_seg = [w for w in words if w[2] == last_row and w[3] > 0]
    if tail_seg:
        seg_id = tail_seg[-1][3]
        seg_words = [w for w in words if w[2] == last_row and w[3] == seg_id]
        texts = [w[0] for w in seg_words]
        if len(new_words) - k > len(texts) and [norm(t) for t in texts] == [norm(t) for t in new_words[-len(texts):]]:
            tail_keep = len(texts)
    if k == len(words) - tail_keep and k == len(new_words) - tail_keep:
        return  # only spacing/markup differed
    keep = set()
    for w in words[:k] + (words[-tail_keep:] if tail_keep else []):
        keep.update(w[1])
    replaced_ids = {i for w in words[k : len(words) - tail_keep] for i in w[1]}
    ordered = [i for r in rows for i in r]
    if replaced_ids:
        lo = min(ordered.index(i) for i in replaced_ids)
        hi = max(ordered.index(i) for i in replaced_ids)
        remove = [i for i in ordered[lo : hi + 1] if i not in keep]
    else:
        remove = []
    if not new_text and k == 0:
        # A deleted line takes its bullet glyph with it.
        before = [
            j for j, g in enumerate(glyphs)
            if g.page == first.page and abs(g.y - first.y) < 0.35 * first.size and g.rect.x1 <= first.rect.x0 + 0.5
            and first.rect.x0 - g.rect.x1 < 3 * first.size and g.c in BULLET_GLYPHS
        ]
        remove += before
    for i in remove:
        plan.removals.setdefault(glyphs[i].page, []).append(glyphs[i].rect)
    # The new text takes the style of the first word it replaces.
    style_ids = [i for i in ordered if i in replaced_ids and not glyphs[i].c.isspace()]
    text_words = new_words[k : len(new_words) - tail_keep]
    if not text_words:
        return

    # Where the new text goes: after the kept prefix, on the same rows the old text used.
    if k:
        last_kept = glyphs[words[k - 1][1][-1]]
        start_row = words[k - 1][2]
        start_x = last_kept.rect.x1 + pymupdf.Font("helv").text_length(" ", last_kept.size)
    else:
        start_row, start_x = 0, glyphs[remove[0]].x if remove else first.x
    # Body style = the line's dominant (most-used) font among the replaced words, so a bold
    # keyword at the start of the replaced part does not make the whole rewrite bold.
    pool = style_ids or [i for i in ids if not glyphs[i].c.isspace()]
    counts: dict[tuple[str, float], int] = {}
    for i in pool:
        key_ = (glyphs[i].font, round(glyphs[i].size, 1))
        counts[key_] = counts.get(key_, 0) + 1
    dominant = max(counts, key=counts.get) if counts else None
    style = next((glyphs[i] for i in pool if (glyphs[i].font, round(glyphs[i].size, 1)) == dominant), first)
    cont_x = min(glyphs[r[0]].x for r in rows[1:]) if len(rows) > 1 else glyphs[rows[0][0]].x
    if k == 0:
        cont_x = min(cont_x, start_x) if len(rows) > 1 else start_x
    page_lines = [lr for lr in all_lines if lr[0] == style.page]
    # Right edge for the new text: the right edge of nearby text in this column (a right-aligned
    # date marks the margin even when no bullet reaches it), but never into text on the same rows
    # (another column, or a date to the right).
    own_right = max(glyphs[r[-1]].rect.x1 for r in rows)
    first_rows = [r for r in rows if glyphs[r[0]].page == style.page]
    y_top = min(glyphs[r[0]].rect.y0 for r in first_rows)
    y_bot = max(glyphs[r[0]].rect.y1 for r in first_rows)
    near = [lr for lr in page_lines if lr[1].y1 > y_top - 250 and lr[1].y0 < y_bot + 250]
    col_right = max(
        [own_right]
        + [lr[1].x1 for lr in near if lr[1].x0 < own_right + 3 * style.size and lr[1].x1 > cont_x]
    )
    blockers = [
        lr[1].x0 for lr in page_lines
        if lr[1].x0 >= own_right - 0.5 and lr[1].y1 > y_top + 1 and lr[1].y0 < y_bot - 1
    ]
    if blockers:
        col_right = min(col_right, min(blockers) - 0.6 * style.size)
    col_right = max(col_right, own_right)
    if len(rows) > 1:
        lh = glyphs[rows[1][0]].y - glyphs[rows[0][0]].y
    else:
        below = [lr[1].y1 for lr in page_lines if lr[1].y0 > first.y and abs(lr[1].x0 - cont_x) < 2.5 * style.size]
        lh = min([b - first.rect.y1 + first.rect.height for b in below if b - first.y < 2.2 * style.size] + [1.2 * style.size])
    baselines = [glyphs[r[0]].y for r in rows][start_row:]
    row_pages = [glyphs[r[0]].page for r in rows][start_row:]  # a line can wrap across a page break
    widths = []
    for n, r in enumerate(rows[start_row:]):
        x0 = start_x if n == 0 else glyphs[r[0]].x
        right = col_right
        if tail_keep and start_row + n == last_row:
            right = glyphs[words[-tail_keep][1][0]].rect.x0 - style.size
        widths.append(right - x0)
    # Free space below the line (no other text in the column) can take extra rows.
    last_bottom = max(glyphs[i].rect.y1 for i in rows[-1])
    last_page = row_pages[-1]
    obstacles = [
        lr[1].y0 for lr in all_lines
        if lr[0] == last_page and lr[1].y0 >= last_bottom - 0.5 and lr[1].x1 > cont_x and lr[1].x0 < col_right
    ]
    page_h = fonts.doc[last_page].rect.height
    limit = min(obstacles + [page_h - 18])
    extra = 0
    while baselines[-1] + (extra + 1) * lh + 0.25 * style.size <= limit - 1 and extra < 3:
        extra += 1
    widths += [col_right - cont_x] * extra
    baselines += [baselines[-1] + (n + 1) * lh for n in range(extra)]
    row_pages += [row_pages[-1]] * extra

    font = fonts.choose(style, " ".join(w for w in text_words if not isinstance(w, BoldWord)) or "a")
    bold_text = " ".join(w for w in text_words if isinstance(w, BoldWord))
    bold_font = font
    if bold_text:
        # The line's own bold font if it has one, else the style's bold counterpart.
        bold_glyph = next((glyphs[i] for i in ids if glyphs[i].flags & 16 or "bold" in glyphs[i].font.lower()), None)
        if bold_glyph is None:
            bold_glyph = copy.copy(style)
            bold_glyph.flags |= 16
            bold_glyph.font = f"{style.font.split('-')[0]}-Bold"
        bold_font = fonts.choose(bold_glyph, bold_text)

    def pick(w: str) -> FontChoice:
        return bold_font if isinstance(w, BoldWord) else font

    color = _rgb(style.color)
    replaced = words[k : len(words) - tail_keep]
    segs = {(w[2], w[3]) for w in replaced}
    if k and len(segs) == 1 and next(iter(segs))[1] > 0:
        # Only a right-hand segment (a date) changed: redraw that whole segment, right-aligned
        # on its old right edge; it may grow leftwards into the gap.
        row, seg = next(iter(segs))
        seg_words = [w for w in words if (w[2], w[3]) == (row, seg)]
        lead = [w for w in seg_words if w in words[:k]]
        for w in lead:
            for i in w[1]:
                plan.removals.setdefault(glyphs[i].page, []).append(glyphs[i].rect)
        texts = [w[0] for w in lead] + text_words
        right = max(glyphs[i].rect.x1 for w in seg_words for i in w[1])
        left_limit = max([glyphs[i].rect.x1 for w in words if w[2] == row and w[3] < seg for i in w[1]] + [0])
        available = right - left_limit - 1.5 * style.size
        for size_scale, x_scale in FIT_STEPS:
            size = style.size * size_scale
            space = _space(font.font, size, glyphs, words)
            width = _row_width(pick, size, space, texts) * x_scale
            if width <= available:
                break
        else:
            raise LayoutUnavailable(f"“{' '.join(texts)[:48]}” is too long for its place")
        y = glyphs[seg_words[0][1][0]].y
        _draw_row(plan, style.page, right - width, y, texts, size, space, color, pick, x_scale)
        return
    for size_scale, x_scale in FIT_STEPS:
        size = style.size * size_scale
        space = _space(font.font, size, glyphs, words)
        lines = _wrap(pick, size, space, text_words, widths, x_scale)
        if lines is not None:
            break
    else:
        raise LayoutUnavailable(f"“{new_text[:48]}…” is too long for the space of the original line")
    for n, row_words in enumerate(lines):
        x = start_x if n == 0 else (glyphs[rows[start_row + n][0]].x if start_row + n < len(rows) else cont_x)
        _draw_row(plan, row_pages[n], x, baselines[n], row_words, size, space, color, pick, x_scale)


def _apply(doc: pymupdf.Document, plan: Plan) -> None:
    for pno, rects in plan.removals.items():
        page = doc[pno]
        # Merge neighbouring glyphs on a row into one band: one annotation per run, not per glyph.
        rows: list[list[pymupdf.Rect]] = []
        for r in sorted(rects, key=lambda r: (r.y0 + r.y1) / 2):
            mid = (r.y0 + r.y1) / 2
            if rows and abs(mid - (rows[-1][0].y0 + rows[-1][0].y1) / 2) < 1.0:
                rows[-1].append(r)
            else:
                rows.append([r])
        merged: list[pymupdf.Rect] = []
        for row in rows:
            start = len(merged)
            for r in sorted(row, key=lambda r: r.x0):
                last = merged[-1] if len(merged) > start else None
                if last is not None and r.x0 <= last.x1 + 0.6 * r.height:
                    last.x1 = max(last.x1, r.x1)
                    last.y0, last.y1 = min(last.y0, r.y0), max(last.y1, r.y1)
                else:
                    merged.append(pymupdf.Rect(r))
        for r in merged:
            # A thin band through the glyph's middle: removes it without touching neighbours.
            mid, h = (r.y0 + r.y1) / 2, max(r.height, 1) * 0.18
            page.add_redact_annot(pymupdf.Rect(r.x0 + 0.15, mid - h, r.x1 - 0.15, mid + h), fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    registered: set[tuple[int, str]] = set()
    for d in plan.draws:
        page = doc[d.page]
        if d.font.buffer is not None and (d.page, d.font.key) not in registered:
            page.insert_font(fontname=d.font.key, fontbuffer=d.font.buffer)
            registered.add((d.page, d.font.key))
        morph = (pymupdf.Point(d.x, d.y), pymupdf.Matrix(d.x_scale, 1)) if d.x_scale != 1 else None
        page.insert_text((d.x, d.y), d.text, fontname=d.font.key, fontsize=d.size, color=d.color, morph=morph)


def edit_pdf(original: bytes, base: Resume, target: Resume) -> bytes:
    """The original PDF with the target version's text in place of the original's."""
    changes = diff_lines(base, target)
    if not changes:
        return original
    doc = pymupdf.open(stream=original, filetype="pdf")
    glyphs = read_glyphs(doc)
    locator = Locator(glyphs)
    located: dict[tuple, list[int] | None] = {}
    for key, text in original_lines(base):
        located[key] = locator.find(text)
    all_lines = sorted({(g.page, tuple(g.line_rect)) for g in glyphs})
    all_lines = [(p, pymupdf.Rect(r)) for p, r in all_lines]
    fonts = Fonts(doc, glyphs)
    plan = Plan()
    lines = dict(original_lines(base))
    for change in changes:
        if len(change.new) > len(change.old):
            raise LayoutUnavailable(f"the edit added {len(change.new) - len(change.old)} line(s) to a section")
        for n, key in enumerate(change.old):
            ids = located.get(key)
            if ids is None:
                if key[0] == "t":  # a synthetic or unprinted heading
                    continue
                raise LayoutUnavailable(f"could not find “{lines[key][:48]}” in the original PDF")
            new = change.new[n] if n < len(change.new) else ""
            # First keep unchanged leading words in place; if the rest no longer fits after
            # them, re-flow the whole line so it can shrink as one.
            for keep_prefix in (True, False):
                attempt = Plan()
                try:
                    _plan_line(glyphs, ids, new, fonts, attempt, all_lines, heading=key[0] == "t", keep_prefix=keep_prefix)
                    break
                except LayoutUnavailable:
                    if not keep_prefix:
                        raise
            for pno, rects in attempt.removals.items():
                plan.removals.setdefault(pno, []).extend(rects)
            plan.draws.extend(attempt.draws)
    _apply(doc, plan)
    meta = doc.metadata or {}
    doc.set_metadata({**meta, "producer": f"{meta.get('producer') or ''} {EDITED_MARK}".strip()})
    out = doc.tobytes(garbage=3, deflate=True)
    _verify(out, base, target, changes)
    return out


def _verify(pdf: bytes, base: Resume, target: Resume, changes: list[Change]) -> None:
    # Read in position order, as layout-aware parsers (and our own uploader) do: replacement
    # text is drawn after the rest of the page in the file, but sits where the old line was.
    text = norm("".join(page.get_text(sort=True) for page in pymupdf.open(stream=pdf, filetype="pdf")))
    lines = dict(original_lines(base))
    for change in changes:
        for new in change.new:
            if norm(new) not in text:
                raise LayoutUnavailable(f"verification failed: “{new[:48]}” is missing from the result")
        for key in change.old:
            old = norm(lines[key])
            if old and old not in norm(" ".join(change.new)) and text.count(old) > norm(_target_text(target)).count(old):
                raise LayoutUnavailable(f"verification failed: “{lines[key][:48]}” is still in the result")


def _target_text(target: Resume) -> str:
    return " ".join([s.title for s in target.sections] + [s.content for s in target.sections])


# --------------------------------------------------------------------------- DOCX


def edit_docx(original: bytes, base: Resume, target: Resume) -> bytes:
    """The original DOCX with changed paragraphs rewritten in place (formatting kept)."""
    from docx import Document

    changes = diff_lines(base, target)
    doc = Document(io.BytesIO(original))
    if not changes:
        return original
    paragraphs = list(_docx_paragraphs(doc))
    keys = {}
    cursor = 0
    for key, text in original_lines(base):
        want = norm(text)
        if not want:
            continue
        order = list(range(cursor, len(paragraphs))) + list(range(0, cursor))
        for i in order:
            if i not in keys.values() and norm(paragraphs[i].text) == want:
                keys[key] = i
                cursor = i + 1
                break
    lines = dict(original_lines(base))
    for change in changes:
        located = [keys.get(k) for k in change.old]
        if None in located:
            if all(k[0] == "t" for k in change.old):
                continue
            missing = change.old[located.index(None)]
            raise LayoutUnavailable(f"could not find “{lines[missing][:48]}” in the original DOCX")
        for n, idx in enumerate(located):
            para = paragraphs[idx]
            if n < len(change.new):
                text = change.new[n]
                if change.old[n][0] == "t" and para.text.isupper():
                    text = text.upper()
                _set_paragraph_text(para, text)
            else:
                para._p.getparent().remove(para._p)
        # Lines beyond the original count are added after the last one, in its format.
        anchor_idx = located[-1] if located else keys.get(change.after)
        anchor = paragraphs[anchor_idx] if anchor_idx is not None else None
        for text in change.new[len(located) :]:
            if anchor is None:
                raise LayoutUnavailable("the edit added lines with nothing to attach them to")
            clone = copy.deepcopy(anchor._p)
            anchor._p.addnext(clone)
            from docx.text.paragraph import Paragraph

            anchor = Paragraph(clone, anchor._parent)
            _set_paragraph_text(anchor, text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _docx_paragraphs(doc):
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _set_paragraph_text(para, text: str) -> None:
    """Replace the text, keeping the first run's formatting, an unchanged bold lead-in
    ("Languages:") and an unchanged tab-separated tail (a right-aligned date)."""
    runs = [r for r in para.runs]
    if not runs:
        para.add_run(text)
        return
    old = para.text
    head, tail = "", ""
    if runs[0].text and text.startswith(runs[0].text) and len(runs) > 1 and runs[0].bold:
        head = runs[0].text
    if "\t" in old:
        after = old.rsplit("\t", 1)[1]
        if after.strip() and norm(text).endswith(norm(after)):
            tail = after
    body = text[len(head) :]
    if tail:
        cut = len(body)
        while cut > 0 and norm(body[cut:]) != norm(tail):
            cut -= 1
        body = body[:cut].rstrip()
    # Rebuild: [head run] [body in the first non-head run] [tab + tail in the last run].
    start = 1 if head else 0
    body_run = runs[start] if start < len(runs) else para.add_run()
    body_run.text = body + ("\t" + tail if tail else "")
    for r in runs[start + 1 :]:
        r._r.getparent().remove(r._r)
