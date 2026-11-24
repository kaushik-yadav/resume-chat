"""Text extraction from uploaded resume files (PDF, DOCX, TXT/MD)."""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import PurePath

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Bullet glyphs commonly emitted by PDF exporters and Word.
_BULLETS = "•●○◦▪▫■□►▸‣⁃∙·➢➤✓✔-–—*"
_BULLET_LINE = re.compile(rf"^\s*[{re.escape(_BULLETS)}]\s+")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
# Symbol / Wingdings bullet glyphs as extracted from Word-made PDFs: •, ▪, ❖, ➢, ✓, ●, □, ➔.
_SYMBOL_BULLETS = "\x7f"
_SYMBOL_BULLET_LINE = re.compile(rf"(?m)^[ \t]*[{_SYMBOL_BULLETS}][ \t]*")
_PRIVATE_GLYPH = re.compile("[\x7f-]")


class ResumeParseError(ValueError):
    """The file could not be turned into resume text."""


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ResumeParseError(
            f"Unsupported file type '{ext or 'unknown'}'. Upload a PDF or DOCX resume."
        )
    if not data:
        raise ResumeParseError("The uploaded file is empty.")

    if ext == ".pdf":
        raw = _extract_pdf(data)
    elif ext == ".docx":
        raw = _extract_docx(data)
    else:
        raw = data.decode("utf-8", errors="replace")

    text = normalize_text(raw)
    if len(text.split()) < 20:
        raise ResumeParseError(
            "Very little text could be extracted. If this is a scanned PDF, "
            "please upload a text-based PDF or a DOCX file."
        )
    return text


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # pragma: no cover - depends on file
                raise ResumeParseError("This PDF is password protected.") from exc
        producer = str((reader.metadata or {}).get("/Producer", ""))
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if "edited in place by Resume Chat" in producer:
                # Our original-layout export draws replaced lines after the rest of the page, so
                # content-stream order (pypdf) would list them last: read in position order.
                text = "\n".join(page.get_text(sort=True) for page in doc)
            else:
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return _mark_drawn_bullets(text, doc)
    except PdfReadError as exc:
        raise ResumeParseError(f"Could not read PDF: {exc}") from exc


def _squash(line: str) -> str:
    # NFKC: pypdf keeps ligatures ("Airﬂow") where PyMuPDF splits them ("Airflow").
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", line))[:40]


def _mark_drawn_bullets(text: str, doc) -> str:
    """Prefix "• " to lines whose bullet is a drawn shape rather than a character.

    Browsers' print-to-PDF (and so most HTML resume builders) draw list markers as small
    filled circles or squares, which leave no trace in the text layer: without this, every
    bullet in such a resume reads as a plain line and runs into the line above it.
    """
    marked: dict[str, int] = {}
    for page in doc:
        dots = [
            d["rect"]
            for d in page.get_drawings()
            if d.get("fill") is not None
            and 1.2 <= d["rect"].width <= 7
            and 1.2 <= d["rect"].height <= 7
            and abs(d["rect"].width - d["rect"].height) <= 1.5
        ]
        if not dots:
            continue
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                x0, y0, _, y1 = line["bbox"]
                if not any(0 <= x0 - r.x1 <= 14 and y0 <= (r.y0 + r.y1) / 2 <= y1 for r in dots):
                    continue
                key = _squash("".join(s["text"] for s in line["spans"]))
                if key:
                    marked[key] = marked.get(key, 0) + 1
    if not marked:
        return text
    out = []
    for line in text.split("\n"):
        key = _squash(line)
        if key and marked.get(key) and not _BULLET_LINE.match(line):
            marked[key] -= 1
            line = "• " + line.strip()
        out.append(line)
    return "\n".join(out)


def _iter_block_items(doc: DocxDocument):
    """Yield paragraphs and tables in document order (python-docx lists them separately)."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _paragraph_text(p: Paragraph) -> str:
    text = p.text.strip()
    if not text:
        return ""
    style = (p.style.name or "").lower() if p.style is not None else ""
    is_list = "list" in style or p._p.pPr is not None and p._p.pPr.numPr is not None
    return f"- {text}" if is_list and not _BULLET_LINE.match(text) else text


def _extract_docx(data: bytes) -> str:
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise ResumeParseError(f"Could not read DOCX: {exc}") from exc

    lines: list[str] = []
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            lines.append(_paragraph_text(block))
        else:  # Tables are often used for two-column layouts: read cell by cell.
            for row in block.rows:
                seen: set[int] = set()
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    lines.extend(_paragraph_text(p) for p in cell.paragraphs)
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """Clean extraction artifacts while preserving line structure."""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    # Word's Symbol/Wingdings bullets come out of PDFs as private-use code points (U+F0B7...)
    # or as DEL (U+007F). At the start of a line they are bullets; anywhere else they are icon
    # glyphs (Font Awesome phone/mail icons) with no readable text, so they are dropped.
    text = _SYMBOL_BULLET_LINE.sub("• ", text)
    text = _PRIVATE_GLYPH.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Re-join words hyphenated across line breaks ("develop-\nment").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    out: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[  ]{2,}", " ", line).strip()
        if _BULLET_LINE.match(line):
            line = "- " + _BULLET_LINE.sub("", line, count=1).strip()
        elif re.match(rf"^[{re.escape(_BULLETS)}](?=\w)", line) and line[0] not in "-*":
            line = "- " + line[1:].strip()
        out.append(line)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
