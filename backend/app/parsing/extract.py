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
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except PdfReadError as exc:
        raise ResumeParseError(f"Could not read PDF: {exc}") from exc


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
