"""Export a resume version as Markdown or DOCX."""

from __future__ import annotations

import io
import re

from docx import Document
from docx.shared import Pt

from app.models.resume import Resume

INLINE = re.compile(r"\*\*(.+?)\*\*")


def _add_runs(paragraph, text: str) -> None:
    """Render **bold** spans; everything else as plain text."""
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])
        paragraph.add_run(m.group(1)).bold = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def to_docx(resume: Resume) -> bytes:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    for section in resume.sections:
        lines = [ln for ln in section.content.split("\n") if ln.strip()]
        if section.kind == "header":
            if lines:
                doc.add_heading(lines[0].strip(), level=0)
                for line in lines[1:]:
                    doc.add_paragraph(line.strip())
            continue
        doc.add_heading(section.title, level=1)
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                _add_runs(doc.add_paragraph(style="List Bullet"), stripped[2:])
            else:
                _add_runs(doc.add_paragraph(), stripped)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_markdown(resume: Resume) -> bytes:
    return resume.to_markdown().encode("utf-8")
