"""Export a resume version as PDF, DOCX or Markdown."""

from __future__ import annotations

import io
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from docx import Document
from docx.shared import Pt

from app.models.conversation import ResumeVersion
from app.models.resume import Resume
from app.services.layout_export import LayoutUnavailable, edit_docx, edit_pdf
from app.services.pdf_export import to_pdf
from app.storage.base import Repository

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


MEDIA = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "md": "text/markdown; charset=utf-8",
}
RENDERERS = {"pdf": to_pdf, "docx": to_docx, "md": to_markdown}

Layout = Literal["original", "clean"]


@dataclass(frozen=True)
class Export:
    body: bytes
    media: str
    layout: Layout  # what was produced
    note: str = ""  # why the original layout could not be kept (empty when it was)


# Versions are immutable, so a rendered file can be served again without re-rendering.
_CACHE: OrderedDict[tuple[str, str, str, str], Export] = OrderedDict()
_CACHE_SIZE = 64
logger = logging.getLogger(__name__)
_lock = threading.Lock()


def _upload_of(repo: Repository, conversation_id: str, version: ResumeVersion) -> ResumeVersion | None:
    """The upload version this version descends from (its original file is the layout)."""
    current: ResumeVersion | None = version
    for _ in range(500):
        if current is None or current.source == "upload":
            return current
        if not current.parent_id:
            return None
        current = repo.get_version(conversation_id, current.parent_id)
    return None


def _render(repo: Repository | None, conversation_id: str, version: ResumeVersion, fmt: str, layout: Layout) -> Export:
    if fmt == "md" or layout == "clean" or repo is None:
        return Export(RENDERERS[fmt](version.resume), MEDIA[fmt], "clean")
    note = "the original file is not stored (uploaded before original-layout export existed)"
    base = _upload_of(repo, conversation_id, version)
    original = repo.get_original(conversation_id, base.id) if base else None
    if base is not None and original is not None:
        name, data = original
        kind = name.lower().rsplit(".", 1)[-1]
        try:
            if fmt == "pdf" and kind == "pdf":
                return Export(edit_pdf(data, base.resume, version.resume), MEDIA[fmt], "original")
            if fmt == "docx" and kind == "docx":
                return Export(edit_docx(data, base.resume, version.resume), MEDIA[fmt], "original")
            note = f"the original is a {kind.upper()} file; its layout is kept in the {kind.upper()} download"
        except LayoutUnavailable as exc:
            note = str(exc)
        except Exception as exc:  # never fail a download: fall back to the clean layout
            logger.exception("Original-layout export failed for version %s", version.id)
            note = f"the original file could not be edited ({type(exc).__name__})"
    return Export(RENDERERS[fmt](version.resume), MEDIA[fmt], "clean", note)


def render_export(
    conversation_id: str,
    version: ResumeVersion,
    fmt: str,
    repo: Repository | None = None,
    layout: Layout = "original",
) -> Export:
    """PDF/DOCX keep the uploaded file's layout when possible, else the clean layout."""
    key = (conversation_id, version.id, fmt, layout if fmt != "md" else "clean")
    with _lock:
        found = _CACHE.get(key)
        if found is not None:
            _CACHE.move_to_end(key)
            return found
    result = _render(repo, conversation_id, version, fmt, layout)
    with _lock:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return result


def prerender(conversation_id: str, version: ResumeVersion, fmt: str = "pdf", repo: Repository | None = None) -> None:
    """Render ahead of the first download (called in the background after a commit)."""
    try:
        render_export(conversation_id, version, fmt, repo)
    except Exception:  # a failed warm-up only means the download renders on demand
        logger.exception("Pre-rendering %s for version %s failed", fmt, version.id)
