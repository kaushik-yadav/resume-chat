"""Resume domain model.

A resume is an ordered list of sections. Section content is kept as lightweight Markdown
(bullets as ``- ``), which is what the agents read and write. Keeping content as text instead
of a deeply nested schema makes the model robust to the huge variety of real resume layouts,
and lets agents edit exactly one section without touching the rest of the document.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

SectionKind = Literal[
    "header",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "awards",
    "publications",
    "languages",
    "volunteer",
    "interests",
    "other",
]


def markdown_block(content: str) -> str:
    """Section content -> Markdown that renders line for line.

    Plain Markdown would glue a role line onto the bullet above it ("lazy continuation") and
    merge consecutive plain lines into one paragraph. A blank line separates bullet runs from
    plain lines, and plain lines get hard breaks.
    """

    def is_bullet(line: str) -> bool:
        return line.lstrip().startswith(("- ", "* "))

    lines = [raw.rstrip() for raw in content.strip().split("\n")]
    out: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            out.append("")
            continue
        if out and out[-1] and is_bullet(out[-1]) != is_bullet(line):
            out.append("")
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        hard_break = not is_bullet(line) and nxt.strip() and not is_bullet(nxt)
        out.append(f"{line}  " if hard_break else line)
    return "\n".join(out).strip()


class ResumeSection(BaseModel):
    id: str = Field(description="Stable identifier, e.g. 'experience' or 'projects-2'.")
    title: str = Field(description="Heading as shown on the resume.")
    kind: SectionKind = "other"
    content: str = ""


class Resume(BaseModel):
    sections: list[ResumeSection] = Field(default_factory=list)
    language: str = Field(default="en", description="ISO 639-1 code of the resume's language.")
    locale: str | None = Field(default=None, description="Target market, e.g. 'es-MX'.")

    # ------------------------------------------------------------------ lookup
    def get(self, section_id: str) -> ResumeSection | None:
        return next((s for s in self.sections if s.id == section_id), None)

    def find_kind(self, kind: str) -> list[ResumeSection]:
        return [s for s in self.sections if s.kind == kind]

    @property
    def section_ids(self) -> list[str]:
        return [s.id for s in self.sections]

    # ----------------------------------------------------------------- editing
    def with_section_content(
        self, section_id: str, content: str, title: str | None = None
    ) -> Resume:
        """Return a copy with one section's content (and optionally title) replaced."""
        copy = self.model_copy(deep=True)
        section = copy.get(section_id)
        if section is None:
            raise KeyError(section_id)
        section.content = content.strip()
        if title:
            section.title = title.strip()
        return copy

    # ------------------------------------------------------------- rendering
    def outline(self, preview_chars: int = 90) -> str:
        """Compact table of contents used in prompts so the LLM knows what exists."""
        lines = []
        for s in self.sections:
            first = " ".join(s.content.split())[:preview_chars]
            lines.append(f"- [{s.id}] {s.title} ({s.kind}): {first}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        parts = []
        for s in self.sections:
            if s.kind == "header":
                parts.append(markdown_block(s.content))
            else:
                parts.append(f"## {s.title}\n\n{markdown_block(s.content)}")
        return "\n\n".join(p for p in parts if p).strip() + "\n"

    def to_text(self) -> str:
        return re.sub(r"[*_`#]", "", self.to_markdown())

    def word_count(self) -> int:
        return len(self.to_text().split())
