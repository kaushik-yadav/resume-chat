"""Find the exact resume lines a request refers to, for precise (line-level) edits.

Users point at lines in three ways:
* selecting text in the resume view (sent as ``selection``),
* pasting the lines into the message ("Optimize these lines: <line> <line>"),
* quoting them ("rewrite "Built a ticket triage agent..." to sound more senior").

Matching is fuzzy (the pasted copy may differ in punctuation or a word) but strict enough that
an instruction sentence or a pasted job description does not match resume lines.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from app.models.resume import Resume

MIN_CHARS = 18  # shorter fragments are too ambiguous to target
MATCH_RATIO = 0.82
MAX_TARGETS = 8  # more than this is a pasted block (resume/JD), not a targeted edit
QUOTED = re.compile(rf"[\"“”']([^\"“”]{{{MIN_CHARS},}})[\"“”']")
BULLET = re.compile(r"^\s*(?:[-*•▪◦–]|\d+[.)])\s+")


@dataclass(frozen=True)
class LineTarget:
    section_id: str
    line_index: int
    text: str  # the line exactly as it currently appears in the resume


def _norm(text: str) -> str:
    text = BULLET.sub("", text.strip())
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip(" .;,:")


def _candidates(message: str, selection: list[str] | None) -> list[str]:
    found = [s for s in (selection or []) if s.strip()]
    for line in message.split("\n"):
        found.append(line)
    found += QUOTED.findall(message)
    return [c for c in found if len(_norm(c)) >= MIN_CHARS]


def _score(candidate: str, line: str) -> float:
    if candidate in line:  # a partial selection inside the line targets the whole line
        return 1.0
    return difflib.SequenceMatcher(a=candidate, b=line, autojunk=False).ratio()


def find_line_targets(
    message: str, resume: Resume, selection: list[str] | None = None
) -> list[LineTarget]:
    lines = [
        (section.id, index, raw, _norm(raw))
        for section in resume.sections
        if section.kind != "header"
        for index, raw in enumerate(section.content.split("\n"))
        if len(_norm(raw)) >= MIN_CHARS
    ]
    if not lines:
        return []
    targets: dict[tuple[str, int], LineTarget] = {}
    for candidate in map(_norm, _candidates(message, selection)):
        best = max(lines, key=lambda item: _score(candidate, item[3]))
        if _score(candidate, best[3]) >= MATCH_RATIO:
            targets[(best[0], best[1])] = LineTarget(best[0], best[1], best[2])
    if len(targets) > MAX_TARGETS:
        return []
    order = {section.id: i for i, section in enumerate(resume.sections)}
    return sorted(targets.values(), key=lambda t: (order[t.section_id], t.line_index))
