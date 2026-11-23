"""Section-level diffs between two resume versions."""

from __future__ import annotations

import difflib

from app.models.conversation import SectionChange
from app.models.resume import Resume


def _word_delta(before: str, after: str) -> tuple[int, int]:
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=before.split(), b=after.split(), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def diff_resumes(old: Resume | None, new: Resume) -> list[SectionChange]:
    """Return per-section changes from ``old`` to ``new`` (in ``new``'s section order)."""
    if old is None:
        return []
    changes: list[SectionChange] = []
    old_by_id = {s.id: s for s in old.sections}
    new_ids = set()
    for section in new.sections:
        new_ids.add(section.id)
        before = old_by_id.get(section.id)
        if before is None:
            added, _ = _word_delta("", section.content)
            changes.append(
                SectionChange(
                    section_id=section.id,
                    title=section.title,
                    change_type="added",
                    after=section.content,
                    words_added=added,
                )
            )
        elif before.content.strip() != section.content.strip() or before.title != section.title:
            added, removed = _word_delta(before.content, section.content)
            changes.append(
                SectionChange(
                    section_id=section.id,
                    title=section.title,
                    change_type="modified",
                    before=before.content,
                    after=section.content,
                    words_added=added,
                    words_removed=removed,
                )
            )
    for section in old.sections:
        if section.id not in new_ids:
            _, removed = _word_delta(section.content, "")
            changes.append(
                SectionChange(
                    section_id=section.id,
                    title=section.title,
                    change_type="removed",
                    before=section.content,
                    words_removed=removed,
                )
            )
    return changes


def unified_diff(before: str, after: str, context: int = 1) -> str:
    """Compact line diff, handy for logs and for showing changes to the LLM."""
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=context)
    return "\n".join(line for line in lines if not line.startswith(("---", "+++")))
