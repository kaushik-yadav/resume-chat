"""Deterministic checks that run on every agent edit (no LLM involved).

They catch the failure modes that matter most for a resume tool: invented numbers,
accidentally deleted content, and style inconsistencies across sections.
"""

from __future__ import annotations

import re
from collections import Counter

from app.models.resume import Resume, ResumeSection

PLACEHOLDER = re.compile(r"\[[^\]]*\]")
# Numbers with optional currency/percent/multiplier, e.g. "40%", "$1.2M", "3x", "10,000".
NUMBER = re.compile(r"[$€£₹]?\d[\d,.]*\s?(?:%|[kKmMbB]\b|x\b)?")
WEAK_OPENERS = re.compile(
    r"^-\s*(worked on|helped( with| to)?|responsible for|participated in|did|was involved in|"
    r"duties included|tasked with|in charge of)\b",
    re.IGNORECASE | re.MULTILINE,
)
FIRST_PERSON = re.compile(r"\b(I|my|me)\b")
DATE_STYLES = {
    "Mon YYYY": re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.? \d{4}\b"
    ),
    "MM/YYYY": re.compile(r"\b\d{1,2}/\d{4}\b"),
    "Month YYYY": re.compile(
        r"\b(?:January|February|March|April|June|July|August|September|October|November|December) \d{4}\b"
    ),
}


NOT_METRICS = re.compile(r"(?<!\d)24/7(?!\d)")  # a phrase, not a metric


def _numbers(text: str) -> Counter[str]:
    text = NOT_METRICS.sub(" ", PLACEHOLDER.sub(" ", text))
    found = (m.group(0).strip().rstrip(".,") for m in NUMBER.finditer(text))
    return Counter(n for n in found if n)


def new_numbers(original: str, edited: str) -> list[str]:
    """Numbers present in ``edited`` that do not appear anywhere in ``original``."""
    allowed = set(_numbers(original))
    return sorted({n for n in _numbers(edited) if n not in allowed})


def dropped_numbers(original: str, edited: str) -> list[str]:
    """Numbers (dates, metrics) that disappeared - used to validate translations."""
    after = _numbers(edited)
    return sorted({n for n, c in _numbers(original).items() if after[n] < c})


# Named things: capitalised or mixed-case words and tech spellings ("BigQuery", "Node.js", "C++").
NAMED = re.compile(r"(?<![\w.+#])[A-Za-z][\w.+#-]*[\w+#]")


def new_terms(original: str, edited: str) -> list[str]:
    """Names (tools, companies, products) in ``edited`` that ``original`` lacks.

    Checked per selected line and per edited section: a skill listed elsewhere in the resume
    ("BigQuery" under Skills) does not mean this piece of work used it. The first word of each
    line and sentence is skipped, since a rewritten bullet starts with a capitalised verb.
    """
    known = {w.lower() for w in NAMED.findall(original)}
    found = []
    for line in edited.splitlines():
        for sentence in re.split(r"(?<=[.!?:;])\s+", re.sub(r"^\s*[-*•]\s*", "", line)):
            for w in NAMED.findall(sentence)[1:]:
                named = w[0].isupper() or any(c.isupper() for c in w[1:])
                if named and w.lower() not in known:
                    found.append(w)
    return list(dict.fromkeys(found))


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER.findall(text)


def check_edit(
    source_resume: Resume, before: ResumeSection, after_content: str, *, names: bool = True
) -> list[str]:
    """Warnings for one edited section, phrased for the end user."""
    warnings: list[str] = []
    invented = new_numbers(source_resume.to_text(), after_content)
    if invented:
        warnings.append(
            f"{before.title}: new figure(s) {', '.join(invented)} not found in your resume - "
            "please verify they are accurate before using this version."
        )
    added = new_terms(before.content, after_content) if names else []
    if added:
        warnings.append(
            f"{before.title}: {', '.join(added)} not in this section before - "
            "check it's true for this work before using this version."
        )
    before_words, after_words = len(before.content.split()), len(after_content.split())
    if before_words >= 20 and after_words < before_words * 0.5:
        warnings.append(
            f"{before.title}: shortened from {before_words} to {after_words} words - "
            "check that nothing important was removed."
        )
    return warnings


def style_issues(resume: Resume, section_ids: list[str] | None = None) -> list[str]:
    """Consistency/style findings, fed to the enhancement agent and shown in the UI."""
    issues: list[str] = []
    sections = [s for s in resume.sections if s.kind != "header"]
    if section_ids:
        sections = [s for s in sections if s.id in section_ids]
    for s in sections:
        weak = [m.group(1) for m in WEAK_OPENERS.finditer(s.content)]
        if weak:
            issues.append(f"{s.title}: weak bullet openers ({', '.join(sorted(set(weak)))})")
        if FIRST_PERSON.search(s.content) and s.kind != "summary":
            issues.append(f"{s.title}: first-person pronouns in bullets")
    styles = {name for name, rx in DATE_STYLES.items() if rx.search(resume.to_text())}
    if len(styles) > 1:
        issues.append(f"Mixed date formats across the resume ({', '.join(sorted(styles))})")
    return issues
