"""Split resume text into typed sections using heading detection.

This is deliberately deterministic: it is instant, free, testable, and handles the vast
majority of resumes. Everything above the first heading becomes the ``header`` section
(name and contact details).
"""

from __future__ import annotations

import re

from app.models.resume import Resume, ResumeSection, SectionKind

# Heading keyword -> section kind. Matched against the whole (normalized) heading line.
HEADING_KINDS: dict[str, SectionKind] = {}
_HEADINGS: dict[SectionKind, list[str]] = {
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "professional profile",
        "about",
        "about me",
        "objective",
        "career objective",
        "overview",
        "executive summary",
        "professional highlights",
        "career highlights",
        "key highlights",
        "highlights",
        "summary of qualifications",
        "qualifications summary",
        "resumen",
        "perfil",
        "perfil profesional",
        "resumen profesional",
        "acerca de mi",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "relevant experience",
        "internships",
        "internship experience",
        "experiencia",
        "experiencia laboral",
        "experiencia profesional",
    ],
    "education": [
        "education",
        "academic background",
        "education and training",
        "academics",
        "relevant coursework",
        "coursework",
        "educación",
        "educacion",
        "formación",
        "formacion",
        "formación académica",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "core competencies",
        "competencies",
        "technologies",
        "tech stack",
        "tools",
        "skills and tools",
        "habilidades",
        "competencias",
        "aptitudes",
        "habilidades técnicas",
    ],
    "projects": [
        "projects",
        "personal projects",
        "selected projects",
        "key projects",
        "academic projects",
        "proyectos",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses",
        "licenses and certifications",
        "certifications and licenses",
        "certificaciones",
    ],
    "awards": [
        "awards",
        "honors",
        "honours",
        "achievements",
        "awards and honors",
        "premios",
        "logros",
    ],
    "publications": ["publications", "papers", "research", "publicaciones"],
    "languages": ["languages", "idiomas"],
    "volunteer": ["volunteer", "volunteering", "volunteer experience", "voluntariado"],
    "interests": ["interests", "hobbies", "intereses"],
    "other": [
        "additional information",
        "personal details",
        "personal information",
        "personal data",
        "extracurricular activities",
        "extra-curricular activities",
        "positions of responsibility",
        "activities",
        "leadership",
        "references",
        "información adicional",
    ],
}
for _kind, _names in _HEADINGS.items():
    for _name in _names:
        HEADING_KINDS[_name] = _kind

_MAX_HEADING_WORDS = 5
_PARTIAL_KEYWORDS: dict[str, SectionKind] = {
    "summary": "summary",
    "profile": "summary",
    "experience": "experience",
    "employment": "experience",
    "education": "education",
    "skills": "skills",
    "projects": "projects",
    "certifications": "certifications",
    "awards": "awards",
    "publications": "publications",
    "volunteer": "volunteer",
    "volunteering": "volunteer",
    "highlights": "summary",
}


def _normalize_heading(line: str) -> str:
    line = re.sub(r"^#+\s*", "", line)  # Markdown headings
    line = line.strip().strip(":").strip()
    line = line.replace("&", "and")
    return re.sub(r"\s+", " ", line).lower()


def classify_heading(line: str) -> SectionKind | None:
    """Return the section kind if ``line`` looks like a section heading."""
    stripped = line.strip()
    if not stripped or stripped.startswith("- ") or len(stripped) > 48:
        return None
    norm = _normalize_heading(stripped)
    if norm in HEADING_KINDS:
        return HEADING_KINDS[norm]
    # Heading-shaped lines (Markdown "##", ALL CAPS, "Title:") that contain a core section
    # keyword, e.g. "RELEVANT WORK EXPERIENCE". ALL-CAPS lines without a keyword are NOT
    # headings: names ("ALEX MORGAN") and job titles ("SENIOR ENGINEER") are often in caps.
    words = norm.split()
    is_markdown = stripped.startswith("#")
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    is_caps = len(letters) >= 4 and letters.isupper() and not re.search(r"[@|/\d]", stripped)
    looks_like_heading = is_markdown or is_caps or stripped.endswith(":")
    if looks_like_heading and 0 < len(words) <= _MAX_HEADING_WORDS:
        for word in words:
            if word in _PARTIAL_KEYWORDS:
                return _PARTIAL_KEYWORDS[word]
        if is_markdown:
            return "other"
    return None


# "Jan 2022 - Present", "2019 - 2021", "May 2025 – Jan 2026", "01/2022 - heute" ...
DATE_RANGE = re.compile(
    r"(?:\b(?:19|20)\d{2}|\b\d{1,2}/(?:19|20)\d{2})\s*(?:-|–|—|to|a|bis)\s*"
    r"(?:present|current|now|today|actualidad|presente|heute"
    r"|[A-Za-zÀ-ÿ]+\.?\s*(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2}|(?:19|20)\d{2})",
    re.IGNORECASE,
)
_WRAP_MIN_CHARS = 50


def _is_wrapped_continuation(previous: str, line: str) -> bool:
    """True when ``line`` is the second visual line of the bullet ``previous``.

    PDF text extraction returns one line per *visual* line, so a long bullet arrives split
    in two. A role line ("Engineer, Acme  May 2025 - Present"), a project title
    ("Name | Stack") or a blank line must never be glued onto the bullet above it.
    """
    if not previous.startswith("- ") or not line or line.startswith("- "):
        return False
    if DATE_RANGE.search(line) or " | " in line or len(previous) < _WRAP_MIN_CHARS:
        return False
    if line[0].islower():
        return True
    last_word = previous.split()[-1]
    # The bullet stopped mid-phrase: trailing comma or a lowercase word ("... architecture to").
    return previous.endswith(",") or (last_word.isalpha() and last_word.islower())


_SKILL_LABEL = re.compile(r"^[\w][\w &/+.()-]{0,40}:\s")


def join_wrapped_lines(lines: list[str], kind: str = "other") -> list[str]:
    """Re-join bullets (and labelled skill lines) that a PDF split across visual lines."""
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        wrapped_skill = (
            kind == "skills"
            and out
            and line
            and _SKILL_LABEL.match(out[-1])
            and not _SKILL_LABEL.match(line)
            and not line.startswith("- ")
        )
        if out and (wrapped_skill or _is_wrapped_continuation(out[-1], line)):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def _slug(kind: str, used: set[str]) -> str:
    base, n = kind, 2
    candidate = base
    while candidate in used:
        candidate = f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


def _title_case(line: str) -> str:
    title = re.sub(r"^#+\s*", "", line).strip().strip(":")
    return title.title() if title.isupper() else title


def sectionize(text: str) -> Resume:
    lines = text.split("\n")
    sections: list[ResumeSection] = []
    used: set[str] = set()

    current_title, current_kind, buffer = "Header", "header", []

    def flush() -> None:
        content = "\n".join(join_wrapped_lines(buffer, current_kind)).strip()
        if not content and current_kind == "header":
            return
        sid = _slug(current_kind, used)
        sections.append(
            ResumeSection(id=sid, title=current_title, kind=current_kind, content=content)
        )

    for line in lines:
        kind = classify_heading(line)
        if kind is not None:
            flush()
            current_title, current_kind, buffer = _title_case(line), kind, []
        else:
            buffer.append(line)
    flush()

    # Drop empty non-header sections created by back-to-back headings.
    sections = [s for s in sections if s.content or s.kind == "header"]
    if len(sections) <= 1:
        only = sections[0].content if sections else text.strip()
        sections = [ResumeSection(id="resume", title="Resume", kind="other", content=only)]
    return Resume(sections=sections, language=_guess_language(text))


_ES_MARKERS = {
    "experiencia",
    "educación",
    "habilidades",
    "desarrollé",
    "responsable",
    "años",
    "empresa",
}


def _guess_language(text: str) -> str:
    words = set(re.findall(r"[a-záéíóúñ]+", text.lower()))
    return "es" if len(words & _ES_MARKERS) >= 2 else "en"


def parse_resume(filename: str, data: bytes) -> Resume:
    """Extract and sectionize an uploaded resume file."""
    from app.parsing.extract import extract_text

    return sectionize(extract_text(filename, data))


def repair_resume(resume: Resume) -> Resume:
    """Re-join PDF-wrapped lines in an already-parsed resume (e.g. stored by an older parser)."""
    repaired = resume.model_copy(deep=True)
    for section in repaired.sections:
        section.content = "\n".join(join_wrapped_lines(section.content.split("\n"), section.kind))
    return repaired
