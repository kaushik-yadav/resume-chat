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
        "summary", "professional summary", "career summary", "profile", "professional profile",
        "about", "about me", "objective", "career objective", "overview", "executive summary",
        "resumen", "perfil", "perfil profesional", "resumen profesional", "acerca de mi",
    ],
    "experience": [
        "experience", "work experience", "professional experience", "employment",
        "employment history", "work history", "career history", "relevant experience",
        "experiencia", "experiencia laboral", "experiencia profesional",
    ],
    "education": [
        "education", "academic background", "education and training", "academics",
        "educación", "educacion", "formación", "formacion", "formación académica",
    ],
    "skills": [
        "skills", "technical skills", "core skills", "key skills", "core competencies",
        "competencies", "technologies", "tech stack", "tools", "skills and tools",
        "habilidades", "competencias", "aptitudes", "habilidades técnicas",
    ],
    "projects": [
        "projects", "personal projects", "selected projects", "key projects",
        "academic projects", "proyectos",
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "licenses and certifications",
        "certifications and licenses", "certificaciones",
    ],
    "awards": ["awards", "honors", "honours", "achievements", "awards and honors", "premios", "logros"],
    "publications": ["publications", "papers", "research", "publicaciones"],
    "languages": ["languages", "idiomas"],
    "volunteer": ["volunteer", "volunteering", "volunteer experience", "voluntariado"],
    "interests": ["interests", "hobbies", "intereses"],
    "other": ["additional information", "activities", "leadership", "references", "información adicional"],
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
        content = "\n".join(buffer).strip()
        if not content and current_kind == "header":
            return
        sid = _slug(current_kind, used)
        sections.append(ResumeSection(id=sid, title=current_title, kind=current_kind, content=content))

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


_ES_MARKERS = {"experiencia", "educación", "habilidades", "desarrollé", "responsable", "años", "empresa"}


def _guess_language(text: str) -> str:
    words = set(re.findall(r"[a-záéíóúñ]+", text.lower()))
    return "es" if len(words & _ES_MARKERS) >= 2 else "en"


def parse_resume(filename: str, data: bytes) -> Resume:
    """Extract and sectionize an uploaded resume file."""
    from app.parsing.extract import extract_text

    return sectionize(extract_text(filename, data))
