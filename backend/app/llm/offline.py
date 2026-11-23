"""Offline (key-less) LLM: deterministic heuristic responders.

Used by the test-suite and by ``LLM_PROVIDER=fake`` so the whole system - routing, agents,
versioning, streaming UI - can be exercised without an API key. Quality is intentionally
basic; real runs use Gemini/OpenAI.
"""

from __future__ import annotations

import re

from app.llm.client import FakeLLM

SECTION_RX = re.compile(
    r'<section id="(?P<id>[^"]+)" title="(?P<title>[^"]*)" kind="(?P<kind>[^"]*)">\n'
    r"(?P<content>.*?)\n</section>",
    re.DOTALL,
)
WEAK = [
    (r"^- Worked on ", "- Developed "),
    (r"^- Helped (to )?", "- Contributed to "),
    (r"^- Responsible for ", "- Owned "),
    (r"^- Participated in ", "- Took part in "),
    (r"^- Did ", "- Performed "),
    (r"^- Wrote ", "- Engineered "),
]
TECH_TERMS = [
    "python", "sql", "pandas", "scikit-learn", "bigquery", "snowflake", "airflow", "docker",
    "aws", "a/b tests", "a/b testing", "machine learning", "pytorch", "xgboost", "lightgbm",
    "statistics", "statistical inference", "experimentation", "fraud detection", "apis",
    "data pipelines", "fastapi", "react", "kubernetes", "go", "java", "spark", "tableau",
]  # fmt: skip
HEADINGS_ES = {
    "summary": "Perfil profesional", "experience": "Experiencia profesional",
    "education": "Formación académica", "skills": "Habilidades", "projects": "Proyectos",
    "certifications": "Certificaciones", "languages": "Idiomas",
}  # fmt: skip


def sections_in(text: str) -> list[dict]:
    return [m.groupdict() for m in SECTION_RX.finditer(text)]


def _block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
    return m.group(1) if m else ""


def _polish(content: str) -> str:
    out = []
    for line in content.split("\n"):
        for pattern, repl in WEAK:
            new = re.sub(pattern, repl, line, flags=re.IGNORECASE)
            if new != line:
                line = new
                break
        out.append(line)
    return "\n".join(out)


def edit_plan(system: str, user: str) -> dict:
    edits = []
    for s in sections_in(_block(user, "edit")):
        new = _polish(s["content"])
        if new != s["content"]:
            edits.append(
                {
                    "section_id": s["id"],
                    "new_content": new,
                    "rationale": "Replaced weak bullet openers with action verbs.",
                }
            )
    return {
        "reasoning": "Offline mode: strengthened bullet openers without changing facts.",
        "edits": edits,
    }


def job_requirements(system: str, user: str) -> dict:
    jd = _block(user, "job_description")
    title = next((ln.strip() for ln in jd.splitlines() if ln.strip()), "Role")
    company = title.split(" - ")[-1].split("(")[0].strip() if " - " in title else ""
    importance = "required"
    reqs = []
    for line in jd.splitlines():
        low = line.strip().lower()
        if low.startswith(("nice to have", "preferred", "bonus")):
            importance = "preferred"
        if line.strip().startswith("- ") and len(reqs) < 12:
            reqs.append({"name": line.strip()[2:80], "category": "skill", "importance": importance})
    keywords = [t for t in TECH_TERMS if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", jd.lower())]
    return {
        "job_title": title.split(" - ")[0][:80],
        "company": company,
        "seniority": "",
        "requirements": reqs,
        "keywords": keywords,
    }


def match_assessment(system: str, user: str) -> dict:
    matches = []
    for line in user.splitlines():
        m = re.match(r"- (.+?) \[(required|preferred), \w+\]: (.*)", line)
        if not m:
            continue
        name, evidence = m.group(1), m.group(3)
        words = {w for w in re.findall(r"[a-z]{4,}", name.lower())}
        overlap = words & set(re.findall(r"[a-z]{4,}", evidence.lower()))
        status = (
            "missing"
            if evidence == "(no close match)"
            else ("met" if len(overlap) >= 2 else "partial")
        )
        matches.append(
            {
                "name": name,
                "status": status,
                "evidence": "" if status == "missing" else evidence[:160],
            }
        )
    return {
        "matches": matches,
        "strengths": ["Relevant hands-on engineering experience"],
        "recommendations": ["Add measurable outcomes to the most relevant bullets"],
    }


def company_profile(system: str, user: str) -> dict:
    company = re.search(r"Company: (.+)", user).group(1).strip()
    return {
        "company": company,
        "summary": f"{company} is the target employer (offline mode: no live profile).",
        "values": ["Customer focus", "Technical excellence", "Ownership"],
        "hiring_signals": ["Measurable impact", "Collaboration", "Strong fundamentals"],
        "emphasis": ["impact at scale", "ownership"],
        "vocabulary": ["impact", "scale", "users"],
        "grounded_in_sources": "(no live research available)" not in user,
    }


def translation(system: str, user: str) -> dict:
    sections = []
    for s in sections_in(user):
        sections.append(
            {
                "section_id": s["id"],
                "title": HEADINGS_ES.get(s["kind"], s["title"]),
                "content": s["content"],
            }
        )
    return {
        "reasoning": "Offline mode: localized headings only; full translation needs a live model.",
        "sections": sections,
        "localization_notes": ["Offline mode cannot translate body text."],
    }


def offline_llm() -> FakeLLM:
    llm = FakeLLM(
        {
            "EditPlan": edit_plan,
            "JobRequirements": job_requirements,
            "MatchAssessment": match_assessment,
            "CompanyProfile": company_profile,
            "Translation": translation,
        }
    )
    try:  # router/conversation responders are registered by the orchestration package
        from app.orchestration.offline import register

        register(llm)
    except ImportError:
        pass
    return llm
