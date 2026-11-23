"""Offline responders for the router and the conversational replies (see app.llm.offline)."""

from __future__ import annotations

import re

from app.llm.client import FakeLLM

LOCALES = {
    "mexic": "es-MX", "spanish": "es-ES", "spain": "es-ES", "german": "de-DE",
    "french": "fr-FR", "brazil": "pt-BR", "portuguese": "pt-BR", "japan": "ja-JP",
}  # fmt: skip
COMPANY_RX = re.compile(r"\b(?:for|at|to)\s+([A-Z][\w&.-]+(?:\s[A-Z][\w&.-]+)?)")
SECTION_WORDS = ["summary", "experience", "skills", "projects", "education", "certifications"]


def _user_message(user: str) -> str:
    m = re.search(r"<user_message>\n(.*?)\n</user_message>", user, re.DOTALL)
    return m.group(1) if m else user


def routing(system: str, user: str) -> dict:
    msg = _user_message(user)
    low = msg.lower()
    is_jd = len(msg.split()) >= 40 and any(
        w in low for w in ("requirements", "responsibilities", "qualifications")
    )
    base = {
        "reasoning": "Offline keyword routing.",
        "steps": [],
        "message_contains_job_description": is_jd,
    }
    if re.match(r"^\s*(undo|revert|go back)", low):
        return {**base, "intent": "undo"}
    if re.match(r"^\s*(redo|put it back)", low):
        return {**base, "intent": "redo"}
    if re.match(r"^\s*(hi|hello|hey|thanks|thank you)\b", low):
        return {**base, "intent": "chitchat"}

    sections = [s for s in SECTION_WORDS if s in low and f"[{s}]" in user]
    steps, company, locale, role = [], "", "", ""
    if is_jd or "job description" in low or "this job" in low:
        steps.append(
            {"agent": "jd_matching", "instruction": msg[:200], "target_sections": sections}
        )
    if "translate" in low or any(k in low for k in ("spanish", "german", "french", "japanese")):
        locale = next((v for k, v in LOCALES.items() if k in low), "")
    m = COMPANY_RX.search(msg)
    wants_company = "optimi" in low or "tailor" in low
    if wants_company and m and not any(k in m.group(1).lower() for k in LOCALES):
        company = m.group(1)
        steps.append(
            {
                "agent": "company_research",
                "instruction": f"Optimize for {company}",
                "target_sections": sections,
            }
        )
    role_m = re.search(r"for an? ([a-z ]+?) role", low)
    if role_m:
        role = role_m.group(1).title()
    if any(w in low for w in ("improve", "stronger", "enhance", "relevant", "rewrite", "punch")):
        steps.append(
            {"agent": "section_enhancement", "instruction": msg[:200], "target_sections": sections}
        )
    if locale:
        steps.append(
            {
                "agent": "translation",
                "instruction": f"Translate for {locale}",
                "target_sections": [],
            }
        )
    has_selection = "<selected_lines>" in user
    edit_verb = re.match(
        r"^\s*(make|rewrite|improve|optimi|shorten|tighten|fix|polish|reword)", low
    )
    if has_selection and edit_verb and not steps:
        steps.append(
            {"agent": "section_enhancement", "instruction": msg[:200], "target_sections": []}
        )
    if steps:
        return {
            **base,
            "intent": "edit",
            "steps": steps,
            "target_company": company,
            "target_locale": locale,
            "target_role": role,
        }
    return {**base, "intent": "question"}


def final_reply(system: str, user: str) -> str:
    agents = re.findall(r"^## (.+)$", user, re.MULTILINE)
    edits = re.findall(r"^- edited \*\*(.+?)\*\*", user, re.MULTILINE)
    warnings = re.findall(r"^WARNING: (.+)$", user, re.MULTILINE)
    text = f"Done - {', '.join(agents)} updated: {', '.join(dict.fromkeys(edits)) or 'nothing'}."
    if warnings:
        text += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in warnings)
    return text


def answer(system: str, user: str) -> str:
    return "Here's what I found in your resume (offline mode). Want me to improve a section?"


def summarize(system: str, user: str) -> str:
    return " ".join(re.findall(r"^(?:user|assistant): (.{0,80})", user, re.MULTILINE))[:600]


def register(llm: FakeLLM) -> None:
    llm.on("RoutingDecision", routing)
    llm.on("final_reply", final_reply)
    llm.on("answer_question", answer)
    llm.on("summarize_history", summarize)
