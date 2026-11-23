"""Suggested actions ("presets") offered as one-click chips in the UI.

A preset already says exactly what to do, so choosing one skips intent classification
entirely: the plan is built here, deterministically, in ~0 ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.resume import Resume
from app.orchestration.router import PlanStep, RoutingDecision


@dataclass(frozen=True)
class Preset:
    id: str
    label: str  # text shown on the chip and sent as the user message
    agent: str
    instruction: str
    target_kinds: tuple[str, ...] = ()
    slots: dict[str, str] = field(default_factory=dict)
    needs_job_description: bool = False


PRESETS: dict[str, Preset] = {
    p.id: p
    for p in [
        Preset(
            id="optimize_google",
            label="Optimize my resume for Google",
            agent="company_research",
            instruction="Research Google's culture, values and hiring signals and align the "
            "summary, experience and skills with them.",
            slots={"target_company": "Google"},
        ),
        Preset(
            id="translate_es_mx",
            label="Translate this to Spanish for the Mexican market",
            agent="translation",
            instruction="Translate the resume into Spanish and localize it for the Mexican job market.",
            slots={"target_locale": "es-MX"},
        ),
        Preset(
            id="match_jd_experience",
            label="Update my experience section to better match this job description",
            agent="jd_matching",
            instruction="Restructure the experience section to highlight the experience most "
            "relevant to the job description.",
            target_kinds=("experience",),
            needs_job_description=True,
        ),
        Preset(
            id="skills_data_scientist",
            label="Make my skills section more relevant for a data scientist role",
            agent="section_enhancement",
            instruction="Reorganize and reword the skills section for a Data Scientist role, "
            "leading with the most relevant skills the candidate actually has.",
            target_kinds=("skills",),
            slots={"target_role": "Data Scientist"},
        ),
    ]
}


def preset_decision(preset_id: str, resume: Resume) -> RoutingDecision | None:
    preset = PRESETS.get(preset_id)
    if preset is None:
        return None
    targets = [s.id for s in resume.sections if s.kind in preset.target_kinds]
    return RoutingDecision(
        intent="edit",
        reasoning=f"Suggested action '{preset.label}' - plan is predefined, no classification needed.",
        steps=[
            PlanStep(agent=preset.agent, instruction=preset.instruction, target_sections=targets)
        ],
        **preset.slots,
    )


def list_presets() -> list[dict]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "agent": p.agent,
            "needs_job_description": p.needs_job_description,
        }
        for p in PRESETS.values()
    ]
