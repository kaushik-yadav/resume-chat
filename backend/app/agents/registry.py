"""Agent registry: the single place that knows which specialized agents exist."""

from __future__ import annotations

from app.agents.base import Agent, AgentDeps
from app.agents.company_research import CompanyResearchAgent
from app.agents.jd_matching import JDMatchingAgent
from app.agents.section_enhancement import SectionEnhancementAgent
from app.agents.translation import TranslationAgent
from app.config import Settings

AGENT_CLASSES: list[type[Agent]] = [
    CompanyResearchAgent,
    JDMatchingAgent,
    TranslationAgent,
    SectionEnhancementAgent,
]
AGENT_NAMES = [cls.name for cls in AGENT_CLASSES]


def build_agents(deps: AgentDeps, settings: Settings | None = None) -> dict[str, Agent]:
    ttl = settings.company_cache_ttl_hours if settings else 72
    agents: dict[str, Agent] = {}
    for cls in AGENT_CLASSES:
        agents[cls.name] = cls(deps, ttl) if cls is CompanyResearchAgent else cls(deps)
    return agents


def describe_agents() -> str:
    """Agent catalogue for the router prompt."""
    return "\n".join(f"- {cls.name}: {cls.description}" for cls in AGENT_CLASSES)
