"""Section Enhancement Agent: stronger verbs, quantified impact, consistent style."""

from __future__ import annotations

from app.agents.base import (
    GROUND_RULES,
    Agent,
    AgentContext,
    AgentResult,
    EditPlan,
    default_targets,
    memory_block,
    render_sections,
)
from app.agents.guardrails import style_issues

SYSTEM = f"""You are the Section Enhancement Agent of a resume assistant.
You rewrite specific resume sections so they read like a top-tier resume:
- start bullets with strong, varied action verbs (past tense for past roles, present for the current one)
- make impact explicit: what changed, for whom, by how much
- tighten wording; one or two lines per bullet; remove filler
- keep style consistent with the rest of the resume (tense, date format, punctuation)
- when a target role is known, emphasize the most relevant experience and terminology
Only edit the sections listed under <edit>. Use <resume_outline> only for consistency.

{GROUND_RULES}"""


class SectionEnhancementAgent(Agent):
    name = "section_enhancement"
    title = "Section Enhancement Agent"
    description = (
        "Improves specific sections (summary, experience, skills, projects...) on request: "
        "action verbs, quantified impact, clarity, consistency, or tailoring a section to a "
        "target role (e.g. 'make my skills section more relevant for a data scientist role')."
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        resume = ctx.resume
        targets = [t for t in ctx.target_sections if resume.get(t)] or default_targets(
            resume, ("summary", "experience", "projects")
        )
        titles = ", ".join(resume.get(t).title for t in targets)

        await ctx.progress(f"Reviewing {titles} for weak verbs and missing impact")
        issues = style_issues(resume, targets)
        guidance = await self.in_thread(
            self.deps.vectors.search_knowledge,
            f"{ctx.instruction} action verbs quantify impact consistency",
            3,
        )

        await ctx.progress(f"Rewriting {titles}")
        user = "\n\n".join(
            part
            for part in [
                f"Request: {ctx.instruction}",
                memory_block(ctx.memory),
                "Detected issues to fix:\n" + "\n".join(f"- {i}" for i in issues) if issues else "",
                "Writing guidance:\n" + "\n\n".join(h.text for h in guidance),
                f"<resume_outline>\n{resume.outline()}\n</resume_outline>",
                f"<edit>\n{render_sections(resume, targets)}\n</edit>",
                "Return an edit for each section you changed, with the full new content.",
            ]
            if part
        )
        plan = await self.deps.llm.structured(EditPlan, SYSTEM, user, name="enhance_sections")
        updated, applied, warnings = self.apply_edits(resume, plan, targets)

        remaining = style_issues(updated, targets)
        return AgentResult(
            agent=self.name,
            title=self.title,
            reasoning=plan.reasoning,
            resume=updated,
            edits=applied,
            warnings=warnings,
            insights={
                "sections_reviewed": [resume.get(t).title for t in targets],
                "issues_found": issues,
                "issues_remaining": remaining,
                "guidance_used": [h.metadata.get("title") for h in guidance],
            },
        )
