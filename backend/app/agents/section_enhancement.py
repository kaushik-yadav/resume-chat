"""Section Enhancement Agent: stronger verbs, quantified impact, consistent style.

Two modes:
* section mode - rewrite whole sections (e.g. "make my skills section fit a data scientist role")
* line mode - the user pointed at exact lines ("optimize these two lines ..."): the model
  returns one replacement per line id and the code swaps exactly those lines. Every other line
  of the resume stays byte-for-byte identical, whatever the model writes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import (
    GROUND_RULES,
    Agent,
    AgentContext,
    AgentResult,
    AppliedEdit,
    EditPlan,
    default_targets,
    memory_block,
    render_sections,
)
from app.agents.guardrails import check_edit, placeholders, style_issues

SYSTEM = f"""You are the Section Enhancement Agent of a resume assistant.
You rewrite specific resume sections so they read like a top-tier resume:
- start bullets with strong, varied action verbs (past tense for past roles, present for the current one)
- make impact explicit: what changed, for whom, by how much
- tighten wording; one or two lines per bullet; remove filler
- keep style consistent with the rest of the resume (tense, date format, punctuation)
- when a target role is known, emphasize the most relevant experience and terminology
Only edit the sections listed under <edit>. Use <resume_outline> only for consistency.

{GROUND_RULES}"""

LINE_SYSTEM = f"""You are the Section Enhancement Agent of a resume assistant.
The user selected specific lines of their resume. Rewrite ONLY those lines, following the
request. Return exactly one rewrite per line id listed under <lines>:
- keep each line's underlying facts; a bullet stays one bullet (one line, no line breaks)
- match the tense, voice and length of the surrounding bullets
- use <context> only to stay consistent - never rewrite lines that are not listed

{GROUND_RULES}"""


class LineRewrite(BaseModel):
    line_id: str = Field(description="Id of the selected line, e.g. L1")
    replacement: str = Field(description="The rewritten line (single line)")
    rationale: str = Field(description="One short sentence: what changed and why")


class LineEditPlan(BaseModel):
    reasoning: str = Field(description="1-2 sentences on the approach")
    rewrites: list[LineRewrite]


def _as_bullet_like(original: str, replacement: str) -> str:
    """Keep the original line's bullet marker and indentation; force a single line."""
    text = " ".join(replacement.split())
    stripped = original.lstrip()
    indent = original[: len(original) - len(stripped)]
    if stripped.startswith("- "):
        text = text.removeprefix("- ").removeprefix("• ")
        return f"{indent}- {text}"
    return indent + text.removeprefix("- ")


class SectionEnhancementAgent(Agent):
    name = "section_enhancement"
    title = "Section Enhancement Agent"
    description = (
        "Improves specific sections (summary, experience, skills, projects...) on request: "
        "action verbs, quantified impact, clarity, consistency, or tailoring a section to a "
        "target role (e.g. 'make my skills section more relevant for a data scientist role')."
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        if ctx.target_lines:
            return await self._edit_lines(ctx)
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

    # ------------------------------------------------------------------ line mode
    async def _edit_lines(self, ctx: AgentContext) -> AgentResult:
        resume = ctx.resume
        # Locate every selected line: (line id, section id, index within the section).
        located: list[tuple[str, str, int, str]] = []
        for section in resume.sections:
            for index, line in enumerate(section.content.split("\n")):
                if line in ctx.target_lines and not any(
                    loc[3] == line and loc[1] == section.id for loc in located
                ):
                    located.append((f"L{len(located) + 1}", section.id, index, line))
        if not located:
            return AgentResult(
                agent=self.name,
                title=self.title,
                reasoning="The selected lines are no longer in the resume, so nothing was changed.",
                resume=resume,
                warnings=["Select the lines again in the current version and retry."],
            )

        noun = "line" if len(located) == 1 else "lines"
        await ctx.progress(f"Rewriting {len(located)} selected {noun}")
        section_ids = list(dict.fromkeys(loc[1] for loc in located))
        user = "\n\n".join(
            part
            for part in [
                f"Request: {ctx.instruction}",
                memory_block(ctx.memory),
                "<lines>\n"
                + "\n".join(f"{lid}: {text.strip()}" for lid, _, _, text in located)
                + "\n</lines>",
                f"<context>\n{render_sections(resume, section_ids)}\n</context>",
            ]
            if part
        )
        plan = await self.deps.llm.structured(LineEditPlan, LINE_SYSTEM, user, name="edit_lines")
        by_id = {r.line_id.strip().upper(): r for r in plan.rewrites}

        updated = resume.model_copy(deep=True)
        applied: list[AppliedEdit] = []
        warnings: list[str] = []
        changed_lines: list[dict[str, str]] = []
        for section_id in section_ids:
            section = updated.get(section_id)
            before = section.model_copy()
            lines = section.content.split("\n")
            rationales = []
            for lid, sid, index, original in located:
                rewrite = by_id.get(lid)
                if sid != section_id or rewrite is None or not rewrite.replacement.strip():
                    continue
                new_line = _as_bullet_like(original, rewrite.replacement)
                if new_line != original:
                    lines[index] = new_line  # only this index is ever written
                    rationales.append(rewrite.rationale)
                    changed_lines.append({"before": original.strip(), "after": new_line.strip()})
            if rationales:
                section.content = "\n".join(lines)
                warnings.extend(check_edit(resume, before, section.content))
                applied.append(
                    AppliedEdit(
                        section_id=section.id, title=section.title, rationale=" ".join(rationales)
                    )
                )
        missing = [lid for lid, *_ in located if lid not in by_id]
        if missing:
            warnings.append(
                f"No rewrite was returned for {len(missing)} selected {noun}; left unchanged."
            )
        holes = sum(len(placeholders(c["after"])) for c in changed_lines)
        if holes:
            warnings.append(
                f"{holes} placeholder(s) like [add metric] mark where a real number would "
                "strengthen a line - replace them with true values or remove them."
            )
        return AgentResult(
            agent=self.name,
            title=self.title,
            reasoning=plan.reasoning,
            resume=updated,
            edits=applied,
            warnings=warnings,
            insights={"mode": "lines", "edited_lines": changed_lines},
        )
