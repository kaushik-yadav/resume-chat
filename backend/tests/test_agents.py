import pytest

from app.agents.base import AgentContext, AgentDeps, EditPlan
from app.agents.guardrails import (
    dropped_numbers,
    new_numbers,
    new_terms,
    placeholders,
    style_issues,
)
from app.agents.jd_matching import keyword_coverage, match_score
from app.agents.registry import AGENT_NAMES, build_agents, describe_agents
from app.agents.translation import describe_locale
from app.llm.offline import offline_llm
from app.models.conversation import ConversationContext
from app.parsing.sectionizer import parse_resume
from app.retrieval.embeddings import HashEmbedder
from app.retrieval.vector_store import VectorStore
from app.tools.web_search import SearchResult, WebSearch


class StubWeb(WebSearch):
    def __init__(self, results=None):
        super().__init__([object()])  # "available"
        self.queries: list[str] = []
        self._results = results or []

    async def search(self, query, max_results=5):
        self.queries.append(query)
        return self._results


@pytest.fixture
def resume(sample_docx):
    return parse_resume("cv.docx", sample_docx)


@pytest.fixture
def deps():
    vectors = VectorStore(HashEmbedder())
    vectors.ensure_knowledge_base()
    web = StubWeb(
        [SearchResult("Google culture", "https://example.com/g", "Google values users and scale.")]
    )
    return AgentDeps(llm=offline_llm(), vectors=vectors, web=web)


def ctx(resume, instruction="improve", targets=None, **memory):
    events = []
    return (
        AgentContext(
            conversation_id="c1",
            resume=resume,
            instruction=instruction,
            user_message=instruction,
            memory=ConversationContext(**memory),
            target_sections=targets or [],
            emit=lambda e, d: events.append(d["message"]),
        ),
        events,
    )


def test_registry_has_all_four_agents(deps):
    assert (
        set(build_agents(deps))
        == set(AGENT_NAMES)
        == {
            "company_research",
            "jd_matching",
            "translation",
            "section_enhancement",
        }
    )
    assert "jd_matching:" in describe_agents()


# ------------------------------------------------------------------ enhancement
async def test_section_enhancement_edits_only_targets(deps, resume):
    agent = build_agents(deps)["section_enhancement"]
    context, events = ctx(resume, "make my experience stronger", ["experience"])
    result = await agent.run(context)
    exp = result.resume.get("experience").content
    assert "- Developed backend services" in exp and "Worked on" not in exp
    assert [e.section_id for e in result.edits] == ["experience"]
    assert result.resume.get("summary").content == resume.get("summary").content
    assert any("weak bullet openers" in i for i in result.insights["issues_found"])
    assert events and "Rewriting" in events[-1]


async def test_invented_numbers_are_flagged(deps, resume):
    deps.llm.on(
        "EditPlan",
        EditPlan(
            reasoning="x",
            edits=[
                {
                    "section_id": "summary",
                    "new_content": "Engineer who cut costs 40% for 2M users [add metric].",
                    "rationale": "r",
                },
                {"section_id": "header", "new_content": "Hacked", "rationale": "not a target"},
            ],
        ),
    )
    result = await build_agents(deps)["section_enhancement"].run(
        ctx(resume, targets=["summary"])[0]
    )
    assert [e.section_id for e in result.edits] == ["summary"]  # header edit rejected
    assert any("40%" in w and "2M" in w for w in result.warnings)
    assert any("placeholder" in w for w in result.warnings)


# --------------------------------------------------------------------- JD match
async def test_jd_matching_scores_and_tailors(deps, resume, sample_jd):
    agent = build_agents(deps)["jd_matching"]
    result = await agent.run(ctx(resume, "match this job", job_description=sample_jd)[0])
    ins = result.insights
    assert 0 <= ins["match_score_before"] <= ins["match_score_after"] <= 100
    assert ins["requirements"] and ins["skill_gaps"]
    assert "python" in ins["keywords_matched"]
    # Keywords the edit introduced are listed for the user to confirm, never silently counted.
    added = ins["keywords_added"]
    assert not set(added) & set(keyword_coverage(added, resume.to_text())[0])
    assert all(k in w for k in added for w in result.warnings if w.startswith("Added from"))
    assert bool(added) == any(w.startswith("Added from the job posting") for w in result.warnings)
    assert result.memory_updates["target_role"].startswith("Data Scientist")
    assert result.memory_updates["target_company"] == "Finlytics"


async def test_jd_matching_without_jd_asks_for_it(deps, resume):
    result = await build_agents(deps)["jd_matching"].run(ctx(resume)[0])
    assert not result.changed and "job description" in result.warnings[0].lower()


def test_keyword_coverage_and_score():
    found, missing = keyword_coverage(["Python", "A/B testing", "Go"], "Python and a/b testing")
    assert found == ["Python", "A/B testing"] and missing == ["Go"]
    assert match_score(50.0, 1, 2) == 50 and match_score(100.0, 2, 2) == 100


# ---------------------------------------------------------------------- company
async def test_company_research_uses_web_then_cache(deps, resume):
    agent = build_agents(deps)["company_research"]
    first = await agent.run(ctx(resume, "optimize for Google", target_company="Google")[0])
    assert deps.web.queries and first.insights["sources"][0]["url"] == "https://example.com/g"
    assert first.insights["company_profile"]["grounded_in_sources"] is True
    deps.web.queries.clear()
    context, events = ctx(resume, "again", target_company="Google")
    await agent.run(context)
    assert deps.web.queries == [] and any("cached" in e for e in events)


async def test_company_research_needs_company(deps, resume):
    result = await build_agents(deps)["company_research"].run(ctx(resume)[0])
    assert not result.changed and result.warnings


# ------------------------------------------------------------------ translation
async def test_translation_localizes_with_exact_conventions(deps, resume):
    result = await build_agents(deps)["translation"].run(
        ctx(resume, "Spanish for Mexico", target_locale="es-MX")[0]
    )
    assert result.resume.language == "es" and result.resume.locale == "es-MX"
    assert result.resume.get("experience").title == "Experiencia profesional"
    assert result.insights["conventions_used"] == ["Mexico (es-MX)"]
    prompt = next(u for name, _, u in deps.llm.calls if name == "SectionTranslation")
    # Sections are translated in separate (parallel) calls; the header is never sent.
    assert sum(name == "SectionTranslation" for name, _, _ in deps.llm.calls) == 6
    assert "computadora" in prompt  # Mexico-specific guidance reached the model


async def test_translation_needs_locale(deps, resume):
    result = await build_agents(deps)["translation"].run(ctx(resume)[0])
    assert not result.changed


def test_describe_locale():
    assert describe_locale("es-MX") == ("es", "Spanish", "Mexico")
    assert describe_locale("de") == ("de", "German", "")


# ------------------------------------------------------------------- guardrails
def test_guardrail_helpers(resume):
    assert new_numbers("cut time from 6 hours to 2 hours", "cut time 66% (6 to 2 hours)") == ["66%"]
    assert new_numbers("Jan 2022", "[add metric: 30%] since Jan 2022") == []
    assert dropped_numbers("2019 - 2021, 3 interns", "2019 - 2021") == ["3"]
    assert placeholders("a [x] b [y]") == ["[x]", "[y]"]
    issues = style_issues(resume)
    assert any("Experience: weak bullet openers" in i for i in issues)


def test_new_terms_flags_tools_borrowed_from_elsewhere():
    before = "- Helped migrate the transactions database from MySQL to PostgreSQL"
    # BigQuery is on the resume (Skills) but not in this line: flagged.
    assert new_terms(
        before,
        "- Led the migration of the transactions database from MySQL to PostgreSQL using BigQuery",
    ) == ["BigQuery"]
    # Rewording, a new capitalised opener and the line's own names are fine.
    assert new_terms(before, "- Migrated the transactions database from MySQL to PostgreSQL") == []
    assert new_terms(
        "- Wrote ETL jobs in Python", "- Built ETL jobs in Python on AWS with Node.js"
    ) == ["AWS", "Node.js"]


def test_check_edit_flags_a_tool_new_to_the_section():
    from app.agents.guardrails import check_edit
    from app.models.resume import Resume, ResumeSection

    before = ResumeSection(
        id="experience", title="Experience", kind="experience",
        content="- Wrote ETL jobs in Python to load shipment data into the warehouse",
    )
    skills = ResumeSection(id="skills", title="Skills", kind="skills", content="Data: BigQuery")
    resume = Resume(sections=[before, skills])
    edited = "- Built ETL pipelines in Python to load shipment data into BigQuery"
    assert any("BigQuery not in this section" in w for w in check_edit(resume, before, edited))
    assert check_edit(resume, before, edited.replace("BigQuery", "the warehouse")) == []
