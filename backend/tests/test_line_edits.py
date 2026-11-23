"""Precise edits: when the user points at lines, only those lines may change."""

import pytest

from app.config import Settings
from app.llm.offline import offline_llm
from app.main import Overrides, build_chat_service
from app.models.conversation import ConversationContext
from app.orchestration.line_targets import find_line_targets
from app.orchestration.route_cache import RouteCache
from app.orchestration.router import route
from app.parsing.sectionizer import parse_resume
from app.retrieval.embeddings import HashEmbedder
from app.retrieval.vector_store import VectorStore
from app.storage.memory import MemoryRepository
from app.tools.web_search import WebSearch

TWO_LINES = (
    "Optimize these lines to show relevant experience in backend + AI research:\n"
    "Helped migrate the transactions database from MySQL to PostgreSQL\n"
    "Wrote ETL jobs in Python and Airflow to load shipment data into the warehouse"
)


@pytest.fixture
def resume(sample_docx):
    return parse_resume("cv.docx", sample_docx)


def lines_of(resume):
    return [(s.id, line) for s in resume.sections for line in s.content.split("\n")]


# --------------------------------------------------------------------- matching
def test_pasted_lines_are_found_and_the_instruction_is_not(resume):
    targets = find_line_targets(TWO_LINES, resume)
    assert [(t.section_id, t.text) for t in targets] == [
        ("experience", "- Helped migrate the transactions database from MySQL to PostgreSQL"),
        (
            "experience",
            "- Wrote ETL jobs in Python and Airflow to load shipment data into the warehouse",
        ),
    ]


def test_matching_tolerates_small_differences_but_not_unrelated_text(resume):
    fuzzy = "Fix this: helped migrate the transaction database from MySQL to Postgres."
    assert [t.line_index for t in find_line_targets(fuzzy, resume)] == [2]
    assert find_line_targets("Make my experience section stronger for backend roles", resume) == []


def test_quoted_line_and_partial_selection_target_the_whole_line(resume):
    quoted = 'Rewrite "Built internal dashboards to track failed payments" with more impact'
    assert (
        find_line_targets(quoted, resume)[0].text
        == "- Built internal dashboards to track failed payments"
    )
    partial = find_line_targets(
        "make this stronger", resume, selection=["code reviews and mentored two"]
    )
    assert partial[0].text == "- Did code reviews and mentored two interns"


def test_pasted_blocks_are_not_treated_as_targeted_edits(resume, sample_jd):
    assert find_line_targets(sample_jd, resume) == []  # a job description matches no resume lines
    whole_section = "Improve this:\n" + resume.get("experience").content
    assert (
        find_line_targets(whole_section, resume) == []
    )  # more than MAX_TARGETS: not "these lines"


# ----------------------------------------------------------------------- routing
async def test_router_scopes_the_plan_to_the_lines_and_never_caches_it(resume):
    cache = RouteCache()
    result = await route(offline_llm(), TWO_LINES, ConversationContext(), resume, "", cache=cache)
    [step] = result.decision.steps
    assert step.agent == "section_enhancement" and len(step.target_lines) == 2
    assert "only the 2 lines" in result.decision.reasoning
    assert cache.get(cache.key(TWO_LINES, ConversationContext(), resume)) is None


# ------------------------------------------------------------------- end to end
@pytest.fixture
async def svc_conv(sample_docx):
    overrides = Overrides(
        repo=MemoryRepository(),
        llm=offline_llm(),
        vectors=VectorStore(HashEmbedder()),
        web=WebSearch([]),
    )
    svc = build_chat_service(Settings(llm_provider="fake", storage_backend="memory"), overrides)
    conv = await svc.create_conversation()
    await svc.upload_resume(conv.id, "alex.docx", sample_docx)
    return svc, conv.id


async def changed_lines(svc, cid, text, **kw):
    before = (await svc.get_state(cid))["current_version"]["resume"]
    events = [e async for e in svc.chat(cid, text, **kw)]
    after = events[-1]["current_version"]["resume"]
    old = [line for s in before["sections"] for line in s["content"].split("\n")]
    new = [line for s in after["sections"] for line in s["content"].split("\n")]
    assert len(old) == len(new)  # no line added or removed
    return [(a, b) for a, b in zip(old, new, strict=True) if a != b], events


async def test_pasted_lines_change_exactly_those_lines(svc_conv):
    svc, cid = svc_conv
    diffs, events = await changed_lines(svc, cid, TWO_LINES)
    assert [a for a, _ in diffs] == [
        "- Helped migrate the transactions database from MySQL to PostgreSQL",
        "- Wrote ETL jobs in Python and Airflow to load shipment data into the warehouse",
    ]
    assert all(b.startswith("- ") for _, b in diffs)  # still bullets
    result = next(e for e in events if e["type"] == "agent_result")
    assert result["insights"]["mode"] == "lines" and len(result["insights"]["edited_lines"]) == 2


async def test_resume_selection_changes_only_the_selected_line(svc_conv):
    svc, cid = svc_conv
    diffs, events = await changed_lines(
        svc, cid, "Make this sound more senior", selection=["Responsible for maintaining REST APIs"]
    )
    assert [a for a, _ in diffs] == [
        "- Responsible for maintaining REST APIs used by the mobile app"
    ]
    assert "> Responsible for maintaining REST APIs" in events[0]["message"]["content"]


async def test_question_about_a_selection_does_not_edit(svc_conv):
    svc, cid = svc_conv
    events = [
        e
        async for e in svc.chat(cid, "Is this line strong enough?", selection=["Did code reviews"])
    ]
    assert events[-1]["message"]["trace"]["intent"] == "question"
    assert events[-1]["current_version"]["number"] == 1
