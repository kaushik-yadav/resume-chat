import io
import json

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.offline import offline_llm
from app.main import Overrides, build_chat_service, create_app
from app.models.conversation import ConversationContext
from app.orchestration.router import PlanStep, RoutingDecision, apply_memory_updates, normalize
from app.parsing.sectionizer import parse_resume
from app.retrieval.embeddings import HashEmbedder
from app.retrieval.vector_store import VectorStore
from app.storage.memory import MemoryRepository
from app.tools.web_search import SearchResult, WebSearch


class StubWeb(WebSearch):
    def __init__(self):
        super().__init__([object()])

    async def search(self, query, max_results=5):
        return [SearchResult("Culture", "https://example.com/c", "Values: users first, scale.")]


def settings(**kw) -> Settings:
    return Settings(llm_provider="fake", storage_backend="memory", **kw)


def overrides() -> Overrides:
    return Overrides(
        repo=MemoryRepository(),
        llm=offline_llm(),
        vectors=VectorStore(HashEmbedder()),
        web=StubWeb(),
    )


# ------------------------------------------------------------------------- router
def test_normalize_orders_dedupes_and_filters(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    decision = RoutingDecision(
        intent="edit",
        reasoning="r",
        steps=[
            PlanStep(agent="translation", instruction="t"),
            PlanStep(
                agent="company_research", instruction="c", target_sections=["summary", "nope"]
            ),
            PlanStep(agent="company_research", instruction="dup"),
        ],
    )
    memory = ConversationContext(target_company="Google", target_locale="es-MX")
    out = normalize(decision, memory, resume)
    assert [s.agent for s in out.steps] == ["company_research", "translation"]
    assert out.steps[0].target_sections == ["summary"]


def test_normalize_asks_for_missing_inputs(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    decision = RoutingDecision(
        intent="edit", reasoning="r", steps=[PlanStep(agent="jd_matching", instruction="match")]
    )
    out = normalize(decision, ConversationContext(), resume)
    assert out.intent == "clarify" and "job description" in out.clarification_question


def test_memory_updates_store_only_real_job_descriptions(sample_jd):
    d = RoutingDecision(
        intent="edit", reasoning="r", target_company="Stripe", message_contains_job_description=True
    )
    assert apply_memory_updates(d, ConversationContext(), "short text").job_description is None
    mem = apply_memory_updates(d, ConversationContext(), sample_jd)
    assert mem.job_description == sample_jd.strip() and mem.target_company == "Stripe"


# ------------------------------------------------------------------ chat service
@pytest.fixture
async def svc_conv(sample_docx):
    svc = build_chat_service(settings(), overrides())
    conv = await svc.create_conversation()
    await svc.upload_resume(conv.id, "alex.docx", sample_docx)
    return svc, conv.id


async def turn(svc, cid, text, jd=None):
    return [e async for e in svc.chat(cid, text, jd)]


async def test_upload_creates_version_and_welcome(svc_conv):
    svc, cid = svc_conv
    state = await svc.get_state(cid)
    assert state["current_version"]["number"] == 1
    assert state["conversation"]["title"] == "Alex Morgan's resume"
    assert "Optimize my resume for Google" in state["messages"][0]["content"]


async def test_single_agent_turn_streams_events_and_commits_version(svc_conv):
    svc, cid = svc_conv
    events = await turn(svc, cid, "Make my experience section stronger")
    types = [e["type"] for e in events]
    for expected in (
        "user_message",
        "plan",
        "agent_start",
        "progress",
        "agent_result",
        "token",
        "version",
        "done",
    ):
        assert expected in types
    plan = next(e for e in events if e["type"] == "plan")
    assert [s["agent"] for s in plan["steps"]] == ["section_enhancement"]
    done = events[-1]
    assert done["current_version"]["number"] == 2 and done["can_undo"]
    assert done["message"]["trace"]["results"][0]["agent"] == "section_enhancement"
    assert "Worked on" not in done["current_version"]["resume"]["sections"][2]["content"]


async def test_multi_agent_turn_chains_company_research_then_translation(svc_conv):
    svc, cid = svc_conv
    events = await turn(
        svc, cid, "Optimize my resume for Google and translate it to Spanish for Mexico"
    )
    starts = [e["agent"] for e in events if e["type"] == "agent_start"]
    assert starts == ["company_research", "translation"]
    done = events[-1]
    assert done["current_version"]["resume"]["locale"] == "es-MX"
    ctx = done["conversation"]["context"]
    assert ctx["target_company"] == "Google" and ctx["target_locale"] == "es-MX"


async def test_jd_attachment_routes_to_matching_and_is_remembered(svc_conv, sample_jd):
    svc, cid = svc_conv
    events = await turn(
        svc, cid, "Update my experience section to better match this job description", sample_jd
    )
    result = next(e for e in events if e["type"] == "agent_result")
    assert result["agent"] == "jd_matching" and "match_score_before" in result["insights"]
    assert events[-1]["conversation"]["context"]["job_description"].startswith("Data Scientist")


async def test_clarify_when_jd_missing_then_undo_and_question(svc_conv):
    svc, cid = svc_conv
    clarify = await turn(svc, cid, "Match my resume to this job description")
    assert clarify[-1]["message"]["trace"]["intent"] == "clarify"
    assert clarify[-1]["current_version"]["number"] == 1

    await turn(svc, cid, "Make my experience section stronger")
    undo = await turn(svc, cid, "undo that")
    assert undo[-1]["current_version"]["number"] == 1 and undo[-1]["can_redo"]

    question = await turn(svc, cid, "What are the weakest parts of my resume?")
    assert question[-1]["message"]["trace"]["intent"] == "question"
    assert not any(e["type"] == "version" for e in question)


async def test_history_is_summarized_when_long(sample_docx):
    svc = build_chat_service(settings(summary_trigger=4, history_window=2), overrides())
    conv = await svc.create_conversation()
    await svc.upload_resume(conv.id, "cv.docx", sample_docx)
    for _ in range(3):
        await turn(svc, conv.id, "hello")
    memory = (await svc.get_state(conv.id))["conversation"]["context"]
    assert memory["summarized_through"] > 0 and memory["summary"]


# ----------------------------------------------------------------------------- API
def parse_sse(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_http_api_end_to_end(sample_docx):
    with TestClient(create_app(settings(), overrides())) as client:
        cid = client.post("/api/conversations").json()["id"]
        up = client.post(
            f"/api/conversations/{cid}/resume", files={"file": ("cv.docx", sample_docx)}
        )
        assert up.status_code == 200 and up.json()["current_version"]["number"] == 1

        stream = client.post(
            f"/api/conversations/{cid}/chat", json={"message": "Make my experience stronger"}
        )
        events = parse_sse(stream.text)
        assert events[-1]["type"] == "done" and events[-1]["current_version"]["number"] == 2

        versions = client.get(f"/api/conversations/{cid}/versions").json()
        assert [v["number"] for v in versions] == [1, 2]
        diff = client.get(
            f"/api/conversations/{cid}/compare",
            params={"from": versions[0]["id"], "to": versions[1]["id"]},
        )
        assert diff.json()[0]["section_id"] == "experience"

        assert (
            client.post(f"/api/conversations/{cid}/undo").json()["current_version"]["number"] == 1
        )
        assert (
            client.post(f"/api/conversations/{cid}/redo").json()["current_version"]["number"] == 2
        )
        assert client.post(f"/api/conversations/{cid}/redo").status_code == 409
        restored = client.post(
            f"/api/conversations/{cid}/versions/{versions[0]['id']}/restore"
        ).json()
        assert restored["current_version"]["number"] == 3

        docx = client.get(f"/api/conversations/{cid}/export", params={"format": "docx"})
        assert docx.status_code == 200 and "cv_v3.docx" in docx.headers["content-disposition"]
        assert "Alex Morgan" in Document(io.BytesIO(docx.content)).paragraphs[0].text
        md = client.get(f"/api/conversations/{cid}/export", params={"format": "md"})
        assert "## Experience" in md.text

        state = client.get(f"/api/conversations/{cid}").json()
        assert len(state["messages"]) == 3
        assert client.get("/api/conversations").json()[0]["id"] == cid
        assert client.get("/api/conversations/missing").status_code == 404


def test_chat_before_upload_returns_error_event():
    with TestClient(create_app(settings(), overrides())) as client:
        cid = client.post("/api/conversations").json()["id"]
        events = parse_sse(
            client.post(f"/api/conversations/{cid}/chat", json={"message": "hi"}).text
        )
        assert events == [{"type": "error", "message": "Upload a resume first."}]


def test_fast_path_handles_bare_version_commands():
    from app.orchestration.router import fast_path

    assert fast_path("Undo that!").intent == "undo"
    assert fast_path("redo").intent == "redo"
    assert fast_path("revert the last change").intent == "undo"
    assert fast_path("undo the translation but keep the summary") is None


def test_guard_ignores_24_7():
    from app.agents.guardrails import new_numbers

    assert new_numbers("on-call rotation", "24/7 on-call rotation") == []
