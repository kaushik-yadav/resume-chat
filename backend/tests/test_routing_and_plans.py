import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.client import TieredLLM, current_plan
from app.llm.offline import offline_llm
from app.main import Overrides, build_chat_service, create_app
from app.models.conversation import ConversationContext
from app.orchestration.classifier import (
    IntentClassifier,
    extract_company,
    extract_locale,
    is_question,
)
from app.orchestration.locales import canonical_locale
from app.orchestration.presets import PRESETS, preset_decision
from app.orchestration.route_cache import RouteCache
from app.parsing.sectionizer import parse_resume, repair_resume
from app.plans import PLANS, call_cost, get_plan
from app.retrieval.embeddings import HashEmbedder
from app.retrieval.vector_store import VectorStore
from app.storage.memory import MemoryRepository
from app.tools.web_search import SearchResult, WebSearch


class StubWeb(WebSearch):
    def __init__(self):
        super().__init__([object()])

    async def search(self, query, max_results=5):
        return [SearchResult("Culture", "https://example.com/c", "Values: users first.")]


def settings(**kw) -> Settings:
    return Settings(llm_provider="fake", storage_backend="memory", **kw)


def overrides(llm=None) -> Overrides:
    return Overrides(
        repo=MemoryRepository(),
        llm=llm or offline_llm(),
        vectors=VectorStore(HashEmbedder()),
        web=StubWeb(),
    )


@pytest.fixture
async def svc_conv(sample_docx):
    svc = build_chat_service(settings(), overrides())
    conv = await svc.create_conversation()
    await svc.upload_resume(conv.id, "alex.docx", sample_docx)
    return svc, conv.id


async def turn(svc, cid, text, **kw):
    return [e async for e in svc.chat(cid, text, **kw)]


def router_node(events):
    flow = [e for e in events if e["type"] == "flow"][-1]
    return next(n for n in flow["nodes"] if n["id"] == "router")


def route_calls(svc) -> int:
    return sum(name == "RoutingDecision" for name, _, _ in svc.llm.calls)


# ---------------------------------------------------------------------- idempotency
async def test_same_client_message_id_replays_instead_of_rerunning(svc_conv):
    svc, cid = svc_conv
    first = await turn(svc, cid, "Make my experience section stronger", client_message_id="m-1")
    calls_after_first = len(svc.llm.calls)
    again = await turn(svc, cid, "Make my experience section stronger", client_message_id="m-1")

    assert again[-1]["replayed"] is True
    assert again[-1]["message"]["id"] == first[-1]["message"]["id"]
    assert len(svc.llm.calls) == calls_after_first  # no agent or LLM work repeated
    state = await svc.get_state(cid)
    assert [m["role"] for m in state["messages"]] == ["assistant", "user", "assistant"]
    assert state["current_version"]["number"] == 2  # no duplicate version


async def test_unanswered_message_is_processed_once_on_retry(svc_conv):
    svc, cid = svc_conv
    from app.models.conversation import ChatMessage

    conv = await svc._io(svc.repo.get_conversation, cid)
    await svc._add_message(conv, ChatMessage(role="user", content="hi", client_id="m-2"))
    await svc._io(svc.repo.save_conversation, conv)  # simulate a crash before the answer
    events = await turn(svc, cid, "hi", client_message_id="m-2")
    assert events[-1]["type"] == "done" and not events[-1].get("replayed")
    roles = [m["role"] for m in (await svc.get_state(cid))["messages"]]
    assert roles == ["assistant", "user", "assistant"]  # the user message was not duplicated


# ------------------------------------------------------------- presets and caching
async def test_preset_skips_intent_classification(svc_conv):
    svc, cid = svc_conv
    preset = PRESETS["translate_es_mx"]
    events = await turn(svc, cid, preset.label, preset=preset.id)
    node = router_node(events)
    assert node["detail"]["decided_by"] == "preset" and node["llm"] == []
    assert route_calls(svc) == 0
    assert events[-1]["current_version"]["resume"]["locale"] == "es-MX"


def test_preset_targets_sections_by_kind(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    decision = preset_decision("skills_data_scientist", resume)
    assert decision.steps[0].target_sections == ["skills"]
    assert decision.target_role == "Data Scientist"
    assert preset_decision("nope", resume) is None


async def test_exact_repeat_reuses_cached_routing(svc_conv):
    svc, cid = svc_conv
    message = "Rewrite my projects with stronger action verbs, please"
    first = router_node(await turn(svc, cid, message))
    second = router_node(await turn(svc, cid, message))
    assert first["detail"]["decided_by"] in ("classifier", "llm")
    assert second["detail"]["decided_by"] == "cache" and second["llm"] == []


def test_route_cache_skips_context_dependent_messages(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    cache = RouteCache()
    memory = ConversationContext()
    assert cache.key("do the same for my projects", memory, resume) is None
    k1 = cache.key("Translate it to Spanish!", memory, resume)
    assert k1 == cache.key("translate it to spanish", memory, resume)  # normalized
    assert k1 != cache.key(
        "translate it to spanish", ConversationContext(target_locale="de-DE"), resume
    )


# ---------------------------------------------------------------------- classifier
def test_slot_extraction():
    assert extract_locale("Translate this to Spanish for the Mexican market") == "es-MX"
    assert extract_locale("I want a German version of my resume") == "de-DE"
    assert extract_locale("localize it for Brazil") == "pt-BR"
    assert extract_locale("make it better") is None
    assert extract_company("I'm applying at Microsoft, adjust my resume") == "Microsoft"
    assert extract_company("Optimize my resume for Google") == "Google"
    assert extract_company("Translate it for the Spanish market") is None
    assert is_question("How good is my match?") and not is_question("Could you translate it?")


async def test_classifier_decides_simple_requests_and_abstains_otherwise(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    # The offline hash embedder is lexical (unrelated texts still score ~0.5), so this test uses
    # its own thresholds; production thresholds are calibrated on Gemini embeddings (evals/).
    clf = IntentClassifier(VectorStore(HashEmbedder()), k=3, min_confidence=0.45)
    clf.ensure_index()
    memory = ConversationContext()
    hit = await clf.classify("Translate my resume into German", memory, resume)
    assert hit.decision.steps[0].agent == "translation" and hit.decision.target_locale == "de-DE"
    # Compound or history-dependent requests are left to the LLM router.
    assert await clf.classify("Optimize for Stripe and then translate it", memory, resume) is None
    assert await clf.classify("do the same for them", memory, resume) is None
    # A JD request without a job description on file is not guessed.
    jd = await clf.classify("Match my resume to this job description", memory, resume)
    assert jd.decision is None


# ---------------------------------------------------------------------- plans/costs
def test_plans_and_costs():
    assert get_plan(None, "pro").id == "pro" and get_plan("bogus", "free").id == "free"
    assert PLANS["free"].agent_model == "gemini-3.5-flash-lite"
    assert call_cost("gemini-3.5-flash-lite", 1_000_000, 0) == pytest.approx(0.30)
    assert call_cost("unknown-model", 10, 10) is None


async def test_tiered_llm_uses_the_current_plan():
    class Stub:
        def __init__(self, name):
            self.name = name

        async def text(self, system, user, *, role="agent", name=""):
            return f"{self.name}:{role}"

    llm = TieredLLM({"free": Stub("free"), "pro": Stub("pro")}, default_plan="pro")
    assert await llm.text("s", "u") == "pro:agent"
    current_plan.set("free")
    assert await llm.text("s", "u", role="reply") == "free:reply"
    current_plan.set("")


async def test_plan_is_applied_and_recorded(svc_conv):
    svc, cid = svc_conv
    await svc.set_plan(cid, "free")
    events = await turn(svc, cid, "hello")
    trace = events[-1]["message"]["trace"]
    assert trace["subscription_plan"] == "free" and "cost_usd" in trace
    assert events[-1]["plan"] == "free"


# ------------------------------------------------------------------ jd + parsing
async def test_jd_requirements_are_cached_per_job_description(svc_conv, sample_jd):
    from app.agents.jd_matching import JDMatchingAgent

    JDMatchingAgent._requirements_cache.clear()  # class-level cache shared across tests
    svc, cid = svc_conv
    await turn(svc, cid, "Match my experience to this job description", job_description=sample_jd)
    await turn(svc, cid, "Now match my skills to this job description")
    extractions = sum(name == "JobRequirements" for name, _, _ in svc.llm.calls)
    assert extractions == 1


def test_repair_resume_rejoins_lines_from_older_parses(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    broken = resume.with_section_content(
        "experience",
        "Engineer, Acme | Jan 2020 - Present\n- Owned delivery of agentic AI products from start to\n"
        "POCs and production deployment",
    )
    fixed = repair_resume(broken).get("experience").content.split("\n")
    assert fixed[1].endswith("from start to POCs and production deployment")


# ----------------------------------------------------------------------------- API
def test_catalog_and_plan_endpoints(sample_docx):
    with TestClient(create_app(settings(), overrides())) as client:
        plans = client.get("/api/plans").json()
        assert plans["default"] == "pro" and {p["id"] for p in plans["plans"]} == set(PLANS)
        assert plans["plans"][0]["prices"]["agent"]["model"]
        assert [p["id"] for p in client.get("/api/presets").json()] == list(PRESETS)
        cid = client.post("/api/conversations").json()["id"]
        client.post(f"/api/conversations/{cid}/resume", files={"file": ("cv.docx", sample_docx)})
        assert (
            client.patch(f"/api/conversations/{cid}", json={"plan": "premium"}).json()["plan"]
            == "premium"
        )
        assert client.patch(f"/api/conversations/{cid}", json={"plan": "gold"}).status_code == 422


# ---------------------------------------------------------------- eval data hygiene
def test_routing_eval_split_has_no_leakage():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
    from routing_cases import CALIBRATION, TEST, leakage

    from app.orchestration.classifier import EXEMPLARS

    exemplars = [text for texts in EXEMPLARS.values() for text in texts]
    assert leakage(TEST, [c.message for c in CALIBRATION], exemplars) == []
    assert len(TEST) >= 30 and len(CALIBRATION) >= 20
    # Locale expectations are canonical, so exact-match scoring is meaningful.
    for case in [*CALIBRATION, *TEST]:
        if "target_locale" in case.slots:
            assert canonical_locale(case.slots["target_locale"]) == case.slots["target_locale"]
