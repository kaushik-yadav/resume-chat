"""LLM client robustness: explicit role mapping, model fallback, and metering of partial calls."""

import asyncio
import contextlib

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel

from app.config import Settings
from app.llm.client import (
    ROLES,
    LangChainLLM,
    LLMError,
    TieredLLM,
    create_llm,
    is_unsupported,
    start_trace,
)
from app.llm.offline import offline_llm
from app.models.conversation import ConversationContext
from app.orchestration.locales import canonical_locale
from app.orchestration.router import RoutingDecision, route
from app.parsing.sectionizer import parse_resume


class Answer(BaseModel):
    value: int


class HTTPError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


class StubChat:
    """Duck-typed chat model: just the three methods LangChainLLM uses."""

    def __init__(self, *, fail_methods=(), fail_with=None, unsupported=False, delay=0.0,
                 chunks=("Hel", "lo ", "there"), stream_fail_at=None) -> None:  # fmt: skip
        self.fail_methods, self.fail_with = set(fail_methods), fail_with
        self.unsupported, self.delay = unsupported, delay
        self.chunks, self.stream_fail_at = chunks, stream_fail_at
        self.seen: list[str] = []

    def with_structured_output(self, schema, *, method, include_raw):
        if self.unsupported:
            raise NotImplementedError("structured output not implemented")
        stub = self

        class Runnable:
            async def ainvoke(self, messages):
                stub.seen.append(method)
                if stub.delay:
                    await asyncio.sleep(stub.delay)
                if method in stub.fail_methods:
                    raise stub.fail_with
                raw = AIMessage(
                    content="{}",
                    usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                )
                return {"raw": raw, "parsed": schema(value=7), "parsing_error": None}

        return Runnable()

    async def ainvoke(self, messages):
        if self.fail_with and "text" in self.fail_methods:
            raise self.fail_with
        return AIMessage(
            content="hi", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
        )

    async def astream(self, messages):
        for i, chunk in enumerate(self.chunks):
            if self.stream_fail_at == i:
                raise self.fail_with
            await asyncio.sleep(0)
            yield AIMessageChunk(content=chunk)


def llm_with(primary: StubChat, fallback: StubChat | None = None) -> LangChainLLM:
    models = {role: primary for role in ROLES}
    names = {role: "primary-model" for role in ROLES}
    fallbacks = {role: (fallback, "fallback-model") for role in ROLES} if fallback else None
    return LangChainLLM(models, names, fallbacks=fallbacks)


# ----------------------------------------------------------------- role mapping
def test_every_role_must_be_mapped_explicitly():
    with pytest.raises(ValueError, match="reply"):
        LangChainLLM({"router": StubChat(), "agent": StubChat()}, {"router": "a", "agent": "b"})


def test_openai_client_has_reply_and_fallback_models():
    llm = create_llm(Settings(llm_provider="openai", openai_api_key="sk-test"))
    assert llm.model_name("reply") == Settings().openai_reply_model
    assert llm._fallbacks["agent"][1] == Settings().openai_fallback_model


def test_gemini_plans_map_every_role_with_a_different_fallback():
    llm = create_llm(Settings(llm_provider="gemini", gemini_api_key="test-key"))
    assert isinstance(llm, TieredLLM)
    for plan in ("free", "pro", "premium"):
        client = llm.for_plan(plan)
        for role in ROLES:
            assert client.model_name(role)
            assert client._fallbacks[role][1] != client.model_name(role)


def test_a_fallback_identical_to_the_primary_is_dropped():
    same = LangChainLLM(
        {r: StubChat() for r in ROLES},
        {r: "m" for r in ROLES},
        fallbacks={r: (StubChat(), "m") for r in ROLES},
    )
    assert same._fallbacks == {}


# --------------------------------------------------------------------- fallback
async def test_structured_falls_back_when_the_model_rejects_the_request():
    primary = StubChat(
        fail_methods={"json_schema", "function_calling"},
        fail_with=HTTPError(400, "INVALID_ARGUMENT: response_schema"),
    )
    fallback = StubChat()
    trace = start_trace()
    assert (await llm_with(primary, fallback).structured(Answer, "s", "u", name="q")).value == 7
    assert primary.seen == ["json_schema", "function_calling"] and fallback.seen == ["json_schema"]
    assert [(c.model, c.status) for c in trace.calls] == [
        ("primary-model", "error"),
        ("primary-model", "error"),
        ("fallback-model", "ok"),
    ]
    assert trace.calls[0].estimated and trace.calls[-1].input_tokens == 11


async def test_function_calling_on_the_same_model_is_tried_first():
    primary = StubChat(fail_methods={"json_schema"}, fail_with=HTTPError(400))
    assert (await llm_with(primary, StubChat()).structured(Answer, "s", "u")).value == 7
    assert primary.seen == ["json_schema", "function_calling"]


async def test_model_without_structured_support_falls_back():
    fallback = StubChat()
    assert (
        await llm_with(StubChat(unsupported=True), fallback).structured(Answer, "s", "u")
    ).value == 7
    assert fallback.seen == ["json_schema"]


@pytest.mark.parametrize("status", [429, 401, 503])
async def test_rate_limits_auth_and_outages_are_not_masked(status):
    primary = StubChat(fail_methods={"json_schema"}, fail_with=HTTPError(status))
    fallback = StubChat()
    with pytest.raises(HTTPError):
        await llm_with(primary, fallback).structured(Answer, "s", "u")
    assert fallback.seen == []


async def test_no_fallback_configured_raises_a_clear_error():
    primary = StubChat(
        fail_methods={"json_schema", "function_calling"},
        fail_with=HTTPError(404, "model not found"),
    )
    with pytest.raises(LLMError, match="No model could produce Answer"):
        await llm_with(primary).structured(Answer, "s", "u")


def test_is_unsupported_classification():
    assert is_unsupported(HTTPError(400)) and is_unsupported(HTTPError(404))
    assert is_unsupported(RuntimeError("400 INVALID_ARGUMENT: model does not support json_schema"))
    assert not is_unsupported(HTTPError(429)) and not is_unsupported(HTTPError(500))
    assert not is_unsupported(RuntimeError("Resource exhausted: quota exceeded"))


# ------------------------------------------------------------- partial metering
async def test_cancelled_stream_is_metered_with_partial_output():
    trace = start_trace()
    llm = llm_with(StubChat(chunks=("a" * 40, "b" * 40, "c" * 40)))
    async with contextlib.aclosing(
        llm.stream("system", "user", role="reply", name="reply")
    ) as tokens:
        async for _ in tokens:
            break  # the client went away after the first chunk
    [call] = trace.calls
    assert call.status == "cancelled" and call.estimated
    assert call.output_tokens == 10  # 40 characters received ~= 10 tokens
    assert call.input_tokens > 0  # prompt the provider may already have billed


async def test_stream_falls_back_only_before_the_first_token():
    early = StubChat(stream_fail_at=0, fail_with=HTTPError(404))
    assert "".join([t async for t in llm_with(early, StubChat()).stream("s", "u")]) == "Hello there"
    late = StubChat(stream_fail_at=1, fail_with=HTTPError(404))
    with pytest.raises(HTTPError):
        _ = [t async for t in llm_with(late, StubChat()).stream("s", "u")]


async def test_cancelled_speculative_router_call_is_recorded(sample_docx):
    """The classifier answers first; the in-flight LLM router call is cancelled and metered."""
    resume = parse_resume("cv.docx", sample_docx)

    class InstantClassifier:
        async def classify(self, message, memory, resume):
            await asyncio.sleep(0.05)
            decision = RoutingDecision(intent="question", reasoning="fast path")
            return type("R", (), {"decision": decision, "confidence": 0.9})()

    trace = start_trace()
    result = await route(
        llm_with(StubChat(delay=5)), "is my resume good?", ConversationContext(), resume, "",
        classifier=InstantClassifier(),
    )  # fmt: skip
    assert result.decided_by == "classifier"
    [call] = trace.calls
    assert (call.name, call.status, call.estimated) == ("route", "cancelled", True)


# ------------------------------------------------------------------- router modes
async def test_llm_only_mode_bypasses_rules_and_classifier(sample_docx):
    resume = parse_resume("cv.docx", sample_docx)
    llm = offline_llm()
    cascade = await route(llm, "undo", ConversationContext(), resume, "")
    assert cascade.decided_by == "rules" and not llm.calls
    baseline = await route(llm, "undo", ConversationContext(), resume, "", mode="llm_only")
    assert baseline.decided_by == "llm" and [c[0] for c in llm.calls] == ["RoutingDecision"]


@pytest.mark.parametrize(
    "raw,expected",
    [("es-MX", "es-MX"), ("es_mx", "es-MX"), ("DE", "de-DE"), ("pt", "pt-BR"),
     ("Spanish (Mexico)", "es-MX"), ("", ""), (None, ""), ("gibberish", "")],
)  # fmt: skip
def test_canonical_locale(raw, expected):
    assert canonical_locale(raw) == expected
