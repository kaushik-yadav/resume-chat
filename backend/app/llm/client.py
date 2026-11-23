"""Provider-agnostic LLM access built on LangChain chat models.

Agents never talk to a provider SDK directly. They call three methods:

* ``structured(schema, system, user)`` - JSON-schema constrained output parsed into Pydantic
* ``text(system, user)``                - free text
* ``stream(system, user)``              - token stream (used for the final chat reply)

Each call is tagged with a ``role`` ("router" or "agent") so cheap/fast and heavier model
settings can be configured independently, and is recorded in a per-request trace (model,
latency, token usage) that the UI shows next to the agent's reasoning.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Role = Literal["router", "agent"]


class LLMError(RuntimeError):
    """The model could not produce a usable response."""


# --------------------------------------------------------------------------- tracing
@dataclass
class LLMCall:
    name: str
    model: str
    ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    node: str = ""  # workflow node that made the call (see ``current_node``)


@dataclass
class LLMTrace:
    calls: list[LLMCall] = field(default_factory=list)

    def total_ms(self) -> int:
        return sum(c.ms for c in self.calls)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "ms": self.total_ms(),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
        }


_current_trace: contextvars.ContextVar[LLMTrace | None] = contextvars.ContextVar(
    "llm_trace", default=None
)


def start_trace() -> LLMTrace:
    trace = LLMTrace()
    _current_trace.set(trace)
    return trace


# Set by each workflow node so LLM calls can be attributed to it in the execution flow.
current_node: contextvars.ContextVar[str] = contextvars.ContextVar("llm_node", default="")


def _record(call: LLMCall) -> None:
    trace = _current_trace.get()
    if trace is not None:
        call.node = call.node or current_node.get()
        trace.calls.append(call)


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


# ------------------------------------------------------------------------- interface
class LLM(Protocol):
    async def structured(
        self, schema: type[T], system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> T: ...

    async def text(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> str: ...

    def stream(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> AsyncIterator[str]: ...


def _messages(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _content_text(content: Any) -> str:
    """Normalize LangChain message content (str or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class LangChainLLM:
    """LLM backed by any LangChain chat model (Gemini or OpenAI here)."""

    def __init__(self, models: dict[Role, BaseChatModel], model_names: dict[Role, str]) -> None:
        self._models = models
        self._names = model_names

    async def structured(
        self, schema: type[T], system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> T:
        runnable = self._models[role].with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        messages = _messages(system, user)
        last_error: Exception | None = None
        for attempt in range(2):  # one repair attempt on malformed output
            start = time.perf_counter()
            result = await runnable.ainvoke(messages)
            ms = int((time.perf_counter() - start) * 1000)
            raw = result.get("raw")
            tokens_in, tokens_out = _usage(raw)
            _record(LLMCall(name or schema.__name__, self._names[role], ms, tokens_in, tokens_out))
            parsed = result.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            last_error = result.get("parsing_error")
            if parsed is not None:
                try:
                    return schema.model_validate(parsed)
                except ValidationError as exc:
                    last_error = exc
            logger.warning(
                "Structured output failed for %s (attempt %d): %s", name, attempt + 1, last_error
            )
            messages = [
                *_messages(system, user),
                HumanMessage(
                    content=f"Your previous answer was invalid ({last_error}). "
                    "Return only a valid JSON object for the requested schema."
                ),
            ]
        raise LLMError(
            f"Model returned invalid structured output for {schema.__name__}: {last_error}"
        )

    async def text(self, system: str, user: str, *, role: Role = "agent", name: str = "") -> str:
        start = time.perf_counter()
        message = await self._models[role].ainvoke(_messages(system, user))
        ms = int((time.perf_counter() - start) * 1000)
        _record(LLMCall(name or "text", self._names[role], ms, *_usage(message)))
        return _content_text(message.content).strip()

    async def stream(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        usage_in = usage_out = 0
        async for chunk in self._models[role].astream(_messages(system, user)):
            text = _content_text(chunk.content)
            i, o = _usage(chunk)
            usage_in, usage_out = max(usage_in, i), max(usage_out, o)
            if text:
                yield text
        ms = int((time.perf_counter() - start) * 1000)
        _record(LLMCall(name or "stream", self._names[role], ms, usage_in, usage_out))


# ------------------------------------------------------------------------ fake model
Responder = Callable[[str, str], Any]


class FakeLLM:
    """Deterministic offline LLM for tests and key-less demos.

    ``responders`` maps a schema class name (for ``structured``) or a call ``name`` (for
    ``text``/``stream``) to a function ``(system, user) -> result``.
    """

    def __init__(self, responders: dict[str, Responder] | None = None) -> None:
        self.responders: dict[str, Responder] = dict(responders or {})
        self.calls: list[tuple[str, str, str]] = []  # (name, system, user)

    def on(self, key: str, responder: Responder | Any) -> FakeLLM:
        self.responders[key] = responder if callable(responder) else (lambda s, u, r=responder: r)
        return self

    def _respond(self, key: str, system: str, user: str) -> Any:
        self.calls.append((key, system, user))
        if key not in self.responders:
            raise LLMError(f"FakeLLM has no responder for '{key}'")
        _record(LLMCall(key, "fake", 0))
        return self.responders[key](system, user)

    async def structured(
        self, schema: type[T], system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> T:
        result = self._respond(schema.__name__, system, user)
        return result if isinstance(result, schema) else schema.model_validate(result)

    async def text(self, system: str, user: str, *, role: Role = "agent", name: str = "") -> str:
        return str(self._respond(name or "text", system, user))

    async def stream(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> AsyncIterator[str]:
        text = str(self._respond(name or "stream", system, user))
        for i in range(0, len(text), 12):
            yield text[i : i + 12]


# --------------------------------------------------------------------------- factory
def create_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider == "fake":
        from app.llm.offline import offline_llm

        return offline_llm()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (or set LLM_PROVIDER=fake for offline mode).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        def build(model: str, effort: str) -> BaseChatModel:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
                reasoning_effort=effort,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )

        names = {"router": settings.router_model, "agent": settings.agent_model}
    else:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        def build(model: str, effort: str) -> BaseChatModel:
            return ChatOpenAI(
                model=model,
                api_key=settings.openai_api_key,
                reasoning_effort=effort,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )

        names = {"router": settings.openai_router_model, "agent": settings.openai_agent_model}

    models = {
        "router": build(names["router"], settings.router_reasoning),
        "agent": build(names["agent"], settings.agent_reasoning),
    }
    logger.info("LLM provider=%s router=%s agent=%s", provider, names["router"], names["agent"])
    return LangChainLLM(models, names)
