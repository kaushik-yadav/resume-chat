"""Provider-agnostic LLM access built on LangChain chat models.

Agents never talk to a provider SDK directly. They call three methods:

* ``structured(schema, system, user)`` - JSON-schema constrained output parsed into Pydantic
* ``text(system, user)``                - free text
* ``stream(system, user)``              - token stream (used for the final chat reply)

Each call is tagged with a ``role`` - "router", "agent" or "reply" - and every client must map
all three explicitly (no silent fallback to another role's model).

Every call is **metered in a ``finally`` block** into the per-request trace (model, latency,
tokens, status), so cancelled and failed calls are recorded too: a reply stream the user
abandons, or a speculative router call cancelled because the fast classifier answered first.
When a call did not complete, the tokens the provider may still bill are estimated and
flagged (``estimated=True``).

If a model rejects a request as unsupported (e.g. it cannot do JSON-schema output, or the
model id is unknown), the call falls back - first to function-calling structured output on the
same model, then to a configured fallback model. Rate limits, auth and server errors are
never masked by a fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Role = Literal["router", "agent", "reply"]
ROLES: tuple[Role, ...] = ("router", "agent", "reply")
CallStatus = Literal["ok", "error", "cancelled"]


class LLMError(RuntimeError):
    """The model could not produce a usable response."""


class ModelUnsupported(LLMError):
    """The model cannot serve this kind of request (e.g. no JSON-schema output)."""


# --------------------------------------------------------------------------- tracing
@dataclass
class LLMCall:
    name: str
    model: str
    ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    node: str = ""  # workflow node that made the call (see ``current_node``)
    status: CallStatus = "ok"
    estimated: bool = False  # token counts estimated because the call did not complete


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
            "incomplete_calls": sum(c.status != "ok" for c in self.calls),
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

# The subscription plan of the conversation being served (see app/plans.py).
current_plan: contextvars.ContextVar[str] = contextvars.ContextVar("llm_plan", default="")


def record_call(call: LLMCall) -> None:
    """Record a call made outside this module (e.g. embeddings) into the current trace."""
    trace = _current_trace.get()
    if trace is not None:
        call.node = call.node or current_node.get()
        trace.calls.append(call)


@dataclass
class _Meter:
    input_tokens: int = 0
    output_tokens: int = 0
    text_chars: int = 0  # streamed so far, to estimate output tokens of an abandoned stream


@contextlib.contextmanager
def _metered(name: str, model: str, prompt: str) -> Iterator[_Meter]:
    """Record the call in a ``finally`` - whether it completes, fails or is cancelled.

    The trace and node are captured when the call *starts*: a stream that is closed later
    (or by the garbage collector, in another context) still lands in the right trace.
    """
    trace, node = _current_trace.get(), current_node.get()
    meter, status, started = _Meter(), "ok", time.perf_counter()
    try:
        yield meter
    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        tokens_in, tokens_out, estimated = meter.input_tokens, meter.output_tokens, False
        if status != "ok":
            if not tokens_in:  # the provider may already have read (and may bill) the prompt
                tokens_in, estimated = len(prompt) // 4, True
            if not tokens_out and meter.text_chars:
                tokens_out, estimated = meter.text_chars // 4, True
        if trace is not None:
            ms = int((time.perf_counter() - started) * 1000)
            trace.calls.append(
                LLMCall(name, model, ms, tokens_in, tokens_out, node, status, estimated)
            )


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


_NOT_A_MODEL_PROBLEM = ("rate limit", "quota", "resource_exhausted", "unauthenticated",
                        "permission", "api key", "timeout", "timed out")  # fmt: skip
_UNSUPPORTED_HINTS = ("invalid_argument", "invalid argument", "not supported", "unsupported",
                      "does not support", "response_schema", "json_schema", "response_format",
                      "structured output", "not found", "no such model", "model_not_found")  # fmt: skip


def is_unsupported(exc: BaseException) -> bool:
    """True when a *different* model or method could succeed (bad request / unknown model).

    Transient or account-level failures (429, 401/403, 5xx, timeouts) return False: falling
    back would only hide them.
    """
    if isinstance(exc, ModelUnsupported | NotImplementedError):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        if status in (400, 404, 422):
            return True
        if status in (401, 403, 408, 429) or status >= 500:
            return False
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(hint in text for hint in _NOT_A_MODEL_PROBLEM):
        return False
    return any(hint in text for hint in _UNSUPPORTED_HINTS)


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

    def __init__(
        self,
        models: dict[Role, BaseChatModel],
        model_names: dict[Role, str],
        *,
        fallbacks: dict[Role, tuple[BaseChatModel, str]] | None = None,
    ) -> None:
        missing = [r for r in ROLES if r not in models or r not in model_names]
        if missing:
            raise ValueError(f"LangChainLLM needs a model for every role; missing: {missing}")
        self._models = dict(models)
        self._names = dict(model_names)
        # A fallback identical to the primary model would just repeat the failure.
        self._fallbacks = {
            role: fb for role, fb in (fallbacks or {}).items() if fb and fb[1] != self._names[role]
        }

    def model_name(self, role: Role) -> str:
        return self._names[role]

    def _candidates(self, role: Role) -> list[tuple[BaseChatModel, str]]:
        primary = [(self._models[role], self._names[role])]
        return primary + ([self._fallbacks[role]] if role in self._fallbacks else [])

    # ------------------------------------------------------------------ structured
    async def structured(
        self, schema: type[T], system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> T:
        (model, model_name), *fallback = self._candidates(role)
        attempts = [(model, model_name, "json_schema"), (model, model_name, "function_calling")]
        attempts += [(m, n, "json_schema") for m, n in fallback]
        last: Exception | None = None
        for chat, chat_name, method in attempts:
            try:
                return await self._structured_with(
                    chat, chat_name, method, schema, system, user, name
                )
            except Exception as exc:
                if not (is_unsupported(exc) or isinstance(exc, LLMError)):
                    raise
                logger.warning(
                    "Structured output for %s failed on %s (%s): %s - trying the next option",
                    name or schema.__name__, chat_name, method, exc,
                )  # fmt: skip
                last = exc
        raise LLMError(f"No model could produce {schema.__name__}: {last}") from last

    async def _structured_with(
        self, chat: BaseChatModel, chat_name: str, method: str, schema: type[T],
        system: str, user: str, name: str,
    ) -> T:  # fmt: skip
        try:
            runnable = chat.with_structured_output(schema, method=method, include_raw=True)
        except (NotImplementedError, ValueError, TypeError) as exc:
            raise ModelUnsupported(f"{chat_name} cannot do {method} output: {exc}") from exc
        messages = _messages(system, user)
        last_error: Exception | None = None
        for attempt in range(2):  # one repair attempt on malformed output
            prompt = "\n".join(str(m.content) for m in messages)
            with _metered(name or schema.__name__, chat_name, prompt) as meter:
                result = await runnable.ainvoke(messages)
                meter.input_tokens, meter.output_tokens = _usage(result.get("raw"))
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
                "Invalid structured output for %s (attempt %d): %s", name, attempt + 1, last_error
            )
            messages = [
                *_messages(system, user),
                HumanMessage(
                    content=f"Your previous answer was invalid ({last_error}). "
                    "Return only a valid JSON object for the requested schema."
                ),
            ]
        raise LLMError(f"{chat_name} returned invalid {schema.__name__}: {last_error}")

    # ------------------------------------------------------------------------ text
    async def text(self, system: str, user: str, *, role: Role = "agent", name: str = "") -> str:
        last: Exception | None = None
        for chat, chat_name in self._candidates(role):
            try:
                with _metered(name or "text", chat_name, system + user) as meter:
                    message = await chat.ainvoke(_messages(system, user))
                    meter.input_tokens, meter.output_tokens = _usage(message)
                return _content_text(message.content).strip()
            except Exception as exc:
                if not is_unsupported(exc):
                    raise
                logger.warning("Text call failed on %s: %s - trying the fallback", chat_name, exc)
                last = exc
        raise LLMError(f"No model could answer: {last}") from last

    # ---------------------------------------------------------------------- stream
    async def stream(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> AsyncIterator[str]:
        last: Exception | None = None
        for chat, chat_name in self._candidates(role):
            started = False
            try:
                with _metered(name or "stream", chat_name, system + user) as meter:
                    async for chunk in chat.astream(_messages(system, user)):
                        tokens_in, tokens_out = _usage(chunk)
                        meter.input_tokens = max(meter.input_tokens, tokens_in)
                        meter.output_tokens = max(meter.output_tokens, tokens_out)
                        text = _content_text(chunk.content)
                        if text:
                            meter.text_chars += len(text)
                            started = True
                            yield text
                return
            except Exception as exc:
                # Once tokens reached the user a different model cannot continue the answer.
                if started or not is_unsupported(exc):
                    raise
                logger.warning("Stream failed on %s: %s - trying the fallback", chat_name, exc)
                last = exc
        raise LLMError(f"No model could stream an answer: {last}") from last


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
        return self.responders[key](system, user)

    async def structured(
        self, schema: type[T], system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> T:
        with _metered(schema.__name__, "fake", system + user):
            await asyncio.sleep(0)  # a real await point, so cancellation behaves as in production
            result = self._respond(schema.__name__, system, user)
        return result if isinstance(result, schema) else schema.model_validate(result)

    async def text(self, system: str, user: str, *, role: Role = "agent", name: str = "") -> str:
        with _metered(name or "text", "fake", system + user):
            return str(self._respond(name or "text", system, user))

    async def stream(
        self, system: str, user: str, *, role: Role = "agent", name: str = ""
    ) -> AsyncIterator[str]:
        with _metered(name or "stream", "fake", system + user) as meter:
            text = str(self._respond(name or "stream", system, user))
            for i in range(0, len(text), 12):
                meter.text_chars += 12
                yield text[i : i + 12]


# --------------------------------------------------------------------------- factory
class TieredLLM:
    """Routes each call to the model of the current plan (``current_plan``) for its role.

    Agents stay plan-agnostic: they call ``structured``/``text``/``stream`` and the plan of the
    conversation being served decides which model answers.
    """

    def __init__(self, per_plan: dict[str, LangChainLLM], default_plan: str) -> None:
        self._per_plan = per_plan
        self._default = default_plan

    def for_plan(self, plan: str | None = None) -> LangChainLLM:
        return (
            self._per_plan.get(plan or current_plan.get() or self._default)
            or self._per_plan[self._default]
        )

    async def structured(self, schema, system, user, *, role: Role = "agent", name: str = ""):
        return await self.for_plan().structured(schema, system, user, role=role, name=name)

    async def text(self, system, user, *, role: Role = "agent", name: str = "") -> str:
        return await self.for_plan().text(system, user, role=role, name=name)

    async def stream(self, system, user, *, role: Role = "agent", name: str = ""):
        # aclosing: if our consumer stops early, the inner stream is closed now (and metered)
        # instead of whenever the garbage collector gets to it.
        inner = self.for_plan().stream(system, user, role=role, name=name)
        async with contextlib.aclosing(inner) as tokens:
            async for token in tokens:
                yield token


def _first_other(candidates: list[str], model: str) -> str | None:
    return next((c for c in candidates if c and c != model), None)


def create_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider
    if provider == "fake":
        from app.llm.offline import offline_llm

        return offline_llm()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (or set LLM_PROVIDER=fake for offline mode).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        from app.plans import PLANS

        cache: dict[tuple[str, str], BaseChatModel] = {}

        def build(model: str, effort: str) -> BaseChatModel:
            if (model, effort) not in cache:
                cache[(model, effort)] = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=settings.gemini_api_key,
                    reasoning_effort=effort,
                    timeout=settings.llm_timeout_s,
                    max_retries=settings.llm_max_retries,
                )
            return cache[(model, effort)]

        per_plan = {}
        for plan in PLANS.values():
            names: dict[Role, str] = {
                "router": plan.router_model,
                "agent": plan.agent_model,
                "reply": plan.reply_model,
            }
            efforts: dict[Role, str] = {
                "router": settings.router_reasoning,
                "agent": plan.agent_effort,
                "reply": settings.router_reasoning,
            }
            fallbacks = {}
            for role in ROLES:
                other = _first_other(settings.fallback_models, names[role])
                if other:
                    fallbacks[role] = (build(other, "low"), other)
            per_plan[plan.id] = LangChainLLM(
                {role: build(names[role], efforts[role]) for role in ROLES},
                names,
                fallbacks=fallbacks,
            )
            logger.info("Plan %s: %s", plan.id, names)
        return TieredLLM(per_plan, settings.default_plan)

    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY is not set.")
    from langchain_openai import ChatOpenAI

    def build_openai(model: str, effort: str) -> BaseChatModel:
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            reasoning_effort=effort,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    names = {
        "router": settings.openai_router_model,
        "agent": settings.openai_agent_model,
        "reply": settings.openai_reply_model,
    }
    efforts = {
        "router": settings.router_reasoning,
        "agent": settings.agent_reasoning,
        "reply": settings.router_reasoning,
    }
    fallbacks = {}
    for role in ROLES:
        other = _first_other([settings.openai_fallback_model], names[role])
        if other:
            fallbacks[role] = (build_openai(other, "low"), other)
    logger.info("LLM provider=openai %s", names)
    return LangChainLLM(
        {role: build_openai(names[role], efforts[role]) for role in ROLES},
        names,
        fallbacks=fallbacks,
    )
