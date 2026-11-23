"""Chat service: one conversational turn end-to-end.

load state -> store user message -> run the LangGraph turn (streaming events) -> commit a new
resume version if agents changed it -> store the assistant message with its full trace ->
update conversation memory -> (maybe) summarize old turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from app.agents.base import Agent
from app.config import Settings
from app.llm.client import LLM, current_plan, start_trace
from app.models.conversation import ChatMessage, Conversation, ResumeVersion, utcnow
from app.orchestration.classifier import IntentClassifier
from app.orchestration.graph import build_graph, result_summary
from app.orchestration.route_cache import RouteCache
from app.parsing.sectionizer import parse_resume, repair_resume
from app.plans import PLANS, get_plan
from app.retrieval.vector_store import VectorStore
from app.services.context import build_history, maybe_summarize
from app.services.export import prerender
from app.services.flow import FlowRecorder
from app.services.versioning import VersionError, VersionService
from app.storage.base import Repository

logger = logging.getLogger(__name__)

AGENT_LABELS = {
    "company_research": "Company research",
    "jd_matching": "Job match",
    "translation": "Translation",
    "section_enhancement": "Section enhancement",
}


class ChatError(ValueError):
    pass


def welcome_message(name: str, sections: list[str]) -> str:
    return (
        f"Thanks{', ' + name if name else ''}! I parsed your resume into {len(sections)} sections "
        f"({', '.join(sections)}).\n\nTry asking me to:\n"
        "- **Optimize my resume for Google**\n"
        "- **Match my resume to this job description** (paste the posting)\n"
        "- **Make my skills section more relevant for a data scientist role**\n"
        "- **Translate this to Spanish for the Mexican market**"
    )


class ChatService:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        llm: LLM,
        vectors: VectorStore,
        agents: dict[str, Agent],
        *,
        route_cache: RouteCache | None = None,
        classifier: IntentClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.llm = llm
        self.vectors = vectors
        self.versions = VersionService(repo)
        self.route_cache = route_cache
        self.graph = build_graph(llm, agents, vectors, cache=route_cache, classifier=classifier)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Re-indexing the new version runs after "done"; the next turn awaits it first.
        self._indexing: dict[str, asyncio.Task] = {}
        self._background: set[asyncio.Task] = set()

    # ------------------------------------------------------------ helpers
    async def _io(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _index_later(self, conversation_id: str, version: ResumeVersion) -> None:
        """Embed the new version in the background so the reply is not held up by it.

        Re-indexes of one conversation are serialized: each waits for the previous one, so an
        edit quickly followed by an undo never runs two delete-and-insert passes at once (and
        whoever waits on the latest task waits on all of them).
        """
        previous = self._indexing.get(conversation_id)

        async def reindex() -> None:
            if previous is not None:
                with contextlib.suppress(Exception):
                    await previous
            await self._io(self.vectors.index_resume, conversation_id, version.id, version.resume)

        task = asyncio.create_task(reindex())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # logged, not raised
        self._indexing[conversation_id] = task
        self._prerender_later(conversation_id, version)

    def _prerender_later(self, conversation_id: str, version: ResumeVersion) -> None:
        """Warm the export cache so "Download PDF" for a new version is served instantly.

        Independent of indexing: nothing waits for it. Tasks are kept referenced until done
        (the event loop only holds weak references to tasks).
        """
        task = asyncio.create_task(self._io(prerender, conversation_id, version, "pdf", self.repo))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _index_ready(self, conversation_id: str) -> None:
        task = self._indexing.pop(conversation_id, None)
        if task is not None:
            try:
                await task
            except Exception:
                logger.exception("Background re-indexing failed")

    async def _add_message(self, conv: Conversation, message: ChatMessage) -> ChatMessage:
        conv.message_count += 1
        message.seq = conv.message_count
        await self._io(self.repo.add_message, conv.id, message)
        return message

    def state_payload(self, conv: Conversation, version: ResumeVersion | None) -> dict[str, Any]:
        return {
            "conversation": conv.model_dump(mode="json"),
            "current_version": version.model_dump(mode="json") if version else None,
            "can_undo": bool(version and version.parent_id),
            "can_redo": bool(conv.redo_stack),
            "plan": self.plan_of(conv),
        }

    def plan_of(self, conv: Conversation) -> str:
        return get_plan(conv.plan, self.settings.default_plan).id

    async def set_plan(self, conversation_id: str, plan: str) -> dict[str, Any]:
        if plan not in PLANS:
            raise ChatError(f"Unknown plan '{plan}'.")
        conv = await self._io(self.repo.get_conversation, conversation_id)
        conv.plan = plan
        await self._io(self.repo.save_conversation, conv)
        return self.state_payload(conv, await self._io(self.versions.current, conv))

    # ------------------------------------------------------- conversations
    async def create_conversation(self) -> Conversation:
        return await self._io(self.repo.save_conversation, Conversation())

    async def get_state(self, conversation_id: str) -> dict[str, Any]:
        conv = await self._io(self.repo.get_conversation, conversation_id)
        messages, version = await asyncio.gather(
            self._io(self.repo.list_messages, conversation_id),
            self._io(self.versions.current, conv),
        )
        return {
            **self.state_payload(conv, version),
            "messages": [m.model_dump(mode="json") for m in messages],
        }

    async def upload_resume(
        self, conversation_id: str, filename: str, data: bytes
    ) -> dict[str, Any]:
        resume = await self._io(parse_resume, filename, data)  # raises ResumeParseError
        conv = await self._io(self.repo.get_conversation, conversation_id)
        async with self._locks[conversation_id]:
            is_replacement = conv.current_version_id is not None
            version = await self._io(
                self.versions.commit,
                conv,
                resume,
                source="upload",
                label=f"Uploaded {filename}" if is_replacement else "Original upload",
            )
            header = resume.find_kind("header")
            name = header[0].content.split("\n")[0].strip()[:40] if header else ""
            conv.filename = filename
            conv.title = f"{name}'s resume" if name else filename
            # Keep the uploaded file: exports edit it in place to preserve its layout.
            await self._io(self.repo.save_original, conv.id, version.id, filename, data)
            await self._io(self.vectors.index_resume, conv.id, version.id, resume)
            self._prerender_later(conv.id, version)
            message = await self._add_message(
                conv,
                ChatMessage(
                    role="assistant",
                    content=welcome_message(name.split(" ")[0], [s.title for s in resume.sections]),
                    version_id=version.id,
                ),
            )
            await self._io(self.repo.save_conversation, conv)
        return {**self.state_payload(conv, version), "message": message.model_dump(mode="json")}

    # -------------------------------------------------------- version ops
    async def version_op(
        self, conversation_id: str, op: str, version_id: str | None = None
    ) -> dict:
        async with self._locks[conversation_id]:
            conv = await self._io(self.repo.get_conversation, conversation_id)
            if op == "undo":
                version = await self._io(self.versions.undo, conv)
            elif op == "redo":
                version = await self._io(self.versions.redo, conv)
            else:
                version = await self._io(self.versions.restore, conv, version_id)
            # Undo/redo only move a pointer; re-embedding for search happens in the background.
            self._index_later(conv.id, version)
            return self.state_payload(conv, version)

    # ---------------------------------------------------------------- chat
    async def chat(
        self,
        conversation_id: str,
        text: str,
        job_description: str | None = None,
        *,
        client_message_id: str | None = None,
        preset: str | None = None,
        selection: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn. ``client_message_id`` makes the call idempotent: a retry or a double
        submit of the same message replays the stored answer instead of running it again."""
        text = (text or "").strip()
        if not text:
            raise ChatError("Message is empty.")
        lock = self._locks[conversation_id]
        resume_from: ChatMessage | None = None
        if client_message_id:
            sent = await self._io(
                self.repo.find_message, conversation_id, client_id=client_message_id
            )
            if sent is not None:
                # Turns are serialized per conversation, so the answer is the next message.
                answer = await self._io(self.repo.find_message, conversation_id, seq=sent.seq + 1)
                if answer is not None and answer.role != "assistant":
                    answer = None
                if answer is not None:
                    async for event in self._replay(conversation_id, sent, answer):
                        yield event
                    return
                if lock.locked():
                    raise ChatError("This message is already being processed - one moment.")
                resume_from = sent  # received earlier but never answered (e.g. a crash): run it
        if lock.locked():
            raise ChatError("Still working on your previous message - one moment.")
        async with lock:
            async for event in self._turn(
                conversation_id,
                text,
                job_description,
                client_message_id,
                preset,
                resume_from,
                selection,
            ):
                yield event

    async def _replay(
        self, conversation_id: str, sent: ChatMessage, answer: ChatMessage
    ) -> AsyncIterator[dict[str, Any]]:
        conv = await self._io(self.repo.get_conversation, conversation_id)
        version = await self._io(self.versions.current, conv)
        yield {"type": "user_message", "message": sent.model_dump(mode="json"), "replayed": True}
        if answer.trace and answer.trace.get("flow"):
            yield {"type": "flow", **answer.trace["flow"]}
        yield {
            "type": "done",
            "replayed": True,
            "message": answer.model_dump(mode="json"),
            **self.state_payload(conv, version),
        }

    async def _turn(
        self,
        conversation_id: str,
        text: str,
        job_description: str | None,
        client_message_id: str | None = None,
        preset: str | None = None,
        resume_from: ChatMessage | None = None,
        selection: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        trace = start_trace()
        conv = await self._io(self.repo.get_conversation, conversation_id)
        current = await self._io(self.versions.current, conv)
        if current is None:
            raise ChatError("Upload a resume first.")

        history_msgs = await self._io(
            self.repo.recent_messages, conversation_id, self.settings.history_window + 1
        )
        memory = conv.context
        if job_description and job_description.strip():
            memory.job_description = job_description.strip()
        content = text if not job_description else f"{text}\n\n[Attached job description]"
        if selection:
            quoted = "\n".join(f"> {s.strip()}" for s in selection if s.strip())
            content = f"{content}\n\n{quoted}"
        if resume_from is not None:
            user_msg = resume_from
            history_msgs = [m for m in history_msgs if m.seq < resume_from.seq]
        else:
            user_msg = await self._add_message(
                conv, ChatMessage(role="user", content=content, client_id=client_message_id)
            )
        yield {"type": "user_message", "message": user_msg.model_dump(mode="json")}
        plan = self.plan_of(conv)
        current_plan.set(plan)  # every LLM call in this turn uses the plan's models
        flow = FlowRecorder(trace, text, bool(job_description and job_description.strip()))
        flow.nodes[0]["detail"].update(
            plan=plan, preset=preset, selected_lines=len(selection or [])
        )
        yield {"type": "flow", **flow.snapshot()}

        state = {
            "conversation_id": conversation_id,
            "user_message": text,
            "history": build_history(history_msgs, memory, self.settings.history_window),
            "memory": memory,
            # Agents work on a repaired copy (old uploads may hold PDF-wrapped lines).
            "resume": repair_resume(current.resume),
            "preset": preset,
            "selection": [s.strip() for s in (selection or []) if s.strip()][:10],
            # Only the answer step reads the resume index; it waits for any re-indexing still
            # running from the previous turn. Every other path starts immediately.
            "index_ready": lambda: self._index_ready(conversation_id),
        }
        prepared = time.perf_counter()
        final: dict[str, Any] = {}
        async for mode, chunk in self.graph.astream(state, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield chunk
                if flow.observe(chunk):
                    yield {"type": "flow", **flow.snapshot()}
            else:
                final = chunk

        graph_done = time.perf_counter()
        decision = final["decision"]
        memory = final["memory"]
        reply = final.get("reply", "")
        flow.response_done(reply)
        version: ResumeVersion | None = None
        if decision.intent in ("undo", "redo"):
            try:
                version = await self._io(getattr(self.versions, decision.intent), conv)
                self._index_later(conv.id, version)
                reply = (
                    f"{'Reverted' if decision.intent == 'undo' else 'Re-applied'} - you're now on "
                    f"version {version.number} ({version.label})."
                )
                flow.version_saved(version.number, version.label, [], op=decision.intent)
            except VersionError as exc:
                reply = str(exc)
                flow.version_failed(reply)
            yield {"type": "token", "text": reply}

        results = final.get("results", [])
        changed = [r for r in results if r.changed]
        if changed:
            flow.version_started()
            agents = [r.agent for r in changed]
            version = await self._io(
                self.versions.commit,
                conv,
                final["working"],
                source="agents",
                label=" + ".join(AGENT_LABELS.get(a, a) for a in agents),
                agents=agents,
            )
            if version is not None:
                self._index_later(conv.id, version)
                flow.version_saved(
                    version.number, version.label, [c.title for c in version.changes]
                )
                yield {
                    "type": "version",
                    "version": version.model_dump(mode="json"),
                    "can_undo": True,
                    "can_redo": False,
                }
                memory.last_action = f"{version.label} edited " + ", ".join(
                    sorted({c.title for c in version.changes})
                )

        flow.close()
        final_flow = flow.snapshot()
        yield {"type": "flow", **final_flow}
        assistant = ChatMessage(
            role="assistant",
            content=reply or "Done.",
            version_id=version.id if version and changed else None,
            trace={
                "intent": decision.intent,
                "reasoning": decision.reasoning,
                "plan": [s.model_dump() for s in decision.steps],
                "results": [result_summary(r) for r in results],
                "llm": trace.as_dict(),
                "ms": int((time.perf_counter() - started) * 1000),
                "flow": final_flow,
                "subscription_plan": plan,
                "timings_ms": {
                    "prepare": int((prepared - started) * 1000),
                    "graph": int((graph_done - prepared) * 1000),
                    "persist": int((time.perf_counter() - graph_done) * 1000),
                },
                "cost_usd": final_flow["cost_usd"],
                "decided_by": next(
                    (
                        n["detail"].get("decided_by")
                        for n in final_flow["nodes"]
                        if n["id"] == "router"
                    ),
                    None,
                ),
            },
        )
        assistant = await self._add_message(conv, assistant)

        try:
            # The full history is loaded only when old turns actually need folding into the
            # rolling summary (rare); a normal turn reads just the recent window.
            if conv.message_count - memory.summarized_through > self.settings.summary_trigger:
                all_messages = await self._io(self.repo.list_messages, conversation_id)
            else:
                all_messages = []
            memory = await maybe_summarize(
                self.llm,
                all_messages,
                memory,
                window=self.settings.history_window,
                trigger=self.settings.summary_trigger,
            )
        except Exception:  # summarization is best-effort
            logger.exception("History summarization failed")
        conv.context = memory
        conv.updated_at = utcnow()
        await self._io(self.repo.save_conversation, conv)
        current_version = await self._io(self.versions.current, conv)
        yield {
            "type": "done",
            "message": assistant.model_dump(mode="json"),
            **self.state_payload(conv, current_version),
        }
