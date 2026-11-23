"""Chat service: one conversational turn end-to-end.

load state -> store user message -> run the LangGraph turn (streaming events) -> commit a new
resume version if agents changed it -> store the assistant message with its full trace ->
update conversation memory -> (maybe) summarize old turns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from app.agents.base import Agent
from app.config import Settings
from app.llm.client import LLM, start_trace
from app.models.conversation import ChatMessage, Conversation, ResumeVersion, utcnow
from app.orchestration.graph import build_graph, result_summary
from app.parsing.sectionizer import parse_resume
from app.retrieval.vector_store import VectorStore
from app.services.context import build_history, maybe_summarize
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
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.llm = llm
        self.vectors = vectors
        self.versions = VersionService(repo)
        self.graph = build_graph(llm, agents, vectors)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ------------------------------------------------------------ helpers
    async def _io(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

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
        }

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
            await self._io(self.vectors.index_resume, conv.id, version.id, resume)
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
            await self._io(self.vectors.index_resume, conv.id, version.id, version.resume)
            return self.state_payload(conv, version)

    # ---------------------------------------------------------------- chat
    async def chat(
        self, conversation_id: str, text: str, job_description: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            raise ChatError("Message is empty.")
        lock = self._locks[conversation_id]
        if lock.locked():
            raise ChatError("Still working on your previous message - one moment.")
        async with lock:
            async for event in self._turn(conversation_id, text, job_description):
                yield event

    async def _turn(
        self, conversation_id: str, text: str, job_description: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        trace = start_trace()
        conv = await self._io(self.repo.get_conversation, conversation_id)
        current = await self._io(self.versions.current, conv)
        if current is None:
            raise ChatError("Upload a resume first.")

        history_msgs = await self._io(self.repo.list_messages, conversation_id)
        memory = conv.context
        if job_description and job_description.strip():
            memory.job_description = job_description.strip()
        content = text if not job_description else f"{text}\n\n[Attached job description]"
        user_msg = await self._add_message(conv, ChatMessage(role="user", content=content))
        yield {"type": "user_message", "message": user_msg.model_dump(mode="json")}

        state = {
            "conversation_id": conversation_id,
            "user_message": text,
            "history": build_history(history_msgs, memory, self.settings.history_window),
            "memory": memory,
            "resume": current.resume,
        }
        final: dict[str, Any] = {}
        async for mode, chunk in self.graph.astream(state, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield chunk
            else:
                final = chunk

        decision = final["decision"]
        memory = final["memory"]
        reply = final.get("reply", "")
        version: ResumeVersion | None = None
        if decision.intent in ("undo", "redo"):
            try:
                version = await self._io(getattr(self.versions, decision.intent), conv)
                await self._io(self.vectors.index_resume, conv.id, version.id, version.resume)
                reply = (
                    f"{'Reverted' if decision.intent == 'undo' else 'Re-applied'} - you're now on "
                    f"version {version.number} ({version.label})."
                )
            except VersionError as exc:
                reply = str(exc)
            yield {"type": "token", "text": reply}

        results = final.get("results", [])
        changed = [r for r in results if r.changed]
        if changed:
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
                await self._io(self.vectors.index_resume, conv.id, version.id, version.resume)
                yield {
                    "type": "version",
                    "version": version.model_dump(mode="json"),
                    "can_undo": True,
                    "can_redo": False,
                }
                memory.last_action = f"{version.label} edited " + ", ".join(
                    sorted({c.title for c in version.changes})
                )

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
            },
        )
        assistant = await self._add_message(conv, assistant)

        all_messages = [*history_msgs, user_msg, assistant]
        try:
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
