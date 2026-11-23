"""Conversation, message and resume-version models (persisted in Firestore)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.resume import Resume


def new_id() -> str:
    return uuid.uuid4().hex[:20]


def utcnow() -> datetime:
    return datetime.now(UTC)


class ConversationContext(BaseModel):
    """Structured working memory carried across turns.

    The router reads these slots instead of re-deriving them from raw history, so follow-ups
    like "now tailor the summary for them too" resolve correctly and cheaply.
    """

    target_company: str | None = None
    target_role: str | None = None
    target_locale: str | None = None
    job_description: str | None = None
    last_action: str | None = None
    # Rolling summary of turns that have dropped out of the verbatim history window.
    summary: str = ""
    summarized_through: int = 0  # number of messages already folded into ``summary``


class Conversation(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str = "New conversation"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    filename: str | None = None
    current_version_id: str | None = None
    redo_stack: list[str] = Field(default_factory=list)
    context: ConversationContext = Field(default_factory=ConversationContext)
    message_count: int = 0


class ChatMessage(BaseModel):
    id: str = Field(default_factory=new_id)
    seq: int = 0
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    # Version produced by this turn (assistant messages that changed the resume).
    version_id: str | None = None
    # Agent plan, reasoning and per-agent results, rendered in the UI's reasoning panel.
    trace: dict[str, Any] | None = None


class SectionChange(BaseModel):
    section_id: str
    title: str
    change_type: Literal["modified", "added", "removed"]
    before: str = ""
    after: str = ""
    words_added: int = 0
    words_removed: int = 0


class ResumeVersion(BaseModel):
    id: str = Field(default_factory=new_id)
    number: int
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    source: Literal["upload", "agents", "restore"] = "agents"
    label: str = ""
    agents: list[str] = Field(default_factory=list)
    resume: Resume
    changes: list[SectionChange] = Field(default_factory=list)
    message_id: str | None = None

    def summary(self) -> dict[str, Any]:
        """Lightweight listing entry (without the full resume body)."""
        return {
            "id": self.id,
            "number": self.number,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "source": self.source,
            "label": self.label,
            "agents": self.agents,
            "changed_sections": [c.section_id for c in self.changes],
        }
