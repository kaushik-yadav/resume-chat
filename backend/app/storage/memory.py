"""In-process repository: used by tests and for running without Firebase."""

from __future__ import annotations

import threading

from app.models.conversation import ChatMessage, Conversation, ResumeVersion
from app.storage.base import NotFoundError, Repository


class MemoryRepository(Repository):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: dict[str, Conversation] = {}
        self._messages: dict[str, list[ChatMessage]] = {}
        self._versions: dict[str, dict[str, ResumeVersion]] = {}

    def save_conversation(self, conversation: Conversation) -> Conversation:
        with self._lock:
            self._conversations[conversation.id] = conversation.model_copy(deep=True)
            self._messages.setdefault(conversation.id, [])
            self._versions.setdefault(conversation.id, {})
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._lock:
            if conversation_id not in self._conversations:
                raise NotFoundError(conversation_id)
            return self._conversations[conversation_id].model_copy(deep=True)

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        with self._lock:
            items = sorted(self._conversations.values(), key=lambda c: c.updated_at, reverse=True)
            return [c.model_copy(deep=True) for c in items[:limit]]

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)
            self._messages.pop(conversation_id, None)
            self._versions.pop(conversation_id, None)

    def add_message(self, conversation_id: str, message: ChatMessage) -> ChatMessage:
        with self._lock:
            self._require(conversation_id)
            self._messages[conversation_id].append(message.model_copy(deep=True))
        return message

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        with self._lock:
            self._require(conversation_id)
            msgs = sorted(self._messages[conversation_id], key=lambda m: m.seq)
            return [m.model_copy(deep=True) for m in msgs]

    def add_version(self, conversation_id: str, version: ResumeVersion) -> ResumeVersion:
        with self._lock:
            self._require(conversation_id)
            self._versions[conversation_id][version.id] = version.model_copy(deep=True)
        return version

    def get_version(self, conversation_id: str, version_id: str) -> ResumeVersion:
        with self._lock:
            self._require(conversation_id)
            version = self._versions[conversation_id].get(version_id)
            if version is None:
                raise NotFoundError(version_id)
            return version.model_copy(deep=True)

    def list_versions(self, conversation_id: str) -> list[ResumeVersion]:
        with self._lock:
            self._require(conversation_id)
            versions = sorted(self._versions[conversation_id].values(), key=lambda v: v.number)
            return [v.model_copy(deep=True) for v in versions]

    def _require(self, conversation_id: str) -> None:
        if conversation_id not in self._conversations:
            raise NotFoundError(conversation_id)
