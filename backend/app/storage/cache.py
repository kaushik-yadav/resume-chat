"""Write-through in-memory cache in front of the persistent repository.

Why it is safe and effective here:

* **Resume versions are immutable** once written, so a cached version can never go stale. Undo,
  redo, restore and compare only read versions they already know about, so they are served from
  memory instead of Firestore.
* **Writes always go to Firestore first**, then into the cache (write-through), so Firestore
  stays the source of truth and a restart loses nothing.
* Conversations and messages are only modified through this process, so the cached copy
  is the latest one.

Scope: one API process. For several replicas behind a load balancer, put the same interface
on Redis, or route each conversation to one replica (sticky sessions); the design does not
change. Memory is bounded by an LRU over conversations.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from app.models.conversation import ChatMessage, Conversation, ResumeVersion
from app.storage.base import Repository


@dataclass
class _Entry:
    conversation: Conversation | None = None
    versions: dict[str, ResumeVersion] = field(default_factory=dict)
    versions_complete: bool = False  # every version of the conversation is in ``versions``
    messages: list[ChatMessage] | None = None  # full, ordered history once loaded


class CachedRepository(Repository):
    def __init__(self, inner: Repository, max_conversations: int = 128) -> None:
        self.inner = inner
        self._max = max_conversations
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self.hits = self.misses = 0

    # --------------------------------------------------------------- internals
    def _entry(self, conversation_id: str) -> _Entry:
        with self._lock:
            entry = self._entries.get(conversation_id)
            if entry is None:
                entry = self._entries[conversation_id] = _Entry()
            self._entries.move_to_end(conversation_id)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
            return entry

    def _hit(self, found: bool) -> bool:
        if found:
            self.hits += 1
        else:
            self.misses += 1
        return found

    # ----------------------------------------------------------- conversations
    def save_conversation(self, conversation: Conversation) -> Conversation:
        self.inner.save_conversation(conversation)
        with self._lock:
            entry = self._entry(conversation.id)
            if (
                entry.conversation is None
                and conversation.message_count == 0
                and conversation.current_version_id is None
            ):
                # Brand-new conversation: we know it has no messages or versions yet.
                entry.messages, entry.versions_complete = [], True
            entry.conversation = conversation.model_copy(deep=True)
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._lock:
            cached = self._entry(conversation_id).conversation
            if self._hit(cached is not None):
                return cached.model_copy(deep=True)
        conversation = self.inner.get_conversation(conversation_id)  # raises NotFoundError
        with self._lock:
            self._entry(conversation_id).conversation = conversation.model_copy(deep=True)
        return conversation

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        return self.inner.list_conversations(limit)  # cross-conversation listing: not cached

    def delete_conversation(self, conversation_id: str) -> None:
        self.inner.delete_conversation(conversation_id)
        with self._lock:
            self._entries.pop(conversation_id, None)

    # ---------------------------------------------------------------- messages
    def add_message(self, conversation_id: str, message: ChatMessage) -> ChatMessage:
        self.inner.add_message(conversation_id, message)
        with self._lock:
            entry = self._entry(conversation_id)
            if entry.messages is not None:
                entry.messages.append(message.model_copy(deep=True))
                entry.messages.sort(key=lambda m: m.seq)
        return message

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        with self._lock:
            cached = self._entry(conversation_id).messages
            if self._hit(cached is not None):
                return [m.model_copy(deep=True) for m in cached]
        messages = self.inner.list_messages(conversation_id)
        with self._lock:
            self._entry(conversation_id).messages = [m.model_copy(deep=True) for m in messages]
        return messages

    def recent_messages(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        with self._lock:
            cached = self._entry(conversation_id).messages
            if self._hit(cached is not None):
                return [m.model_copy(deep=True) for m in cached[-limit:]] if limit else []
        return self.inner.recent_messages(conversation_id, limit)

    def find_message(
        self, conversation_id: str, *, client_id: str | None = None, seq: int | None = None
    ) -> ChatMessage | None:
        with self._lock:
            cached = self._entry(conversation_id).messages
            if self._hit(cached is not None):
                found = next(
                    (
                        m
                        for m in cached
                        if (client_id is None or m.client_id == client_id)
                        and (seq is None or m.seq == seq)
                    ),
                    None,
                )
                return found.model_copy(deep=True) if found else None
        return self.inner.find_message(conversation_id, client_id=client_id, seq=seq)

    # ---------------------------------------------------------------- versions
    def add_version(self, conversation_id: str, version: ResumeVersion) -> ResumeVersion:
        self.inner.add_version(conversation_id, version)
        with self._lock:
            self._entry(conversation_id).versions[version.id] = version.model_copy(deep=True)
        return version

    def get_version(self, conversation_id: str, version_id: str) -> ResumeVersion:
        with self._lock:
            cached = self._entry(conversation_id).versions.get(version_id)
            if self._hit(cached is not None):
                return cached.model_copy(deep=True)
        version = self.inner.get_version(conversation_id, version_id)
        with self._lock:
            self._entry(conversation_id).versions[version.id] = version.model_copy(deep=True)
        return version

    def list_versions(self, conversation_id: str) -> list[ResumeVersion]:
        with self._lock:
            entry = self._entry(conversation_id)
            if self._hit(entry.versions_complete):
                ordered = sorted(entry.versions.values(), key=lambda v: v.number)
                return [v.model_copy(deep=True) for v in ordered]
        versions = self.inner.list_versions(conversation_id)
        with self._lock:
            entry = self._entry(conversation_id)
            entry.versions.update({v.id: v.model_copy(deep=True) for v in versions})
            entry.versions_complete = True
        return versions

    def latest_version_number(self, conversation_id: str) -> int:
        with self._lock:
            entry = self._entry(conversation_id)
            if self._hit(entry.versions_complete):
                return max((v.number for v in entry.versions.values()), default=0)
        return self.inner.latest_version_number(conversation_id)
