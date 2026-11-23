"""Persistence interface for conversations, messages and resume versions.

Firestore layout (mirrored by the in-memory backend):

    conversations/{conversation_id}                      Conversation
    conversations/{conversation_id}/messages/{id}        ChatMessage   (ordered by seq)
    conversations/{conversation_id}/versions/{id}        ResumeVersion (ordered by number)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.conversation import ChatMessage, Conversation, ResumeVersion


class NotFoundError(KeyError):
    pass


class Repository(ABC):
    # ------------------------------------------------------------ conversations
    @abstractmethod
    def save_conversation(self, conversation: Conversation) -> Conversation: ...

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation:
        """Raise ``NotFoundError`` when missing."""

    @abstractmethod
    def list_conversations(self, limit: int = 50) -> list[Conversation]: ...

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None: ...

    # ----------------------------------------------------------------- messages
    @abstractmethod
    def add_message(self, conversation_id: str, message: ChatMessage) -> ChatMessage: ...

    @abstractmethod
    def list_messages(self, conversation_id: str) -> list[ChatMessage]: ...

    # Targeted reads used on every chat turn. The defaults are correct for any backend;
    # Firestore overrides them with indexed queries so a turn never loads the whole history.
    def recent_messages(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        return self.list_messages(conversation_id)[-limit:] if limit else []

    def find_message(
        self, conversation_id: str, *, client_id: str | None = None, seq: int | None = None
    ) -> ChatMessage | None:
        return next(
            (
                m
                for m in self.list_messages(conversation_id)
                if (client_id is None or m.client_id == client_id) and (seq is None or m.seq == seq)
            ),
            None,
        )

    # ----------------------------------------------------------------- versions
    @abstractmethod
    def add_version(self, conversation_id: str, version: ResumeVersion) -> ResumeVersion: ...

    @abstractmethod
    def get_version(self, conversation_id: str, version_id: str) -> ResumeVersion:
        """Raise ``NotFoundError`` when missing."""

    @abstractmethod
    def list_versions(self, conversation_id: str) -> list[ResumeVersion]: ...

    def latest_version_number(self, conversation_id: str) -> int:
        return max((v.number for v in self.list_versions(conversation_id)), default=0)

    # ---------------------------------------------------------------- originals
    # The uploaded file itself, keyed by the version its upload created: exports edit it in
    # place to keep the original layout. Backends that cannot store files return None.
    def save_original(self, conversation_id: str, upload_id: str, filename: str, data: bytes) -> None:
        return None

    def get_original(self, conversation_id: str, upload_id: str) -> tuple[str, bytes] | None:
        return None
