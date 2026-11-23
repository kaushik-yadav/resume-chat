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

    # ----------------------------------------------------------------- versions
    @abstractmethod
    def add_version(self, conversation_id: str, version: ResumeVersion) -> ResumeVersion: ...

    @abstractmethod
    def get_version(self, conversation_id: str, version_id: str) -> ResumeVersion:
        """Raise ``NotFoundError`` when missing."""

    @abstractmethod
    def list_versions(self, conversation_id: str) -> list[ResumeVersion]: ...
