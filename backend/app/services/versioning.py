"""Resume version history: commit, undo, redo, restore and compare.

History model
-------------
Every change creates an immutable ``ResumeVersion`` that points at its parent. The
conversation keeps a ``current_version_id`` pointer plus a ``redo_stack``:

* commit  - new version (parent = current), pointer moves to it, redo stack is cleared
* undo    - pointer moves to the parent, the old current is pushed on the redo stack
* redo    - pointer moves to the top of the redo stack
* restore - copies any earlier version into a *new* version, so history is never rewritten
"""

from __future__ import annotations

from app.models.conversation import Conversation, ResumeVersion, SectionChange, utcnow
from app.models.resume import Resume
from app.services.diff import diff_resumes
from app.storage.base import Repository


class VersionError(ValueError):
    pass


class VersionService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    # ------------------------------------------------------------------ reads
    def current(self, conversation: Conversation) -> ResumeVersion | None:
        if not conversation.current_version_id:
            return None
        return self.repo.get_version(conversation.id, conversation.current_version_id)

    def current_resume(self, conversation: Conversation) -> Resume | None:
        version = self.current(conversation)
        return version.resume if version else None

    def history(self, conversation: Conversation) -> list[ResumeVersion]:
        return self.repo.list_versions(conversation.id)

    # ----------------------------------------------------------------- writes
    def commit(
        self,
        conversation: Conversation,
        resume: Resume,
        *,
        source: str = "agents",
        label: str = "",
        agents: list[str] | None = None,
        message_id: str | None = None,
    ) -> ResumeVersion | None:
        """Store ``resume`` as the new current version. Returns None if nothing changed."""
        parent = self.current(conversation)
        changes = diff_resumes(parent.resume if parent else None, resume)
        if parent is not None and not changes and parent.resume.language == resume.language:
            return None

        versions = self.repo.list_versions(conversation.id)
        version = ResumeVersion(
            number=(max((v.number for v in versions), default=0) + 1),
            parent_id=parent.id if parent else None,
            source=source,  # type: ignore[arg-type]
            label=label or ("Original upload" if parent is None else "Updated resume"),
            agents=agents or [],
            resume=resume,
            changes=changes,
            message_id=message_id,
        )
        self.repo.add_version(conversation.id, version)
        conversation.current_version_id = version.id
        conversation.redo_stack = []
        self._touch(conversation)
        return version

    def undo(self, conversation: Conversation) -> ResumeVersion:
        current = self.current(conversation)
        if current is None or current.parent_id is None:
            raise VersionError("Nothing to undo - this is the original resume.")
        conversation.redo_stack.append(current.id)
        conversation.current_version_id = current.parent_id
        self._touch(conversation)
        return self.repo.get_version(conversation.id, current.parent_id)

    def redo(self, conversation: Conversation) -> ResumeVersion:
        if not conversation.redo_stack:
            raise VersionError("Nothing to redo.")
        version_id = conversation.redo_stack.pop()
        conversation.current_version_id = version_id
        self._touch(conversation)
        return self.repo.get_version(conversation.id, version_id)

    def restore(self, conversation: Conversation, version_id: str) -> ResumeVersion:
        target = self.repo.get_version(conversation.id, version_id)
        if target.id == conversation.current_version_id:
            raise VersionError(f"Version {target.number} is already the current version.")
        version = self.commit(
            conversation,
            target.resume,
            source="restore",
            label=f"Restored version {target.number}",
        )
        if version is None:  # identical content: just move the pointer
            conversation.current_version_id = target.id
            self._touch(conversation)
            return target
        return version

    def compare(self, conversation: Conversation, from_id: str, to_id: str) -> list[SectionChange]:
        old = self.repo.get_version(conversation.id, from_id)
        new = self.repo.get_version(conversation.id, to_id)
        return diff_resumes(old.resume, new.resume)

    # ---------------------------------------------------------------- helpers
    def can_undo(self, conversation: Conversation) -> bool:
        current = self.current(conversation)
        return bool(current and current.parent_id)

    def _touch(self, conversation: Conversation) -> None:
        conversation.updated_at = utcnow()
        self.repo.save_conversation(conversation)
