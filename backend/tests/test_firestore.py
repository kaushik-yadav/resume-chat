"""Round-trip tests against the Firestore emulator.

Skipped unless FIRESTORE_EMULATOR_HOST is set, e.g.:

    npm run emulators            # from the repo root
    FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 pytest tests/test_firestore.py
"""

import os
import uuid

import pytest

from app.config import Settings
from app.models.conversation import ChatMessage, Conversation
from app.models.resume import Resume, ResumeSection
from app.services.versioning import VersionService
from app.storage.base import NotFoundError
from app.storage.firestore import FirestoreRepository, build_client

EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST")
pytestmark = pytest.mark.skipif(not EMULATOR, reason="Firestore emulator not running")


@pytest.fixture
def repo():
    settings = Settings(
        firestore_emulator_host=EMULATOR,
        firebase_project_id="demo-resume-chat",
        firestore_transport=os.environ.get("FIRESTORE_TRANSPORT", "grpc"),
    )
    # Unique root collection per test run keeps tests isolated.
    return FirestoreRepository(build_client(settings), f"test-{uuid.uuid4().hex[:8]}")


def test_conversation_messages_and_versions_round_trip(repo):
    conv = repo.save_conversation(Conversation(title="Test", filename="cv.pdf"))
    assert repo.get_conversation(conv.id).filename == "cv.pdf"

    repo.add_message(conv.id, ChatMessage(seq=2, role="assistant", content="Done", trace={"a": 1}))
    repo.add_message(conv.id, ChatMessage(seq=1, role="user", content="Hi"))
    assert [m.content for m in repo.list_messages(conv.id)] == ["Hi", "Done"]

    service = VersionService(repo)
    base = Resume(
        sections=[ResumeSection(id="skills", title="Skills", kind="skills", content="Go")]
    )
    v1 = service.commit(conv, base, source="upload")
    v2 = service.commit(conv, base.with_section_content("skills", "Go, Rust"))
    assert [v.number for v in repo.list_versions(conv.id)] == [1, 2]
    assert repo.get_version(conv.id, v2.id).changes[0].after == "Go, Rust"

    service.undo(conv)
    stored = repo.get_conversation(conv.id)
    assert stored.current_version_id == v1.id and stored.redo_stack == [v2.id]
    assert repo.list_conversations()[0].id == conv.id

    repo.delete_conversation(conv.id)
    with pytest.raises(NotFoundError):
        repo.get_conversation(conv.id)
