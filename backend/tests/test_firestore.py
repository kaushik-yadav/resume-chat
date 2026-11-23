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


def test_targeted_queries(repo):
    conv = repo.save_conversation(Conversation(title="Q"))
    for seq in range(1, 6):
        role = "user" if seq % 2 else "assistant"
        repo.add_message(
            conv.id, ChatMessage(seq=seq, role=role, content=str(seq), client_id=f"c{seq}")
        )
    assert [m.seq for m in repo.recent_messages(conv.id, 2)] == [4, 5]
    assert repo.find_message(conv.id, client_id="c3").seq == 3
    assert repo.find_message(conv.id, seq=4).content == "4"
    assert repo.find_message(conv.id, client_id="nope") is None
    assert repo.latest_version_number(conv.id) == 0
    service = VersionService(repo)
    base = Resume(sections=[ResumeSection(id="s", title="S", kind="skills", content="Go")])
    service.commit(conv, base, source="upload")
    service.commit(conv, base.with_section_content("s", "Rust"))
    assert repo.latest_version_number(conv.id) == 2


def test_heavy_fields_are_json_and_legacy_docs_still_load(repo):
    conv = repo.save_conversation(Conversation(title="J"))
    trace = {"intent": "edit", "flow": {"nodes": [{"id": "router", "detail": {"a": [1, 2]}}]}}
    repo.add_message(
        conv.id, ChatMessage(id="m1", seq=1, role="assistant", content="x", trace=trace)
    )
    raw = repo._conv_ref(conv.id).collection("messages").document("m1").get().to_dict()
    assert "trace_json" in raw and "trace" not in raw
    assert repo.list_messages(conv.id)[0].trace == trace
    # A document written before the JSON encoding (nested map) is still readable.
    legacy = ChatMessage(id="m0", seq=0, role="user", content="old", trace={"intent": "question"})
    repo._conv_ref(conv.id).collection("messages").document("m0").set(
        legacy.model_dump(mode="python")
    )
    assert repo.list_messages(conv.id)[0].trace == {"intent": "question"}
