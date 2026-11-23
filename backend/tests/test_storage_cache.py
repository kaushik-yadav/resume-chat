import pytest

from app.models.conversation import ChatMessage, Conversation
from app.models.resume import Resume, ResumeSection
from app.services.versioning import VersionService
from app.storage.base import NotFoundError
from app.storage.cache import CachedRepository
from app.storage.memory import MemoryRepository


class CountingRepo(MemoryRepository):
    """Memory repository that counts calls, to prove which reads reach the backing store."""

    def __init__(self):
        super().__init__()
        self.reads = 0
        self.writes = 0

    def _count(self, name):
        if name.startswith(("get_", "list_", "recent_", "find_", "latest_")):
            self.reads += 1
        else:
            self.writes += 1

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr) and not name.startswith("_") and name not in ("reads", "writes"):
            super().__getattribute__("_count")(name)
        return attr


def resume(skills: str) -> Resume:
    return Resume(
        sections=[ResumeSection(id="skills", title="Skills", kind="skills", content=skills)]
    )


@pytest.fixture
def stores():
    inner = CountingRepo()
    return inner, CachedRepository(inner)


def test_undo_redo_are_served_from_memory(stores):
    inner, repo = stores
    conv = repo.save_conversation(Conversation())
    service = VersionService(repo)
    service.commit(conv, resume("Go"), source="upload")
    service.commit(conv, resume("Rust"))
    reads_before = inner.reads

    for _ in range(3):
        service.undo(conv)
        service.redo(conv)
    assert inner.reads == reads_before  # every version/conversation read hit the cache
    assert repo.get_version(conv.id, conv.current_version_id).resume.get("skills").content == "Rust"
    # Writes still went through to the backing store.
    assert inner.get_conversation(conv.id).current_version_id == conv.current_version_id


def test_new_conversation_is_known_complete(stores):
    inner, repo = stores
    conv = repo.save_conversation(Conversation())
    reads_before = inner.reads
    assert repo.list_messages(conv.id) == []
    assert repo.latest_version_number(conv.id) == 0
    repo.add_message(conv.id, ChatMessage(seq=1, role="user", content="hi", client_id="c1"))
    assert repo.find_message(conv.id, client_id="c1").content == "hi"
    assert [m.content for m in repo.recent_messages(conv.id, 5)] == ["hi"]
    assert inner.reads == reads_before


def test_cold_reads_fill_the_cache_once(stores):
    inner, repo = stores
    conv = inner.save_conversation(Conversation())  # written by "another process"
    inner.add_message(conv.id, ChatMessage(seq=1, role="user", content="hello"))
    repo.list_messages(conv.id)
    reads = inner.reads
    repo.list_messages(conv.id)
    repo.get_conversation(conv.id)
    repo.get_conversation(conv.id)
    assert inner.reads == reads + 1  # messages cached; conversation fetched once


def test_cached_objects_are_copies(stores):
    _, repo = stores
    conv = repo.save_conversation(Conversation(title="Original"))
    fetched = repo.get_conversation(conv.id)
    fetched.title = "Mutated by caller"
    assert repo.get_conversation(conv.id).title == "Original"


def test_lru_eviction_and_delete(stores):
    inner, _ = stores
    repo = CachedRepository(inner, max_conversations=2)
    ids = [repo.save_conversation(Conversation()).id for _ in range(3)]
    reads = inner.reads
    repo.get_conversation(ids[0])  # evicted -> read from the backing store
    assert inner.reads == reads + 1
    repo.delete_conversation(ids[1])
    with pytest.raises(NotFoundError):
        repo.get_conversation(ids[1])
