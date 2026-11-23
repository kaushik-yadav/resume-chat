import pytest

from app.models.conversation import Conversation
from app.models.resume import Resume, ResumeSection
from app.services.diff import diff_resumes, unified_diff
from app.services.versioning import VersionError, VersionService
from app.storage.memory import MemoryRepository


def resume(skills: str = "Python, SQL", summary: str = "Engineer.") -> Resume:
    return Resume(
        sections=[
            ResumeSection(id="summary", title="Summary", kind="summary", content=summary),
            ResumeSection(id="skills", title="Skills", kind="skills", content=skills),
        ]
    )


@pytest.fixture
def setup():
    repo = MemoryRepository()
    conv = repo.save_conversation(Conversation())
    service = VersionService(repo)
    v1 = service.commit(conv, resume(), source="upload")
    return repo, conv, service, v1


def test_first_commit_is_original_upload(setup):
    _, conv, service, v1 = setup
    assert v1.number == 1 and v1.parent_id is None and v1.label == "Original upload"
    assert conv.current_version_id == v1.id
    assert not service.can_undo(conv)


def test_commit_records_section_changes(setup):
    _, conv, service, v1 = setup
    v2 = service.commit(conv, resume(skills="Python, SQL, pandas"), agents=["section_enhancement"])
    assert v2.number == 2 and v2.parent_id == v1.id
    assert [c.section_id for c in v2.changes] == ["skills"]
    # "SQL" -> "SQL," counts as a replacement, plus the inserted "pandas".
    assert v2.changes[0].words_added == 2 and v2.changes[0].words_removed == 1


def test_commit_without_changes_is_a_noop(setup):
    _, conv, service, v1 = setup
    assert service.commit(conv, resume()) is None
    assert conv.current_version_id == v1.id


def test_undo_redo_round_trip(setup):
    repo, conv, service, v1 = setup
    v2 = service.commit(conv, resume(skills="Go"))
    v3 = service.commit(conv, resume(skills="Rust"))

    assert service.undo(conv).id == v2.id
    assert service.undo(conv).id == v1.id
    with pytest.raises(VersionError):
        service.undo(conv)

    assert service.redo(conv).id == v2.id
    assert service.redo(conv).id == v3.id
    with pytest.raises(VersionError):
        service.redo(conv)
    # Pointer changes are persisted.
    assert repo.get_conversation(conv.id).current_version_id == v3.id


def test_new_commit_after_undo_clears_redo_stack(setup):
    _, conv, service, v1 = setup
    service.commit(conv, resume(skills="Go"))
    service.undo(conv)
    v3 = service.commit(conv, resume(skills="Kotlin"))
    assert v3.number == 3 and v3.parent_id == v1.id
    assert conv.redo_stack == []


def test_restore_creates_new_version(setup):
    _, conv, service, v1 = setup
    service.commit(conv, resume(skills="Go"))
    restored = service.restore(conv, v1.id)
    assert restored.number == 3 and restored.source == "restore"
    assert restored.resume.get("skills").content == "Python, SQL"
    with pytest.raises(VersionError):
        service.restore(conv, restored.id)


def test_compare_versions(setup):
    _, conv, service, v1 = setup
    v2 = service.commit(conv, resume(skills="Go", summary="Senior engineer."))
    changes = service.compare(conv, v1.id, v2.id)
    assert {c.section_id for c in changes} == {"summary", "skills"}


def test_diff_detects_added_and_removed_sections():
    old = resume()
    new = Resume(
        sections=[
            ResumeSection(id="summary", title="Summary", kind="summary", content="Engineer."),
            ResumeSection(id="projects", title="Projects", kind="projects", content="A B C"),
        ]
    )
    kinds = {c.section_id: c.change_type for c in diff_resumes(old, new)}
    assert kinds == {"projects": "added", "skills": "removed"}


def test_unified_diff_is_compact():
    out = unified_diff("a\nb\nc", "a\nB\nc")
    assert "-b" in out and "+B" in out and "---" not in out
