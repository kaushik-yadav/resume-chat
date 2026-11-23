import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.client import FakeLLM, LLMError, start_trace
from app.models.resume import Resume, ResumeSection
from app.retrieval.embeddings import HashEmbedder, create_embedder
from app.retrieval.vector_store import VectorStore, chunk_resume, company_key
from app.tools.web_search import SearchResult, WebSearch, create_web_search


class Answer(BaseModel):
    value: int


# ------------------------------------------------------------------------------ LLM
async def test_fake_llm_structured_text_and_stream_are_traced():
    llm = FakeLLM().on("Answer", {"value": 42}).on("reply", "Hello there, friend")
    trace = start_trace()
    assert (await llm.structured(Answer, "sys", "user")).value == 42
    assert await llm.text("sys", "user", name="reply") == "Hello there, friend"
    chunks = [c async for c in llm.stream("sys", "user", name="reply")]
    assert "".join(chunks) == "Hello there, friend" and len(chunks) > 1
    assert [c.name for c in trace.calls] == ["Answer", "reply", "reply"]
    assert llm.calls[0] == ("Answer", "sys", "user")


async def test_fake_llm_requires_responder():
    with pytest.raises(LLMError):
        await FakeLLM().structured(Answer, "s", "u")


# ------------------------------------------------------------------------ retrieval
@pytest.fixture
def store() -> VectorStore:
    return VectorStore(HashEmbedder())


def sample_resume() -> Resume:
    return Resume(
        sections=[
            ResumeSection(id="header", title="Header", kind="header", content="Alex Morgan"),
            ResumeSection(
                id="experience",
                title="Experience",
                kind="experience",
                content="Engineer, Acme | 2020 - Present\n- Built fraud detection models in Python\n"
                "- Migrated the billing database to PostgreSQL",
            ),
            ResumeSection(
                id="skills", title="Skills", kind="skills", content="Python, SQL, Docker"
            ),
        ]
    )


def test_chunk_resume_skips_header_and_adds_role_context():
    chunks = chunk_resume(sample_resume())
    texts = [t for t, _ in chunks]
    assert not any("Alex Morgan" in t for t in texts)
    assert "Engineer, Acme | 2020 - Present — Built fraud detection models in Python" in texts
    assert chunks[-1][1]["section_id"] == "skills"


def test_resume_index_is_scoped_per_conversation_and_replaced(store):
    store.index_resume("c1", "v1", sample_resume())
    store.index_resume("c2", "v1", sample_resume().with_section_content("skills", "Rust, Go, Zig"))
    hits = store.search_resume("c1", "fraud detection models python")
    assert "fraud detection" in hits[0].text
    assert all(h.metadata["conversation_id"] == "c1" for h in hits)
    # Re-indexing replaces the previous version's chunks.
    updated = sample_resume().with_section_content("skills", "Kubernetes, Terraform, Helm")
    store.index_resume("c1", "v2", updated)
    assert all(h.metadata["version_id"] == "v2" for h in store.search_resume("c1", "skills", k=10))


def test_knowledge_base_indexes_once_and_finds_locale(store):
    n = store.ensure_knowledge_base()
    assert n >= 15
    assert store.ensure_knowledge_base() == n  # unchanged files -> no re-index
    exact = store.knowledge_for_locale("es-MX")
    assert [h.metadata["title"] for h in exact] == ["Mexico (es-MX)"]
    assert store.knowledge_for_locale("xx-XX") == []
    hits = store.search_knowledge("action verbs for stronger bullets")
    assert any("verb" in h.metadata["title"].lower() for h in hits[:2])


def test_company_research_cache(store):
    assert store.company_research_age_hours("Google LLC") is None
    store.add_company_research(
        "Google LLC", [("Google values user focus and scale", {"url": "https://x"})]
    )
    assert store.company_research_age_hours("google llc") < 0.01
    assert store.search_company("Google LLC", "values")[0].metadata["url"] == "https://x"
    assert company_key("Google LLC") == "google-llc"


def test_embedder_falls_back_to_hash_without_key():
    assert create_embedder(Settings(gemini_api_key=None)).name == "hash256"


# ----------------------------------------------------------------------- web search
class StubProvider:
    def __init__(self, name, results=None, error=None):
        self.name, self._results, self._error = name, results or [], error

    def search(self, query, max_results):
        if self._error:
            raise self._error
        return self._results[:max_results]


async def test_web_search_falls_back_and_dedupes():
    good = StubProvider(
        "backup",
        [
            SearchResult("A", "https://a", "alpha"),
            SearchResult("A dup", "https://a", "alpha again"),
            SearchResult("Empty", "https://e", ""),
        ],
    )
    search = WebSearch([StubProvider("primary", error=RuntimeError("quota")), good])
    assert [r.url for r in await search.search("q")] == ["https://a", "https://a", "https://e"]
    merged = await search.search_many(["q1", "q2"])
    assert [r.url for r in merged] == ["https://a"]


def test_web_search_provider_selection():
    names = lambda s: [p.name for p in create_web_search(s).providers]  # noqa: E731
    assert names(Settings(tavily_api_key=None, search_provider="auto")) == ["duckduckgo"]
    assert names(Settings(tavily_api_key="k", search_provider="auto")) == ["tavily", "duckduckgo"]
    assert names(Settings(search_provider="none")) == []
