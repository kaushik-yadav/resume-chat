"""Chroma vector store with three collections.

* ``knowledge``        - curated hiring knowledge (localization conventions, best practices)
* ``resume_chunks``    - the current resume of each conversation, one chunk per bullet/paragraph;
                          the JD agent retrieves evidence for each job requirement from here
* ``company_research`` - web research snippets per company, reused across conversations
                          until they expire (saves search calls and latency)

Collection names carry the embedder name so switching embedding models never mixes vectors
of different dimensions.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.resume import Resume
from app.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


@dataclass
class Hit:
    text: str
    score: float  # cosine similarity in [−1, 1]; higher is closer
    metadata: dict = field(default_factory=dict)


def company_key(company: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")


def chunk_resume(resume: Resume) -> list[tuple[str, dict]]:
    """Split a resume into retrievable chunks: bullets, or short paragraphs."""
    chunks: list[tuple[str, dict]] = []
    for section in resume.sections:
        if section.kind == "header":
            continue
        context = ""
        for line in section.content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("- "):
                text = f"{context} — {line[2:]}" if context else line[2:]
            else:
                context = line[:120]  # role/title line gives bullets their context
                text = line
            if len(text.split()) >= 3:
                chunks.append((text, {"section_id": section.id, "section_kind": section.kind}))
    return chunks


def _markdown_chunks(path: Path) -> list[tuple[str, dict]]:
    """One chunk per '## ' heading; the heading's '<!-- tags: ... -->' line becomes metadata."""
    text = path.read_text(encoding="utf-8")
    chunks = []
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block.startswith("## "):
            continue
        title = block.split("\n", 1)[0][3:].strip()
        tags = re.search(r"<!--\s*tags:\s*(.*?)\s*-->", block)
        tag_text = tags.group(1) if tags else ""
        body = re.sub(r"<!--.*?-->", "", block).strip()
        # Tags are embedded with the text so locale codes and synonyms are searchable.
        if tag_text:
            body = f"{body}\n(Keywords: {tag_text})"
        chunks.append((body, {"source": path.name, "title": title, "tags": tag_text}))
    return chunks


class VectorStore:
    def __init__(self, embedder: Embedder, path: str | None = None) -> None:
        settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        self._client = (
            chromadb.PersistentClient(path=path, settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        self.embedder = embedder
        # In-memory Chroma clients share one process-wide store, so each in-memory instance
        # (tests, offline mode) gets its own namespace instead of seeing other instances' data.
        self._namespace = "" if path else f"m{uuid.uuid4().hex[:10]}_"

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            f"{self._namespace}{name}__{self.embedder.name}", metadata={"hnsw:space": "cosine"}
        )

    def _add(self, name: str, items: list[tuple[str, dict]], ids: list[str]) -> None:
        if not items:
            return
        texts = [t for t, _ in items]
        self._collection(name).upsert(
            ids=ids,
            documents=texts,
            metadatas=[m for _, m in items],
            embeddings=self.embedder.embed_documents(texts),
        )

    def _query(self, name: str, query: str, k: int, where: dict | None = None) -> list[Hit]:
        collection = self._collection(name)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, collection.count()),
            where=where,
        )
        return [
            Hit(text=doc, score=round(1 - dist, 4), metadata=meta or {})
            for doc, dist, meta in zip(
                result["documents"][0], result["distances"][0], result["metadatas"][0], strict=False
            )
        ]

    # ---------------------------------------------------------------- knowledge
    def ensure_knowledge_base(self, directory: Path = KNOWLEDGE_DIR) -> int:
        """(Re)index the knowledge base when its files changed. Returns the chunk count."""
        files = sorted(directory.glob("*.md"))
        digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
        collection = self._collection("knowledge")
        if (collection.metadata or {}).get("digest") == digest and collection.count():
            return collection.count()
        self._client.delete_collection(collection.name)
        collection = self._client.create_collection(
            collection.name, metadata={"hnsw:space": "cosine", "digest": digest}
        )
        chunks = [c for f in files for c in _markdown_chunks(f)]
        self._add("knowledge", chunks, [f"kb-{i}" for i in range(len(chunks))])
        logger.info("Indexed %d knowledge-base chunks", len(chunks))
        return len(chunks)

    def search_knowledge(self, query: str, k: int = 4) -> list[Hit]:
        return self._query("knowledge", query, k)

    def knowledge_for_locale(self, locale: str) -> list[Hit]:
        """Exact lookup of the guidance chunk(s) tagged with a locale code like 'es-MX'.

        Locale guidance must not depend on similarity ranking: "Spain" and "Mexico" chunks are
        semantically close, but giving Peninsular conventions to a Mexican CV is wrong.
        """
        result = self._collection("knowledge").get(
            where_document={"$contains": f"Keywords: {locale}"}, include=["documents", "metadatas"]
        )
        return [
            Hit(text=doc, score=1.0, metadata=meta or {})
            for doc, meta in zip(result["documents"], result["metadatas"], strict=False)
        ]

    # ------------------------------------------------------------------- resume
    def index_resume(self, conversation_id: str, version_id: str, resume: Resume) -> int:
        """Replace the conversation's indexed resume with ``resume``."""
        collection = self._collection("resume_chunks")
        collection.delete(where={"conversation_id": conversation_id})
        chunks = [
            (text, {**meta, "conversation_id": conversation_id, "version_id": version_id})
            for text, meta in chunk_resume(resume)
        ]
        self._add(
            "resume_chunks",
            chunks,
            [f"{conversation_id}-{version_id}-{i}" for i in range(len(chunks))],
        )
        return len(chunks)

    def search_resume(self, conversation_id: str, query: str, k: int = 3) -> list[Hit]:
        return self._query("resume_chunks", query, k, where={"conversation_id": conversation_id})

    def search_resume_many(
        self, conversation_id: str, queries: list[str], k: int = 2
    ) -> list[list[Hit]]:
        """Batch variant: one embedding request and one Chroma query for all queries."""
        collection = self._collection("resume_chunks")
        if not queries or collection.count() == 0:
            return [[] for _ in queries]
        embed_many = getattr(self.embedder, "embed_queries", self.embedder.embed_documents)
        result = collection.query(
            query_embeddings=embed_many(queries),
            n_results=min(k, collection.count()),
            where={"conversation_id": conversation_id},
        )
        return [
            [
                Hit(text=doc, score=round(1 - dist, 4), metadata=meta or {})
                for doc, dist, meta in zip(docs, dists, metas, strict=False)
            ]
            for docs, dists, metas in zip(
                result["documents"], result["distances"], result["metadatas"], strict=False
            )
        ]

    # ------------------------------------------------------------- static sets
    def ensure_static(self, name: str, items: list[tuple[str, dict]], digest: str) -> int:
        """Index a fixed set of texts once; re-index only when ``digest`` changes."""
        collection = self._collection(name)
        if (collection.metadata or {}).get("digest") == digest and collection.count():
            return collection.count()
        self._client.delete_collection(collection.name)
        self._client.create_collection(
            collection.name, metadata={"hnsw:space": "cosine", "digest": digest}
        )
        self._add(name, items, [f"{name}-{i}" for i in range(len(items))])
        return len(items)

    def query_static(self, name: str, query: str, k: int) -> list[Hit]:
        return self._query(name, query, k)

    # ------------------------------------------------------------------ company
    def add_company_research(self, company: str, snippets: list[tuple[str, dict]]) -> None:
        key = company_key(company)
        now = time.time()
        items = [(text, {**meta, "company": key, "fetched_at": now}) for text, meta in snippets]
        self._add("company_research", items, [f"{key}-{int(now)}-{i}" for i in range(len(items))])

    def company_research_age_hours(self, company: str) -> float | None:
        result = self._collection("company_research").get(
            where={"company": company_key(company)}, include=["metadatas"]
        )
        if not result["ids"]:
            return None
        newest = max(m.get("fetched_at", 0) for m in result["metadatas"])
        return (time.time() - newest) / 3600

    def search_company(self, company: str, query: str, k: int = 6) -> list[Hit]:
        return self._query("company_research", query, k, where={"company": company_key(company)})
