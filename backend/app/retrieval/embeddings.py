"""Embedding providers for the vector store."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Gemini embeddings: multilingual, so English and Spanish resume text share one space."""

    def __init__(self, api_key: str, model: str, dimensions: int = 768) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        self._dims = dimensions
        self.name = f"gemini{dimensions}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed_documents(
            texts, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=self._dims
        )

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(
            text, task_type="RETRIEVAL_QUERY", output_dimensionality=self._dims
        )


class HashEmbedder:
    """Offline bag-of-words embedding (feature hashing). Used in tests and key-less mode.

    It captures lexical overlap only, which is enough for deterministic tests.
    """

    name = "hash256"

    def __init__(self, dims: int = 256) -> None:
        self._dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for token in re.findall(r"[a-z0-9+#]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dims] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "gemini" and settings.gemini_api_key:
        return GeminiEmbedder(settings.gemini_api_key, settings.embedding_model)
    return HashEmbedder()
