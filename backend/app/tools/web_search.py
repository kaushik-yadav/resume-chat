"""Web research tool: Tavily (LangChain integration) with a key-less DuckDuckGo fallback."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    content: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str) -> None:
        from langchain_tavily import TavilySearch

        self._tool = TavilySearch(max_results=5, search_depth="basic", tavily_api_key=api_key)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        response = self._tool.invoke({"query": query})
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(str(response["error"]))
        results = response.get("results", []) if isinstance(response, dict) else []
        return [
            SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in results[:max_results]
        ]


class DuckDuckGoProvider:
    name = "duckduckgo"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=max_results)
        return [
            SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows
        ]


class WebSearch:
    """Runs a query against the primary provider and falls back on failure.

    Blocking provider calls run in a worker thread; several queries run concurrently.
    """

    def __init__(self, providers: list[SearchProvider], timeout_s: float = 20.0) -> None:
        self.providers = providers
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.providers)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for provider in self.providers:
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(provider.search, query, max_results), self.timeout_s
                )
                if results:
                    return results
            except Exception as exc:  # network errors, quota, timeouts
                logger.warning("Search provider %s failed for %r: %s", provider.name, query, exc)
        return []

    async def search_many(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        """Run queries concurrently and de-duplicate results by URL."""
        batches = await asyncio.gather(*(self.search(q, max_results) for q in queries))
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for batch in batches:
            for result in batch:
                if result.url and result.url not in seen and result.content:
                    seen.add(result.url)
                    merged.append(result)
        return merged


def create_web_search(settings: Settings) -> WebSearch:
    mode = settings.search_provider
    providers: list[SearchProvider] = []
    if mode in ("auto", "tavily") and settings.tavily_api_key:
        providers.append(TavilyProvider(settings.tavily_api_key))
    if mode in ("auto", "duckduckgo") or (mode == "tavily" and not settings.tavily_api_key):
        providers.append(DuckDuckGoProvider())
    logger.info("Web search providers: %s", [p.name for p in providers] or "none")
    return WebSearch(providers)
