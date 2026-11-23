"""Offline (key-less) LLM: deterministic heuristic responders for demos and tests."""

from app.llm.client import FakeLLM


def offline_llm() -> FakeLLM:
    return FakeLLM()
