"""Application settings, loaded from environment variables and ``backend/.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Resume Chat"
    environment: Literal["development", "test", "production"] = "development"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    max_upload_mb: int = 10

    # --- LLM -------------------------------------------------------------
    # "gemini" (default), "openai", or "fake" (deterministic, offline; used by tests).
    llm_provider: Literal["gemini", "openai", "fake"] = "gemini"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    # Gemini models per role are chosen by the conversation's plan (app/plans.py).
    default_plan: Literal["free", "pro", "premium"] = "pro"
    openai_agent_model: str = "gpt-5.4-mini"
    openai_router_model: str = "gpt-5.4-mini"
    openai_reply_model: str = "gpt-5.4-mini"
    # Used when a model rejects a request as unsupported (e.g. no JSON-schema output).
    openai_fallback_model: str = "gpt-4.1-mini"
    # Gemini: the first entry that differs from a role's primary model is its fallback.
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    )
    # Gemini 3 thinking level / OpenAI reasoning effort: "low" keeps chat responsive.
    agent_reasoning: Literal["minimal", "low", "medium", "high"] = "low"
    router_reasoning: Literal["minimal", "low", "medium", "high"] = "low"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2

    # --- Retrieval -------------------------------------------------------
    embedding_provider: Literal["gemini", "hash"] = "gemini"
    embedding_model: str = "models/gemini-embedding-001"
    chroma_path: str = str(BACKEND_DIR / ".chroma")

    # --- Web research ----------------------------------------------------
    tavily_api_key: str | None = None
    search_provider: Literal["auto", "tavily", "duckduckgo", "none"] = "auto"
    company_cache_ttl_hours: int = 72

    # --- Persistence -----------------------------------------------------
    storage_backend: Literal["firestore", "memory"] = "firestore"
    firebase_project_id: str = "demo-resume-chat"
    # Path to a service-account JSON. Not needed when using the emulator.
    firebase_credentials: str | None = None
    # e.g. "localhost:8080" - when set, the Firestore emulator is used.
    firestore_emulator_host: str | None = None
    # "grpc" (default) or "rest" for networks where gRPC/HTTP2 connections are blocked.
    firestore_transport: Literal["grpc", "rest"] = "grpc"
    # Write-through in-memory cache of conversations, versions and messages (one process).
    storage_cache: bool = True

    # --- Context engineering --------------------------------------------
    history_window: int = 8  # recent messages passed verbatim to the router
    # Embedding-based intent classifier in front of the LLM router.
    fast_classifier: bool = True
    summary_trigger: int = 14  # summarize older turns once history exceeds this


@lru_cache
def get_settings() -> Settings:
    return Settings()
