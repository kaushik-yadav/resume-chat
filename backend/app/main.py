"""FastAPI application factory and service wiring."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.base import AgentDeps
from app.agents.registry import build_agents
from app.api.conversations import router as conversations_router
from app.api.health import router as health_router
from app.api.resumes import router as resumes_router
from app.config import Settings, get_settings
from app.llm.client import LLM, create_llm
from app.retrieval.embeddings import create_embedder
from app.retrieval.vector_store import VectorStore
from app.services.chat import ChatService
from app.storage.base import Repository
from app.storage.firestore import create_repository
from app.tools.web_search import WebSearch, create_web_search

logger = logging.getLogger(__name__)


@dataclass
class Overrides:
    """Dependency overrides for tests (any field left as None is built from settings)."""

    repo: Repository | None = None
    llm: LLM | None = None
    vectors: VectorStore | None = None
    web: WebSearch | None = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Third-party clients are chatty at INFO.
    for noisy in ("httpx", "httpcore", "google_genai", "chromadb", "urllib3", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_chat_service(settings: Settings, overrides: Overrides | None = None) -> ChatService:
    o = overrides or Overrides()
    repo = o.repo or create_repository(settings)
    llm = o.llm or create_llm(settings)
    vectors = o.vectors or VectorStore(create_embedder(settings), settings.chroma_path)
    vectors.ensure_knowledge_base()
    web = o.web or create_web_search(settings)
    agents = build_agents(AgentDeps(llm=llm, vectors=vectors, web=web), settings)
    return ChatService(settings, repo, llm, vectors, agents)


def create_app(settings: Settings | None = None, overrides: Overrides | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.chat_service = build_chat_service(settings, overrides)
        logger.info(
            "Resume Chat API ready (llm=%s, storage=%s)",
            settings.llm_provider,
            settings.storage_backend,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Chat with your resume: multi-agent resume optimization.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(resumes_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    return app


app = create_app()
