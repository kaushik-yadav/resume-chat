from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "storage_backend": settings.storage_backend,
        "search_provider": settings.search_provider,
    }
