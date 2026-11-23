"""Product catalogue: subscription plans and suggested actions."""

from fastapi import APIRouter, Request

from app.orchestration.presets import list_presets
from app.plans import PLANS

router = APIRouter(tags=["catalog"])


@router.get("/plans")
def plans(request: Request) -> dict:
    return {
        "default": request.app.state.settings.default_plan,
        "plans": [p.as_dict() for p in PLANS.values()],
    }


@router.get("/presets")
def presets() -> list[dict]:
    return list_presets()
