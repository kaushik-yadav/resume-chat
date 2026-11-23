"""Subscription plans: which model serves each role, and what a request costs.

A SaaS product cannot run its most expensive model for every free user. Plans map the three
LLM roles to models:

* ``router`` - intent routing (every turn; must be fast; a lite model scores 100% on our eval)
* ``agent``  - the specialist rewrite/translate/analyze calls (where quality matters most)
* ``reply``  - the streamed chat summary (easy task; a lite model is plenty)

Prices are USD per 1M tokens for the Gemini Developer API (standard tier), from
https://ai.google.dev/gemini-api/docs/pricing as of 2026-11-23. gemini-3.8-flash is at its
introductory rate through 2026-12-31 ($1.50 / $7.50 from 2027-01-01). Update ``PRICES`` when
pricing changes; costs shown in the UI are computed from real token usage x these prices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

PlanId = Literal["free", "pro", "premium"]

# model -> (input USD / 1M tokens, output USD / 1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    # Embeddings: input only ($0.15 / 1M tokens, standard tier).
    "gemini-embedding-001": (0.15, 0.0),
}

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.8-flash"


@dataclass(frozen=True)
class Plan:
    id: PlanId
    name: str
    tagline: str
    router_model: str
    agent_model: str
    reply_model: str
    agent_effort: Literal["minimal", "low", "medium", "high"] = "low"
    features: tuple[str, ...] = field(default_factory=tuple)

    def model_for(self, role: str) -> str:
        return {"router": self.router_model, "agent": self.agent_model}.get(role, self.reply_model)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["features"] = list(self.features)
        data["prices"] = {
            role: {
                "model": m,
                "input_per_1m": PRICES.get(m, (None, None))[0],
                "output_per_1m": PRICES.get(m, (None, None))[1],
            }
            for role, m in (
                ("router", self.router_model),
                ("agent", self.agent_model),
                ("reply", self.reply_model),
            )
        }
        return data


PLANS: dict[str, Plan] = {
    "free": Plan(
        id="free",
        name="Free",
        tagline="Fast and lightweight: every step on our lite model",
        router_model=LITE,
        agent_model=LITE,
        reply_model=LITE,
        features=("All 4 agents", "Version history & undo", "Typical edit in ~2-10 s"),
    ),
    "pro": Plan(
        id="pro",
        name="Pro",
        tagline="Our best writer for the agents, lite model for routing and replies",
        router_model=LITE,
        agent_model=FLASH,
        reply_model=LITE,
        features=("Higher-quality rewrites and translations", "Everything in Free"),
    ),
    "premium": Plan(
        id="premium",
        name="Premium",
        tagline="Deeper reasoning on every agent step and richer replies",
        router_model=LITE,
        agent_model=FLASH,
        reply_model=FLASH,
        agent_effort="medium",
        features=("More thorough analysis (higher reasoning effort)", "Everything in Pro"),
    ),
}


def get_plan(plan_id: str | None, default: str = "pro") -> Plan:
    return PLANS.get(plan_id or default) or PLANS[default]


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
