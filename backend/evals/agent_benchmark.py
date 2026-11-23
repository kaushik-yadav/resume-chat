"""Agent benchmark: latency, cost and guardrail signals per agent and model.

Live (uses GEMINI_API_KEY). Run from backend/:

    uv run python evals/agent_benchmark.py --models gemini-3.8-flash,gemini-3.5-flash-lite --repeats 3

Methodology:
* One warm-up run per (model, agent) is discarded: it absorbs cold-start costs (client set-up,
  connection pools) and fills the company-research web cache, so every timed run measures the
  same steady state.
* Each (model, agent) pair then runs ``--repeats`` times (default 3); runs are interleaved
  across models per round so provider-side drift affects both equally.
* Reported per pair: success rate (agent produced edits), median and p95 latency, billed cost
  per run (LLM + embeddings at app/plans.py prices), and guardrail warnings per run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentContext, AgentDeps
from app.agents.registry import build_agents
from app.config import get_settings
from app.llm.client import ROLES, LangChainLLM, start_trace
from app.models.conversation import ConversationContext
from app.parsing.sectionizer import parse_resume
from app.plans import call_cost
from app.retrieval.embeddings import create_embedder
from app.retrieval.vector_store import VectorStore
from app.tools.web_search import create_web_search

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
RESULTS = Path(__file__).resolve().parent / "results"
JD = (SAMPLES / "job_description_data_scientist.txt").read_text(encoding="utf-8")
CASES = [
    ("translation", "Translate this to Spanish for the Mexican market", [], {"target_locale": "es-MX"}),
    ("section_enhancement", "Make my skills section more relevant for a data scientist role", ["skills"], {"target_role": "Data Scientist"}),
    ("jd_matching", "Update my experience section to better match this job description", ["experience"], {"job_description": JD}),
    ("company_research", "Optimize my resume for Google", [], {"target_company": "Google"}),
]  # fmt: skip


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def build_llm(settings, model: str) -> LangChainLLM:
    chat = ChatGoogleGenerativeAI(
        model=model, google_api_key=settings.gemini_api_key, reasoning_effort="low",
        timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
    )  # fmt: skip
    return LangChainLLM({r: chat for r in ROLES}, {r: model for r in ROLES})


async def run_once(agents, model: str, resume, case) -> dict:
    name, instruction, targets, memory = case
    trace = start_trace()
    started = time.perf_counter()
    result = await agents[name].run(
        AgentContext(
            f"bench-{model}",
            resume,
            instruction,
            instruction,
            ConversationContext(**memory),
            targets,
        )
    )
    ms = (time.perf_counter() - started) * 1000
    billed = sum(
        call_cost(c.model, c.input_tokens, c.output_tokens) or 0.0
        for c in trace.calls
        if c.status == "ok"
    )
    return {"ms": ms, "billed_usd": billed, "edits": len(result.edits), "warnings": len(result.warnings),
            "llm_calls": sum(c.name != "embed" for c in trace.calls)}  # fmt: skip


async def main_async(models: list[str], resume_path: Path, repeats: int) -> dict:
    settings = get_settings()
    resume = parse_resume(resume_path.name, resume_path.read_bytes())
    vectors = VectorStore(create_embedder(settings), settings.chroma_path)
    vectors.ensure_knowledge_base()
    web = create_web_search(settings)
    agents = {
        m: build_agents(AgentDeps(build_llm(settings, m), vectors, web), settings) for m in models
    }

    print("Warm-up: one discarded run per (model, agent)")
    for model in models:
        for case in CASES:
            await run_once(agents[model], model, resume, case)

    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rep in range(repeats):
        for case in CASES:
            for model in models if rep % 2 == 0 else list(reversed(models)):
                rows[(model, case[0])].append(await run_once(agents[model], model, resume, case))
        print(f"  round {rep + 1}/{repeats} done")

    print(f"\n{repeats} timed runs per (model, agent) after warm-up\n")
    print(
        f"{'model':24s} {'agent':20s} {'success':>8s} {'median':>8s} {'p95':>8s} {'$/run':>9s} {'warnings':>9s}"
    )
    summary = {}
    for (model, agent), runs in rows.items():
        latencies = [r["ms"] for r in runs]
        s = {
            "success_rate": sum(r["edits"] > 0 for r in runs) / len(runs),
            "median_ms": statistics.median(latencies),
            "p95_ms": p95(latencies),
            "billed_usd_per_run": statistics.mean(r["billed_usd"] for r in runs),
            "warnings_per_run": statistics.mean(r["warnings"] for r in runs),
        }
        summary[f"{model}/{agent}"] = s
        print(f"{model:24s} {agent:20s} {s['success_rate']:8.0%} {s['median_ms'] / 1000:7.1f}s "
              f"{s['p95_ms'] / 1000:7.1f}s {s['billed_usd_per_run']:9.4f} {s['warnings_per_run']:9.1f}")  # fmt: skip
    return {"summary": summary, "runs": {f"{m}/{a}": r for (m, a), r in rows.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", default="gemini-3.8-flash,gemini-3.5-flash-lite")
    parser.add_argument("--resume", type=Path, default=SAMPLES / "alex_morgan_resume.pdf")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = asyncio.run(main_async(args.models.split(","), args.resume, max(args.repeats, 1)))
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%MZ")
    out = RESULTS / "agent_benchmark.json"
    out.write_text(
        json.dumps(
            {"at": stamp, "repeats": args.repeats, "models": args.models, **result}, indent=1
        )
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
