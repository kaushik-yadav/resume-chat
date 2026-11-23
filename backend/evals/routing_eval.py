"""Routing eval: accuracy, latency and cost of the routing cascade vs. a genuine LLM-only router.

Live (uses GEMINI_API_KEY). Run from backend/:

    uv run python evals/routing_eval.py --calibrate      # tune classifier thresholds (calibration set only)
    uv run python evals/routing_eval.py --repeats 5      # evaluate on the held-out test set

Methodology (see routing_cases.py for the data):
* Thresholds are chosen on CALIBRATION only; accuracy is reported on TEST only. The split is
  checked for exact and near-duplicate leakage before anything runs.
* Correctness is strict: exact intent, exact ordered agent list, exact canonical slot values.
* The baseline uses ``route(mode="llm_only")``, which bypasses presets, rules, cache and the
  classifier - otherwise "LLM only" would silently include the zero-cost tiers.
* Latency: warm-up calls first (not counted), every case repeated N times with the two modes
  interleaved, reported as median and p95 (nearest rank). The route cache is disabled so
  repeats measure the real routing path rather than cache hits.
* Cost: billed cost from completed calls (LLM + embeddings) at app/plans.py prices; cancelled
  speculative router calls are reported separately as an upper bound (estimated input tokens).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import itertools
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from routing_cases import CALIBRATION, SAMPLES, TEST, Case, leakage

from app.config import get_settings
from app.llm.client import create_llm, current_plan, start_trace
from app.models.conversation import ConversationContext
from app.orchestration.classifier import EXEMPLARS, IntentClassifier
from app.orchestration.locales import canonical_locale
from app.orchestration.router import apply_memory_updates, normalize, route
from app.parsing.sectionizer import parse_resume
from app.plans import call_cost
from app.retrieval.embeddings import create_embedder
from app.retrieval.vector_store import VectorStore

RESULTS = Path(__file__).resolve().parent / "results"


def correct(d, intent: str, agents: tuple[str, ...], slots: dict[str, str]) -> bool:
    """Strict: exact intent, exact ordered agents, exact slot values (case-insensitive)."""
    if d.intent != intent:
        return False
    if [x.agent for x in d.steps] != list(agents):
        return False
    return all((getattr(d, k, "") or "").lower() == v.lower() for k, v in slots.items())


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def memory_of(case: Case) -> ConversationContext:
    return ConversationContext(**case.memory)


def costs(trace) -> tuple[float, float]:
    """(billed USD from completed calls, upper bound USD for incomplete calls)."""
    billed = incomplete = 0.0
    for call in trace.calls:
        cost = call_cost(call.model, call.input_tokens, call.output_tokens) or 0.0
        if call.status == "ok":
            billed += cost
        else:
            incomplete += cost
    return billed, incomplete


# ------------------------------------------------------------------------ calibrate
def calibrate(vectors: VectorStore, resume) -> dict:
    """Sweep classifier thresholds on the calibration set; pick max coverage at 100% precision."""
    # Every message is embedded once; the sweep then re-scores the cached neighbours offline.
    vectors.query_static = functools.lru_cache(maxsize=None)(vectors.query_static)
    grid = []
    for min_conf, min_sim in itertools.product(
        [round(0.50 + 0.05 * i, 2) for i in range(10)],
        [round(0.70 + 0.02 * i, 2) for i in range(11)],
    ):
        clf = IntentClassifier(vectors, min_confidence=min_conf, min_similarity=min_sim)
        decided = right = 0
        for case in CALIBRATION:
            memory = memory_of(case)
            result = asyncio.run(clf.classify(case.message, memory, resume))
            if result is None or result.decision is None:
                continue
            decision = result.decision
            decision.target_locale = canonical_locale(decision.target_locale)
            decision = normalize(
                decision, apply_memory_updates(decision, memory, case.message), resume
            )
            decided += 1
            right += correct(decision, case.intent, case.agents, case.slots)
        precision = right / decided if decided else 1.0
        grid.append({"min_confidence": min_conf, "min_similarity": min_sim, "decided": decided,
                     "correct": right, "precision": precision, "coverage": decided / len(CALIBRATION)})  # fmt: skip
    perfect = [g for g in grid if g["precision"] == 1.0 and g["decided"]]
    # Highest coverage; among ties the most conservative (highest) thresholds.
    best = max(perfect, key=lambda g: (g["coverage"], g["min_confidence"], g["min_similarity"]))
    print("Calibration set:", len(CALIBRATION), "cases")
    print("Top settings at 100% precision (coverage = share decided by the classifier):")
    for g in sorted(
        perfect, key=lambda g: (-g["coverage"], -g["min_confidence"], -g["min_similarity"])
    )[:5]:
        print(f"  min_confidence={g['min_confidence']:.2f} min_similarity={g['min_similarity']:.2f}"
              f"  coverage={g['coverage']:.0%} ({g['decided']}/{len(CALIBRATION)})")  # fmt: skip
    print(
        f"Selected: min_confidence={best['min_confidence']:.2f}, min_similarity={best['min_similarity']:.2f}"
    )
    return {"selected": best, "grid": grid}


# ------------------------------------------------------------------------- evaluate
async def evaluate(llm, classifier, resume, repeats: int, warmup: int) -> dict:
    modes = ("cascade", "llm_only")
    print(f"Warm-up: {warmup} call(s) per mode (not counted)")
    for case in CALIBRATION[:warmup]:
        for mode in modes:
            await route(
                llm, case.message, memory_of(case), resume, "", classifier=classifier, mode=mode
            )

    runs: dict[str, list[dict]] = defaultdict(list)
    for rep in range(repeats):
        for case in TEST:
            for mode in modes if rep % 2 == 0 else reversed(modes):  # alternate order per round
                trace = start_trace()
                started = time.perf_counter()
                result = await route(llm, case.message, memory_of(case), resume, "",
                                     classifier=classifier, mode=mode)  # fmt: skip
                ms = (time.perf_counter() - started) * 1000
                billed, incomplete = costs(trace)
                runs[mode].append({
                    "case": case.message[:70], "rep": rep, "ms": ms, "decided_by": result.decided_by,
                    "correct": correct(result.decision, case.intent, case.agents, case.slots),
                    "got": [result.decision.intent, [s.agent for s in result.decision.steps],
                            result.decision.target_company, result.decision.target_locale],
                    "billed_usd": billed, "incomplete_usd": incomplete,
                })  # fmt: skip
        print(f"  round {rep + 1}/{repeats} done")

    summary = {}
    for mode, rows in runs.items():
        by_case = defaultdict(list)
        for r in rows:
            by_case[r["case"]].append(r["correct"])
        latencies = [r["ms"] for r in rows]
        tiers = Counter(r["decided_by"] for r in rows)
        clf_rows = [r for r in rows if r["decided_by"] == "classifier"]
        summary[mode] = {
            "requests": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "always_correct_cases": sum(all(v) for v in by_case.values()),
            "unstable_cases": sum(0 < sum(v) < len(v) for v in by_case.values()),
            "median_ms": statistics.median(latencies),
            "p95_ms": p95(latencies),
            "billed_usd_per_request": statistics.mean(r["billed_usd"] for r in rows),
            "incomplete_usd_upper_bound_per_request": statistics.mean(r["incomplete_usd"] for r in rows),
            "tiers": dict(tiers),
            "classifier_precision": (sum(r["correct"] for r in clf_rows) / len(clf_rows)) if clf_rows else None,
            "misses": sorted({(r["case"], json.dumps(r["got"])) for r in rows if not r["correct"]}),
        }  # fmt: skip
    return {"runs": runs, "summary": summary}


def report(summary: dict, cases: int, repeats: int) -> None:
    print(f"\nHeld-out test set: {cases} cases x {repeats} repeats per mode (strict scoring)\n")
    print(
        f"{'mode':10s} {'accuracy':>8s} {'stable':>7s} {'median':>8s} {'p95':>8s} {'$/request':>10s} {'+cancelled<=':>13s}  tiers"
    )
    for mode, s in summary.items():
        print(f"{mode:10s} {s['accuracy']:8.1%} {s['always_correct_cases']:4d}/{cases:<2d} "
              f"{s['median_ms'] / 1000:7.2f}s {s['p95_ms'] / 1000:7.2f}s {s['billed_usd_per_request']:10.5f} "
              f"{s['incomplete_usd_upper_bound_per_request']:13.5f}  {s['tiers']}")  # fmt: skip
        if s["classifier_precision"] is not None:
            print(
                f"{'':10s} classifier precision on held-out decisions: {s['classifier_precision']:.1%}"
            )
        for case, got in s["misses"]:
            print(f"{'':10s} miss: {case!r} -> {got}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="tune thresholds on the calibration set"
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per case per mode (>= 3)")
    parser.add_argument("--warmup", type=int, default=3, help="uncounted warm-up calls per mode")
    args = parser.parse_args()

    exemplars = [t for texts in EXEMPLARS.values() for t in texts]
    leaks = leakage(TEST, [c.message for c in CALIBRATION], exemplars)
    if leaks:
        sys.exit(f"Test set leaks into calibration/exemplars: {leaks}")

    settings = get_settings()
    resume = parse_resume("cv.docx", (SAMPLES / "alex_morgan_resume.docx").read_bytes())
    vectors = VectorStore(create_embedder(settings), settings.chroma_path)
    classifier = IntentClassifier(vectors)
    classifier.ensure_index()
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%MZ")

    if args.calibrate:
        out = calibrate(vectors, resume)
        (RESULTS / "routing_calibration.json").write_text(
            json.dumps({"at": stamp, **out}, indent=1)
        )
        return

    current_plan.set("free")  # the router model is the lite model on every plan
    llm = create_llm(settings)
    print(f"Classifier thresholds: min_confidence={classifier.min_confidence}, "
          f"min_similarity={classifier.min_similarity}")  # fmt: skip
    result = asyncio.run(evaluate(llm, classifier, resume, max(args.repeats, 1), args.warmup))
    report(result["summary"], len(TEST), args.repeats)
    payload = {"at": stamp, "repeats": args.repeats, "warmup": args.warmup,
               "test_cases": [asdict(c) | {"message": c.message[:120]} for c in TEST], **result}  # fmt: skip
    (RESULTS / "routing_eval.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nSaved {RESULTS / 'routing_eval.json'}")


if __name__ == "__main__":
    main()
