"""Execution-flow recorder: turns a chat turn's stream events into an n8n-style node graph.

Nodes (in execution order):

    trigger (user message) -> router -> agent:<name> ... -> respond -> version

Each node carries a status (pending / running / done / error / skipped), start/end offsets
in ms from the start of the turn, sub-steps (the agent's progress messages, timed), the LLM
calls made while it ran (model, latency, tokens) and node-specific details. The UI renders
snapshots live while the turn runs; the final snapshot is stored in the message trace so
past turns can be replayed.

Timing uses each event's emit timestamp (``ts``), and LLM calls are attributed through the
``node`` tag set by the workflow (``app.llm.client.current_node``). Both matter because the
graph runs ahead of the consumer of its event stream.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from app.llm.client import LLMTrace
from app.plans import call_cost

RESPOND_LABELS = {
    "edit": "Summarize changes",
    "question": "Answer question (RAG)",
    "chitchat": "Reply",
    "clarify": "Ask clarifying question",
}


class FlowRecorder:
    def __init__(self, trace: LLMTrace, message: str, job_description_attached: bool) -> None:
        self._t0 = time.perf_counter()
        self._trace = trace
        self.nodes: list[dict[str, Any]] = []
        self._add(
            "trigger",
            "trigger",
            "User message",
            status="done",
            start=0,
            end=0,
            detail={"message": message[:500], "job_description_attached": job_description_attached},
        )

    # ------------------------------------------------------------------ helpers
    def _ms(self, ts: float | None = None) -> int:
        return max(0, int(((ts or time.perf_counter()) - self._t0) * 1000))

    def _get(self, node_id: str) -> dict[str, Any] | None:
        return next((n for n in self.nodes if n["id"] == node_id), None)

    def _add(
        self,
        node_id: str,
        kind: str,
        label: str,
        *,
        status: str = "pending",
        start: int | None = None,
        end: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        node = {
            "id": node_id,
            "kind": kind,
            "label": label,
            "status": status,
            "start_ms": start,
            "end_ms": end,
            "detail": detail or {},
            "steps": [],
        }
        self.nodes.append(node)
        return node

    def _start(
        self, node_id: str, kind: str, label: str, ts: float | None = None, **detail: Any
    ) -> dict[str, Any]:
        node = self._get(node_id) or self._add(node_id, kind, label)
        node.update(status="running", start_ms=self._ms(ts), end_ms=None)
        node["detail"].update(detail)
        return node

    def _close_steps(self, node: dict[str, Any], at: int, status: str = "done") -> None:
        for step in node["steps"]:
            if step["status"] == "running":
                step.update(status=status, end_ms=at)

    def _finish(
        self, node_id: str, status: str = "done", ts: float | None = None, **detail: Any
    ) -> None:
        node = self._get(node_id)
        if node is None or node["status"] not in ("running", "pending"):
            return
        at = self._ms(ts)
        if node["start_ms"] is None:
            node["start_ms"] = at
        self._close_steps(node, at, status)
        node.update(status=status, end_ms=at)
        node["detail"].update(detail)

    def _step(self, node_id: str, label: str, ts: float | None) -> None:
        node = self._get(node_id)
        if node is None:
            return
        at = self._ms(ts)
        self._close_steps(node, at)
        node["steps"].append({"label": label, "status": "running", "start_ms": at, "end_ms": None})

    # ------------------------------------------------------------------- events
    def observe(self, event: dict[str, Any]) -> bool:
        """Update the flow from a graph stream event. Returns True if the flow changed."""
        kind, ts = event.get("type"), event.get("ts")
        if kind == "status" and event.get("stage") == "routing":
            self._start("router", "router", "Router", ts)
        elif kind == "plan":
            intent = event["intent"]
            self._finish(
                "router",
                ts=ts,
                intent=intent,
                reasoning=event.get("reasoning", ""),
                plan=[s["agent"] for s in event.get("steps", [])],
                decided_by=event.get("decided_by", "llm"),
                confidence=event.get("confidence"),
            )
            for step in event.get("steps", []):
                self._add(
                    f"agent:{step['agent']}",
                    "agent",
                    step.get("title") or step["agent"],
                    detail={
                        "agent": step["agent"],
                        "instruction": step.get("instruction", ""),
                        "target_sections": step.get("target_sections", []),
                    },
                )
            if intent in RESPOND_LABELS:
                self._add("respond", "respond", RESPOND_LABELS[intent])
            if intent == "edit":
                self._add("version", "version", "Save version")
            elif intent in ("undo", "redo"):
                self._add("version", "version", intent.capitalize())
        elif kind == "agent_start":
            agent = event["agent"]
            self._start(
                f"agent:{agent}", "agent", event.get("title", agent), ts, index=event.get("index")
            )
        elif kind == "progress":
            self._step(f"agent:{event['agent']}", event.get("message", ""), ts)
        elif kind == "agent_result":
            failed = any(" failed: " in w for w in event.get("warnings", []))
            insights = event.get("insights", {})
            self._finish(
                f"agent:{event['agent']}",
                status="error" if failed else "done",
                ts=ts,
                reasoning=event.get("reasoning", ""),
                edited=[e["title"] for e in event.get("edits", [])],
                warnings=event.get("warnings", []),
                **{
                    k: insights[k]
                    for k in ("match_score_before", "match_score_after", "locale")
                    if k in insights
                },
            )
        elif kind == "status" and event.get("stage") == "responding":
            node = self._get("respond")
            self._start("respond", "respond", node["label"] if node else "Reply", ts)
        elif kind == "token":
            node = self._get("respond")
            if node is None or node["status"] != "pending":
                return False
            self._start("respond", "respond", node["label"], ts)  # clarify streams no status
        else:
            return False
        return True

    def response_done(self, reply: str) -> None:
        self._finish("respond", reply_preview=reply[:240])

    def version_started(self) -> None:
        self._start("version", "version", "Save version")

    def version_saved(self, number: int, label: str, changed: list[str], op: str = "save") -> None:
        title = {"undo": "Undo", "redo": "Redo"}.get(op, "Save version")
        node = self._get("version")
        if node is None or node["status"] != "running":
            self._start("version", "version", title)
        self._finish("version", number=number, label=label, changed_sections=changed)

    def version_failed(self, message: str) -> None:
        self._start("version", "version", "Version")
        self._finish("version", status="skipped", reason=message)

    def close(self) -> None:
        """End of turn: finish anything still running; mark never-started nodes as skipped."""
        for node in self.nodes:
            if node["status"] == "running":
                self._finish(node["id"])
            elif node["status"] == "pending":
                node["status"] = "skipped"
                if node["kind"] == "version":
                    node["detail"]["reason"] = "No resume changes to save"

    def snapshot(self) -> dict[str, Any]:
        """Nodes with their LLM/embedding calls and cost.

        ``cost_usd`` sums completed calls (billed amounts). Calls that were cancelled or failed
        (e.g. the speculative router call cancelled when the fast classifier answered first)
        carry *estimated* tokens; their cost is reported separately as an upper bound in
        ``incomplete_cost_usd`` rather than being mixed into the billed figure.
        """
        nodes = copy.deepcopy(self.nodes)
        billed = incomplete = 0.0
        for node in nodes:
            node["llm"] = []
            for c in self._trace.calls:
                if c.node != node["id"]:
                    continue
                cost = call_cost(c.model, c.input_tokens, c.output_tokens)
                if c.status == "ok":
                    billed += cost or 0.0
                else:
                    incomplete += cost or 0.0
                node["llm"].append(
                    {
                        "name": c.name,
                        "model": c.model,
                        "ms": c.ms,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "cost_usd": cost,
                        "status": c.status,
                        "estimated": c.estimated,
                    }
                )
            node["cost_usd"] = sum(c["cost_usd"] or 0.0 for c in node["llm"] if c["status"] == "ok")
        return {
            "nodes": nodes,
            "total_ms": self._ms(),
            "cost_usd": billed,
            "incomplete_cost_usd": incomplete,
        }
