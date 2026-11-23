"""Live acceptance check against the assignment brief (needs the running API).

Drives the HTTP/SSE API exactly as the UI does, with the fictional sample resume, and checks
every behaviour the brief names: the four example requests, routing, multi-agent coordination,
follow-ups, clarifications, questions, versions (undo/redo/compare/restore), reasoning shown
with each result, streamed replies (typing) and Firestore persistence.

    uv run python evals/brief_check.py [--base http://127.0.0.1:8000] [--plan pro]

Costs a few cents of LLM usage per run.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
RESUME = ROOT / "samples" / "alex_morgan_resume.pdf"
JD = (ROOT / "samples" / "job_description_data_scientist.txt").read_text(encoding="utf-8")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


class Client:
    def __init__(self, base: str):
        self.http = httpx.Client(base_url=base + "/api", timeout=180)

    def new(self, plan: str) -> str:
        cid = self.http.post("/conversations").json()["id"]
        with RESUME.open("rb") as f:
            r = self.http.post(f"/conversations/{cid}/resume", files={"file": (RESUME.name, f, "application/pdf")})
        r.raise_for_status()
        self.http.patch(f"/conversations/{cid}", json={"plan": plan}).raise_for_status()
        return cid

    def chat(self, cid: str, message: str, jd: str | None = None) -> dict:
        body = {"message": message, "job_description": jd, "client_message_id": uuid.uuid4().hex}
        turn: dict = {"tokens": 0, "events": [], "results": [], "plan": None, "version": None,
                      "done": None, "error": None, "first_token_s": None}
        t0 = time.perf_counter()
        with self.http.stream("POST", f"/conversations/{cid}/chat", json=body,
                              headers={"Accept": "text/event-stream"}) as r:
            event = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    kind = data.get("type") or event
                    turn["events"].append(kind)
                    if kind == "plan":
                        turn["plan"] = data
                    elif kind == "agent_result":
                        turn["results"].append(data)
                    elif kind == "token":
                        turn["tokens"] += 1
                        if turn["first_token_s"] is None:
                            turn["first_token_s"] = time.perf_counter() - t0
                    elif kind == "version":
                        turn["version"] = data["version"]
                    elif kind == "done":
                        turn["done"] = data
                    elif kind == "error":
                        turn["error"] = data.get("message")
        turn["seconds"] = time.perf_counter() - t0
        return turn

    def versions(self, cid: str) -> list[dict]:
        return self.http.get(f"/conversations/{cid}/versions").json()


def agents_of(turn: dict) -> list[str]:
    return [s["agent"] for s in (turn["plan"] or {}).get("steps", [])]


def changed(turn: dict) -> list[str]:
    return [c["section_id"] for c in (turn["version"] or {}).get("changes", [])]


def summary(turn: dict) -> str:
    return (f"agents={agents_of(turn)} changed={changed(turn)} "
            f"{turn['seconds']:.1f}s first-token={turn['first_token_s'] or 0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--plan", default="pro")
    args = ap.parse_args()
    c = Client(args.base)
    health = c.http.get("/health").json()
    print(f"Backend: {health}")
    check("Firebase (Firestore) is the storage backend", health.get("storage_backend") == "firestore")

    print("\nA. The brief's four example requests (one conversation)")
    cid = c.new(args.plan)
    t = c.chat(cid, "Optimize my resume for Google")
    r = t["results"][0] if t["results"] else {}
    ins = r.get("insights", {})
    check("'Optimize my resume for Google' -> Company Research agent", agents_of(t) == ["company_research"], summary(t))
    check("  researched with web APIs (sources shown)", bool(ins.get("sources")), f"{len(ins.get('sources', []))} sources")
    check("  company values/culture profile", bool((ins.get("company_profile") or {}).get("values")))
    check("  reasoning shown with the result", bool(r.get("reasoning")))
    check("  new resume version saved", t["version"] is not None)
    check("  reply streamed token by token (typing)", t["tokens"] > 1, f"{t['tokens']} token events")

    t = c.chat(cid, "Make my skills section more relevant for a data scientist role")
    check("'Make my skills section more relevant…' -> Section Enhancement", agents_of(t) == ["section_enhancement"], summary(t))
    check("  only the skills section changed", changed(t) and all(s.startswith("skills") for s in changed(t)))

    t = c.chat(cid, "Update my experience section to better match this job description", jd=JD)
    r = t["results"][0] if t["results"] else {}
    ins = r.get("insights", {})
    check("'Update my experience section … job description' -> JD Matching", agents_of(t) == ["jd_matching"], summary(t))
    check("  match score generated", isinstance(ins.get("match_score_after"), int | float),
          f"{ins.get('match_score_before')} -> {ins.get('match_score_after')}")
    gaps = [q for q in ins.get("requirements", []) if q.get("status") in ("missing", "partial")]
    check("  skill gaps identified", bool(gaps), f"{len(gaps)} gaps/partials")
    check("  only the experience section changed", changed(t) and all(s.startswith("experience") for s in changed(t)))

    print("\nB. Follow-ups, questions and context")
    t = c.chat(cid, "What are my biggest gaps for this role?")
    check("question answered without editing (no new version)", t["version"] is None and t["tokens"] > 1,
          f"intent={(t['plan'] or {}).get('intent')}")
    t = c.chat(cid, "make it shorter")
    check("follow-up 'make it shorter' resolves to the last edited section", bool(changed(t)), summary(t))

    print("\nC. Versions")
    before = len(c.versions(cid))
    t = c.chat(cid, "undo that")
    check("'undo that' in chat moves back one version", (t["done"] or {}).get("current_version", {}).get("number") == before - 1,
          f"{t['seconds']:.2f}s")
    state = c.http.post(f"/conversations/{cid}/redo").json()
    check("redo restores it", state["current_version"]["number"] == before)
    vs = c.versions(cid)
    diff = c.http.get(f"/conversations/{cid}/compare", params={"from": vs[0]["id"], "to": vs[-1]["id"]}).json()
    check("compare any two versions", len(diff) > 0, f"{len(diff)} sections differ v1 -> v{len(vs)}")
    restored = c.http.post(f"/conversations/{cid}/versions/{vs[0]['id']}/restore").json()
    check("restore creates a new version (history kept)", restored["current_version"]["number"] == len(vs) + 1)

    t = c.chat(cid, "Translate this to Spanish for the Mexican market")
    r = t["results"][0] if t["results"] else {}
    ins = r.get("insights", {})
    check("'Translate this to Spanish for the Mexican market' -> Translation", agents_of(t) == ["translation"], summary(t))
    check("  localized for es-MX with notes", (t["version"] or {}).get("resume", {}).get("locale") == "es-MX"
          and bool(ins.get("localization_notes")), f"{len(ins.get('localization_notes', []))} notes")

    print("\nD. Multi-agent coordination")
    cid2 = c.new(args.plan)
    t = c.chat(cid2, "Optimize my resume for Google and then translate it to Spanish for the Mexican market")
    check("one request -> two agents, translation last", agents_of(t) == ["company_research", "translation"], summary(t))
    check("  both agents report results", len(t["results"]) == 2)

    print("\nE. Clarification")
    cid3 = c.new(args.plan)
    t = c.chat(cid3, "Match my resume to the job description")
    check("JD request without a JD asks for it (no edit)", t["version"] is None and (t["plan"] or {}).get("intent") == "clarify",
          f"intent={(t['plan'] or {}).get('intent')}")
    t = c.chat(cid3, "Here it is", jd=JD)
    check("the follow-up with the JD runs JD Matching", agents_of(t) == ["jd_matching"], summary(t))

    print("\nF. Persistence")
    conv = c.http.get(f"/conversations/{cid}").json()
    check("conversation history stored and reloadable", len(conv.get("messages", [])) >= 12, f"{len(conv.get('messages', []))} messages")
    check("versions stored", len(c.versions(cid)) >= 6, f"{len(c.versions(cid))} versions")

    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} {detail}")


if __name__ == "__main__":
    main()
