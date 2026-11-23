# Resume Chat: a conversational, multi-agent resume optimizer

Upload a resume (PDF or DOCX) and improve it by chatting:

- *"Optimize my resume for Google"*
- *"Translate this to Spanish for the Mexican market"*
- *"Update my experience section to better match this job description"*
- *"Make my skills section more relevant for a data scientist role"*

A router reads each message, plans which specialist agents to run (one or several, in
order), and streams their progress, reasoning and results into the chat. Every change is
saved as a resume version in Firestore, so you can diff, undo, redo, restore and export.

**All four agents from the brief are implemented** (the brief asked for three of four):
Company Research & Optimization, Job Description Matching, Translation & Localization, and
Section Enhancement.

| Doc | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Agent architecture, turn lifecycle, context-engineering choices, tool/API choices, measured latency |
| [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) | Responsible Use of AI statement, and how prompt traces are provided |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | A 5-minute walkthrough script for the demo video |

---

## Feature checklist

| Requirement | Where |
|---|---|
| Chat interface with PDF/DOCX upload and parsing | `frontend/`, `backend/app/parsing/` |
| Conversation router (intent → agents) | `backend/app/orchestration/router.py` |
| Context management (history + resume state across turns) | `backend/app/services/context.py`, conversation memory slots |
| Agent coordination for multi-step requests | LangGraph workflow, `backend/app/orchestration/graph.py` |
| Resume version tracking | `backend/app/services/versioning.py` (Firestore) |
| Display resume changes and agent reasoning | Resume panel (highlights + word diffs), expandable "Agent reasoning" cards, **Agent flow** tab: an n8n-style live execution graph (router → agents → reply → version) with per-node steps, timings and LLM calls |
| Follow-up questions and clarifications | Router memory and follow-up resolution, clarifying questions, Q&A over the resume (RAG) |
| **Bonus:** undo/redo, compare versions | Toolbar undo/redo, "undo that" in chat, Changes tab compares any two versions, History tab restores |
| **Bonus:** real-time conversation with typing indicators | Server-Sent Events: plan, per-agent progress, typing indicator, token-streamed reply |
| LangChain / LangGraph orchestration | LangChain chat models, structured output and Tavily tool; LangGraph `StateGraph` |
| FastAPI backend with a chat API | `backend/app/api/` (REST + SSE) |
| Firebase for conversation history and versions | Firestore (`backend/app/storage/firestore.py`), emulator supported |
| LLM / vector DB / other services (own choice) | Gemini 3.8 Flash, Chroma with Gemini embeddings, Tavily (DuckDuckGo fallback) |

---

## Quick start

**Prerequisites:** Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node 20+, and Java 21+
(only needed for the local Firestore emulator).

```bash
# 1. Install dependencies
scripts/setup.sh            # macOS/Linux
scripts\setup.ps1           # Windows PowerShell
# (or manually: cd backend && uv sync;  cd frontend && npm install)

# 2. Configure: backend/.env is created from backend/.env.example
#    Set GEMINI_API_KEY (required for live mode) and optionally TAVILY_API_KEY.

# 3. Run, in three terminals from the repo root
npm run emulators           # Firestore emulator on :8080, UI on :4000
npm run backend             # FastAPI on :8000 (docs at http://localhost:8000/docs)
npm run frontend            # Chat UI on http://localhost:5173
```

Open http://localhost:5173 and upload `samples/alex_morgan_resume.pdf` (a fictional
candidate). Paste `samples/job_description_data_scientist.txt` to try JD matching.

### Running without keys or Firebase

| Setting | Effect |
|---|---|
| `LLM_PROVIDER=fake` | Deterministic offline responders: the whole pipeline runs with no API key |
| `STORAGE_BACKEND=memory` | In-process storage instead of Firestore |
| `EMBEDDING_PROVIDER=hash` | Offline lexical embeddings instead of Gemini embeddings |
| `SEARCH_PROVIDER=duckduckgo` | Key-less web search (or `none`) |

### Using a real Firebase project

1. Create a project in the Firebase console and enable **Cloud Firestore**.
2. Project settings → Service accounts → **Generate new private key**. Save the JSON outside
   the repo.
3. In `backend/.env`: set `FIREBASE_PROJECT_ID=<your-project-id>`,
   `FIREBASE_CREDENTIALS=<path-to-json>`, and remove `FIRESTORE_EMULATOR_HOST`.

### Troubleshooting

- **Firestore calls hang or fail with `WSA Error 10049` / `UNAVAILABLE` on Windows.** Some
  network/VPN filter drivers break gRPC's dual-stack sockets. Set `FIRESTORE_TRANSPORT=rest`
  (HTTP/1.1 REST transport). If the emulator itself returns `503 io exception`, start it with
  `npm run emulators:ipv4`, which forces Java onto IPv4.
- **Agents say live research is unavailable.** Tavily is optional; DuckDuckGo is used when
  there is no key, and the company agent labels profiles not grounded in live sources.

---

## Tests and checks

```bash
cd backend
uv run pytest -q                                   # 71 tests, offline (fake LLM, memory store)
FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 uv run pytest tests/test_firestore.py   # emulator round trip
uv run ruff check app tests
cd ../frontend && npm run build                    # type-check + production build
```

The suite runs fully offline through the same code paths as production. It covers the
parsers, sectionizer, versioning/undo/redo, vector store, web-search fallback, all four
agents, guardrails, router validation, multi-agent LangGraph turns, history summarization,
and the HTTP API, including SSE streaming and DOCX export.

---

## Code components

```
backend/app
├── main.py                 App factory; wires repository, LLM, vector store, web search, agents
├── config.py               All settings (env / backend/.env)
├── api/
│   ├── conversations.py    REST + SSE: conversations, upload, chat stream, versions, undo/redo, compare, export
│   ├── resumes.py          Stateless parse endpoint
│   └── health.py
├── models/
│   ├── resume.py           Resume = ordered typed sections (Markdown content); outline/markdown rendering
│   └── conversation.py     Conversation (+ memory slots), ChatMessage (+ agent trace), ResumeVersion, SectionChange
├── parsing/
│   ├── extract.py          PDF (pypdf) / DOCX (python-docx, incl. tables) text extraction + normalization
│   └── sectionizer.py      Deterministic heading detection → typed sections
├── orchestration/
│   ├── router.py           LLM intent router → validated plan (+ undo/redo fast path)
│   ├── graph.py            LangGraph turn workflow: route → agents → synthesize | answer | clarify
│   └── offline.py          Offline router/reply responders
├── agents/
│   ├── base.py             Agent contract, EditPlan schema, edit application, shared ground rules
│   ├── company_research.py Web research (cached) → company profile → tailoring
│   ├── jd_matching.py      Requirements → vector evidence → deterministic score/gaps → tailoring
│   ├── translation.py      Locale conventions (KB, web fallback) → translation + localization
│   ├── section_enhancement.py  Style lint + guidance → section rewrite
│   ├── guardrails.py       Invented-number, deletion, lost-figure and style checks (no LLM)
│   └── registry.py         Agent catalogue (also feeds the router prompt)
├── llm/
│   ├── client.py           LangChain chat-model wrapper: structured output, streaming, tracing, FakeLLM
│   └── offline.py          Offline agent responders
├── retrieval/
│   ├── vector_store.py     Chroma: knowledge base, per-conversation resume chunks, company research cache
│   ├── embeddings.py       Gemini embeddings (multilingual) / offline hash embeddings
│   └── knowledge/*.md      Curated hiring conventions (12 markets) + resume best practices
├── services/
│   ├── chat.py             One chat turn end-to-end (stream, commit version, persist trace, memory)
│   ├── flow.py             Execution-flow recorder (nodes, timings, per-node LLM calls) for the Agent flow view
│   ├── context.py          History window + rolling summary
│   ├── versioning.py       Commit / undo / redo / restore / compare
│   ├── diff.py             Section-level diffs
│   └── export.py           DOCX / Markdown export
├── storage/                Repository interface; Firestore (gRPC or REST) and in-memory backends
└── tools/web_search.py     Tavily (LangChain) with DuckDuckGo fallback, concurrent + de-duplicated

frontend/src
├── App.tsx                 State, SSE event handling, toolbar (undo/redo/export/new/recent)
├── api.ts                  Typed REST client + SSE stream parser (POST streaming via fetch)
├── types.ts                Mirrors backend models and stream events
└── components/             UploadPane, ChatPanel, Composer, LiveActivity, MessageBubble, AgentCard,
                            ResumePanel (resume / changes / history / agent flow), FlowView,
                            ResumeSectionBody (line-faithful resume rendering), DiffView
```

### API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/conversations` | Create a conversation |
| GET | `/api/conversations` · `/{id}` | List conversations / get state (messages, current version, undo/redo availability) |
| POST | `/api/conversations/{id}/resume` | Upload and parse a resume (creates a version) |
| POST | `/api/conversations/{id}/chat` | Send a message; **SSE stream** of `plan`, `agent_start`, `progress`, `agent_result`, `flow`, `token`, `version`, `done`, `error` |
| GET | `/api/conversations/{id}/versions[/{vid}]` | Version list / one version |
| POST | `/api/conversations/{id}/undo` · `/redo` · `/versions/{vid}/restore` | Version control |
| GET | `/api/conversations/{id}/compare?from=&to=` | Section diffs between two versions |
| GET | `/api/conversations/{id}/export?format=docx\|md` | Download |
| POST | `/api/resumes/parse` | Stateless parse (debugging) |

