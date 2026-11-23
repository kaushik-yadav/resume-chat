"""Fast intent classifier: embedding nearest-neighbour voting + rule-based slot extraction.

Sits in front of the LLM router. For simple, single-intent requests ("translate my resume to
German", "improve my experience bullets") it decides in one embedding call (~0.3 s) instead of
an LLM call, and costs ~nothing. It abstains - and the LLM router takes over - whenever it is
not confident, the request is compound ("... and then translate"), it refers back to earlier
turns, or a required slot (company, language) cannot be extracted reliably.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from app.models.conversation import ConversationContext
from app.models.resume import Resume
from app.orchestration.locales import LANGUAGES, extract_locale
from app.orchestration.router import PlanStep, RoutingDecision
from app.retrieval.vector_store import VectorStore

EXEMPLARS: dict[str, list[str]] = {
    "company_research": [
        "Optimize my resume for Google",
        "Tailor my resume for Amazon",
        "Make my resume a better fit for Microsoft's culture",
        "Adapt my CV to what Netflix looks for in candidates",
        "Optimize it for Stripe",
        "Customize my resume for a job at Meta",
        "Align my resume with Apple's values",
        "Make my resume more appealing to Deloitte recruiters",
        "I'm applying to Airbnb, update my resume for that company",
        "Research Shopify and adjust my resume",
    ],
    "jd_matching": [
        "Match my resume to this job description",
        "Update my experience section to better match this job description",
        "Tailor my resume to the job posting I pasted",
        "Align my skills with this JD",
        "Optimize my resume for this job posting",
        "Rewrite my summary to fit the job description",
        "Compare my resume to the job requirements and adjust it",
        "Make my resume match the requirements in the job ad",
    ],
    "translation": [
        "Translate this to Spanish for the Mexican market",
        "Translate my resume into German",
        "Convert my CV to French for jobs in France",
        "Localize my resume for Brazil in Portuguese",
        "Can you translate it to Japanese",
        "I need a Spanish version of my resume for Spain",
        "Translate my resume to English for the UK market",
        "Make a Dutch version of my CV",
    ],
    "section_enhancement": [
        "Make my skills section more relevant for a data scientist role",
        "Improve my experience bullets",
        "Make the summary punchier",
        "Rewrite my projects with stronger action verbs",
        "Add more quantified impact to my work experience",
        "Polish my summary section",
        "Make my resume bullets stronger",
        "Shorten my summary",
        "Fix the wording in my education section",
        "Make my skills section better for a backend engineer role",
    ],
    "question": [
        "What's the weakest part of my resume?",
        "How good is my match for the job?",
        "What skills am I missing?",
        "Is my resume ATS friendly?",
        "Which section should I improve first?",
        "What do you think of my summary?",
        "How long should my resume be?",
        "Does my resume have any typos?",
    ],
    "chitchat": [
        "hi",
        "hello there",
        "thanks!",
        "thank you so much",
        "great, that's perfect",
        "ok cool",
        "good morning",
    ],
}

SECTION_WORDS = {
    "summary": "summary", "profile": "summary", "experience": "experience", "bullets": "experience",
    "skills": "skills", "projects": "projects", "education": "education",
    "certifications": "certifications",
}  # fmt: skip
NOT_COMPANIES = {"the", "a", "an", "my", "this", "data", "senior", "junior", "backend", "frontend",
                 "jobs", "job", "role", "market", *(w.capitalize() for w in LANGUAGES)}  # fmt: skip
COMPANY_RX = re.compile(r"\b(?:for|at|to|with)\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,2})")
ROLE_RX = re.compile(r"\bfor an? ([a-z][a-z /&+-]{2,40}?) (?:role|position|job)\b", re.IGNORECASE)
ACTION_VERBS = (
    r"translate|locali[sz]e|optimi[sz]e|tailor|adapt|rewrite|improve|polish|make|add|update|"
    r"match|shorten|trim|fix|put|convert|strengthen"
)
# Two requests in one message: explicit sequencing ("and then", "; ") or "and <action verb>"
# ("optimize it for Google and translate it to Spanish"). The single-label classifier cannot
# plan several steps, so it abstains and the LLM router plans them.
COMPOUND_RX = re.compile(
    rf"\b(and then|then|and also|as well as|plus|after that)\b|;|\band\s+(?:{ACTION_VERBS})\b",
    re.IGNORECASE,
)
REFERS_BACK = re.compile(
    r"\b(same|again|them|they|their|that one|previous|instead)\b", re.IGNORECASE
)
MAX_WORDS = 40


@dataclass
class Classification:
    label: str
    confidence: float  # share of the top-k similarity mass held by the winning label
    similarity: float  # cosine similarity of the best-matching exemplar
    decision: RoutingDecision | None  # None = abstain (slots missing / not confident)
    reason: str = ""


def extract_company(message: str) -> str | None:
    for match in COMPANY_RX.finditer(message):
        name = re.sub(r"'s$", "", match.group(1)).strip(" .,")
        first = name.split()[0]
        if first.lower() not in NOT_COMPANIES and first not in NOT_COMPANIES:
            return name
    return None


JD_CUES = re.compile(
    r"\b(job description|jd|job posting|posting|job ad|this job|this role|the role|requirements)\b",
    re.IGNORECASE,
)
INTERROGATIVE = re.compile(
    r"^\s*(what|how|which|why|when|where|who|is|are|does|do|did|should|would it|am i)\b",
    re.IGNORECASE,
)
POLITE_REQUEST = re.compile(r"^\s*(can|could|would|will) you\b|^\s*please\b", re.IGNORECASE)


def is_question(message: str) -> bool:
    """A real question ("how good is my match?") rather than a polite request ("could you...?")."""
    if POLITE_REQUEST.search(message):
        return False
    return bool(INTERROGATIVE.search(message)) or message.rstrip().endswith("?")


class IntentClassifier:
    def __init__(
        self, vectors: VectorStore, *, k: int = 7, min_confidence: float = 0.55,
        min_similarity: float | None = None,
    ) -> None:  # fmt: skip
        self.vectors = vectors
        self.k = k
        self.min_confidence = min_confidence
        # Thresholds chosen by `evals/routing_eval.py --calibrate` on the calibration split only
        # (highest coverage at 100% precision); accuracy is reported on the held-out split.
        # Similarity scales differ by embedding model, hence a separate value for the offline
        # hash embedder.
        self.min_similarity = min_similarity or (
            0.76 if vectors.embedder.name.startswith("gemini") else 0.45
        )

    def ensure_index(self) -> int:
        items = [(text, {"label": label}) for label, texts in EXEMPLARS.items() for text in texts]
        digest = hashlib.sha256(repr(items).encode()).hexdigest()[:16]
        return self.vectors.ensure_static("intent_exemplars", items, digest)

    async def classify(
        self, message: str, memory: ConversationContext, resume: Resume
    ) -> Classification | None:
        """Return a classification, or None when the message is out of scope for the fast path."""
        if (
            len(message.split()) > MAX_WORDS
            or COMPOUND_RX.search(message)
            or REFERS_BACK.search(message)
        ):
            return None
        hits = await asyncio.to_thread(
            self.vectors.query_static, "intent_exemplars", message, self.k
        )
        if not hits:
            return None
        votes: dict[str, float] = defaultdict(float)
        for hit in hits:
            votes[hit.metadata["label"]] += max(hit.score, 0.0)
        total = sum(votes.values()) or 1.0
        # "Tailor my resume" reads the same to an embedding whether the target is a company or a
        # job posting; the difference is structural, so it is decided by explicit cues.
        tailoring = votes["company_research"] + votes["jd_matching"]
        if tailoring / total >= self.min_confidence:
            if JD_CUES.search(message):
                votes = defaultdict(float, {"jd_matching": tailoring})
            elif extract_company(message):
                votes = defaultdict(float, {"company_research": tailoring})
        label = max(votes, key=votes.get)
        result = Classification(label, votes[label] / total, hits[0].score, None)
        if result.confidence < self.min_confidence or result.similarity < self.min_similarity:
            result.reason = "low confidence"
            return result
        if label != "question" and is_question(message):
            result.reason = "phrased as a question"  # "how good is my match?" is not an edit
            return result
        if label != "translation" and extract_locale(message):
            # A language/market is named but the winner is another agent ("optimize for Google
            # in German"): likely a multi-step request - let the LLM plan it.
            result.reason = "mentions a language but not classified as translation"
            return result
        result.decision, result.reason = self._decide(label, message, memory, resume, result)
        return result

    def _decide(self, label, message, memory, resume, result) -> tuple[RoutingDecision | None, str]:
        why = f"Fast classifier: '{label}' ({result.confidence:.0%} of votes, similarity {result.similarity:.2f})"
        if label in ("question", "chitchat"):
            return RoutingDecision(intent=label, reasoning=why), "ok"

        slots: dict[str, str] = {}
        if label == "translation":
            locale = extract_locale(message) or memory.target_locale
            if not locale:
                return None, "no target language"
            slots["target_locale"] = locale
        if label == "company_research":
            company = extract_company(message)  # never guess: memory-based follow-ups go to the LLM
            if not company:
                return None, "no company name"
            slots["target_company"] = company
        if label == "jd_matching" and not memory.job_description:
            return None, "no job description on file"
        if label == "section_enhancement":
            role = ROLE_RX.search(message)
            if role:
                slots["target_role"] = role.group(1).strip().title()

        low = message.lower()
        kinds = {kind for word, kind in SECTION_WORDS.items() if re.search(rf"\b{word}\b", low)}
        targets = [s.id for s in resume.sections if s.kind in kinds]
        step = PlanStep(agent=label, instruction=message, target_sections=targets)
        decision = RoutingDecision(intent="edit", reasoning=why, steps=[step], **slots)
        return decision, "ok"
