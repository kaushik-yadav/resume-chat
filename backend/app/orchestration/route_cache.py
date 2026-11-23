"""Routing cache: an exact repeat of a request in the same context reuses its routing decision.

The *work* is always redone (the resume may have changed since), but deciding what the request
means is not. The key covers everything routing depends on - the normalized message, the
memory slots, the resume's language and sections - and requests that refer back to earlier
turns ("do the same", "again", "for them") are never cached, because their meaning depends on
history that the key does not capture.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict

from app.models.conversation import ConversationContext
from app.models.resume import Resume
from app.orchestration.router import RoutingDecision

CONTEXT_DEPENDENT = re.compile(
    r"\b(same|again|that one|those|them|they|their|previous|earlier|above|last one|instead)\b",
    re.IGNORECASE,
)


def normalize_message(message: str) -> str:
    return re.sub(r"[\s.!?]+$", "", " ".join(message.lower().split()))


class RouteCache:
    def __init__(self, max_entries: int = 512) -> None:
        self._entries: OrderedDict[str, RoutingDecision] = OrderedDict()
        self._max = max_entries
        self.hits = self.misses = 0

    def key(self, message: str, memory: ConversationContext, resume: Resume) -> str | None:
        if CONTEXT_DEPENDENT.search(message) or len(message) > 4000:
            return None
        jd_hash = hashlib.sha256((memory.job_description or "").encode()).hexdigest()[:12]
        parts = [
            normalize_message(message),
            (memory.target_company or "").lower(),
            (memory.target_role or "").lower(),
            (memory.target_locale or "").lower(),
            jd_hash,
            resume.language,
            ",".join(resume.section_ids),
        ]
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, key: str | None) -> RoutingDecision | None:
        if key is None or key not in self._entries:
            self.misses += key is not None
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key].model_copy(deep=True)

    def put(self, key: str | None, decision: RoutingDecision) -> None:
        if key is None or decision.intent in ("undo", "redo"):
            return
        self._entries[key] = decision.model_copy(deep=True)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
