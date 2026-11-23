"""Labelled routing cases, split so that threshold tuning never sees the test set.

* ``CALIBRATION`` - the only cases used to choose the fast classifier's thresholds
  (``routing_eval.py --calibrate``).
* ``TEST`` - held out: unseen paraphrases and new requests, used only to report accuracy.

Expected slots are exact canonical values (case-insensitive): "fr-FR", not "fr".
``leakage()`` guards the split: no test case may equal or nearly equal (token Jaccard >= 0.8)
a calibration case or one of the classifier's exemplars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
JD = (SAMPLES / "job_description_data_scientist.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Case:
    message: str
    intent: str
    agents: tuple[str, ...] = ()
    slots: dict[str, str] = field(default_factory=dict)
    memory: dict[str, str] = field(default_factory=dict)  # ConversationContext fields


def edit(message, *agents, memory=None, **slots) -> Case:
    return Case(message, "edit", tuple(agents), slots, memory or {})


def other(message, intent, memory=None) -> Case:
    return Case(message, intent, (), {}, memory or {})


WITH_JD = {"job_description": JD}

CALIBRATION: list[Case] = [
    edit("Optimize my resume for Google", "company_research", target_company="Google"),
    edit("Translate this to Spanish for the Mexican market", "translation", target_locale="es-MX"),
    edit("Update my experience section to better match this job description", "jd_matching", memory=WITH_JD),
    edit("Make my skills section more relevant for a data scientist role", "section_enhancement", target_role="Data Scientist"),
    edit("Optimize for Stripe and then translate it to German", "company_research", "translation", target_company="Stripe", target_locale="de-DE"),
    edit("tailor it to this job " + JD, "jd_matching"),
    other("What's the weakest part of my resume?", "question"),
    edit("make the summary punchier", "section_enhancement"),
    other("undo the last change please", "undo"),
    other("thanks a lot!", "chitchat"),
    edit("Make it better for them", "company_research", memory={"target_company": "Netflix"}),
    other("Translate my resume", "clarify"),
    other("How good is my match for the job I pasted?", "question", memory=WITH_JD),
    edit("Rewrite my projects with stronger verbs and then localize for Brazil", "section_enhancement", "translation", target_locale="pt-BR"),
    edit("Tailor my resume for Amazon", "company_research", target_company="Amazon"),
    edit("Make a French version of my resume for France", "translation", target_locale="fr-FR"),
    edit("Improve the bullets in my projects section", "section_enhancement"),
    other("Does my resume pass ATS filters?", "question"),
    edit("Match my resume to this posting", "jd_matching", memory=WITH_JD),
    other("good morning", "chitchat"),
    edit("Optimize my resume for Microsoft", "company_research", target_company="Microsoft"),
    edit("Translate it to Portuguese for Brazil", "translation", target_locale="pt-BR"),
    # Added while fixing the classifier's "and <verb>" compound guard: calibration, not test.
    edit("Tailor my resume for Apple and translate it into French", "company_research", "translation", target_company="Apple", target_locale="fr-FR"),
]  # fmt: skip

TEST: list[Case] = [
    # Paraphrases first written for the original eval - never used for tuning now.
    edit("could you put my CV into French?", "translation", target_locale="fr-FR"),
    edit("I want a German version of my resume", "translation", target_locale="de-DE"),
    edit("tune my resume towards Amazon", "company_research", target_company="Amazon"),
    edit("I'm applying at Microsoft, adjust my resume", "company_research", target_company="Microsoft"),
    edit("strengthen the bullets in my experience", "section_enhancement"),
    edit("my summary is too long, trim it", "section_enhancement"),
    other("which skills should I learn next?", "question"),
    other("is this resume good enough for FAANG?", "question"),
    edit("fit my resume to the posting", "jd_matching", memory=WITH_JD),
    other("hello", "chitchat"),
    edit("Translate it to Japanese for jobs in Japan", "translation", target_locale="ja-JP"),
    edit("please make my education section cleaner", "section_enhancement"),
    # New held-out requests.
    edit("Get my resume ready for a role at Netflix", "company_research", target_company="Netflix"),
    edit("Adapt my resume to Atlassian's culture", "company_research", target_company="Atlassian"),
    edit("Localize my CV for the job market in Spain", "translation", target_locale="es-ES"),
    edit("Need this resume in Italian", "translation", target_locale="it-IT"),
    edit("Can you rewrite my summary to sound more senior?", "section_enhancement"),
    edit("Add metrics to my work experience bullets", "section_enhancement"),
    edit("Tailor my experience to the job description I shared", "jd_matching", memory=WITH_JD),
    other("How well do I match the posting?", "question", memory=WITH_JD),
    other("What would Google recruiters think of my resume?", "question"),
    other("Is two pages too long for a resume?", "question"),
    other("redo", "redo"),
    other("revert that", "undo"),
    other("cheers, that's great", "chitchat"),
    edit("Optimize my resume for Spotify and then put it into Dutch", "company_research", "translation", target_company="Spotify", target_locale="nl-NL"),
    edit("Translate my resume to Spanish for Mexico, then polish the skills section", "section_enhancement", "translation", target_locale="es-MX"),
    edit("Make my resume stronger for a product manager role", "section_enhancement", target_role="Product Manager"),
    other("Update my summary for this job description", "clarify"),  # no JD on file
    other("Optimize it for the company", "clarify"),  # no company named or remembered
]  # fmt: skip


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def leakage(
    test: list[Case], *references: list[str], threshold: float = 0.8
) -> list[tuple[str, str]]:
    """Test messages that equal or nearly equal any reference text (token Jaccard)."""
    hits = []
    for case in test:
        mine = _tokens(case.message)
        for ref in (r for group in references for r in group):
            theirs = _tokens(ref)
            overlap = len(mine & theirs) / (len(mine | theirs) or 1)
            if overlap >= threshold:
                hits.append((case.message, ref))
    return hits
