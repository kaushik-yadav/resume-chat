"""Extraction and export check over real resumes (offline, nothing leaves the machine).

For each file it runs the product's upload path (text extraction + sectionizer) and then
round-trips the parsed resume through the PDF and DOCX exporters, re-reading each export with
the same upload path. It reports structure, timings and anything suspicious:

- text lost between extraction and the parsed sections (word recall < 99%);
- no headings recognized at all (the whole resume in one block);
- role or degree lines glued onto a bullet ("... | Jan 2024 - Present" inside a bullet);
- short fragments left after bullets (a wrapped line that was not re-joined);
- a header without a name, or contact details outside the header;
- exports that lose words or sections when read back.

    uv run python evals/extraction_check.py FILE_OR_DIR [...]   [--json out.json]

Only file names and counts are printed; resume text is never written out.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.resume import Resume  # noqa: E402
from app.parsing.extract import ResumeParseError, extract_text  # noqa: E402
from app.parsing.sectionizer import sectionize  # noqa: E402
from app.services.export import to_docx  # noqa: E402
from app.services.pdf_export import DATE_RANGE, parse_entries, to_pdf  # noqa: E402

WORD = re.compile(r"[\w'+#.%/-]+", re.UNICODE)
BULLET = re.compile(r"^\s*[-*•▪◦]\s+")
EMAIL = re.compile(r"[^\s@|]+@[^\s@|]+\.[a-z]{2,}", re.I)


def words(text: str) -> Counter:
    """Word multiset; punctuation-only tokens ("-", "|") are layout, not content."""
    tokens = (w.lower().strip(".,;:()") for w in WORD.findall(text.replace("**", "")))
    return Counter(t for t in tokens if any(ch.isalnum() for ch in t))


def recall(source: Counter, target: Counter) -> float:
    total = sum(source.values())
    if not total:
        return 1.0
    return sum(min(n, target[w]) for w, n in source.items()) / total


def body_text(resume: Resume) -> str:
    return "\n".join([s.title for s in resume.sections] + [s.content for s in resume.sections])


def structure_issues(resume: Resume) -> list[str]:
    issues = []
    kinds = [s.kind for s in resume.sections]
    if len(resume.sections) <= 1:
        issues.append("no section headings recognized (whole resume in one block)")
    header = next((s for s in resume.sections if s.kind == "header"), None)
    first = header.content.split("\n")[0].strip() if header and header.content else ""
    if not first or EMAIL.search(first) or len(first.split()) > 6 or re.search(r"\d{3}", first):
        issues.append("header does not start with a name")
    if header is None or not EMAIL.search(header.content):
        elsewhere = [s.title for s in resume.sections if s.kind != "header" and EMAIL.search(s.content)]
        issues.append(
            f"email not in header (found in: {', '.join(elsewhere)})" if elsewhere else "no email found"
        )
    for s in resume.sections:
        if s.kind not in {"experience", "projects", "education", "volunteer"}:
            continue
        lines = [ln for ln in s.content.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if BULLET.match(ln) and DATE_RANGE.search(ln.strip()) and " | " in ln:
                issues.append(f"{s.title}: a role/date line is inside a bullet")
            prev = lines[i - 1] if i else ""
            if (
                not BULLET.match(ln)
                and BULLET.match(prev)
                and len(ln.split()) <= 3
                and not DATE_RANGE.search(ln)
                and ln.strip()[:1].islower()
            ):
                issues.append(f"{s.title}: wrapped fragment not re-joined ('{ln.strip()[:24]}…')")
    if "experience" in kinds:
        exp = next(s for s in resume.sections if s.kind == "experience")
        entries = parse_entries(exp.content)
        undated = [e for e in entries if e.title and not e.date]
        if entries and len(undated) == len(entries):
            issues.append("experience: no dates detected on any role")
    return issues


def roundtrip(resume: Resume, render, ext: str) -> dict:
    t = time.perf_counter()
    data = render(resume)
    ms = (time.perf_counter() - t) * 1000
    back = sectionize(extract_text(f"export{ext}", data))
    original = words(body_text(resume))
    reread = words(body_text(back))
    lost_kinds = sorted({s.kind for s in resume.sections} - {s.kind for s in back.sections})
    return {
        "ms": round(ms),
        "kb": len(data) // 1024,
        "word_recall": round(recall(original, reread), 4),
        "sections_in": len(resume.sections),
        "sections_back": len(back.sections),
        "lost_section_kinds": lost_kinds,
    }


def check(path: Path) -> dict:
    data = path.read_bytes()
    row: dict = {"file": path.name, "type": path.suffix.lower().lstrip(".")}
    t = time.perf_counter()
    try:
        text = extract_text(path.name, data)
        resume = sectionize(text)
    except ResumeParseError as exc:
        row["error"] = str(exc)
        return row
    row["parse_ms"] = round((time.perf_counter() - t) * 1000)
    row["sections"] = [s.kind for s in resume.sections]
    exp = [s for s in resume.sections if s.kind == "experience"]
    row["roles"] = len(parse_entries(exp[0].content)) if exp else 0
    row["bullets"] = sum(
        1 for s in resume.sections for ln in s.content.split("\n") if BULLET.match(ln)
    )
    row["extraction_recall"] = round(recall(words(text), words(body_text(resume))), 4)
    row["issues"] = structure_issues(resume)
    if row["extraction_recall"] < 0.99:
        row["issues"].append(f"text lost while sectioning (recall {row['extraction_recall']:.1%})")
    for name, render, ext in (("pdf", to_pdf, ".pdf"), ("docx", to_docx, ".docx")):
        try:
            rt = roundtrip(resume, render, ext)
        except Exception as exc:  # report, keep going
            row["issues"].append(f"{name} export failed: {type(exc).__name__}: {exc}")
            continue
        row[name] = rt
        if rt["word_recall"] < 0.99:
            row["issues"].append(f"{name} export loses words (recall {rt['word_recall']:.1%})")
        if rt["lost_section_kinds"]:
            row["issues"].append(f"{name} export: sections not recognized on re-upload: {rt['lost_section_kinds']}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    files: list[Path] = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in {".pdf", ".docx"}) if p.is_dir() else [p]

    rows = [check(f) for f in files]
    print(f"{'file':44} {'type':4} {'parse':>6} {'secs':>4} {'roles':>5} {'bul':>4} {'recall':>7} "
          f"{'pdf ms':>6} {'pdf rt':>7} {'docx rt':>7}  issues")
    for r in rows:
        if "error" in r:
            print(f"{r['file'][:44]:44} {r['type']:4}  ERROR: {r['error']}")
            continue
        pdf, docx = r.get("pdf", {}), r.get("docx", {})
        print(
            f"{r['file'][:44]:44} {r['type']:4} {r['parse_ms']:>4}ms {len(r['sections']):>4} {r['roles']:>5} "
            f"{r['bullets']:>4} {r['extraction_recall']:>7.1%} {pdf.get('ms', '-'):>6} "
            f"{pdf.get('word_recall', 0):>7.1%} {docx.get('word_recall', 0):>7.1%}  {len(r['issues'])}"
        )
        for issue in r["issues"]:
            print(f"{'':6}- {issue}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        pdf_ms = [r["pdf"]["ms"] for r in ok if "pdf" in r]
        print(
            f"\n{len(rows)} files, {len(rows) - len(ok)} unreadable, "
            f"{sum(1 for r in ok if not r['issues'])} with no issues. "
            f"Parse median {statistics.median(r['parse_ms'] for r in ok):.0f} ms; "
            f"PDF render median {statistics.median(pdf_ms):.0f} ms, max {max(pdf_ms)} ms."
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
