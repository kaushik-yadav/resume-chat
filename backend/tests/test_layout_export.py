"""Original-layout export: the uploaded PDF/DOCX edited in place."""

import io

import pymupdf
import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.models.conversation import ResumeVersion
from app.parsing.extract import extract_text
from app.parsing.sectionizer import parse_resume, sectionize
from app.services import export
from app.services.layout_export import LayoutUnavailable, edit_docx, edit_pdf, norm


def edited(resume, section_id: str, old: str, new: str):
    copy = resume.model_copy(deep=True)
    section = next(s for s in copy.sections if s.id == section_id)
    assert old in section.content
    section.content = section.content.replace(old, new)
    return copy


def spans(pdf: bytes) -> dict[str, tuple]:
    """Text of each span -> (font, size, bbox), page 1."""
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    return {
        s["text"].strip(): (s["font"], round(s["size"], 2), tuple(round(v, 1) for v in s["bbox"]))
        for b in page.get_text("dict")["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip()
    }


def sorted_text(pdf: bytes) -> str:
    return norm("".join(p.get_text(sort=True) for p in pymupdf.open(stream=pdf, filetype="pdf")))


def test_unchanged_version_is_the_original_file(sample_pdf):
    base = parse_resume("cv.pdf", sample_pdf)
    assert edit_pdf(sample_pdf, base, base.model_copy(deep=True)) == sample_pdf


def test_changed_line_replaced_in_place_everything_else_untouched(sample_pdf):
    base = parse_resume("cv.pdf", sample_pdf)
    old = "Helped migrate the transactions database from MySQL to PostgreSQL"
    new = "Led the migration of the transactions database from MySQL to PostgreSQL"
    out = edit_pdf(sample_pdf, base, edited(base, "experience", old, new))

    text = sorted_text(out)
    assert norm(new) in text and norm(old) not in text
    before, after = spans(sample_pdf), spans(out)
    # Every other span keeps its font, size and position exactly.
    untouched = [k for k in before if k and k not in old and old not in k]
    assert untouched and all(after.get(k) == before[k] for k in untouched if k in after)
    assert sum(1 for k in untouched if k in after) >= len(untouched) - 1
    assert len(pymupdf.open(stream=out, filetype="pdf")) == len(pymupdf.open(stream=sample_pdf, filetype="pdf"))


def test_date_change_stays_right_aligned(sample_pdf):
    base = parse_resume("cv.pdf", sample_pdf)
    out = edit_pdf(sample_pdf, base, edited(base, "experience", "Jan 2022 - Present", "Jan 2022 - Dec 2025"))
    page = pymupdf.open(stream=out, filetype="pdf")[0]
    orig = pymupdf.open(stream=sample_pdf, filetype="pdf")[0]
    new_box = page.search_for("Dec 2025")[0]
    old_box = orig.search_for("Present")[0]
    assert abs(new_box.x1 - old_box.x1) < 1.5 and abs(new_box.y1 - old_box.y1) < 1.5


def test_layout_that_cannot_hold_the_edit_is_refused(sample_pdf):
    base = parse_resume("cv.pdf", sample_pdf)
    grown = base.model_copy(deep=True)
    grown.sections[-1].content += "\nAzure Fundamentals (2024)\nGCP Associate Engineer (2024)"
    with pytest.raises(LayoutUnavailable, match="added"):
        edit_pdf(sample_pdf, base, grown)


def test_edited_pdf_reuploads_in_reading_order(sample_pdf):
    base = parse_resume("cv.pdf", sample_pdf)
    target = edited(base, "skills", "Languages: Python, SQL, JavaScript, Go", "Languages: Python, SQL, TypeScript, Go")
    again = sectionize(extract_text("cv.pdf", edit_pdf(sample_pdf, base, target)))
    skills = next(s for s in again.sections if s.kind == "skills")
    assert skills.content.split("\n")[0].startswith("Languages: Python, SQL, TypeScript, Go")


def test_docx_paragraph_rewritten_with_its_formatting(sample_docx):
    base = parse_resume("cv.docx", sample_docx)
    old = "Helped migrate the transactions database from MySQL to PostgreSQL"
    new = "Led the migration of the transactions database from MySQL to PostgreSQL"
    target = edited(base, "experience", old, new)
    target_exp = next(s for s in target.sections if s.id == "experience")
    target_exp.content += "\n- Added a new bullet that only a DOCX can make room for"
    out = Document(io.BytesIO(edit_docx(sample_docx, base, target)))
    texts = [p.text for p in out.paragraphs]
    before = {p.text: p.style.name for p in Document(io.BytesIO(sample_docx)).paragraphs}
    replaced = next(p for p in out.paragraphs if p.text == new)
    assert replaced.style.name == before[old]
    assert old not in texts and "Added a new bullet that only a DOCX can make room for" in texts


def test_export_falls_back_to_clean_layout_with_a_reason(sample_pdf):
    from app.models.conversation import Conversation
    from app.storage.memory import MemoryRepository

    repo = MemoryRepository()
    conv = repo.save_conversation(Conversation())
    base = parse_resume("cv.pdf", sample_pdf)
    upload = repo.add_version(conv.id, ResumeVersion(number=1, source="upload", resume=base))
    repo.save_original(conv.id, upload.id, "cv.pdf", sample_pdf)
    grown = base.model_copy(deep=True)
    grown.sections[-1].content += "\nAzure Fundamentals (2024)\nGCP Associate Engineer (2024)"
    child = repo.add_version(conv.id, ResumeVersion(number=2, parent_id=upload.id, resume=grown))

    same = export.render_export(conv.id, upload, "pdf", repo)
    assert same.layout == "original" and same.body == sample_pdf
    fallback = export.render_export(conv.id, child, "pdf", repo)
    assert fallback.layout == "clean" and "added" in fallback.note
    assert export.render_export(conv.id, child, "pdf", repo, layout="clean").layout == "clean"


def test_api_reports_layout_and_serves_original(sample_pdf):
    from app.main import create_app
    from tests.test_orchestration import overrides, settings

    with TestClient(create_app(settings(), overrides())) as client:
        cid = client.post("/api/conversations").json()["id"]
        state = client.post(
            f"/api/conversations/{cid}/resume", files={"file": ("cv.pdf", sample_pdf, "application/pdf")}
        ).json()
        vid = state["current_version"]["id"]
        pdf = client.get(f"/api/conversations/{cid}/export", params={"format": "pdf"})
        assert pdf.headers["x-export-layout"] == "original" and pdf.content == sample_pdf
        clean = client.get(f"/api/conversations/{cid}/export", params={"format": "pdf", "layout": "clean"})
        assert clean.headers["x-export-layout"] == "clean" and clean.content != sample_pdf
        info = client.get(f"/api/conversations/{cid}/export/info", params={"version_id": vid}).json()
        assert info["pdf"]["layout"] == "original" and info["docx"]["layout"] == "clean"
