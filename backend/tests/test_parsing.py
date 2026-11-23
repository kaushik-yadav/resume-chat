import io

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.parsing.extract import ResumeParseError, extract_text, normalize_text
from app.parsing.sectionizer import classify_heading, parse_resume, sectionize

EXPECTED_IDS = [
    "header",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
]


@pytest.mark.parametrize("fixture,name", [("sample_pdf", "r.pdf"), ("sample_docx", "r.docx")])
def test_sample_resumes_split_into_expected_sections(request, fixture, name):
    resume = parse_resume(name, request.getfixturevalue(fixture))
    assert resume.section_ids == EXPECTED_IDS
    assert resume.language == "en"
    assert "Alex Morgan" in resume.get("header").content
    assert "PostgreSQL" in resume.get("experience").content
    assert "AWS Certified" in resume.get("certifications").content


def test_docx_list_paragraphs_become_bullets(sample_docx):
    text = extract_text("resume.docx", sample_docx)
    assert "- Helped migrate the transactions database from MySQL to PostgreSQL" in text


def test_docx_tables_are_read_in_order():
    doc = Document()
    doc.add_paragraph("Jordan Lee")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "EXPERIENCE"
    table.cell(0, 1).text = "Built data pipelines for analytics teams at Acme Corp " * 3
    doc.add_paragraph("SKILLS")
    doc.add_paragraph("Python, SQL, dbt, Snowflake, Looker and more tools " * 2)
    buf = io.BytesIO()
    doc.save(buf)
    resume = parse_resume("table.docx", buf.getvalue())
    assert resume.section_ids == ["header", "experience", "skills"]


def test_rejects_unsupported_and_empty_files():
    with pytest.raises(ResumeParseError, match="Unsupported"):
        extract_text("resume.png", b"data")
    with pytest.raises(ResumeParseError, match="empty"):
        extract_text("resume.pdf", b"")
    with pytest.raises(ResumeParseError, match="Very little text"):
        extract_text("resume.txt", b"just a few words")


def test_normalize_text_fixes_bullets_ligatures_and_hyphenation():
    raw = "• Led the eﬃcient develop-\nment of APIs\n▪ Shipped   features\n\n\n\nEnd"
    assert (
        normalize_text(raw) == "- Led the efficient development of APIs\n- Shipped features\n\nEnd"
    )


@pytest.mark.parametrize(
    "line,kind",
    [
        ("EXPERIENCE", "experience"),
        ("Work Experience:", "experience"),
        ("## Technical Skills", "skills"),
        ("RELEVANT WORK EXPERIENCE", "experience"),
        ("Experiencia Laboral", "experience"),
        ("## Side Quests", "other"),
        ("ALEX MORGAN", None),  # names in caps are not headings
        ("SENIOR SOFTWARE ENGINEER", None),  # neither are job titles
        ("Skills: Python, SQL", None),  # inline label, not a heading
        ("- Experience with Python", None),
    ],
)
def test_classify_heading(line, kind):
    assert classify_heading(line) == kind


def test_duplicate_sections_get_unique_ids():
    text = "Name\nPROJECTS\nA project with enough words\nPROJECTS\nAnother project here"
    assert sectionize(text).section_ids == ["header", "projects", "projects-2"]


def test_unstructured_text_falls_back_to_single_section():
    resume = sectionize("Just a paragraph about me and my work history without headings.")
    assert resume.section_ids == ["resume"]


def test_spanish_resume_language_detected():
    text = "Ana Ruiz\nExperiencia\n- Desarrollé APIs en la empresa durante 3 años\nEducación\nUNAM"
    assert sectionize(text).language == "es"


def test_parse_endpoint(sample_docx):
    client = TestClient(create_app(Settings(llm_provider="fake", storage_backend="memory")))
    ok = client.post("/api/resumes/parse", files={"file": ("cv.docx", sample_docx)})
    assert ok.status_code == 200
    assert [s["id"] for s in ok.json()["sections"]] == EXPECTED_IDS
    bad = client.post("/api/resumes/parse", files={"file": ("cv.exe", b"MZ")})
    assert bad.status_code == 422
