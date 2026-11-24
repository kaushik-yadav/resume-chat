"""PDF export: layout parsing, fidelity (read back through the upload parser), fonts, cache."""

import io
import os

import pytest
from pypdf import PdfReader

from app.models.conversation import ResumeVersion
from app.models.resume import Resume, ResumeSection
from app.parsing.extract import extract_text, normalize_text
from app.parsing.sectionizer import classify_heading, sectionize
from app.services import export
from app.services.pdf_export import _fallback_fonts, _header, _split_date, parse_entries, to_pdf


def resume(*sections: tuple[str, str, str], locale: str | None = None) -> Resume:
    return Resume(
        sections=[ResumeSection(id=k, kind=k, title=t, content=c) for k, t, c in sections],
        locale=locale,
    )


SAMPLE = resume(
    ("header", "Header", "Jake Ryan\n123-456-7890 | jake@su.edu | linkedin.com/in/jake"),
    ("summary", "Summary", "Backend engineer focused on\nreliable data systems."),
    (
        "education",
        "Education",
        "Southwestern University Georgetown, TX\nBachelor of Arts in CS Aug. 2018 – May 2021\n"
        "Blinn College Bryan, TX\nAssociate's in Liberal Arts Aug. 2014 – May 2018",
    ),
    (
        "experience",
        "Experience",
        "Research Assistant | Texas A&M | College Station | June 2020 – Present\n"
        "- Built a REST API with **FastAPI** and PostgreSQL\n- Cut query latency by 40%",
    ),
    (
        "projects",
        "Projects",
        "Gitlytics | Python, Flask, React (GitHub) Dec – Jan 2024\n- Shipped v1",
    ),
    ("skills", "Technical Skills", "Languages: Java, Python, C/C++\n- Docker\n- Git"),
)


def words(text: str) -> set[str]:
    return {w.strip(".,()*").lower() for w in text.split() if any(c.isalnum() for c in w)}


def test_second_dates_start_new_entries():
    entries = parse_entries(SAMPLE.sections[2].content)
    assert [(e.title, e.date) for e in entries] == [
        ("Southwestern University Georgetown, TX", "Aug. 2018 – May 2021"),
        ("Blinn College Bryan, TX", "Aug. 2014 – May 2018"),
    ]


def test_month_span_dates_and_verbatim_tech_stack():
    (entry,) = parse_entries(SAMPLE.sections[4].content)
    assert entry.date == "Dec – Jan 2024"
    assert entry.tags == ["Python", "Flask", "React (GitHub)"]


def test_pipe_separated_headline_is_not_contact_details():
    _, free, contacts = _header(
        ResumeSection(
            id="header",
            kind="header",
            title="Header",
            content="Jaya K\nSenior Recruiter | Leadership Hiring\njaya@mail.com | +91-8958488326",
        )
    )
    assert free == ["Senior Recruiter | Leadership Hiring"]
    assert [c for c, _ in contacts] == ["jaya@mail.com", "+91-8958488326"]


def test_pdf_reads_back_with_every_word_and_section():
    data = to_pdf(SAMPLE)
    assert data.startswith(b"%PDF") and len(PdfReader(io.BytesIO(data)).pages) == 1
    back = sectionize(extract_text("export.pdf", data))
    source = words("\n".join(s.content for s in SAMPLE.sections))
    assert source <= words("\n".join(s.content for s in back.sections))
    assert {s.kind for s in SAMPLE.sections} <= {s.kind for s in back.sections}


def test_page_size_follows_locale():
    a4 = PdfReader(io.BytesIO(to_pdf(SAMPLE))).pages[0].mediabox
    letter = (
        PdfReader(
            io.BytesIO(
                to_pdf(
                    resume(*[(s.kind, s.title, s.content) for s in SAMPLE.sections], locale="en-US")
                )
            )
        )
        .pages[0]
        .mediabox
    )
    assert round(float(a4.height)) == 842 and round(float(letter.height)) == 792


def test_icons_are_dropped_and_need_no_fallback_font():
    r = resume(
        ("header", "Header", "Jaya K\n✉ jaya@mail.com | 📞 +91 89584 88326"),
        ("skills", "Skills", "○ Sourcing"),
    )
    assert _fallback_fonts("✉ 📞 ○") == []
    text = PdfReader(io.BytesIO(to_pdf(r))).pages[0].extract_text()
    assert "jaya@mail.com" in text and "📞" not in text and "Sourcing" in text


@pytest.mark.skipif(
    not any(
        os.path.exists(p)
        for p in (
            r"C:\Windows\Fonts\YuGothR.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )
    ),
    reason="no CJK font installed",
)
def test_japanese_uses_a_fallback_font():
    r = resume(
        ("header", "Header", "山田 太郎\ntaro@example.jp"),
        ("summary", "職務要約", "バックエンド開発者"),
    )
    assert _fallback_fonts("山田") != []
    text = PdfReader(io.BytesIO(to_pdf(r))).pages[0].extract_text()
    assert "山田" in text


def test_rendered_files_are_cached_per_version():
    version = ResumeVersion(id="v-cache", number=1, label="Upload", resume=SAMPLE)
    first = export.render_export("c-cache", version, "pdf")
    again = export.render_export("c-cache", version, "pdf")
    assert first.media == "application/pdf" and first.layout == "clean" and again is first


def test_symbol_font_bullets_from_word_pdfs():
    text = normalize_text(
        "Experience\n\uf0b7 Led data profiling\n\x7f Built dashboards\n\uf095 +1 555 0100"
    )
    assert "- Led data profiling" in text and "- Built dashboards" in text
    assert "\uf095" not in text and "+1 555 0100" in text


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("PERSONAL DETAILS", "other"),
        ("PROFESSIONAL HIGHLIGHTS", "summary"),
        ("Internships", "experience"),
        ("Relevant Coursework", "education"),
    ],
)
def test_headings_found_in_real_resumes(line, kind):
    assert classify_heading(line) == kind


@pytest.mark.parametrize(
    "line,title,date",
    [
        (
            "B.S. Computer Science, University of Texas at Dallas 2015 - 2019",
            "B.S. Computer Science, University of Texas at Dallas",
            "2015 - 2019",
        ),
        (
            "Engineer, Acme, Austin, TX Jan 2022 - Present",
            "Engineer, Acme, Austin, TX",
            "Jan 2022 - Present",
        ),
        ("Ingeniero, Acme ene. 2022 - actualidad", "Ingeniero, Acme", "ene. 2022 - actualidad"),
        ("Entwickler, X März 2021 - Okt 2023", "Entwickler, X", "März 2021 - Okt 2023"),
        ("Intern, Foo Dec - Jan 2024", "Intern, Foo", "Dec - Jan 2024"),
        (
            "MSc, University of Normandy Expected May 2026",
            "MSc, University of Normandy",
            "Expected May 2026",
        ),
        ("Analyst, Marketing Co 2019", "Analyst, Marketing Co", "2019"),
    ],
)
def test_only_a_month_may_precede_the_year(line, title, date):
    # A place name before the year ("Dallas 2015") stays in the title.
    assert _split_date(line) == (title, date)
