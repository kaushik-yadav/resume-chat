import pytest

from app.models.resume import Resume, ResumeSection


def make_resume() -> Resume:
    return Resume(
        sections=[
            ResumeSection(
                id="header", title="Header", kind="header", content="Alex Morgan\nAustin, TX"
            ),
            ResumeSection(id="skills", title="Skills", kind="skills", content="Python, SQL"),
        ]
    )


def test_with_section_content_returns_copy():
    original = make_resume()
    updated = original.with_section_content("skills", "  Python, SQL, pandas  ")
    assert updated.get("skills").content == "Python, SQL, pandas"
    assert original.get("skills").content == "Python, SQL"


def test_with_section_content_unknown_section():
    with pytest.raises(KeyError):
        make_resume().with_section_content("nope", "x")


def test_markdown_and_outline():
    resume = make_resume()
    # Hard break ("  ") keeps the two header lines on separate lines when rendered.
    assert resume.to_markdown() == "Alex Morgan  \nAustin, TX\n\n## Skills\n\nPython, SQL\n"
    assert "[skills] Skills (skills): Python, SQL" in resume.outline()
    assert resume.word_count() == 7
