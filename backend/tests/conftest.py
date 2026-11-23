from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture
def sample_pdf() -> bytes:
    return (SAMPLES / "alex_morgan_resume.pdf").read_bytes()


@pytest.fixture
def sample_docx() -> bytes:
    return (SAMPLES / "alex_morgan_resume.docx").read_bytes()


@pytest.fixture
def sample_jd() -> str:
    return (SAMPLES / "job_description_data_scientist.txt").read_text(encoding="utf-8")
