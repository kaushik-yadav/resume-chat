"""Build the sample resume (fictional candidate) as DOCX and HTML.

The PDF is printed from the HTML with a headless Chromium browser:

    python scripts/build_samples.py
    msedge --headless --print-to-pdf=samples/alex_morgan_resume.pdf samples/alex_morgan_resume.html
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt

OUT = Path(__file__).resolve().parent.parent / "samples"

NAME = "Alex Morgan"
CONTACT = (
    "Austin, TX | alex.morgan@example.com | (512) 555-0147 | linkedin.com/in/alexmorgan-example"
)
SUMMARY = (
    "Software engineer with 5 years of experience building backend services and data "
    "pipelines in Python. Comfortable across APIs, SQL and cloud infrastructure, and "
    "interested in moving toward machine learning and data science work."
)
EXPERIENCE = [
    (
        "Software Engineer II, Brightline Payments, Austin, TX",
        "Jan 2022 - Present",
        [
            "Worked on backend services for the payments team using Python and FastAPI",
            "Helped migrate the transactions database from MySQL to PostgreSQL",
            "Built internal dashboards to track failed payments",
            "Did code reviews and mentored two interns",
        ],
    ),
    (
        "Software Engineer, Northwind Logistics, Dallas, TX",
        "Jun 2019 - Dec 2021",
        [
            "Wrote ETL jobs in Python and Airflow to load shipment data into the warehouse",
            "Responsible for maintaining REST APIs used by the mobile app",
            "Reduced nightly batch runtime from 6 hours to 2 hours by rewriting slow SQL queries",
            "Participated in on-call rotation",
        ],
    ),
]
EDUCATION = [
    ("B.S. Computer Science, University of Texas at Dallas", "2015 - 2019", "GPA 3.6/4.0"),
]
SKILLS = [
    ("Languages", "Python, SQL, JavaScript, Go"),
    ("Frameworks", "FastAPI, Flask, Airflow, React"),
    ("Data", "PostgreSQL, MySQL, BigQuery, pandas"),
    ("Cloud and tools", "AWS (Lambda, S3, RDS), Docker, Git, GitHub Actions"),
]
PROJECTS = [
    (
        "Churn Predictor (personal project)",
        "Trained a scikit-learn model on a public telecom dataset to predict customer churn "
        "and served it with a small FastAPI app.",
    ),
    (
        "Open-source contributor, Apache Airflow",
        "Fixed two bugs in the Postgres provider and improved its documentation.",
    ),
]
CERTIFICATIONS = ["AWS Certified Developer - Associate (2023)"]


def build_docx(path: Path) -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading(NAME, level=0)
    doc.add_paragraph(CONTACT)

    doc.add_heading("Summary", level=1)
    doc.add_paragraph(SUMMARY)

    doc.add_heading("Experience", level=1)
    for title, dates, bullets in EXPERIENCE:
        doc.add_paragraph(f"{title} | {dates}").runs[0].bold = True
        for b in bullets:
            doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("Education", level=1)
    for degree, dates, extra in EDUCATION:
        doc.add_paragraph(f"{degree} | {dates}")
        doc.add_paragraph(extra)

    doc.add_heading("Skills", level=1)
    for label, items in SKILLS:
        doc.add_paragraph(f"{label}: {items}")

    doc.add_heading("Projects", level=1)
    for title, desc in PROJECTS:
        doc.add_paragraph(title).runs[0].bold = True
        doc.add_paragraph(desc, style="List Bullet")

    doc.add_heading("Certifications", level=1)
    for c in CERTIFICATIONS:
        doc.add_paragraph(c, style="List Bullet")
    doc.save(path)


def build_html(path: Path) -> None:
    def ul(items):
        return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

    exp = "".join(
        f"<div class='row'><strong>{t}</strong><span>{d}</span></div>{ul(b)}"
        for t, d, b in EXPERIENCE
    )
    edu = "".join(
        f"<div class='row'><strong>{d}</strong><span>{dt}</span></div><p>{e}</p>"
        for d, dt, e in EDUCATION
    )
    skills = "".join(f"<p><strong>{label}:</strong> {items}</p>" for label, items in SKILLS)
    projects = "".join(f"<p><strong>{t}</strong></p>{ul([d])}" for t, d in PROJECTS)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{NAME}</title>
<style>
 body {{ font-family: Calibri, Arial, sans-serif; font-size: 10.5pt; margin: 0.6in; color: #111; }}
 h1 {{ font-size: 22pt; margin: 0; }} h2 {{ font-size: 11.5pt; text-transform: uppercase;
 letter-spacing: .08em; border-bottom: 1px solid #999; margin: 14px 0 6px; }}
 .row {{ display: flex; justify-content: space-between; margin-top: 6px; }}
 ul {{ margin: 3px 0 3px 18px; padding: 0; }} p {{ margin: 3px 0; }}
</style></head><body>
<h1>{NAME}</h1><p>{CONTACT}</p>
<h2>Summary</h2><p>{SUMMARY}</p>
<h2>Experience</h2>{exp}
<h2>Education</h2>{edu}
<h2>Skills</h2>{skills}
<h2>Projects</h2>{projects}
<h2>Certifications</h2>{ul(CERTIFICATIONS)}
</body></html>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_docx(OUT / "alex_morgan_resume.docx")
    build_html(OUT / "alex_morgan_resume.html")
    print("Wrote", OUT / "alex_morgan_resume.docx", "and", OUT / "alex_morgan_resume.html")
