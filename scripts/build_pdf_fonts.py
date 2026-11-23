"""Build the trimmed fonts used by the PDF exporter (run once; output is committed).

Noto Sans ships ~4,000 hinted glyphs. The PDF exporter needs Latin (incl. extended and
Vietnamese), Greek, Cyrillic and common punctuation/symbols; other scripts use fallback fonts.
Dropping hinting and unused glyphs makes each font ~5x smaller, which cuts per-document font
parsing and subsetting time (the bulk of PDF render latency).

    cd backend && uv run python ../scripts/build_pdf_fonts.py   (fonttools comes with fpdf2)
"""

import tempfile
import urllib.request
from pathlib import Path

from fontTools import subset

FONTS = Path(__file__).resolve().parent.parent / "backend" / "app" / "assets" / "fonts"
SOURCE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-{}.ttf"
UNICODES = (
    "U+0000-024F,U+0250-02FF,U+0300-036F,U+0370-03FF,U+0400-052F,U+1E00-1EFF,"
    "U+2000-206F,U+20A0-20CF,U+2100-214F,U+2150-218F,U+2190-21FF,U+2200-22FF,"
    "U+2460-24FF,U+25A0-25FF,U+FB00-FB06"
)

tmp = Path(tempfile.mkdtemp())
for style in ("Regular", "Bold"):
    src = tmp / f"NotoSans-{style}.ttf"
    urllib.request.urlretrieve(SOURCE.format(style), src)  # full upstream font (not committed)
    out = FONTS / f"NotoSans-{style}-pdf.ttf"
    subset.main([
        str(src), f"--unicodes={UNICODES}", f"--output-file={out}",
        "--no-hinting", "--layout-features=", "--name-IDs=*", "--notdef-outline",
    ])
    print(f"{out.name}: {src.stat().st_size // 1024} KB -> {out.stat().st_size // 1024} KB")
