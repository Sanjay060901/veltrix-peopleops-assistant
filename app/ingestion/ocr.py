"""OCR for scanned pages (pages without a usable text layer)."""
from __future__ import annotations

import io
import logging
import re
import shutil

log = logging.getLogger(__name__)

OCR_DPI = 300


def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def ocr_png_bytes(png: bytes) -> str:
    """Run Tesseract on one rendered page image. Returns '' when OCR is unavailable."""
    if not tesseract_available():
        log.error("tesseract is not installed; scanned page cannot be read")
        return ""
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(png))
    # psm 6 would treat the page as a single block; the default (3) keeps paragraph breaks.
    return pytesseract.image_to_string(image, lang="eng")


def ocr_text_to_paragraphs(text: str) -> list[str]:
    """Join wrapped OCR lines into paragraphs; blank lines separate paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    # Drop OCR noise such as a scanned signature rendered as a few stray characters.
    return [p for p in cleaned if sum(ch.isalpha() for ch in p) >= 4 and _word_ratio(p) > 0.5]


def _word_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    real = sum(1 for w in words if re.fullmatch(r"[A-Za-z][a-z]+[.,:;]?|[A-Z]+[.,:;]?|[\d@./:,-]+\w*", w))
    return real / len(words)
