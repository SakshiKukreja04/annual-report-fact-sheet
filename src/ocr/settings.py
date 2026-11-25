"""OCR settings. The confidence cutoff is a starting point, not a final choice."""

from pathlib import Path

from src.config.ocr import (
    OCR_MIN_CONTIGUOUS_WORDS as MIN_CONTIGUOUS_WORDS,
    OCR_MIN_RUN_CONFIDENCE as MIN_RUN_CONFIDENCE,
    OCR_WORD_CONFIDENCE as WORD_CONFIDENCE_THRESHOLD,
)

WHITE_BORDER_PX = 20
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
