"""Tesseract path: binarized image, word confidence, and word boxes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytesseract

from src.ocr.binarize import adaptive_gaussian, illumination_std, otsu, sauvola, tesseract_variants
from src.ocr.settings import TESSERACT_EXE

if TESSERACT_EXE.is_file():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)


def tesseract_available() -> bool:
    return Path(pytesseract.pytesseract.tesseract_cmd).is_file()


def psm_for_page(classifications: list[str], document_like: bool) -> int:
    """Document pages use the configured PSM. PSM 11 reads scattered labels."""
    from src.config.ocr import OCR_PSM_DOCUMENT

    if "IMAGE_ONLY_VISUAL" in classifications or not document_like:
        return 11
    return int(OCR_PSM_DOCUMENT)


def recognize_words(binary: np.ndarray, psm: int) -> dict:
    """Keep every recognized word, including low-confidence words."""
    data = pytesseract.image_to_data(
        binary,
        lang="eng",
        config=f"--oem 1 --psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    count = len(data["text"])
    for index in range(count):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        confidence = float(data["conf"][index])
        words.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": [
                    int(data["left"][index]),
                    int(data["top"][index]),
                    int(data["width"][index]),
                    int(data["height"][index]),
                ],
                "line_id": (
                    f"{int(data['block_num'][index])}-"
                    f"{int(data['par_num'][index])}-"
                    f"{int(data['line_num'][index])}"
                ),
            }
        )
    return {
        "raw_text": " ".join(word["text"] for word in words),
        "words": words,
        "lines": line_confidence(words),
    }


def line_confidence(words: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for word in words:
        grouped.setdefault(word["line_id"], []).append(word)
    lines = []
    for line_id, items in grouped.items():
        scores = [float(item["confidence"]) for item in items if float(item["confidence"]) >= 0]
        boxes = [item["bbox"] for item in items]
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[0] + box[2] for box in boxes)
        y1 = max(box[1] + box[3] for box in boxes)
        lines.append(
            {
                "line_id": line_id,
                "text": " ".join(item["text"] for item in items),
                "line_confidence": round(sum(scores) / len(scores), 2) if scores else None,
                "word_count": len(items),
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            }
        )
    return lines


_THRESHOLDS = {
    "adaptive_gaussian": adaptive_gaussian,
    "sauvola": sauvola,
    "otsu": otsu,
}


def recognize_variant(gray: np.ndarray, name: str, psm: int) -> dict | None:
    """Recognize one threshold. Otsu returns None when the page lighting is uneven."""
    binary = _THRESHOLDS[name](gray)
    if binary is None:
        return None
    recognized = recognize_words(binary, psm)
    return {
        "engine": "tesseract",
        "binarization": name,
        "psm": psm,
        "processing_method": f"{name}_psm{psm}",
        "raw_text": recognized["raw_text"],
        "words": recognized["words"],
        "lines": recognized["lines"],
    }


def run_tesseract(gray: np.ndarray, psm: int) -> dict:
    """Try each allowed threshold. The caller keeps every raw result."""
    variants = []
    for name, binary in tesseract_variants(gray):
        recognized = recognize_words(binary, psm)
        variants.append(
            {
                "engine": "tesseract",
                "binarization": name,
                "psm": psm,
                "processing_method": f"{name}_psm{psm}",
                "raw_text": recognized["raw_text"],
                "words": recognized["words"],
                "lines": recognized["lines"],
            }
        )
    return {
        "available": True,
        "engine": "tesseract",
        "psm": psm,
        "illumination_std": round(illumination_std(gray), 2),
        "otsu_used": any(item["binarization"] == "otsu" for item in variants),
        "variants": variants,
    }
