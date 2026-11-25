"""Fast OCR path: one binarization, one Tesseract pass, Paddle only if needed."""

from __future__ import annotations

import time

import numpy as np

from src.ocr.assess import cleaned_text, is_document_like, kept_word_runs, recoverability
from src.ocr.binarize import illumination_std
from src.ocr.paddle_engine import run_paddle
from src.ocr.sample import _skipped_paddle
from src.ocr.settings import MIN_CONTIGUOUS_WORDS, MIN_RUN_CONFIDENCE, WORD_CONFIDENCE_THRESHOLD
from src.ocr.tesseract_engine import psm_for_page, recognize_variant


def choose_binarization(std: float, uniform_max: float = 28.0) -> str:
    """One threshold from illumination. Even light → Otsu; otherwise Sauvola."""
    if std <= uniform_max:
        return "otsu"
    return "sauvola"


def mean_line_confidence(runs: list[dict]) -> float:
    scores = [float(run["line_confidence"]) for run in runs if run.get("line_confidence") is not None]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def assess_half_fast(
    gray: np.ndarray,
    classifications: list[str],
    threshold: float = WORD_CONFIDENCE_THRESHOLD,
    ticker: str = "",
    pdf_page: int = 0,
    half: str = "FULL",
) -> dict:
    """Single Tesseract variant; Paddle only when mean kept-line confidence is low."""
    timers: dict[str, float] = {}
    document_like = is_document_like(gray)
    psm = psm_for_page(classifications, document_like)

    started = time.perf_counter()
    std = illumination_std(gray)
    timers["illumination_std"] = round(time.perf_counter() - started, 3)
    name = choose_binarization(std)
    # Caption / photo pages still use Sauvola (cheap and local).
    if "IMAGE_ONLY_VISUAL" in classifications or not document_like:
        name = "sauvola"

    started = time.perf_counter()
    variant = recognize_variant(gray, name, psm)
    # Otsu can refuse uneven pages; fall back once.
    if variant is None:
        name = "sauvola"
        variant = recognize_variant(gray, name, psm)
    timers[f"tesseract_{name}"] = round(time.perf_counter() - started, 3)

    runs = kept_word_runs(variant["words"], threshold=threshold) if variant else []
    mean_conf = mean_line_confidence(runs)
    need_paddle = (
        document_like
        and "IMAGE_ONLY_VISUAL" not in classifications
        and (not runs or mean_conf < MIN_RUN_CONFIDENCE)
    )
    started = time.perf_counter()
    paddle = run_paddle(gray) if need_paddle else _skipped_paddle()
    timers["paddle"] = round(time.perf_counter() - started, 3)

    decision = recoverability(classifications, document_like, runs)
    method = variant["processing_method"] if variant else "none"
    evidence = cleaned_text(runs)
    return {
        "ticker": ticker,
        "pdf_page": pdf_page,
        "half": half,
        "classifications": classifications,
        "document_like": document_like,
        "word_confidence_threshold": threshold,
        "min_run_confidence": MIN_RUN_CONFIDENCE,
        "min_contiguous_words": MIN_CONTIGUOUS_WORDS,
        "threshold_finalized": False,
        "path": "fast",
        "tesseract": {
            "engine": "tesseract",
            "psm": psm,
            "illumination_std": round(std, 2),
            "otsu_used": name == "otsu",
            "selected_processing_method": method,
            "raw_variants": [] if variant is None else [variant],
            "cleaned_text": evidence,
            "kept_lines": runs,
            "mean_line_confidence": round(mean_conf, 2),
        },
        "paddleocr": paddle,
        "primary_engine": "paddleocr" if paddle.get("available") else "tesseract",
        "evidence_engine": "tesseract",
        "processing_method": method,
        "recoverability": decision["recoverability"],
        "failure_reason": decision["failure_reason"],
        "pass_to_business_extraction": decision["pass_to_extraction"],
        "use_ocr_text": decision["use_ocr_text"],
        "caption_text": decision["caption_text"],
        "qwen_rescue": False,
        "raw_ocr_retained": True,
        "quote_rule": "A quote must be a contiguous run of reliable words.",
        "timers": timers,
        "processing_duration_sec": round(sum(timers.values()), 3),
    }
