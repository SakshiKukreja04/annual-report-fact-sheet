"""OCR a representative sample. This does not process every page and does not call a model."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.config.paths import CLASSIFICATION_DIR, INPUT_DIR, LOG_DIR, OCR_DIR, PREPROCESS_CACHE_DIR
from src.ingestion.audit import locate_pdfs
from src.ocr.assess import (
    IMAGE_ONLY_VISUAL,
    character_error_rate,
    cleaned_text,
    is_document_like,
    kept_word_runs,
    recoverability,
)
from src.ocr.paddle_engine import run_paddle
from src.ocr.settings import MIN_CONTIGUOUS_WORDS, MIN_RUN_CONFIDENCE, WORD_CONFIDENCE_THRESHOLD
from src.ocr.tesseract_engine import illumination_std, psm_for_page, recognize_variant, run_tesseract
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")

# Cached Stage 3 halves. One readable scan, one photograph, one portrait page,
# Coal India captions, and Hindalco digital tables.
SAMPLE_HALVES = [
    ("POWERGRID", 100, "LEFT"),
    ("POWERGRID", 100, "RIGHT"),
    ("POWERGRID", 26, "LEFT"),
    ("POWERGRID", 25, "FULL"),
    ("COALINDIA", 82, "LEFT"),
    ("COALINDIA", 38, "LEFT"),
    ("COALINDIA", 314, "FULL"),
    ("HINDALCO", 180, "LEFT"),
    ("HINDALCO", 180, "RIGHT"),
    ("HINDALCO", 45, "LEFT"),
]


def confidence_threshold() -> float:
    raw = os.environ.get("OCR_WORD_CONFIDENCE", str(WORD_CONFIDENCE_THRESHOLD))
    return float(raw)


def load_classification(ticker: str, pdf_page: int) -> list[str]:
    path = CLASSIFICATION_DIR / f"{ticker}_pages.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["pdf_page"]) == pdf_page:
                return list(row["classifications"])
    return []


def select_variant(variants: list[dict], threshold: float) -> dict:
    """Prefer the binarization that yields the most reliable word runs."""
    scored = []
    for variant in variants:
        runs = kept_word_runs(variant["words"], threshold=threshold)
        average = (
            sum(run["line_confidence"] for run in runs) / len(runs) if runs else 0.0
        )
        scored.append((len(runs), average, variant, runs))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _count, _average, variant, runs = scored[0]
    return {"variant": variant, "runs": runs}


def threshold_order(classifications: list[str], document_like: bool) -> tuple[str, ...]:
    """Sauvola first. A caption page gets that one pass and nothing else."""
    if "IMAGE_ONLY_VISUAL" in classifications or not document_like:
        return ("sauvola",)
    return ("sauvola", "adaptive_gaussian", "otsu")


def paddle_needed(runs: list[dict], classifications: list[str], document_like: bool) -> bool:
    """Paddle is the fallback when a text page has no quote-grade Tesseract line."""
    if "IMAGE_ONLY_VISUAL" in classifications or not document_like:
        return False
    return not runs


def _skipped_paddle() -> dict:
    return {
        "available": False,
        "engine": "paddleocr",
        "error": None,
        "processing_method": "skipped",
        "raw_text": "",
        "lines": [],
    }


def _first_kept_variant(gray, classifications: list[str], document_like: bool, threshold: float) -> dict:
    """Stop once a threshold keeps a quote-grade line. Only that result is returned."""
    psm = psm_for_page(classifications, document_like)
    selected = None
    selected_runs: list[dict] = []
    otsu_used = False
    for name in threshold_order(classifications, document_like):
        variant = recognize_variant(gray, name, psm)
        if variant is None:
            continue
        otsu_used = otsu_used or name == "otsu"
        runs = kept_word_runs(variant["words"], threshold=threshold)
        selected = variant
        selected_runs = runs
        if runs:
            break
    return {
        "psm": psm,
        "variant": selected,
        "runs": selected_runs,
        "otsu_used": otsu_used,
        "illumination_std": round(illumination_std(gray), 2),
    }


def assess_half(
    gray,
    classifications: list[str],
    threshold: float,
    ticker: str,
    pdf_page: int,
    half: str,
) -> dict:
    document_like = is_document_like(gray)
    chosen = _first_kept_variant(gray, classifications, document_like, threshold)
    variant = chosen["variant"]
    runs = chosen["runs"]
    decision = recoverability(classifications, document_like, runs)
    paddle = run_paddle(gray) if paddle_needed(runs, classifications, document_like) else _skipped_paddle()
    evidence = cleaned_text(runs)
    method = variant["processing_method"] if variant else "none"
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
        "tesseract": {
            "engine": "tesseract",
            "psm": chosen["psm"],
            "illumination_std": chosen["illumination_std"],
            "otsu_used": chosen["otsu_used"],
            "selected_processing_method": method,
            "raw_variants": [] if variant is None else [variant],
            "cleaned_text": evidence,
            "kept_lines": runs,
        },
        "paddleocr": paddle,
        "primary_engine": "paddleocr" if paddle["available"] else "tesseract",
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
    }


def half_image_path(ticker: str, pdf_page: int, half: str) -> Path:
    return PREPROCESS_CACHE_DIR / ticker / f"p{pdf_page:04d}_{half}" / "full.png"


def digital_half_text(ticker: str, pdf_page: int, half: str) -> str:
    """Text-layer words on one half. This is a reference, not a manual transcript."""
    import pymupdf

    pdf_path = locate_pdfs(INPUT_DIR)[ticker]
    with pymupdf.open(pdf_path) as document:
        page = document[pdf_page - 1]
        midpoint = float(page.rect.width) / 2.0
        words = page.get_text("words")
    chosen = []
    for word in words:
        center = (float(word[0]) + float(word[2])) / 2.0
        if half == "LEFT" and center >= midpoint:
            continue
        if half == "RIGHT" and center < midpoint:
            continue
        chosen.append(word)
    chosen.sort(key=lambda word: (word[5], word[6], word[7]))
    return " ".join(str(word[4]) for word in chosen)


def load_manual_transcripts() -> dict:
    path = Path(__file__).with_name("manual_transcripts.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ocr_crop(record: dict, crop_box: list[int], threshold: float) -> dict:
    """OCR the same rectangle that was transcribed, so extra page text is not scored."""
    gray = cv2.imread(record["source_image"], cv2.IMREAD_GRAYSCALE)
    x, y, width, height = [int(value) for value in crop_box]
    crop = gray[y : y + height, x : x + width]
    psm = 6 if record["document_like"] else 11
    tesseract = run_tesseract(crop, psm)
    chosen = select_variant(tesseract["variants"], threshold)
    paddle = run_paddle(crop)
    return {
        "tesseract_text": cleaned_text(chosen["runs"]),
        "paddle_text": paddle.get("raw_text") or "",
    }


def cer_cases(records: list[dict]) -> list[dict]:
    """Compare OCR with a text layer or a short manual transcript. The threshold stays provisional."""
    manual = load_manual_transcripts()
    cases = []
    for record in records:
        key = f"{record['ticker']}:{record['pdf_page']}:{record['half']}"
        reference = None
        source = None
        tesseract_text = record["tesseract"]["cleaned_text"]
        paddle_text = record["paddleocr"].get("raw_text") or ""
        if key in manual:
            reference = manual[key]["text"]
            source = "manual_transcript"
            crop_box = manual[key].get("crop_box")
            if crop_box:
                cropped = _ocr_crop(record, crop_box, record["word_confidence_threshold"])
                tesseract_text = cropped["tesseract_text"]
                paddle_text = cropped["paddle_text"]
            cases.append(_cer_row(record, source, reference, tesseract_text, paddle_text))
        if record["ticker"] == "HINDALCO":
            reference = digital_half_text(record["ticker"], record["pdf_page"], record["half"])
            if reference:
                cases.append(
                    _cer_row(
                        record,
                        "digital_text_layer",
                        reference,
                        record["tesseract"]["cleaned_text"],
                        record["paddleocr"].get("raw_text") or "",
                    )
                )
    return cases


def _cer_row(record: dict, source: str, reference: str, tesseract_text: str, paddle_text: str) -> dict:
    return {
        "ticker": record["ticker"],
        "pdf_page": record["pdf_page"],
        "half": record["half"],
        "reference_source": source,
        "reference_chars": len(reference),
        "tesseract_cer": character_error_rate(reference, tesseract_text),
        "paddle_cer": character_error_rate(reference, paddle_text) if paddle_text else None,
        "recoverability": record["recoverability"],
    }


def apply_gate(record: dict, threshold: float) -> dict:
    """Recompute the quality gate from stored raw words. Raw OCR is left in place."""
    chosen = select_variant(record["tesseract"]["raw_variants"], threshold)
    decision = recoverability(record["classifications"], record["document_like"], chosen["runs"])
    record["word_confidence_threshold"] = threshold
    record["min_run_confidence"] = MIN_RUN_CONFIDENCE
    record["min_contiguous_words"] = MIN_CONTIGUOUS_WORDS
    record["threshold_finalized"] = False
    record["tesseract"]["selected_processing_method"] = chosen["variant"]["processing_method"]
    record["tesseract"]["cleaned_text"] = cleaned_text(chosen["runs"])
    record["tesseract"]["kept_lines"] = chosen["runs"]
    record["processing_method"] = chosen["variant"]["processing_method"]
    record["recoverability"] = decision["recoverability"]
    record["failure_reason"] = decision["failure_reason"]
    record["pass_to_business_extraction"] = decision["pass_to_extraction"]
    record["use_ocr_text"] = decision["use_ocr_text"]
    record["caption_text"] = decision["caption_text"]
    return record


def rescore_sample(output_dir: Path | None = None, threshold: float | None = None) -> dict:
    """Apply the current gate to OCR that has already been stored."""
    threshold = confidence_threshold() if threshold is None else threshold
    output_dir = output_dir or (OCR_DIR / "sample")
    records = []
    for path in sorted(output_dir.glob("*_p*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record = apply_gate(record, threshold)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    if len(records) != len(SAMPLE_HALVES):
        raise RuntimeError("A sample page is missing from the stored OCR records.")
    summary = _summary(records, threshold)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_sample(output_dir: Path | None = None, threshold: float | None = None) -> dict:
    threshold = confidence_threshold() if threshold is None else threshold
    output_dir = output_dir or (OCR_DIR / "sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ticker, pdf_page, half in SAMPLE_HALVES:
        image_path = half_image_path(ticker, pdf_page, half)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Missing preprocessed image: {image_path}")
        LOGGER.info("OCR %s pdf %s %s", ticker, pdf_page, half)
        record = assess_half(
            gray,
            load_classification(ticker, pdf_page),
            threshold,
            ticker,
            pdf_page,
            half,
        )
        record["source_image"] = image_path.as_posix()
        destination = output_dir / f"{ticker}_p{pdf_page:04d}_{half}.json"
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        LOGGER.info(
            "%s p%s %s -> %s via %s",
            ticker,
            pdf_page,
            half,
            record["recoverability"],
            record["processing_method"],
        )

    if len(records) != len(SAMPLE_HALVES):
        raise RuntimeError("A sample page was dropped before it received an OCR record.")

    summary = _summary(records, threshold)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _summary(records: list[dict], threshold: float) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(records),
        "requested_count": len(SAMPLE_HALVES),
        "word_confidence_threshold": threshold,
        "min_run_confidence": MIN_RUN_CONFIDENCE,
        "threshold_finalized": False,
        "threshold_note": (
            "A word must reach 60 and a kept line must average 80. "
            "Both numbers are pilot settings, not a final cutoff. "
            "POWERGRID page 100 was not transcribed: at native size the body glyphs are only about 8-12 pixels tall."
        ),
        "qwen_called": False,
        "llm_called": False,
        "dropped_page_count": 0,
        "status_counts": _count(records, "recoverability"),
        "pages": [
            {
                "ticker": record["ticker"],
                "pdf_page": record["pdf_page"],
                "half": record["half"],
                "recoverability": record["recoverability"],
                "failure_reason": record["failure_reason"],
                "pass_to_business_extraction": record["pass_to_business_extraction"],
                "use_ocr_text": record.get("use_ocr_text", bool(record["tesseract"]["kept_lines"])),
                "primary_engine": record["primary_engine"],
                "processing_method": record["processing_method"],
                "paddle_available": record["paddleocr"]["available"],
                "kept_line_count": len(record["tesseract"]["kept_lines"]),
                "caption_text": record["caption_text"],
            }
            for record in records
        ],
        "cer_pilot": cer_cases(records),
    }


def _count(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record[key]] = counts.get(record[key], 0) + 1
    counts.setdefault(IMAGE_ONLY_VISUAL, counts.get(IMAGE_ONLY_VISUAL, 0))
    return counts


def main() -> None:
    configure_logging(LOG_DIR / "ocr.log")
    summary = run_sample()
    print(json.dumps({
        "page_count": summary["page_count"],
        "status_counts": summary["status_counts"],
        "threshold_finalized": summary["threshold_finalized"],
        "cer_pilot": summary["cer_pilot"],
        "dropped_page_count": summary["dropped_page_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
