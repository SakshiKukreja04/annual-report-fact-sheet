"""Retry short OCR pages at higher DPI with alternate engine/PSM. No LLM/vision."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config.paths import OCR_DIR, PAGE_DIR, PREPROCESS_CACHE_DIR
from src.ocr.paddle_engine import run_paddle
from src.ocr.tesseract_engine import run_tesseract, tesseract_available
from src.preprocessing.pipeline import process_rendered_fast, render_embedded_native, render_page

MIN_CHARS = 80
TARGET_DPI = 300
PSMS = (6, 11)


def retry_short_ocr_pages(
    ticker: str,
    pdf_path: Path,
    pdf_pages: list[int],
    *,
    ocr_dir: Path | None = None,
    page_dir: Path | None = None,
    preprocess_dir: Path | None = None,
) -> list[dict]:
    ocr_dir = ocr_dir or (OCR_DIR / ticker)
    page_dir = page_dir or (PAGE_DIR / ticker)
    preprocess_dir = preprocess_dir or PREPROCESS_CACHE_DIR
    ocr_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf_page in pdf_pages:
        page_path = page_dir / f"p{pdf_page:04d}.json"
        if not page_path.is_file():
            results.append({"pdf_page": pdf_page, "status": "missing_page_record"})
            continue
        page = json.loads(page_path.read_text(encoding="utf-8"))
        selected_chars = _selected_chars(page)
        if selected_chars >= MIN_CHARS:
            results.append({"pdf_page": pdf_page, "status": "skip_already_ok", "chars": selected_chars})
            continue
        attempt = _retry_page(pdf_path, ticker, pdf_page, preprocess_dir, ocr_dir)
        results.append(attempt)
        if attempt.get("chars", 0) >= MIN_CHARS:
            _merge_into_page(page_path, page, attempt)
        else:
            page.setdefault("warnings", [])
            if "ocr_retry_failed" not in page["warnings"]:
                page["warnings"].append("ocr_retry_failed")
            qm = dict(page.get("quality_metadata") or {})
            qm["ocr_retry"] = {k: attempt.get(k) for k in ("status", "chars", "engine", "psm", "reason")}
            page["quality_metadata"] = qm
            if page.get("content_status") != "recoverable":
                page["content_status"] = "image_only_visual"
            page_path.write_text(json.dumps(page, indent=2), encoding="utf-8")
    return results


def _selected_chars(page: dict) -> int:
    total = 0
    for block in page.get("text_blocks") or []:
        if block.get("selected", True):
            total += len((block.get("text") or "").strip())
    return total


def _retry_page(
    pdf_path: Path,
    ticker: str,
    pdf_page: int,
    preprocess_dir: Path,
    ocr_dir: Path,
) -> dict:
    best: dict = {
        "pdf_page": pdf_page,
        "status": "skipped_short",
        "chars": 0,
        "engine": None,
        "psm": None,
        "attempts": [],
    }
    try:
        color, width_pt, height_pt, _metrics = render_embedded_native(pdf_path, pdf_page)
    except Exception:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            color = render_page(document[pdf_page - 1])
            rect = document[pdf_page - 1].rect
            width_pt, height_pt = float(rect.width), float(rect.height)
    records = process_rendered_fast(
        color,
        width_pt,
        height_pt,
        pdf_page,
        ticker,
        preprocess_dir,
        target_dpi=TARGET_DPI,
    )
    for record in records:
        half = record.get("half") or "FULL"
        image_path = Path(
            record.get("output_image_path")
            or record.get("full_path")
            or (preprocess_dir / ticker / f"p{pdf_page:04d}_{half}" / "full.png")
        )
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for psm in PSMS:
            if tesseract_available():
                tess = run_tesseract(gray, psm)
                texts = [(variant.get("raw_text") or "") for variant in tess.get("variants") or []]
                text = max(texts, key=len) if texts else ""
                best["attempts"].append({"engine": "tesseract", "psm": psm, "half": half, "chars": len(text)})
                if len(text) > best["chars"]:
                    best.update(
                        chars=len(text),
                        engine="tesseract",
                        psm=psm,
                        text=text,
                        half=half,
                        status="recovered" if len(text) >= MIN_CHARS else "skipped_short",
                    )
            paddle = run_paddle(gray)
            ptext = (paddle.get("raw_text") or paddle.get("cleaned_text") or "").strip()
            best["attempts"].append({"engine": "paddleocr", "psm": None, "half": half, "chars": len(ptext)})
            if len(ptext) > best["chars"]:
                best.update(
                    chars=len(ptext),
                    engine="paddleocr",
                    psm=None,
                    text=ptext,
                    half=half,
                    status="recovered" if len(ptext) >= MIN_CHARS else "skipped_short",
                )
        if best.get("text"):
            payload = {
                "ticker": ticker,
                "pdf_page": pdf_page,
                "half": half,
                "retry": True,
                "target_dpi": TARGET_DPI,
                "tesseract": {
                    "cleaned_text": best["text"] if best.get("engine") == "tesseract" else "",
                    "psm": best.get("psm"),
                },
                "paddleocr": {"raw_text": best["text"] if best.get("engine") == "paddleocr" else ""},
                "use_ocr_text": best["chars"] >= MIN_CHARS,
                "recoverability": "RECOVERABLE" if best["chars"] >= MIN_CHARS else "IMAGE_ONLY_VISUAL",
            }
            dest = ocr_dir / f"{ticker}_p{pdf_page:04d}_{half}_retry300.json"
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best["chars"] < MIN_CHARS:
        best["status"] = "skipped_short"
        best["reason"] = f"ocr_chars<{MIN_CHARS}_after_retry"
    return best


def _merge_into_page(page_path: Path, page: dict, attempt: dict) -> None:
    text = attempt.get("text") or ""
    if not text:
        return
    block = {
        "text": text,
        "bbox": None,
        "bbox_space": "pdf_points",
        "source": attempt.get("engine") or "tesseract",
        "ocr_confidence": None,
        "half": attempt.get("half") or "FULL",
        "reading_order": 0,
        "selected": True,
        "pdf_page": attempt.get("pdf_page"),
    }
    existing = [b for b in (page.get("text_blocks") or []) if (b.get("text") or "").strip()]
    if _selected_chars(page) < MIN_CHARS:
        for item in existing:
            item["selected"] = False
        page["text_blocks"] = existing + [block]
    else:
        page.setdefault("text_blocks", []).append(block)
    page["content_status"] = "recoverable"
    qm = dict(page.get("quality_metadata") or {})
    qm["selected_text_source"] = "ocr"
    qm["ocr_retry"] = {k: attempt.get(k) for k in ("status", "chars", "engine", "psm", "reason")}
    page["quality_metadata"] = qm
    page_path.write_text(json.dumps(page, indent=2), encoding="utf-8")
