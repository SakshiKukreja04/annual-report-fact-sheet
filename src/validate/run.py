"""Validate saved candidates and write the reports. Nothing unsupported is marked validated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, INPUT_DIR, LOG_DIR, OCR_DIR, PAGE_DIR, PREPROCESS_CACHE_DIR
from src.utils.logging_config import configure_logging
from src.validate.items import validate_candidate
from src.validate.quotes import pdf_page_count, read_pdf_page


def validate_many(candidates: list[dict], context: dict) -> dict:
    results = [validate_candidate(candidate, context) for candidate in candidates]
    grouped = {"validated": [], "rejected": [], "review": []}
    for result in results:
        grouped[result["outcome"]].append(result)
    return {
        "ticker": context.get("ticker"),
        "validated": grouped["validated"],
        "rejected": grouped["rejected"],
        "review": grouped["review"],
        "quote_report": [quote for result in results for quote in result["quotes"]],
        "financial_report": [
            {"candidate_id": result["coverage"].get("candidate_id"), "name": result["coverage"].get("name"), **result["financial"]}
            for result in results
        ],
        "coverage": [result["coverage"] for result in results],
        "validated_count": len(grouped["validated"]),
        "rejected_count": len(grouped["rejected"]),
        "review_count": len(grouped["review"]),
    }


def build_context(ticker: str, pages: list[int], pdf_path: Path | None = None) -> dict:
    pdf_path = pdf_path or _pdf_path(ticker)
    count, reader = pdf_page_count(pdf_path)
    native = {}
    for pdf_page in pages:
        if 1 <= pdf_page <= count:
            native[pdf_page], reader = read_pdf_page(pdf_path, pdf_page)
    ocr_text, ocr_only, images = _ocr_context(ticker, pages)
    return {
        "ticker": ticker,
        "page_count": count,
        "native_text": native,
        "ocr_text": ocr_text,
        "ocr_only_pages": ocr_only,
        "page_images": images,
        "pdf_reader": reader,
        "pdf_path": str(pdf_path),
    }


def load_candidates(ticker: str, folder: Path | None = None) -> list[dict]:
    folder = folder or (CANDIDATE_DIR / ticker)
    resolution = folder / "dedup" / "resolution.json"
    if resolution.is_file():
        record = json.loads(resolution.read_text(encoding="utf-8"))
        return list(record.get("merged") or []) + list(record.get("unchanged") or [])
    items = []
    parsed = folder / "parsed"
    if not parsed.is_dir():
        return items
    for path in sorted(parsed.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for index, candidate in enumerate(record.get("candidates") or []):
            items.append({**candidate, "candidate_id": f"{path.stem}:{index}"})
    return items


def write_reports(ticker: str, report: dict, output_dir: Path | None = None) -> Path:
    destination = output_dir or (CANDIDATE_DIR / ticker / "validation")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("validated", "rejected", "review", "quote_report", "financial_report", "coverage"):
        (destination / f"{name}.json").write_text(json.dumps(report[name], indent=2), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps({
        "ticker": ticker,
        "validated_count": report["validated_count"],
        "rejected_count": report["rejected_count"],
        "review_count": report["review_count"],
        "pdf_reader": report.get("pdf_reader"),
    }, indent=2), encoding="utf-8")
    return destination


def _pdf_path(ticker: str) -> Path:
    folder = INPUT_DIR / ticker
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    if len(pdfs) != 1:
        raise FileNotFoundError(f"{ticker} needs one PDF in {folder}")
    return pdfs[0]


def _ocr_context(ticker: str, pages: list[int]) -> tuple[dict, set[int], dict]:
    text = {}
    ocr_only = set()
    images = {}
    for pdf_page in pages:
        page_path = PAGE_DIR / ticker / f"p{pdf_page:04d}.json"
        parts = []
        if page_path.is_file():
            page = json.loads(page_path.read_text(encoding="utf-8"))
            labels = set(page.get("page_type") or [])
            selected = (page.get("quality_metadata") or {}).get("selected_text_source")
            if "SCANNED_IMAGE" in labels and "DIGITAL_TEXT" not in labels or selected == "ocr":
                ocr_only.add(pdf_page)
            for block in page.get("text_blocks") or []:
                if block.get("source") in {"tesseract", "paddleocr"} and block.get("text"):
                    parts.append(block["text"])
            for caption in page.get("captions") or []:
                if caption.get("text"):
                    parts.append(caption["text"])
        for path in (OCR_DIR / "sample").glob(f"{ticker}_p{pdf_page:04d}_*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            cleaned = ((record.get("tesseract") or {}).get("cleaned_text") or "").strip()
            if cleaned:
                parts.append(cleaned)
            if record.get("recoverability") == "RECOVERABLE" and record.get("use_ocr_text"):
                ocr_only.add(pdf_page)
            half = record.get("half") or "FULL"
            image = PREPROCESS_CACHE_DIR / ticker / f"p{pdf_page:04d}_{half}" / "full.png"
            if image.is_file():
                images[pdf_page] = str(image)
        if parts:
            text[pdf_page] = "\n".join(parts)
    return text, ocr_only, images


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidates against the original PDF.")
    parser.add_argument("ticker")
    args = parser.parse_args()
    configure_logging(LOG_DIR / "validate.log")
    ticker = args.ticker.upper()
    candidates = load_candidates(ticker)
    pages = sorted({int(source["pdf_page"]) for item in candidates for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)})
    context = build_context(ticker, pages)
    report = validate_many(candidates, context)
    report["pdf_reader"] = context["pdf_reader"]
    destination = write_reports(ticker, report)
    print(json.dumps({
        "ticker": ticker,
        "pdf_reader": context["pdf_reader"],
        "validated_count": report["validated_count"],
        "rejected_count": report["rejected_count"],
        "review_count": report["review_count"],
        "report": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
