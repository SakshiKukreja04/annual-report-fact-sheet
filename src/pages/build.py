"""Build and cache one JSON record per PDF page. Existing files are reused."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config.paths import CLASSIFICATION_DIR, INPUT_DIR, LOG_DIR, OCR_DIR, PAGE_DIR, TABLE_DIR
from src.ingestion.audit import locate_pdfs
from src.pages.assemble import REQUIRED_KEYS, assemble_page
from src.ocr_mistral import mistral_blocks_for_page
from src.pages.extract import (
    digital_content,
    load_jsonl_by_page,
    load_ocr_records,
    load_tables,
    ocr_blocks_from_record,
    ocr_printed_folio,
)
from src.tables.provenance import spread_folio_inference
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")


def page_cache_path(output_dir: Path, ticker: str, pdf_page: int) -> Path:
    return output_dir / ticker / f"p{pdf_page:04d}.json"


def write_page(path: Path, page: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in page]
    if missing:
        raise ValueError(f"Page record is missing {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page, indent=2), encoding="utf-8")


def read_cached_page(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_company(
    ticker: str,
    pages: list[int] | None = None,
    resume: bool = True,
    output_dir: Path | None = None,
    pdf_path: Path | None = None,
    classification_path: Path | None = None,
    ocr_dir: Path | None = None,
    table_dir: Path | None = None,
) -> dict:
    """Write a page file for each requested PDF page. Cached pages are left as they are."""
    import pymupdf

    output_dir = output_dir or PAGE_DIR
    pdf_path = pdf_path or locate_pdfs(INPUT_DIR)[ticker]
    classification = load_jsonl_by_page(classification_path or (CLASSIFICATION_DIR / f"{ticker}_pages.jsonl"))
    ocr = load_ocr_records(ocr_dir or (OCR_DIR / "sample"), ticker)
    for pdf_page, records in load_ocr_records(OCR_DIR / ticker, ticker).items():
        ocr.setdefault(pdf_page, []).extend(records)
    tables = load_tables(table_dir or TABLE_DIR, ticker)
    written = 0
    reused = 0
    with pymupdf.open(pdf_path) as document:
        wanted = pages or list(range(1, document.page_count + 1))
        for pdf_page in wanted:
            path = page_cache_path(output_dir, ticker, pdf_page)
            if resume and path.is_file():
                reused += 1
                continue
            if pdf_page < 1 or pdf_page > document.page_count:
                raise ValueError(f"{ticker} has no PDF page {pdf_page}")
            page = document[pdf_page - 1]
            record = build_one_page(ticker, page, classification.get(pdf_page), ocr.get(pdf_page, []), tables.get(pdf_page, []))
            write_page(path, record)
            written += 1
            if written and written % 100 == 0:
                LOGGER.info("%s cached %s pages", ticker, written)
    folio = apply_spread_folio_inference(output_dir, ticker)
    index_path = write_index(output_dir, ticker)
    return {"ticker": ticker, "written": written, "reused": reused, "index": str(index_path), "folio": folio}


def build_one_page(ticker: str, page: pymupdf.Page, classification: dict | None, ocr_records: list[dict], tables: list[dict]) -> dict:
    pdf_page = page.number + 1
    digital = digital_content(page)
    labels = list((classification or {}).get("classifications") or [])
    indicators = (classification or {}).get("indicators") or {}
    ocr_blocks = []
    captions = []
    recoverability = {}
    for record in ocr_records:
        blocks, caption_rows, status = ocr_blocks_from_record(record)
        ocr_blocks.extend(blocks)
        captions.extend(caption_rows)
        if status:
            recoverability[str(record.get("half") or "FULL")] = status
    mistral_blocks = mistral_blocks_for_page(ticker, pdf_page)
    if mistral_blocks:
        ocr_blocks.extend(mistral_blocks)
        # Treat Mistral text as recoverable so pages can select it without a Tesseract half status.
        recoverability.setdefault("FULL", "RECOVERABLE")
    printed = digital["printed_page"]
    folio_meta: dict = {}
    if printed not in (None, ""):
        folio_meta = {"folio_source": "digital"}
    else:
        detected = ocr_printed_folio(ocr_records, spread="TWO_PAGE_SPREAD" in labels)
        if detected and detected.get("printed_page"):
            printed = detected["printed_page"]
            folio_meta = {
                "folio_source": "ocr",
                "folio_confidence": detected.get("confidence"),
                "folio_half": detected.get("half"),
            }
        else:
            for table in tables:
                if table.get("printed_page"):
                    printed = table["printed_page"]
                    folio_meta = {"folio_source": "table"}
                    break
    page = assemble_page(
        ticker=ticker,
        pdf_page=pdf_page,
        printed_page=printed,
        page_type=labels,
        digital_blocks=digital["text_blocks"],
        ocr_blocks=ocr_blocks,
        captions=captions,
        images=digital["images"],
        tables=tables,
        ocr_recoverability=recoverability,
        garbled=bool(indicators.get("garbled_or_low_quality")),
        classification_confidence=(classification or {}).get("confidence"),
        recommended_method=(classification or {}).get("recommended_extraction_method"),
    )
    page["quality_metadata"].update(folio_meta)
    return page


def apply_spread_folio_inference(output_dir: Path, ticker: str) -> dict:
    """Fill spread folios from one offset when five confident OCR detections agree."""
    folder = output_dir / ticker
    paths = sorted(folder.glob("p*.json")) if folder.is_dir() else []
    loaded = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        loaded.append((path, record))
    pending = any(
        record.get("printed_page") in (None, "") and "TWO_PAGE_SPREAD" in (record.get("page_type") or [])
        for _path, record in loaded
    )
    if not pending:
        return {"filled": 0, "offset": None, "anchors": []}
    result = spread_folio_inference([record for _path, record in loaded])
    filled = result.get("filled") or {}
    for path, record in loaded:
        printed = filled.get(int(record["pdf_page"]))
        if not printed:
            continue
        record["printed_page"] = printed
        meta = record.setdefault("quality_metadata", {})
        meta["folio_source"] = "spread_offset"
        meta["folio_offset"] = result.get("offset")
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    anchors = [
        {
            "pdf_page": anchor["pdf_page"],
            "printed_page": anchor["printed_page"],
            "confidence": anchor["confidence"],
            "offset": anchor["offset"],
        }
        for anchor in result.get("anchors") or []
    ]
    return {"filled": len(filled), "offset": result.get("offset"), "anchors": anchors}


def write_index(output_dir: Path, ticker: str) -> Path:
    folder = output_dir / ticker
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / f"{ticker}_pages.jsonl"
    pages = sorted(folder.glob("p*.json"))
    with index.open("w", encoding="utf-8") as handle:
        for path in pages:
            record = json.loads(path.read_text(encoding="utf-8"))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return index


def load_company(ticker: str, output_dir: Path | None = None) -> dict:
    """Return usable pages. Illegible pages stay out of candidate extraction."""
    folder = (output_dir or PAGE_DIR) / ticker
    if not folder.is_dir():
        raise FileNotFoundError(f"No page cache for {ticker}. Run python -m src.pages.build {ticker} first.")
    candidate = []
    image_only = []
    excluded = []
    for path in sorted(folder.glob("p*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        status = page["content_status"]
        if status == "recoverable":
            candidate.append(page)
        elif status == "image_only_visual":
            image_only.append(page)
        else:
            excluded.append(
                {
                    "pdf_page": page["pdf_page"],
                    "content_status": status,
                    "warnings": page["warnings"],
                }
            )
    return {
        "ticker": ticker,
        "candidate_pages": candidate,
        "image_only_pages": image_only,
        "excluded_pages": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached unified page records.")
    parser.add_argument("ticker")
    parser.add_argument("--pages", default="", help="Comma-separated 1-based PDF pages. Default: all pages.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the page cache and write every page again.")
    args = parser.parse_args()
    configure_logging(LOG_DIR / "pages.log")
    selected = [int(part) for part in args.pages.split(",") if part.strip()] or None
    summary = build_company(args.ticker.upper(), pages=selected, resume=not args.rebuild)
    loaded = load_company(args.ticker.upper())
    print(json.dumps({
        **summary,
        "candidate_pages": len(loaded["candidate_pages"]),
        "image_only_pages": len(loaded["image_only_pages"]),
        "excluded_pages": len(loaded["excluded_pages"]),
    }, indent=2))


if __name__ == "__main__":
    main()
