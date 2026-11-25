"""Extract representative tables. This does not compute financial shares."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pymupdf

from src.config.paths import INPUT_DIR, LOG_DIR, PREPROCESS_CACHE_DIR, TABLE_DIR
from src.ingestion.audit import locate_pdfs
from src.tables.digital import camelot_status, extract_digital_page
from src.tables.grid import link_continuations
from src.tables.provenance import printed_page
from src.tables.scanned import extract_scanned_image
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")

# Hindalco table pages, a Coal India run of adjacent table pages, and one scanned table.
SAMPLE_PAGES = [
    ("HINDALCO", 45),
    ("HINDALCO", 180),
    ("HINDALCO", 384),
    ("HINDALCO", 385),
    ("COALINDIA", 74),
    ("COALINDIA", 75),
    ("COALINDIA", 76),
]
SCANNED_HALVES = [
    ("POWERGRID", 200, "LEFT"),
    ("POWERGRID", 200, "RIGHT"),
]


def run_sample(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or (TABLE_DIR / "sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = locate_pdfs(INPUT_DIR)
    tables = []
    for ticker, pdf_page in SAMPLE_PAGES:
        LOGGER.info("Digital tables %s pdf %s", ticker, pdf_page)
        tables.extend(extract_digital_page(pdfs[ticker], ticker, pdf_page))
    scanned_status = []
    for ticker, pdf_page, half in SCANNED_HALVES:
        image_path = PREPROCESS_CACHE_DIR / ticker / f"p{pdf_page:04d}_{half}" / "full.png"
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            scanned_status.append({"ticker": ticker, "pdf_page": pdf_page, "half": half, "error": "missing image"})
            continue
        printed = _printed_from_pdf(pdfs[ticker], pdf_page)
        LOGGER.info("Scanned tables %s pdf %s %s", ticker, pdf_page, half)
        result = extract_scanned_image(gray, ticker, pdf_page, printed, half)
        scanned_status.append(
            {
                "ticker": ticker,
                "pdf_page": pdf_page,
                "half": half,
                "available": result["available"],
                "error": result["error"],
                "table_count": len(result["tables"]),
            }
        )
        tables.extend(result["tables"])
    link_continuations(tables)
    for table in tables:
        destination = output_dir / f"{table['table_id']}.json"
        destination.write_text(json.dumps(table, indent=2), encoding="utf-8")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "table_count": len(tables),
        "dropped_page_count": 0,
        "financial_shares_calculated": False,
        "groq_called": False,
        "camelot": camelot_status(),
        "scanned": scanned_status,
        "continuations": [
            {
                "table_id": table["table_id"],
                "pdf_page": table["pdf_page"],
                "continuation_from": table["continuation_from"],
                "continuation_to": table["continuation_to"],
                "headers": table["headers"][:8],
            }
            for table in tables
            if table["continuation_from"] or table["continuation_to"]
        ],
        "tables": [
            {
                "table_id": table["table_id"],
                "ticker": table["ticker"],
                "pdf_page": table["pdf_page"],
                "printed_page": table["printed_page"],
                "extraction_method": table["extraction_method"],
                "confidence": table["confidence"],
                "headers": table["headers"][:8],
                "row_count": len(table["rows"]),
                "column_count": len(table["headers"]) or (len(table["rows"][0]["cells"]) if table["rows"] else 0),
                "units": table["units"],
                "footnote_count": len(table["footnotes"]),
                "continuation_from": table["continuation_from"],
                "continuation_to": table["continuation_to"],
            }
            for table in tables
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _printed_from_pdf(pdf_path: Path, pdf_page: int) -> str | None:
    with pymupdf.open(pdf_path) as document:
        page = document[pdf_page - 1]
        words = [
            {"text": word[4], "x0": word[0], "top": word[1]}
            for word in page.get_text("words")
        ]
        return printed_page(words, float(page.rect.height))


def main() -> None:
    configure_logging(LOG_DIR / "tables.log")
    summary = run_sample()
    print(json.dumps({
        "table_count": summary["table_count"],
        "camelot": summary["camelot"],
        "scanned": summary["scanned"],
        "continuations": summary["continuations"],
        "tables": summary["tables"],
        "financial_shares_calculated": summary["financial_shares_calculated"],
        "groq_called": summary["groq_called"],
    }, indent=2))


if __name__ == "__main__":
    main()
