"""Stage 1: inventory the FY25 annual reports and audit every page.

This stage records page geometry and preliminary text statistics.
It does not run OCR, call an API, or judge whether a text layer is garbled.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.config.paths import AUDIT_DIR, INPUT_DIR, LOG_DIR, PROJECT_ROOT
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")

# Stripped text shorter than this is empty or nearly empty.
# The cutoff is a size check only. It does not measure text quality.
NEARLY_EMPTY_TEXT_CHARS = 50

EXPECTED_PAGE_COUNTS = {
    "POWERGRID": 364,
    "HINDALCO": 413,
    "COALINDIA": 371,
}


def locate_pdfs(input_dir: Path | None = None) -> dict[str, Path]:
    """Return the single PDF inside each company input folder."""
    root = input_dir or INPUT_DIR
    found: dict[str, Path] = {}
    for ticker in EXPECTED_PAGE_COUNTS:
        folder = root / ticker
        pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
        if len(pdfs) != 1:
            names = ", ".join(path.name for path in pdfs) or "(none)"
            raise FileNotFoundError(
                f"{ticker} needs exactly one PDF in {folder}. Found: {names}"
            )
        found[ticker] = pdfs[0]
    return found


def page_orientation(width_pt: float, height_pt: float) -> str:
    """Classify the displayed page. A two-page spread is landscape."""
    if abs(width_pt - height_pt) < 0.5:
        return "square"
    if width_pt > height_pt:
        return "landscape"
    return "portrait"


def stripped_text_length(text: str) -> int:
    return len(text.strip())


def usable_text_present(stripped_chars: int, threshold: int = NEARLY_EMPTY_TEXT_CHARS) -> bool:
    """Preliminary flag: the page has more than a few characters of text.

    A true value does not mean the text is reliable. Garbled text can be long.
    """
    return stripped_chars >= threshold


def is_empty_or_nearly_empty(stripped_chars: int, threshold: int = NEARLY_EMPTY_TEXT_CHARS) -> bool:
    return stripped_chars < threshold


def nearly_empty_percent(stripped_lengths: Iterable[int], threshold: int = NEARLY_EMPTY_TEXT_CHARS) -> float:
    lengths = list(stripped_lengths)
    if not lengths:
        return 0.0
    empty = sum(1 for count in lengths if count < threshold)
    return round(100.0 * empty / len(lengths), 2)


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _images_shown(page: pymupdf.Page) -> int:
    """Count images painted on this page.

    page.get_images() follows the page resource dictionary. In these files
    that dictionary is shared, so it reports every image in the PDF on each
    scanned page. get_image_info() lists images actually placed on the page.
    """
    return len(page.get_image_info())


def audit_page(page: pymupdf.Page, ticker: str) -> dict:
    """Collect geometry and preliminary text stats for one 1-based page."""
    text = page.get_text("text") or ""
    raw_chars = len(text)
    stripped_chars = stripped_text_length(text)
    width_pt = round(float(page.rect.width), 2)
    height_pt = round(float(page.rect.height), 2)
    return {
        "ticker": ticker,
        "pdf_page": page.number + 1,
        "width_pt": width_pt,
        "height_pt": height_pt,
        "rotation": int(page.rotation),
        "text_char_count": raw_chars,
        "text_stripped_char_count": stripped_chars,
        "image_count": _images_shown(page),
        "usable_text_present": usable_text_present(stripped_chars),
        "orientation": page_orientation(width_pt, height_pt),
    }


def display_path(path: Path) -> str:
    """Prefer a project-relative path. Fall back to absolute for temp test dirs."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def audit_document(pdf_path: Path, ticker: str) -> tuple[list[dict], dict]:
    """Inspect every page. The PDF is opened read-only and is not rewritten."""
    import pymupdf

    rows: list[dict] = []
    with pymupdf.open(pdf_path) as document:
        metadata = dict(document.metadata or {})
        metadata["page_count"] = document.page_count
        LOGGER.info("%s opened %s pages=%s", ticker, pdf_path.name, document.page_count)
        for index, page in enumerate(document):
            rows.append(audit_page(page, ticker))
            if (index + 1) % 100 == 0:
                LOGGER.info("%s audited %s/%s pages", ticker, index + 1, document.page_count)
    return rows, metadata


def summarize_company(ticker: str, pdf_path: Path, pages: list[dict], metadata: dict) -> dict:
    expected = EXPECTED_PAGE_COUNTS[ticker]
    stripped = [row["text_stripped_char_count"] for row in pages]
    empty = sum(1 for count in stripped if count == 0)
    nearly = sum(1 for count in stripped if 0 < count < NEARLY_EMPTY_TEXT_CHARS)
    orientations = {"landscape": 0, "portrait": 0, "square": 0}
    for row in pages:
        orientations[row["orientation"]] += 1
    relative = display_path(pdf_path)
    page_count = len(pages)
    return {
        "ticker": ticker,
        "filename": pdf_path.name,
        "path": relative,
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count": page_count,
        "expected_page_count": expected,
        "page_count_matches_expected": page_count == expected,
        "metadata": metadata,
        "empty_text_pages": empty,
        "nearly_empty_text_pages": nearly,
        "empty_or_nearly_empty_pages": empty + nearly,
        "empty_or_nearly_empty_pct": nearly_empty_percent(stripped),
        "nearly_empty_threshold_chars": NEARLY_EMPTY_TEXT_CHARS,
        "usable_text_pages_preliminary": sum(1 for row in pages if row["usable_text_present"]),
        "landscape_pages": orientations["landscape"],
        "portrait_pages": orientations["portrait"],
        "square_pages": orientations["square"],
        "median_text_char_count": median([row["text_char_count"] for row in pages]),
        "pages_with_images": sum(1 for row in pages if row["image_count"] > 0),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, companies: list[dict]) -> dict:
    import pymupdf

    summary = {
        "stage": "pdf_audit",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": platform.python_version(),
        "pymupdf": pymupdf.__version__,
        "nearly_empty_threshold_chars": NEARLY_EMPTY_TEXT_CHARS,
        "notes": [
            "pdf_page is 1-based and matches the PDF viewer page number.",
            "width_pt and height_pt come from the displayed page rectangle, after rotation.",
            "image_count is the number of images painted on the page.",
            "usable_text_present means the stripped text has at least "
            f"{NEARLY_EMPTY_TEXT_CHARS} characters. It is not a quality judgment. "
            "Garbled text is not classified in this stage.",
            "Original PDFs were opened read-only and were not modified.",
        ],
        "companies": companies,
        "all_page_counts_match_expected": all(
            company["page_count_matches_expected"] for company in companies
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def run_audit(input_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    pdfs = locate_pdfs(input_dir)
    destination = output_dir or AUDIT_DIR
    companies = []
    for ticker, pdf_path in pdfs.items():
        pages, metadata = audit_document(pdf_path, ticker)
        page_path = destination / f"{ticker}_pages.jsonl"
        write_jsonl(page_path, pages)
        company = summarize_company(ticker, pdf_path, pages, metadata)
        company["page_audit_path"] = display_path(page_path)
        companies.append(company)
        LOGGER.info(
            "%s pages=%s expected=%s match=%s empty_or_nearly_empty=%s (%.2f%%)",
            ticker,
            company["page_count"],
            company["expected_page_count"],
            company["page_count_matches_expected"],
            company["empty_or_nearly_empty_pages"],
            company["empty_or_nearly_empty_pct"],
        )
    summary = write_summary(destination / "summary.json", companies)
    LOGGER.info("Wrote %s", destination / "summary.json")
    return summary


def main() -> int:
    configure_logging(LOG_DIR / "audit.log")
    summary = run_audit()
    return 0 if summary["all_page_counts_match_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
