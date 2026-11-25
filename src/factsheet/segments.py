"""Read a segment-revenue table from page records and digital tables.

The unit and the total come from the document. Share is segment revenue
divided by the printed total. Nothing is filled in when the table is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config.factsheet_keywords import (
    ELIMINATION_LABEL,
    EXTERNAL_HEADER,
    INTER_HEADER,
    NO_SEPARATE_SEGMENTS,
    REVENUE_HEADER,
    SEGMENT_PAGE,
    TOTAL_LABEL,
    UNIT_PHRASE,
)
from src.config.paths import PAGE_DIR, TABLE_DIR
from src.extract.select import page_text
from src.tables.grid import clean_cell

_NUMBER = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?")


def locate_segment_tables(ticker: str, pages: list[dict] | None = None, pdf_path: Path | None = None) -> dict:
    """Return segments plus the provenance report. segments is empty when no table is usable."""
    pages = pages if pages is not None else _load_pages(ticker)
    hits = []
    for page in pages:
        text = page_text(page)
        if text and SEGMENT_PAGE.search(text):
            hits.append(page)
    tables = []
    errors = []
    for page in hits:
        number = int(page["pdf_page"])
        tables.extend(_saved_tables(ticker, number))
        if pdf_path is not None and not any(int(table.get("pdf_page") or 0) == number for table in tables):
            try:
                from src.tables.digital import extract_digital_page

                tables.extend(extract_digital_page(pdf_path, ticker, number))
            except Exception as exc:
                errors.append({"pdf_page": number, "error": f"{type(exc).__name__}: {exc}"})
    parsed = []
    no_separate = []
    for page in hits:
        if NO_SEPARATE_SEGMENTS.search(page_text(page)):
            no_separate.append(int(page["pdf_page"]))
    for table in tables:
        page = _page_for(pages, int(table.get("pdf_page") or 0))
        row = _parse_table(table, page)
        if row and row["segments"]:
            parsed.append(row)
    if not parsed:
        if no_separate:
            reason = "document_states_no_separate_reportable_segments"
        elif not hits:
            reason = "no_page_matched_segment_keywords"
        elif errors and not tables:
            reason = "table_extraction_failed"
        else:
            reason = "no_segment_revenue_table"
        return {
            "segments": [],
            "table_found": False,
            "reason": reason,
            "pages_checked": [int(page["pdf_page"]) for page in hits],
            "no_separate_pages": no_separate,
            "errors": errors,
        }
    chosen = max(parsed, key=lambda item: (item["has_total"], len(item["segments"])))
    return {
        "segments": chosen["segments"],
        "table_found": True,
        "reason": "",
        "pdf_page": chosen["pdf_page"],
        "printed_page": chosen["printed_page"],
        "unit": chosen["unit"],
        "denominator": chosen["denominator"],
        "printed_total": chosen["printed_total"],
        "pages_checked": [int(page["pdf_page"]) for page in hits],
        "errors": errors,
    }


def write_segment_report(ticker: str, report: dict, staging_root: Path) -> Path:
    path = staging_root / ticker / "segments_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _parse_table(table: dict, page: dict | None) -> dict | None:
    headers = [clean_cell(cell) for cell in (table.get("headers") or [])]
    header_blob = " ".join(headers)
    value_idx, basis = _value_column(headers)
    # A revenue column has to be labelled. A bare two-column grid is not a segment table.
    if value_idx is None or not (REVENUE_HEADER.search(header_blob) or SEGMENT_PAGE.search(header_blob)):
        return None
    unit = _unit_from_table(table) or _unit_from_page(page)
    segments = []
    printed_total = None
    for row in table.get("rows") or []:
        cells = [clean_cell(cell) for cell in (row.get("cells") or [])]
        if not cells or not cells[0]:
            continue
        label = cells[0]
        number = _amount(cells, value_idx)
        if TOTAL_LABEL.search(label):
            printed_total = number
            continue
        if ELIMINATION_LABEL.search(label) or _is_header_label(label):
            continue
        if number is None or not _short_label(label):
            continue
        segments.append({"name": label, "revenue": number})
    if len(segments) < 2 or printed_total is None or printed_total == 0:
        return None
    shares = []
    for item in segments:
        share = None
        if printed_total and printed_total > 0 and item["revenue"] is not None:
            share = round(100.0 * float(item["revenue"]) / float(printed_total), 2)
        shares.append({
            "name": item["name"],
            "revenue": item["revenue"],
            "unit": unit,
            "share_pct": share,
        })
    printed = table.get("printed_page")
    return {
        "segments": shares,
        "pdf_page": int(table.get("pdf_page") or 0),
        "printed_page": None if printed in (None, "") else str(printed),
        "unit": unit,
        "denominator": basis if printed_total else "no_printed_total",
        "printed_total": printed_total,
        "has_total": printed_total is not None,
    }


def _value_column(headers: list[str]) -> tuple[int | None, str]:
    revenue_idx = None
    external_idx = None
    inter_idx = None
    for index, header in enumerate(headers):
        if INTER_HEADER.search(header):
            inter_idx = index
        elif EXTERNAL_HEADER.search(header) and REVENUE_HEADER.search(header):
            external_idx = index
        elif REVENUE_HEADER.search(header) and revenue_idx is None:
            revenue_idx = index
    if revenue_idx is not None:
        basis = "before_inter_segment_elimination" if inter_idx is not None else "printed_total"
        return revenue_idx, basis
    if external_idx is not None:
        return external_idx, "after_inter_segment_elimination"
    return None, "printed_total"


def _amount(cells: list[str], index: int) -> float | None:
    if index >= len(cells):
        return None
    text = cells[index].replace(",", "")
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = match.group(0)
    negative = raw.startswith("(") or raw.startswith("-")
    digits = re.sub(r"[()]", "", raw)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -abs(value) if negative else value


def _unit_from_table(table: dict) -> str | None:
    units = table.get("units")
    if isinstance(units, str):
        match = UNIT_PHRASE.search(units)
        if match:
            return match.group(0)
    return None


def _unit_from_page(page: dict | None) -> str | None:
    if not page:
        return None
    match = UNIT_PHRASE.search(page_text(page))
    return match.group(0) if match else None


def _is_header_label(label: str) -> bool:
    folded = label.casefold()
    return folded in {"particulars", "segment", "description", "year ended", "for the year ended"}


def _short_label(label: str) -> bool:
    text = " ".join(label.split())
    if len(text) < 2 or len(text) > 60 or text[0].isdigit():
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _page_for(pages: list[dict], number: int) -> dict | None:
    for page in pages:
        if int(page.get("pdf_page") or 0) == number:
            return page
    return None


def _saved_tables(ticker: str, pdf_page: int) -> list[dict]:
    folder = TABLE_DIR / ticker
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob(f"{ticker}-p{pdf_page:04d}-t*.json")):
        found.append(json.loads(path.read_text(encoding="utf-8")))
    return found


def _load_pages(ticker: str) -> list[dict]:
    folder = PAGE_DIR / ticker
    pages = []
    if not folder.is_dir():
        return pages
    for path in sorted(folder.glob("p*.json")):
        pages.append(json.loads(path.read_text(encoding="utf-8")))
    return pages
