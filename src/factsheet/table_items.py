"""Turn subsidiary, AOC-1, JV, and associate tables into candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config.factsheet_keywords import (
    ACTIVITY_HEADER,
    LISTING_HEADER,
    LISTING_PAGE,
    NAME_HEADER,
    RELATED_PARTY_HEADER,
)
from src.config.paths import CANDIDATE_DIR, INPUT_DIR, PAGE_DIR
from src.extract.prompt import PROMPT_VERSION
from src.extract.rules import extract_rules
from src.extract.select import page_text
from src.factsheet.names import assignment_key
from src.tables.grid import clean_cell, is_numeric_cell
from src.validate.quotes import snap_quote

_LEGAL_END = re.compile(
    r"(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH)\s*$",
    re.I,
)


def write_table_candidates(ticker: str, pages: list[dict] | None = None, pdf_path: Path | None = None) -> dict:
    """Write a parsed file dedup and validate already know how to read. Rules names are skipped."""
    pages = pages if pages is not None else _load_pages(ticker)
    pdf_path = pdf_path or _pdf(ticker)
    covered = _rules_keys(pages)
    candidates = []
    skipped_related = 0
    skipped_covered = 0
    pages_used = []
    for page in pages:
        text = page_text(page)
        if not text or not LISTING_PAGE.search(text):
            continue
        number = int(page["pdf_page"])
        try:
            from src.tables.digital import extract_digital_page

            tables = extract_digital_page(pdf_path, ticker, number) if pdf_path else []
        except Exception:
            tables = []
        page_added = False
        for table in tables:
            if _related_only(table):
                skipped_related += 1
                continue
            if not _is_listing_table(table):
                continue
            rows = _join_wrapped([list(row.get("cells") or []) for row in table.get("rows") or []])
            headers = [clean_cell(cell) for cell in (table.get("headers") or [])]
            name_idx, activity_idx = _columns(headers, rows)
            for cells in rows:
                item = _row_candidate(page, cells, name_idx, activity_idx, text)
                if item is None:
                    continue
                if assignment_key(item["name"]) in covered:
                    skipped_covered += 1
                    continue
                covered.add(assignment_key(item["name"]))
                candidates.append(item)
                page_added = True
        if page_added:
            pages_used.append(number)
    destination = CANDIDATE_DIR / ticker / "parsed" / f"{ticker}_tables_{PROMPT_VERSION}.json"
    record = {
        "ticker": ticker,
        "target_pages": pages_used,
        "model": None,
        "prompt_version": PROMPT_VERSION,
        "request_status": "success",
        "source": "tables",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "candidates": candidates,
        "rejected": [],
        "error": None,
        "tools_enabled": False,
        "name_only_names": [item["name"] for item in candidates if item.get("descriptor_source") == "name_only_no_activity_column"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "path": str(destination),
        "count": len(candidates),
        "pages": pages_used,
        "skipped_related": skipped_related,
        "skipped_covered": skipped_covered,
        "name_only_names": record["name_only_names"],
    }


def join_wrapped_rows(rows: list[list[str]]) -> list[list[str]]:
    """Join a name that the grid split onto the following row."""
    return _join_wrapped(rows)


def _join_wrapped(rows: list[list[str]]) -> list[list[str]]:
    joined: list[list[str]] = []
    for row in rows:
        cells = [clean_cell(cell) for cell in row]
        if not any(cells):
            continue
        other_text = any(cell and not is_numeric_cell(cell) for cell in cells[1:])
        if joined and cells[0] and not _LEGAL_END.search(joined[-1][0]) and not other_text:
            joined[-1][0] = clean_cell(joined[-1][0] + " " + cells[0])
            continue
        joined.append(cells)
    return joined


def _row_candidate(page: dict, cells: list[str], name_idx: int, activity_idx: int | None, text: str) -> dict | None:
    if name_idx >= len(cells):
        return None
    name = " ".join(cells[name_idx].split())
    if not name or not _LEGAL_END.search(name):
        return None
    if name.count("(") != name.count(")") or name.endswith((",", ";", ":")):
        return None
    if name.casefold().startswith(("total", "name of", "particulars")):
        return None
    activity = ""
    if activity_idx is not None and activity_idx < len(cells):
        activity = cells[activity_idx].strip()
    quote = _verbatim(text, name, activity)
    if not quote:
        return None
    if activity and activity.casefold() != name.casefold() and activity.casefold() in quote.casefold():
        descriptor = activity
        source = "activity_column"
    else:
        descriptor = name
        source = "name_only_no_activity_column"
    printed = page.get("printed_page")
    return {
        "name": name,
        "kind": "subsidiary",
        "descriptor": descriptor,
        "entity": name,
        "status": "current",
        "revenue_share_pct": None,
        "descriptor_source": source,
        "sources": [{
            "pdf_page": int(page["pdf_page"]),
            "printed_page": None if printed in (None, "") else str(printed),
            "quote": quote,
        }],
        "supporting_figures": [],
    }


def _verbatim(text: str, name: str, activity: str) -> str | None:
    if name and name in text:
        if activity and activity in text:
            start = text.find(name)
            window = text[max(0, start - 80): start + max(len(name), len(activity)) + 80]
            if activity in window and name in window:
                return window.strip()
        return name
    snapped = snap_quote(name, text, minimum_ratio=0.98)
    if snapped.get("reason") == "normalized_window" and snapped.get("quote"):
        return snapped["quote"]
    return None


def _columns(headers: list[str], rows: list[list[str]]) -> tuple[int, int | None]:
    name_idx = 0
    activity_idx = None
    for index, header in enumerate(headers):
        if NAME_HEADER.search(header):
            name_idx = index
        if ACTIVITY_HEADER.search(header):
            activity_idx = index
    if not headers and rows:
        name_idx = 0
    return name_idx, activity_idx


def _related_only(table: dict) -> bool:
    blob = " ".join(clean_cell(cell) for cell in (table.get("headers") or []))
    if not RELATED_PARTY_HEADER.search(blob):
        return False
    return LISTING_HEADER.search(blob) is None


def _is_listing_table(table: dict) -> bool:
    blob = " ".join(clean_cell(cell) for cell in (table.get("headers") or []))
    if LISTING_HEADER.search(blob) or NAME_HEADER.search(blob):
        return True
    for row in (table.get("rows") or [])[:8]:
        cells = " ".join(row.get("cells") or [])
        if _LEGAL_END.search(cells):
            return True
    return False


def _rules_keys(pages: list[dict]) -> set[str]:
    keys = set()
    for item in extract_rules(pages):
        key = assignment_key(item.get("name") or "")
        if key:
            keys.add(key)
    return keys


def _pdf(ticker: str) -> Path | None:
    folder = INPUT_DIR / ticker
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    return pdfs[0] if len(pdfs) == 1 else None


def _load_pages(ticker: str) -> list[dict]:
    folder = PAGE_DIR / ticker
    if not folder.is_dir():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("p*.json"))]
