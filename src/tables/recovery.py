"""Recovery status for extracted tables. This does not read the PDF or invent cells."""

from __future__ import annotations

import re

from src.tables.grid import clean_cell

STRUCTURE_AND_TEXT_VALID = "STRUCTURE_AND_TEXT_VALID"
STRUCTURE_ONLY = "STRUCTURE_ONLY"
PARTIAL = "PARTIAL"
FAILED = "FAILED"
MANUAL_REVIEW = "MANUAL_REVIEW"
NOT_DETECTED = "NOT_DETECTED"

DEFAULT_THRESHOLDS = {
    "min_readable_cell_ratio": 0.6,
    "min_ocr_confidence": 0.75,
}

_COMMON = {
    "amount", "assets", "average", "balance", "capital", "company", "country",
    "date", "details", "director", "excess", "financial", "fund", "limited",
    "name", "particulars", "percent", "previous", "profit", "projects",
    "revenue", "section", "spent", "subsidiary", "surplus", "total", "transfer",
    "transferred", "unspent", "year", "years",
}
_NUMBER = re.compile(r"^[\(\)\-–—]?[\d,.\s/%]+$")
_INDEX = re.compile(r"^[\(\[]?(?:\d+|[ivx]+)[\)\].:]?$", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']+")


def cell_is_readable(value: str) -> bool:
    """True when a cell is a number, an index, or ordinary words. Garbled OCR is not readable."""
    text = clean_cell(value)
    if not text:
        return True
    if text.casefold() in {"na", "n.a.", "n.a", "nil", "-", "–", "—"}:
        return True
    if _NUMBER.match(text) or _INDEX.match(text):
        return True
    long_tokens = []
    for token in _TOKEN.findall(text):
        if len(token) < 3:
            continue
        long_tokens.append(token.casefold() in _COMMON)
    if not long_tokens:
        return True
    return all(long_tokens)


def normalize_cell(value) -> dict:
    """Return a cleaned value, or null when the text is unreadable. The raw value is not changed."""
    raw = "" if value is None else str(value)
    cleaned = clean_cell(raw)
    if cleaned and not cell_is_readable(cleaned):
        return {"value": None, "reason": "unreadable"}
    return {"value": cleaned, "reason": None}


def _column_count(table: dict) -> int:
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    widths = [len(headers)]
    widths.extend(len(row.get("cells") or []) for row in rows)
    return max(widths) if widths else 0


def _expected_width(expected_columns) -> int | None:
    if expected_columns is None:
        return None
    if isinstance(expected_columns, int):
        return expected_columns
    return len(expected_columns)


def _is_ocr_method(method: str) -> bool:
    name = (method or "").casefold()
    return "paddle" in name or "ocr" in name or "tesseract" in name


def assess_table(
    table: dict,
    expected_columns=None,
    expected_body_rows: int | None = None,
    thresholds: dict | None = None,
    ambiguous: bool = False,
) -> dict:
    """Score one extracted table. Raw cells are copied through unchanged."""
    settings = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    headers = list(table.get("headers") or [])
    rows = list(table.get("rows") or [])
    column_count = _column_count(table)
    method = table.get("extraction_method")
    confidence = table.get("confidence")
    cells = [cell for row in rows for cell in (row.get("cells") or [])]
    judged = [cell for cell in headers + cells if clean_cell(cell)]
    prose = [cell for cell in judged if any(len(token) >= 4 for token in _TOKEN.findall(cell))]
    basis = prose or judged
    readable = [cell for cell in basis if cell_is_readable(cell)]
    ratio = (len(readable) / len(basis)) if basis else 1.0
    text_reliable = ratio >= settings["min_readable_cell_ratio"]
    reason = None
    status = STRUCTURE_AND_TEXT_VALID

    if not headers and not rows:
        status = FAILED
        reason = "Extraction produced no header and no body."
    else:
        expected_width = _expected_width(expected_columns)
        missing_columns = expected_width is not None and column_count < expected_width
        missing_rows = expected_body_rows is not None and len(rows) < expected_body_rows
        if missing_columns or missing_rows:
            status = PARTIAL
            parts = []
            if missing_columns:
                parts.append(f"{column_count} columns were detected and {expected_width} were expected")
            if missing_rows:
                parts.append(f"{len(rows)} body rows were detected and {expected_body_rows} were expected")
            reason = "Partial table: " + "; ".join(parts) + "."
        elif ambiguous:
            status = MANUAL_REVIEW
            reason = "The table structure is ambiguous and needs human inspection."
        elif not text_reliable:
            status = STRUCTURE_ONLY
            reason = "A grid was detected, but the cell text is not reliable."
            if _is_ocr_method(method) and confidence is not None and float(confidence) < settings["min_ocr_confidence"]:
                reason += f" OCR confidence {confidence} is also below {settings['min_ocr_confidence']}."
        elif _is_ocr_method(method) and confidence is None:
            status = MANUAL_REVIEW
            reason = "OCR confidence was not stored, so the cell text was not accepted as valid."
        elif _is_ocr_method(method) and float(confidence) < settings["min_ocr_confidence"]:
            status = STRUCTURE_ONLY
            reason = (
                f"OCR confidence {confidence} is below {settings['min_ocr_confidence']}. "
                "The grid was kept and the cell text was not accepted."
            )

    return {
        "table_id": table.get("table_id"),
        "ticker": table.get("ticker"),
        "pdf_page": table.get("pdf_page"),
        "printed_page": table.get("printed_page"),
        "half": table.get("half"),
        "bbox": table.get("bbox"),
        "detected_row_count": len(rows),
        "detected_column_count": column_count,
        "raw_headers": headers,
        "normalized_headers": [normalize_cell(header) for header in headers],
        "raw_rows": [
            {
                "pdf_page": row.get("pdf_page", table.get("pdf_page")),
                "printed_page": row.get("printed_page", table.get("printed_page")),
                "cells": list(row.get("cells") or []),
            }
            for row in rows
        ],
        "normalized_rows": [
            {
                "pdf_page": row.get("pdf_page", table.get("pdf_page")),
                "printed_page": row.get("printed_page", table.get("printed_page")),
                "cells": [normalize_cell(cell) for cell in (row.get("cells") or [])],
            }
            for row in rows
        ],
        "cell_confidence": None,
        "cell_confidence_reason": "Cell-level OCR confidence was not stored by the extractor.",
        "table_confidence": confidence,
        "extraction_method": method,
        "recovery_status": status,
        "rejection_reason": reason,
        "text_reliable": text_reliable,
        "readable_cell_ratio": round(ratio, 4),
        "thresholds": settings,
        "values_fabricated": False,
    }


def undetected_table(expectation: dict, pdf_page: int, printed_page, half: str) -> dict:
    """A table a person saw, which the extractor did not return. No cells are invented."""
    return {
        "table_id": None,
        "label": expectation.get("label"),
        "ticker": expectation.get("ticker"),
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "half": half,
        "bbox": None,
        "detected_row_count": 0,
        "detected_column_count": 0,
        "raw_headers": [],
        "normalized_headers": [],
        "raw_rows": [],
        "normalized_rows": [],
        "cell_confidence": None,
        "cell_confidence_reason": "No cells were extracted.",
        "table_confidence": None,
        "extraction_method": expectation.get("extraction_method"),
        "recovery_status": NOT_DETECTED,
        "rejection_reason": expectation.get("note") or "A visible table was not returned by the extractor.",
        "text_reliable": False,
        "values_fabricated": False,
    }


def audit_half(
    company: str,
    pdf_page: int,
    printed_page,
    half: str,
    detected_tables: list[dict],
    expectations: list[dict] | None,
    source_image: str | None,
    extraction_method: str | None,
    recovery_notes: list[str] | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Compare extractor output with an explicit expected list. None means the count is unknown."""
    notes = list(recovery_notes or [])
    if expectations is None:
        assessed = [assess_table(table, thresholds=thresholds) for table in detected_tables]
        missing = []
        expected_count = None
        missing_count = None
        detection_status = "unknown_expected_count"
    else:
        assessed = []
        missing = []
        matched = set()
        for expectation in expectations:
            table_id = expectation.get("matched_table_id")
            found = next((table for table in detected_tables if table.get("table_id") == table_id), None)
            if found is None:
                missing.append(undetected_table(expectation, pdf_page, printed_page, half))
                continue
            matched.add(found.get("table_id"))
            assessed.append(
                assess_table(
                    found,
                    expected_columns=expectation.get("expected_columns"),
                    expected_body_rows=expectation.get("expected_body_rows"),
                    thresholds=thresholds,
                    ambiguous=bool(expectation.get("ambiguous")),
                )
            )
        for table in detected_tables:
            if table.get("table_id") not in matched:
                assessed.append(assess_table(table, thresholds=thresholds))
        expected_count = len(expectations)
        missing_count = len(missing)
        if expected_count == 0 and not detected_tables:
            detection_status = "no_table"
        elif missing_count == 0 and assessed and all(
            item["recovery_status"] == STRUCTURE_AND_TEXT_VALID for item in assessed
        ):
            detection_status = "complete"
        else:
            detection_status = "incomplete"

    recovered = sum(1 for item in assessed if item["recovery_status"] == STRUCTURE_AND_TEXT_VALID)
    review_statuses = {STRUCTURE_ONLY, PARTIAL, FAILED, MANUAL_REVIEW, NOT_DETECTED}
    needs_review = any(item["recovery_status"] in review_statuses for item in assessed + missing)
    return {
        "company": company,
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "half": half,
        "expected_table_count": expected_count,
        "detected_table_count": len(detected_tables),
        "recovered_table_count": recovered,
        "missing_table_count": missing_count,
        "table_detection_status": detection_status,
        "manual_review_required": needs_review,
        "recovery_notes": notes,
        "source_image": source_image,
        "extraction_method": extraction_method,
        "tables": assessed + missing,
    }
