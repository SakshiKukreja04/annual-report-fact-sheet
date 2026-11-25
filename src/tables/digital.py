"""Digital table path: pdfplumber, with Camelot lattice and stream when they load."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from src.preprocessing.pipeline import is_two_page_spread
from src.tables.grid import grid_score, prepare_grid, split_header
from src.tables.provenance import footnotes_below, printed_page, units_above

_PLUMBER_SETTINGS = (
    (
        "pdfplumber_text_lines",
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
        },
    ),
    (
        "pdfplumber_lines",
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
        },
    ),
    (
        "pdfplumber_text",
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
        },
    ),
)

_CAMELOT_ERROR: str | None = None
_CAMELOT_CHECKED = False


def camelot_status() -> dict:
    """Import Camelot once. A blocked native library is reported, not hidden."""
    global _CAMELOT_ERROR, _CAMELOT_CHECKED
    if not _CAMELOT_CHECKED:
        _CAMELOT_CHECKED = True
        try:
            import camelot  # noqa: F401
        except Exception as exc:
            _CAMELOT_ERROR = f"{type(exc).__name__}: {exc}"
    return {"available": _CAMELOT_ERROR is None, "error": _CAMELOT_ERROR}


def _candidate(method: str, rows: list[list[str]], bbox, error: str | None = None) -> dict:
    score = grid_score(rows) if rows else 0.0
    return {
        "method": method,
        "rows": rows,
        "bbox": bbox,
        "score": score,
        "row_count": len(rows),
        "column_count": max((len(row) for row in rows), default=0),
        "error": error,
    }


def _plumber_candidates(page) -> list[dict]:
    found = []
    for method, settings in _PLUMBER_SETTINGS:
        try:
            tables = page.find_tables(table_settings=settings) or []
        except Exception as exc:
            found.append(_candidate(method, [], None, f"{type(exc).__name__}: {exc}"))
            continue
        if not tables:
            found.append(_candidate(method, [], None))
            continue
        for table in tables:
            grid = prepare_grid(table.extract() or [])
            found.append(_candidate(method, grid, tuple(table.bbox)))
    return found


def _camelot_candidates(pdf_path: Path, pdf_page: int) -> list[dict]:
    status = camelot_status()
    if not status["available"]:
        return [
            _candidate("camelot_lattice", [], None, status["error"]),
            _candidate("camelot_stream", [], None, status["error"]),
        ]
    import camelot

    found = []
    for flavor, method in (("lattice", "camelot_lattice"), ("stream", "camelot_stream")):
        try:
            result = camelot.read_pdf(str(pdf_path), pages=str(pdf_page), flavor=flavor)
        except Exception as exc:
            found.append(_candidate(method, [], None, f"{type(exc).__name__}: {exc}"))
            continue
        if not result:
            found.append(_candidate(method, [], None))
            continue
        for table in result:
            values = table.df.values.tolist()
            grid = prepare_grid(values)
            bbox = None
            if getattr(table, "_bbox", None):
                bbox = tuple(float(value) for value in table._bbox)
            accuracy = None
            report = getattr(table, "parsing_report", None) or {}
            if report.get("accuracy") is not None:
                accuracy = float(report["accuracy"]) / 100.0
            candidate = _candidate(method, grid, bbox)
            if accuracy is not None:
                candidate["score"] = round(0.5 * candidate["score"] + 0.5 * accuracy, 4)
            found.append(candidate)
    return found


def _words(page) -> list[dict]:
    return page.extract_words(use_text_flow=False, keep_blank_chars=False) or []


def extract_digital_page(pdf_path: Path, ticker: str, pdf_page: int) -> list[dict]:
    """Return one record per table. The best grid keeps its row and column order."""
    with pdfplumber.open(pdf_path) as document:
        page = document.pages[pdf_page - 1]
        words = _words(page)
        printed = printed_page(
            words,
            float(page.height),
            page_width=float(page.width),
            allow_pair=is_two_page_spread(float(page.width), float(page.height)),
        )
        candidates = _plumber_candidates(page) + _camelot_candidates(pdf_path, pdf_page)
    usable = [item for item in candidates if item["score"] >= 0.45 and item["rows"]]
    # One winner per distinct bbox, so three strategies do not emit the same table three times.
    chosen = _unique_tables(usable)
    records = []
    for index, item in enumerate(chosen, start=1):
        headers, body = split_header(item["rows"])
        if not body:
            continue
        bbox = item["bbox"] or (0, 0, 0, 0)
        records.append(
            {
                "ticker": ticker,
                "pdf_page": pdf_page,
                "printed_page": printed,
                "table_id": f"{ticker}-p{pdf_page:04d}-t{index:02d}",
                "headers": headers,
                "rows": [
                    {"pdf_page": pdf_page, "printed_page": printed, "cells": row}
                    for row in body
                ],
                "units": units_above(words, bbox) if item["bbox"] else None,
                "footnotes": footnotes_below(words, bbox) if item["bbox"] else [],
                "extraction_method": item["method"],
                "confidence": item["score"],
                "continuation_from": None,
                "continuation_to": None,
                "bbox": [round(float(value), 2) for value in bbox],
                "candidate_methods": _method_summary(candidates),
            }
        )
    return records


def _unique_tables(candidates: list[dict]) -> list[dict]:
    """Keep the higher-scoring grid when two methods found the same region."""
    ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept = []
    for candidate in ordered:
        if any(_same_region(candidate, prior) for prior in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item["bbox"][1] if item["bbox"] else 0, item["bbox"][0] if item["bbox"] else 0))
    return kept


def _method_summary(candidates: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for candidate in candidates:
        current = best.get(candidate["method"])
        if current is None or candidate["score"] > current["score"]:
            best[candidate["method"]] = {
                "method": candidate["method"],
                "score": candidate["score"],
                "row_count": candidate["row_count"],
                "column_count": candidate["column_count"],
                "error": candidate["error"],
            }
    return list(best.values())


def _same_region(left: dict, right: dict) -> bool:
    if not left["bbox"] or not right["bbox"]:
        return False
    ax0, atop, ax1, abottom = left["bbox"]
    bx0, btop, bx1, bbottom = right["bbox"]
    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_y = min(abottom, bbottom) - max(atop, btop)
    if overlap_x <= 0 or overlap_y <= 0:
        return False
    area = overlap_x * overlap_y
    left_area = max((ax1 - ax0) * (abottom - atop), 1)
    right_area = max((bx1 - bx0) * (bbottom - btop), 1)
    return area / min(left_area, right_area) >= 0.5
