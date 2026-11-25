"""Stage 5 table structure. Real pages are Hindalco and Coal India."""

from pathlib import Path

from src.ingestion.audit import locate_pdfs
from src.tables.digital import extract_digital_page
from src.tables.grid import (
    header_similarity,
    html_to_grid,
    link_continuations,
    prepare_grid,
    split_header,
)
from src.tables.provenance import footnotes_below, printed_page, units_above
from src.tables.sample import SAMPLE_PAGES


def _table(ticker, page, table_id, headers):
    return {
        "ticker": ticker,
        "pdf_page": page,
        "table_id": table_id,
        "headers": headers,
        "rows": [{"pdf_page": page, "printed_page": None, "cells": ["kept"]}],
    }


def test_header_row_stays_out_of_the_body():
    grid = prepare_grid(
        [
            ["Name of the Subsidiary", "Country", "Capital"],
            ["Novelis Inc.", "USA", "10"],
            ["Utkal Alumina", "India", "20"],
        ]
    )
    headers, body = split_header(grid)
    assert headers[0] == "Name of the Subsidiary"
    assert body[0][0] == "Novelis Inc."
    assert all(row[0] != "Name of the Subsidiary" for row in body)


def test_empty_columns_do_not_shift_later_cells():
    grid = prepare_grid([["Sr.", "", "Country"], ["1", "", "India"]])
    assert grid == [["Sr.", "Country"], ["1", "India"]]


def test_html_rowspan_keeps_the_column_count():
    html = """
    <table>
      <tr><th rowspan="2">Particulars</th><th colspan="2">Amount</th></tr>
      <tr><th>2025</th><th>2024</th></tr>
      <tr><td>Revenue</td><td>10</td><td>8</td></tr>
    </table>
    """
    grid = html_to_grid(html)
    assert grid[0] == ["Particulars", "Amount", ""]
    assert grid[1] == ["", "2025", "2024"]
    assert grid[2] == ["Revenue", "10", "8"]
    assert all(len(row) == 3 for row in grid)


def test_matching_headers_link_adjacent_pages_without_moving_rows():
    first = _table("COALINDIA", 74, "COALINDIA-p0074-t01", ["Particulars", "Coal offtake", "Closing stock"])
    second = _table("COALINDIA", 75, "COALINDIA-p0075-t01", ["Particulars", "Coal offtake", "Closing stock"])
    link_continuations([first, second])
    assert first["continuation_to"] == "COALINDIA-p0075-t01"
    assert second["continuation_from"] == "COALINDIA-p0074-t01"
    assert first["rows"][0]["pdf_page"] == 74
    assert second["rows"][0]["pdf_page"] == 75


def test_similar_column_count_does_not_link_different_headers():
    production = _table("COALINDIA", 74, "a", ["Coal production", "Offtake", "Stock"])
    remuneration = _table("COALINDIA", 75, "b", ["Director", "Salary", "Commission"])
    assert header_similarity(production["headers"], remuneration["headers"]) == 0
    link_continuations([production, remuneration])
    assert production["continuation_to"] is None
    assert remuneration["continuation_from"] is None


def test_printed_page_units_and_footnotes():
    words = [
        {"text": "12", "x0": 500, "top": 10},
        {"text": "(Rs.", "x0": 40, "top": 90},
        {"text": "in", "x0": 70, "top": 90},
        {"text": "crore)", "x0": 90, "top": 90},
        {"text": "Note:", "x0": 40, "top": 410},
        {"text": "Includes", "x0": 80, "top": 410},
        {"text": "joint", "x0": 140, "top": 410},
        {"text": "ventures.", "x0": 180, "top": 410},
    ]
    assert printed_page(words, 800) == "12"
    assert units_above(words, (30, 120, 400, 400)) == "(Rs. in crore)"
    assert footnotes_below(words, (30, 120, 400, 400)) == ["Note: Includes joint ventures."]


def test_ocr_bbox_maps_to_the_margin_and_a_non_consecutive_pair_is_not_invented():
    # Image is 1000 px tall. y=900 is inside the bottom 12% and outside a tighter 8% strip.
    words = [
        {"text": "8", "confidence": 96, "bbox": [20, 970, 30, 20]},
        {"text": "4", "confidence": 60, "bbox": [900, 890, 30, 20]},
        {"text": "body", "confidence": 95, "bbox": [100, 400, 40, 20]},
    ]
    assert printed_page(words, 1000, page_width=1000, allow_pair=True) == "8"
    assert printed_page(words, 1000, page_width=1000, allow_pair=False) == "8"


def test_a_single_half_does_not_return_a_consecutive_pair():
    words = [
        {"text": "7", "confidence": 80, "bbox": [180, 20, 20, 16]},
        {"text": "8", "confidence": 92, "bbox": [10, 20, 20, 16]},
    ]
    assert printed_page(words, 1000, page_width=400, allow_pair=False) == "8"


def test_the_best_scoring_consecutive_pair_wins():
    words = [
        {"text": "2", "confidence": 70, "bbox": [10, 20, 20, 16]},
        {"text": "3", "confidence": 70, "bbox": [900, 20, 20, 16]},
        {"text": "40", "confidence": 96, "bbox": [12, 960, 30, 16]},
        {"text": "41", "confidence": 97, "bbox": [910, 960, 30, 16]},
    ]
    assert printed_page(words, 1000, page_width=1000, allow_pair=True) == "40/41"


def test_spread_offset_fills_only_after_five_confident_anchors():
    from src.tables.provenance import spread_folio_inference

    def page(number, printed, confidence):
        return {
            "pdf_page": number,
            "printed_page": printed,
            "page_type": ["TWO_PAGE_SPREAD", "SCANNED_IMAGE"],
            "quality_metadata": {
                "folio_source": "ocr",
                "folio_confidence": confidence,
                "folio_half": "PAIR",
            },
        }

    short = [page(30 + index, f"{40 + 2 * index}/{41 + 2 * index}", 90) for index in range(4)]
    short.append({"pdf_page": 40, "printed_page": None, "page_type": ["TWO_PAGE_SPREAD"], "quality_metadata": {}})
    assert spread_folio_inference(short)["filled"] == {}
    long = [page(30 + index, f"{40 + 2 * index}/{41 + 2 * index}", 90) for index in range(5)]
    long.append({"pdf_page": 90, "printed_page": None, "page_type": ["TWO_PAGE_SPREAD"], "quality_metadata": {}})
    long.append({"pdf_page": 91, "printed_page": None, "page_type": ["SCANNED_IMAGE"], "quality_metadata": {}})
    filled = spread_folio_inference(long)
    assert filled["offset"] == 40 - 2 * 30
    assert filled["filled"][90] == f"{2 * 90 + filled['offset']}/{2 * 90 + filled['offset'] + 1}"
    assert 91 not in filled["filled"]
    assert len(filled["anchors"]) >= 5


def test_sample_covers_hindalco_and_coal_india_runs():
    tickers = {ticker for ticker, _page in SAMPLE_PAGES}
    assert "HINDALCO" in tickers
    assert "COALINDIA" in tickers
    coal_pages = [page for ticker, page in SAMPLE_PAGES if ticker == "COALINDIA"]
    assert coal_pages == [74, 75, 76]


def test_hindalco_subsidiary_table_keeps_rows_on_the_pdf_page():
    pdfs = locate_pdfs()
    tables = extract_digital_page(pdfs["HINDALCO"], "HINDALCO", 180)
    matched = [
        table
        for table in tables
        if any("subsidiary" in header.casefold() for header in table["headers"])
    ]
    assert matched
    table = max(matched, key=lambda item: len(item["rows"]))
    assert len(table["rows"]) >= 5
    assert all(row["pdf_page"] == 180 for row in table["rows"])
    assert table["extraction_method"].startswith("pdfplumber")
    assert "revenue_share" not in table


def test_coal_india_pages_keep_their_own_row_pages():
    pdfs = locate_pdfs()
    tables = []
    for page in (74, 75, 76):
        tables.extend(extract_digital_page(pdfs["COALINDIA"], "COALINDIA", page))
    assert tables
    for table in tables:
        assert all(row["pdf_page"] == table["pdf_page"] for row in table["rows"])


def test_stage_does_not_call_groq_or_compute_shares():
    root = Path(__file__).resolve().parents[1] / "src" / "tables"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "import groq" not in text
        assert "revenue_share" not in text
        assert "qwen" not in text
