"""Recovery tracking. Sample JSON is read and not rewritten."""

import json
from pathlib import Path

from src.tables.recovery import (
    FAILED,
    NOT_DETECTED,
    PARTIAL,
    STRUCTURE_AND_TEXT_VALID,
    STRUCTURE_ONLY,
    assess_table,
    audit_half,
    normalize_cell,
)
from src.tables.recovery_report import build_powergrid_audit


def _table(headers, rows, method="pdfplumber_text_lines", confidence=0.93, pdf_page=180, half=None):
    return {
        "ticker": "HINDALCO",
        "pdf_page": pdf_page,
        "printed_page": "300/301",
        "half": half,
        "table_id": f"HINDALCO-p{pdf_page:04d}-t01",
        "headers": headers,
        "rows": [{"pdf_page": pdf_page, "printed_page": "300/301", "cells": row} for row in rows],
        "extraction_method": method,
        "confidence": confidence,
        "bbox": [10, 20, 30, 40],
    }


def test_reliable_structure_and_text():
    table = assess_table(_table(["Particulars", "Amount"], [["Total", "10"], ["Capital", "20"]]))
    assert table["recovery_status"] == STRUCTURE_AND_TEXT_VALID
    assert table["rejection_reason"] is None
    assert table["values_fabricated"] is False


def test_garbled_cell_text_is_structure_only():
    source = _table(
        ["S:NO,", "Corticblan", "Amount(int)"],
        [["(i", "Jola ama zt upent trir", "14418.78099"]],
        method="paddleocr_pp_structure",
        confidence=0.64,
        pdf_page=200,
        half="LEFT",
    )
    table = assess_table(source)
    assert table["recovery_status"] == STRUCTURE_ONLY
    assert table["text_reliable"] is False
    assert "cell text" in table["rejection_reason"]
    assert table["raw_headers"][1] == "Corticblan"
    assert table["normalized_headers"][1] == {"value": None, "reason": "unreadable"}
    assert table["cell_confidence"] is None


def test_missing_rows_are_partial():
    table = assess_table(
        _table(["Particulars", "Amount"], [["Total", "10"]]),
        expected_body_rows=5,
    )
    assert table["recovery_status"] == PARTIAL
    assert "body rows" in table["rejection_reason"]
    assert table["raw_rows"][0]["cells"] == ["Total", "10"]


def test_missing_columns_are_partial():
    table = assess_table(
        _table(["Particulars", "Amount"], [["Total", "10"]]),
        expected_columns=["S.No.", "Particulars", "Amount"],
    )
    assert table["recovery_status"] == PARTIAL
    assert table["detected_column_count"] == 2
    assert "columns" in table["rejection_reason"]


def test_visible_table_that_was_not_detected_has_no_invented_cells():
    page = audit_half(
        "POWERGRID",
        200,
        None,
        "LEFT",
        [],
        [{"label": "three_year_unspent_csr", "ticker": "POWERGRID", "matched_table_id": None, "note": "Visible and not returned."}],
        "p0200_LEFT/full.png",
        "paddleocr_pp_structure",
    )
    missing = page["tables"][0]
    assert missing["recovery_status"] == NOT_DETECTED
    assert missing["pdf_page"] == 200
    assert missing["half"] == "LEFT"
    assert missing["raw_rows"] == []
    assert missing["raw_headers"] == []
    assert page["missing_table_count"] == 1
    assert page["detected_table_count"] == 0
    assert "2021" not in json.dumps(missing)


def test_page_with_no_table_is_not_a_failure():
    page = audit_half("POWERGRID", 26, None, "RIGHT", [], [], "p0026_RIGHT/full.png", "paddleocr_pp_structure")
    assert page["table_detection_status"] == "no_table"
    assert page["expected_table_count"] == 0
    assert page["missing_table_count"] == 0
    assert page["manual_review_required"] is False
    assert all(table["recovery_status"] != FAILED for table in page["tables"])


def test_unknown_expected_count_stays_null():
    page = audit_half("POWERGRID", 50, None, "FULL", [], None, None, None)
    assert page["expected_table_count"] is None
    assert page["missing_table_count"] is None
    assert page["table_detection_status"] == "unknown_expected_count"


def test_pdf_page_and_spread_half_are_preserved():
    source = _table(["Particulars", "Amount"], [["Total", "10"]], pdf_page=200, half="LEFT")
    table = assess_table(source)
    assert table["pdf_page"] == 200
    assert table["half"] == "LEFT"
    assert table["raw_rows"][0]["pdf_page"] == 200
    page = audit_half("POWERGRID", 200, None, "RIGHT", [source], None, None, "paddleocr_pp_structure")
    assert page["pdf_page"] == 200
    assert page["half"] == "RIGHT"
    assert page["tables"][0]["half"] == "LEFT"
    assert page["tables"][0]["pdf_page"] == 200


def test_raw_cells_stay_after_normalization():
    raw = "  Total   amount  "
    assert normalize_cell(raw) == {"value": "Total amount", "reason": None}
    table = assess_table(_table([raw], [["10"]]))
    assert table["raw_headers"] == [raw]
    assert table["normalized_headers"][0]["value"] == "Total amount"


def test_unreadable_cells_are_not_replaced_with_a_guess():
    result = normalize_cell("Corticblan")
    assert result["value"] is None
    assert result["reason"] == "unreadable"
    assert result["value"] != "Particulars"


def test_powergrid_page_200_audit_uses_the_stored_sample():
    sample = Path(__file__).resolve().parents[1] / "data" / "processed" / "tables" / "sample"
    before = {
        path.name: path.read_bytes()
        for path in sample.glob("POWERGRID-p0200-*.json")
    }
    audit = build_powergrid_audit(sample)
    after = {path.name: path.read_bytes() for path in sample.glob("POWERGRID-p0200-*.json")}
    assert before == after
    left, right = audit["halves"]
    assert left["pdf_page"] == 200 and left["half"] == "LEFT"
    assert left["expected_table_count"] == 3
    assert left["detected_table_count"] == 2
    assert left["recovered_table_count"] == 0
    assert left["missing_table_count"] == 1
    by_status = {table.get("label") or table["table_id"]: table["recovery_status"] for table in left["tables"]}
    assert by_status["POWERGRID-p0200-s01-LEFT"] == STRUCTURE_ONLY
    assert by_status["POWERGRID-p0200-s02-LEFT"] == STRUCTURE_ONLY
    assert by_status["three_year_unspent_csr"] == NOT_DETECTED
    missing = next(table for table in left["tables"] if table["recovery_status"] == NOT_DETECTED)
    assert missing["raw_rows"] == []
    assert right["half"] == "RIGHT"
    assert right["detected_table_count"] == 0
    assert right["expected_table_count"] == 1
    assert right["table_detection_status"] == "incomplete"
    assert audit["financial_shares_calculated"] is False
    assert audit["language_model_called"] is False
