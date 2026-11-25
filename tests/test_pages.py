"""Stage 6 page records. Digital text and OCR stay separate."""

import json
from pathlib import Path

import pymupdf

from src.ingestion.audit import locate_pdfs
from src.pages.assemble import (
    IMAGE_ONLY,
    ILLEGIBLE,
    RECOVERABLE,
    REQUIRED_KEYS,
    assemble_page,
    is_candidate,
    text_block,
)
from src.pages.build import build_company, load_company, page_cache_path


def _page(**kwargs):
    defaults = dict(
        ticker="POWERGRID",
        pdf_page=10,
        printed_page=None,
        page_type=["DIGITAL_TEXT"],
        digital_blocks=[],
        ocr_blocks=[],
        captions=[],
        images=[],
        tables=[],
    )
    defaults.update(kwargs)
    return assemble_page(**defaults)


def test_printed_page_stays_separate_from_the_pdf_page():
    page = _page(pdf_page=180, printed_page="300/301", digital_blocks=[
        text_block("Subsidiary holdings " * 8, [10, 40, 400, 60], "pymupdf", "LEFT", 0),
    ])
    assert page["pdf_page"] == 180
    assert page["printed_page"] == "300/301"
    assert page["printed_page"] != str(page["pdf_page"])
    assert all(block["pdf_page"] == 180 for block in page["text_blocks"])


def test_spread_halves_share_one_pdf_page_and_keep_reading_order():
    left = text_block("Left half of the spread report", [20, 200, 400, 230], "pymupdf", "LEFT", 99)
    right = text_block("Right half of the spread report", [700, 50, 1100, 80], "pymupdf", "RIGHT", 0)
    page = _page(
        pdf_page=10,
        page_type=["DIGITAL_TEXT", "TWO_PAGE_SPREAD"],
        digital_blocks=[right, left],
    )
    assert page["pdf_page"] == 10
    assert [block["half"] for block in page["text_blocks"]] == ["LEFT", "RIGHT"]
    assert [block["reading_order"] for block in page["text_blocks"]] == [0, 1]
    assert {block["pdf_page"] for block in page["text_blocks"]} == {10}


def test_missing_text_is_recorded_and_not_sent_to_extraction():
    page = _page(page_type=["SCANNED_IMAGE"], digital_blocks=[], ocr_blocks=[])
    assert page["content_status"] == "pending_ocr"
    assert "ocr_not_run" in page["warnings"]
    assert is_candidate(page) is False

    empty = _page(page_type=["UNKNOWN"], digital_blocks=[])
    assert "missing_text" in empty["warnings"]
    assert is_candidate(empty) is False


def test_digital_text_and_ocr_are_both_kept_and_only_one_is_selected():
    digital = text_block("Board of Directors " * 8, [30, 80, 400, 110], "pymupdf", "LEFT", 0)
    tesseract = text_block("Board of Directors", [10, 10, 200, 40], "tesseract", "LEFT", 0, ocr_confidence=91.0, bbox_space="preprocessed_image_px")
    paddle = text_block("Board of Directors", [12, 12, 210, 42], "paddleocr", "LEFT", 0, ocr_confidence=88.0, bbox_space="preprocessed_image_px")
    page = _page(
        digital_blocks=[digital],
        ocr_blocks=[tesseract, paddle],
        ocr_recoverability={"LEFT": "RECOVERABLE"},
    )
    selected = [block for block in page["text_blocks"] if block["selected"]]
    assert [block["source"] for block in selected] == ["pymupdf"]
    assert {block["source"] for block in page["text_blocks"]} == {"pymupdf", "tesseract", "paddleocr"}
    assert "digital_and_ocr_both_present" in page["warnings"]
    assert "ocr_engines_kept_separate" in page["warnings"]
    assert page["quality_metadata"]["selected_text_source"] == "pymupdf"
    joined = " ".join(block["text"] for block in selected)
    assert "Board of Directors Board of Directors" not in joined or selected[0]["source"] == "pymupdf"
    assert tesseract["text"] not in "".join(block["text"] for block in selected if block["source"] != "pymupdf")


def test_weak_text_layer_selects_one_ocr_engine():
    digital = text_block("x", [0, 0, 10, 10], "pymupdf", "FULL", 0)
    tesseract = text_block("Coal India Limited letterhead", [0, 0, 80, 20], "tesseract", "FULL", 0, ocr_confidence=90)
    paddle = text_block("Coal India Limited letterhead extra", [0, 20, 90, 40], "paddleocr", "FULL", 0, ocr_confidence=70)
    page = _page(
        page_type=["SCANNED_IMAGE"],
        digital_blocks=[digital],
        ocr_blocks=[tesseract, paddle],
        ocr_recoverability={"FULL": "RECOVERABLE"},
    )
    selected = [block["source"] for block in page["text_blocks"] if block["selected"]]
    assert selected == ["tesseract"]
    assert any(block["source"] == "paddleocr" and not block["selected"] for block in page["text_blocks"])


def test_illegible_page_is_stored_but_excluded():
    ocr = text_block("sere ford noise", [0, 0, 40, 10], "tesseract", "LEFT", 0, ocr_confidence=40)
    page = _page(
        pdf_page=100,
        page_type=["SCANNED_IMAGE", "TWO_PAGE_SPREAD"],
        ocr_blocks=[ocr],
        ocr_recoverability={"LEFT": "ILLEGIBLE"},
    )
    assert page["content_status"] == ILLEGIBLE
    assert "illegible_low_dpi" in page["warnings"]
    assert all(not block["selected"] for block in page["text_blocks"])
    assert page["text_blocks"][0]["text"] == "sere ford noise"
    assert is_candidate(page) is False


def test_one_illegible_half_keeps_the_other_half():
    bad = text_block("sere ford noise", [0, 0, 40, 10], "tesseract", "RIGHT", 0, ocr_confidence=40)
    good = text_block("Power Grid transmission lines and substations", [0, 0, 200, 20], "tesseract", "LEFT", 0, ocr_confidence=90)
    page = _page(
        page_type=["SCANNED_IMAGE", "TWO_PAGE_SPREAD"],
        ocr_blocks=[bad, good],
        ocr_recoverability={"LEFT": "RECOVERABLE", "RIGHT": "ILLEGIBLE"},
    )
    selected = [block for block in page["text_blocks"] if block["selected"]]
    assert [block["half"] for block in selected] == ["LEFT"]
    assert page["content_status"] == RECOVERABLE
    assert "illegible_low_dpi" not in page["warnings"]
    assert "partial_half_illegible" in page["warnings"]
    assert is_candidate(page) is True


def test_a_caption_does_not_turn_a_visual_page_into_body_text():
    caption = "Transportation of Coal through Pipe Conveyor to Power Station at WCL"
    page = _page(
        ticker="COALINDIA",
        pdf_page=82,
        page_type=["IMAGE_ONLY_VISUAL", "SCANNED_IMAGE", "TWO_PAGE_SPREAD"],
        digital_blocks=[text_block(caption, [10, 40, 500, 80], "pymupdf", "LEFT", 0)],
    )
    assert page["content_status"] == IMAGE_ONLY
    assert is_candidate(page) is False
    assert page["captions"][0]["text"] == caption
    assert page["captions"][0]["pdf_page"] == 82
    assert all(not block["selected"] for block in page["text_blocks"])


def test_image_only_page_keeps_captions_apart_from_candidate_text():
    page = _page(
        pdf_page=26,
        page_type=["IMAGE_ONLY_VISUAL", "SCANNED_IMAGE"],
        digital_blocks=[text_block("Aerial view of a station", [40, 700, 300, 730], "pymupdf", "LEFT", 0)],
        captions=[{"text": "INTEGRATED ANNUAL REPORT", "half": "FULL", "source": "tesseract"}],
    )
    assert page["content_status"] == IMAGE_ONLY
    assert is_candidate(page) is False
    assert any(caption["text"] == "INTEGRATED ANNUAL REPORT" and caption["pdf_page"] == 26 for caption in page["captions"])
    assert any(caption["text"] == "Aerial view of a station" for caption in page["captions"])
    assert all(not block["selected"] for block in page["text_blocks"])


def test_table_rows_keep_their_pdf_page():
    table = {
        "table_id": "HINDALCO-p0180-t01",
        "pdf_page": 180,
        "printed_page": "300/301",
        "headers": ["Sr.", "Name of the Subsidiary Company"],
        "rows": [
            {"pdf_page": 180, "printed_page": "300/301", "cells": ["46", "Novelis Casthouse Germany GmbH"]},
        ],
        "units": "Figures Indian Rupee in Crore",
        "footnotes": ["Source: annual report"],
        "extraction_method": "pdfplumber_text_lines",
        "confidence": 0.93,
        "continuation_from": None,
        "continuation_to": None,
    }
    page = _page(
        ticker="HINDALCO",
        pdf_page=180,
        printed_page="300/301",
        page_type=["DIGITAL_TEXT", "TABLE_HEAVY", "TWO_PAGE_SPREAD"],
        digital_blocks=[text_block("Subsidiary table " * 6, [10, 40, 500, 70], "pymupdf", "LEFT", 0)],
        tables=[table],
    )
    stored = page["tables"][0]
    assert stored["pdf_page"] == 180
    assert stored["printed_page"] == "300/301"
    assert stored["rows"][0]["pdf_page"] == 180
    assert stored["rows"][0]["cells"][1] == "Novelis Casthouse Germany GmbH"
    assert stored["units"].startswith("Figures")
    assert stored["footnotes"] == ["Source: annual report"]
    assert page["content_status"] == RECOVERABLE


def test_page_record_serializes_the_required_fields():
    page = _page(
        pdf_page=10,
        printed_page="22",
        digital_blocks=[text_block("Notice of the annual general meeting text", [10, 40, 400, 70], "pymupdf", "FULL", 0)],
    )
    restored = json.loads(json.dumps(page))
    assert tuple(restored) == REQUIRED_KEYS or set(REQUIRED_KEYS).issubset(restored)
    for key in REQUIRED_KEYS:
        assert key in restored
    assert restored["ticker"] == "POWERGRID"
    assert restored["pdf_page"] == 10
    assert restored["text_blocks"][0]["bbox"] == [10.0, 40.0, 400.0, 70.0]
    assert restored["text_blocks"][0]["source"] == "pymupdf"
    assert restored["text_blocks"][0]["half"] == "FULL"


def test_resume_reuses_a_cached_page(tmp_path: Path):
    pdf = locate_pdfs()["HINDALCO"]
    first = build_company("HINDALCO", pages=[180], resume=False, output_dir=tmp_path, pdf_path=pdf)
    assert first["written"] == 1
    cached = page_cache_path(tmp_path, "HINDALCO", 180)
    original = cached.read_text(encoding="utf-8")
    second = build_company("HINDALCO", pages=[180], resume=True, output_dir=tmp_path, pdf_path=pdf)
    assert second["reused"] == 1
    assert second["written"] == 0
    assert cached.read_text(encoding="utf-8") == original


def test_real_spread_page_keeps_both_halves_and_its_table():
    pdf = locate_pdfs()["HINDALCO"]
    with pymupdf.open(pdf) as document:
        from src.pages.build import build_one_page
        from src.pages.extract import load_tables
        from src.config.paths import TABLE_DIR

        tables = load_tables(TABLE_DIR, "HINDALCO").get(180, [])
        page = build_one_page("HINDALCO", document[179], {"classifications": ["DIGITAL_TEXT", "TABLE_HEAVY", "TWO_PAGE_SPREAD"], "indicators": {}, "confidence": 0.9, "recommended_extraction_method": "digital_text_and_tables"}, [], tables)
    assert page["pdf_page"] == 180
    assert page["printed_page"] not in (None, "180")
    halves = {block["half"] for block in page["text_blocks"] if block["selected"]}
    assert "LEFT" in halves and "RIGHT" in halves
    assert all(block["pdf_page"] == 180 for block in page["text_blocks"])
    assert any(table["pdf_page"] == 180 and table["rows"] and table["rows"][0]["pdf_page"] == 180 for table in page["tables"])
    assert all(block["source"] == "pymupdf" for block in page["text_blocks"] if block["selected"])


def test_loader_splits_recoverable_visual_and_illegible(tmp_path: Path):
    from src.pages.build import write_page

    write_page(page_cache_path(tmp_path, "COALINDIA", 38), _page(ticker="COALINDIA", pdf_page=38, page_type=["IMAGE_ONLY_VISUAL"], captions=[{"text": "Coal India Limited", "half": "LEFT", "source": "tesseract"}]))
    write_page(page_cache_path(tmp_path, "COALINDIA", 82), _page(
        ticker="COALINDIA",
        pdf_page=82,
        page_type=["DIGITAL_TEXT"],
        digital_blocks=[text_block("Transportation of coal through the conveyor " * 3, [10, 40, 400, 80], "pymupdf", "LEFT", 0)],
    ))
    write_page(page_cache_path(tmp_path, "COALINDIA", 314), _page(
        ticker="COALINDIA",
        pdf_page=314,
        page_type=["SCANNED_IMAGE"],
        ocr_blocks=[text_block("noise only here", [0, 0, 10, 10], "tesseract", "FULL", 0)],
        ocr_recoverability={"FULL": "ILLEGIBLE"},
    ))
    loaded = load_company("COALINDIA", output_dir=tmp_path)
    assert [page["pdf_page"] for page in loaded["candidate_pages"]] == [82]
    assert [page["pdf_page"] for page in loaded["image_only_pages"]] == [38]
    assert loaded["excluded_pages"][0]["pdf_page"] == 314
    assert loaded["excluded_pages"][0]["content_status"] == ILLEGIBLE


def test_stage_does_not_call_a_language_model():
    root = Path(__file__).resolve().parents[1] / "src" / "pages"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "import groq" not in source
    assert "qwen" not in source.lower()
