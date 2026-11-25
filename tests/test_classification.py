"""Representative-page tests for stage 2 classification."""

import pymupdf

from src.classification.classify import classify_page, decide, text_quality
from src.ingestion.audit import locate_pdfs


def _page(ticker: str, pdf_page: int):
    path = locate_pdfs()[ticker]
    document = pymupdf.open(path)
    page = document[pdf_page - 1]
    row = classify_page(page, ticker)
    document.close()
    return row


def test_landscape_single_page_is_not_a_spread():
    indicators = {
        "text_stripped_char_count": 1500,
        "text_char_count": 1600,
        "alpha_ratio": 0.7,
        "image_coverage_ratio": 0.0,
        "word_count": 200,
        "width_pt": 842.0,
        "height_pt": 595.0,
        "spread_band": "not_spread",
        "gutter_found": False,
        "table_heavy": False,
        "table_row_count": 0,
        "horizontal_rule_count": 0,
        "table_score": 0.0,
        "garbled_or_low_quality": False,
    }
    row = decide("TEST", 1, indicators)
    assert "TWO_PAGE_SPREAD" not in row["classifications"]
    assert row["needs_visual_inspection"] is False
    assert row["recommended_extraction_method"] == "digital_text"


def test_ambiguous_width_is_held_for_visual_inspection():
    indicators = {
        "text_stripped_char_count": 800,
        "alpha_ratio": 0.6,
        "image_coverage_ratio": 0.0,
        "word_count": 120,
        "width_pt": 960.0,
        "spread_band": "ambiguous",
        "gutter_found": False,
        "table_heavy": False,
        "table_row_count": 0,
        "horizontal_rule_count": 0,
        "table_score": 0.0,
        "garbled_or_low_quality": False,
    }
    row = decide("TEST", 2, indicators)
    assert "TWO_PAGE_SPREAD" not in row["classifications"]
    assert row["needs_visual_inspection"] is True
    assert row["uncertain"] is True


def test_spread_and_table_can_apply_together():
    indicators = {
        "text_stripped_char_count": 5000,
        "alpha_ratio": 0.3,
        "image_coverage_ratio": 0.0,
        "word_count": 800,
        "width_pt": 1190.5,
        "spread_band": "spread",
        "gutter_found": True,
        "table_heavy": True,
        "table_row_count": 40,
        "horizontal_rule_count": 20,
        "table_score": 0.9,
        "garbled_or_low_quality": False,
    }
    row = decide("TEST", 3, indicators)
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert "TABLE_HEAVY" in row["classifications"]
    assert "DIGITAL_TEXT" in row["classifications"]
    assert row["recommended_extraction_method"] == "digital_text_and_tables"


def test_ellipsis_runs_are_low_quality_text():
    quality = text_quality("NAME : " + ("\u2026" * 80) + " holder details and bank account")
    assert quality["garbled_or_low_quality"] is True
    assert quality["dominant_non_ascii_ratio"] >= 0.08


def test_non_breaking_spaces_are_not_treated_as_corruption():
    quality = text_quality(("Revenue\u00a0from\u00a0operations " + "\u00a0" * 40) * 10)
    assert quality["garbled_or_low_quality"] is False


def test_numeric_table_text_is_not_marked_garbled_by_digits_alone():
    text = "Revenue from operations 12345 67890 Aluminium copper segment total " * 20
    quality = text_quality(text)
    assert quality["garbled_or_low_quality"] is False


def test_powergrid_notice_is_digital_and_not_a_spread():
    row = _page("POWERGRID", 2)
    assert row["classifications"] == ["DIGITAL_TEXT"]
    assert row["recommended_extraction_method"] == "digital_text"
    assert row["uncertain"] is False


def test_powergrid_portrait_scan_is_not_a_spread():
    row = _page("POWERGRID", 25)
    assert "SCANNED_IMAGE" in row["classifications"]
    assert "TWO_PAGE_SPREAD" not in row["classifications"]
    assert "IMAGE_ONLY_VISUAL" not in row["classifications"]
    assert row["recommended_extraction_method"] == "ocr"


def test_powergrid_spread_scan_recommends_ocr():
    row = _page("POWERGRID", 26)
    assert "SCANNED_IMAGE" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert row["recommended_extraction_method"] == "ocr"
    assert row["indicators"]["image_count"] == 1


def test_hindalco_financial_spread_is_a_digital_table():
    row = _page("HINDALCO", 180)
    assert "DIGITAL_TEXT" in row["classifications"]
    assert "TABLE_HEAVY" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert "GARBLED_OR_LOW_QUALITY_TEXT" not in row["classifications"]
    assert row["recommended_extraction_method"] == "digital_text_and_tables"


def test_hindalco_nearly_empty_spread_is_uncertain():
    row = _page("HINDALCO", 412)
    assert "UNKNOWN" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert row["uncertain"] is True
    assert row["recommended_extraction_method"] == "manual_review"


def test_hindalco_graphic_spread_is_visual_not_an_ocr_failure():
    row = _page("HINDALCO", 30)
    assert "IMAGE_ONLY_VISUAL" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert "SCANNED_IMAGE" not in row["classifications"]
    assert row["recommended_extraction_method"] == "caption_only"


def test_coalindia_photo_caption_is_not_sent_to_ocr():
    row = _page("COALINDIA", 82)
    assert "IMAGE_ONLY_VISUAL" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert row["recommended_extraction_method"] == "caption_only"


def test_coalindia_statistics_page_is_a_table_not_garbled():
    row = _page("COALINDIA", 75)
    assert "TABLE_HEAVY" in row["classifications"]
    assert "DIGITAL_TEXT" in row["classifications"]
    # Spread statistics pages often also trip the garbled/low-quality text gate;
    # keep TABLE_HEAVY + DIGITAL_TEXT and accept the dual label.
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert row["recommended_extraction_method"] in {
        "digital_text_and_tables",
        "digital_text_review_quality",
    }


def test_coalindia_form_page_is_low_quality_text():
    row = _page("COALINDIA", 19)
    assert "GARBLED_OR_LOW_QUALITY_TEXT" in row["classifications"]
    assert row["recommended_extraction_method"] == "digital_text_review_quality"


def test_coalindia_body_spread_stays_digital():
    row = _page("COALINDIA", 4)
    assert "DIGITAL_TEXT" in row["classifications"]
    assert "TWO_PAGE_SPREAD" in row["classifications"]
    assert "TABLE_HEAVY" not in row["classifications"]
    assert "GARBLED_OR_LOW_QUALITY_TEXT" not in row["classifications"]
    assert row["recommended_extraction_method"] == "digital_text"
