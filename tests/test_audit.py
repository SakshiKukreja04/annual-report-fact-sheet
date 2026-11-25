"""Tests for the stage 1 page audit."""

from pathlib import Path

import pymupdf
import pytest

from src.config.paths import INPUT_DIR, PROJECT_ROOT
from src.ingestion.audit import (
    EXPECTED_PAGE_COUNTS,
    NEARLY_EMPTY_TEXT_CHARS,
    audit_document,
    audit_page,
    is_empty_or_nearly_empty,
    locate_pdfs,
    median,
    nearly_empty_percent,
    page_orientation,
    run_audit,
    usable_text_present,
)


def test_project_root_contains_requirements():
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "data" / "input").is_dir()


def test_orientation_uses_displayed_dimensions():
    assert page_orientation(1190.5, 841.9) == "landscape"
    assert page_orientation(595.3, 841.9) == "portrait"
    assert page_orientation(500.0, 500.2) == "square"


def test_usable_text_flag_is_only_a_length_check():
    assert usable_text_present(NEARLY_EMPTY_TEXT_CHARS) is True
    assert usable_text_present(NEARLY_EMPTY_TEXT_CHARS - 1) is False
    assert is_empty_or_nearly_empty(0) is True
    assert is_empty_or_nearly_empty(49) is True
    assert is_empty_or_nearly_empty(50) is False


def test_nearly_empty_percent_and_median():
    assert nearly_empty_percent([0, 10, 80, 100]) == 50.0
    assert nearly_empty_percent([]) == 0.0
    assert median([1, 3, 10]) == 3
    assert median([1, 2, 3, 4]) == 2.5


def test_input_pdfs_are_located_without_using_the_shell_cwd():
    found = locate_pdfs(INPUT_DIR)
    assert set(found) == set(EXPECTED_PAGE_COUNTS)
    for ticker, path in found.items():
        assert path.is_file()
        assert path.parent.name == ticker


def test_page_counts_match_the_assignment():
    found = locate_pdfs(INPUT_DIR)
    for ticker, expected in EXPECTED_PAGE_COUNTS.items():
        with pymupdf.open(found[ticker]) as document:
            assert document.page_count == expected


def test_powergrid_scanned_page_counts_painted_images_not_shared_resources():
    pdf_path = locate_pdfs(INPUT_DIR)["POWERGRID"]
    with pymupdf.open(pdf_path) as document:
        page = document[25]  # PDF page 26, the first full spread
        row = audit_page(page, "POWERGRID")
    assert row["pdf_page"] == 26
    assert row["orientation"] == "landscape"
    assert row["text_char_count"] == 0
    assert row["image_count"] == 1
    assert row["usable_text_present"] is False


def test_powergrid_notice_page_has_preliminary_text():
    pdf_path = locate_pdfs(INPUT_DIR)["POWERGRID"]
    with pymupdf.open(pdf_path) as document:
        row = audit_page(document[1], "POWERGRID")
    assert row["pdf_page"] == 2
    assert row["orientation"] == "portrait"
    assert row["text_stripped_char_count"] >= NEARLY_EMPTY_TEXT_CHARS
    assert row["usable_text_present"] is True


def test_full_audit_writes_one_record_per_page(tmp_path: Path):
    summary = run_audit(input_dir=INPUT_DIR, output_dir=tmp_path)
    assert summary["all_page_counts_match_expected"] is True
    assert summary["python"].startswith("3.12")
    for company in summary["companies"]:
        ticker = company["ticker"]
        assert company["page_count"] == EXPECTED_PAGE_COUNTS[ticker]
        assert company["page_count_matches_expected"] is True
        lines = (tmp_path / f"{ticker}_pages.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == company["page_count"]
        assert company["empty_or_nearly_empty_pct"] == pytest.approx(
            round(100.0 * company["empty_or_nearly_empty_pages"] / company["page_count"], 2)
        )
