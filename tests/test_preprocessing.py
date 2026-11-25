"""Tests for spread splitting and the grayscale preprocessing pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf

from src.ingestion.audit import locate_pdfs
from src.preprocessing.benchmark import BENCHMARK_PAGES, benchmark_page_count
from src.preprocessing.pipeline import (
    HALF_FULL,
    HALF_LEFT,
    HALF_RIGHT,
    build_variants,
    find_gutter_x,
    is_two_page_spread,
    process_page,
    should_deskew,
    speckle_ratio,
)


def test_ordinary_landscape_and_portrait_are_not_spreads():
    assert is_two_page_spread(842, 595) is False
    assert is_two_page_spread(595.3, 841.9) is False
    assert is_two_page_spread(1190.5, 841.9) is True


def test_bright_center_is_used_as_the_gutter():
    image = np.full((80, 400), 30, dtype=np.uint8)
    image[:, 196:204] = 255
    cut, method = find_gutter_x(image)
    assert method == "gutter"
    assert 180 <= cut <= 220


def test_deskew_ignores_tiny_and_implausible_angles():
    assert should_deskew(0.2) is False
    assert should_deskew(1.2) is True
    assert should_deskew(40) is False


def test_ruled_lines_are_not_treated_as_speckles():
    ruled = np.full((80, 220), 240, dtype=np.uint8)
    ruled[40, :] = 0
    assert speckle_ratio(ruled) < 0.008
    salt = np.full((80, 220), 210, dtype=np.uint8)
    salt[::6, ::6] = 0
    salt[3::6, 4::6] = 255
    assert speckle_ratio(salt) >= 0.008


def test_clean_gradient_is_not_speckled_or_binary():
    gradient = np.tile(np.linspace(20, 220, 200, dtype=np.uint8), (80, 1))
    assert speckle_ratio(gradient) < 0.008
    variants = build_variants(cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB))
    full = variants["full"]
    assert full.shape[1] == gradient.shape[1] * 5
    assert len(np.unique(full)) > 2
    assert "median_3x3" not in variants["meta"]["applied_operations"]
    assert "binarize" not in " ".join(variants["meta"]["applied_operations"])
    assert variants["meta"]["deskew_applied"] is False


def test_benchmark_set_is_about_twenty_to_thirty_pages():
    assert 20 <= benchmark_page_count() <= 30
    assert len(BENCHMARK_PAGES["POWERGRID"]) >= 8
    assert len(BENCHMARK_PAGES["COALINDIA"]) >= 4
    assert len(BENCHMARK_PAGES["HINDALCO"]) >= 4


def test_powergrid_portrait_scan_stays_one_full_page(tmp_path: Path):
    path = locate_pdfs()["POWERGRID"]
    with pymupdf.open(path) as document:
        records = process_page(document[24], "POWERGRID", tmp_path)
    assert [record["half"] for record in records] == [HALF_FULL]
    assert records[0]["pdf_page"] == 25
    assert records[0]["scale_factor"] == 5
    assert Path(records[0]["variant_paths"]["D_lanczos5_clahe_sharp"]).is_file()


def test_powergrid_spread_splits_before_upscale_and_keeps_the_page_number(tmp_path: Path):
    path = locate_pdfs()["POWERGRID"]
    with pymupdf.open(path) as document:
        records = process_page(document[25], "POWERGRID", tmp_path)
    halves = {record["half"]: record for record in records}
    assert set(halves) == {HALF_LEFT, HALF_RIGHT}
    assert halves[HALF_LEFT]["pdf_page"] == halves[HALF_RIGHT]["pdf_page"] == 26
    for record in records:
        original_width = record["original_dimensions"][0]
        processed_width = record["processed_dimensions"][0]
        assert processed_width == original_width * 5
        assert "lanczos_5x" in record["applied_operations"]
        assert "clahe" in record["applied_operations"]
        assert "unsharp" in record["applied_operations"]
        assert record["processing_duration_sec"] >= 0
        assert Path(record["output_image_path"]).is_file()
