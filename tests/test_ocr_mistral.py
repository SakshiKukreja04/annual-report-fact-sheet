"""Mistral OCR layer: cache, page merge, folio consensus, check_factsheet grounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extract.select import page_text
from src.ocr_mistral import (
    detect_folio_from_markdown,
    infer_printed_folios,
    merge_mistral_into_pages,
    mistral_blocks_for_page,
    ocr_pdf_pages,
    page_paths,
    parse_page_spec,
)
from src.pages.assemble import preferred_ocr_source, text_block
from src.validate.quotes import resolve_quote_on_page


def test_parse_page_spec_ranges():
    assert parse_page_spec("361-362,10,12-13") == [361, 362, 10, 12, 13]


def test_cache_skips_existing_pages(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ocr_mistral.OCR_MISTRAL_DIR", tmp_path)
    monkeypatch.setattr("src.ocr_mistral.mistral_dir", lambda ticker: tmp_path / ticker.upper())
    ticker = "POWERGRID"
    json_path, md_path = page_paths(ticker, 361)
    json_path.parent.mkdir(parents=True)
    json_path.write_text(json.dumps({"markdown": "hello", "pdf_page": 361}), encoding="utf-8")
    md_path.write_text("hello\n", encoding="utf-8")

    def _boom(*_a, **_k):
        raise AssertionError("API must not be called for cached pages")

    monkeypatch.setattr("src.ocr_mistral._client", _boom)
    result = ocr_pdf_pages(ticker, [361], force=False, pdf_path=Path("unused.pdf"))
    assert result["cached"] == [361]
    assert result["written"] == []


def test_merge_into_page_json_selects_mistral(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ocr_mistral.OCR_MISTRAL_DIR", tmp_path / "ocr_mistral")
    monkeypatch.setattr("src.ocr_mistral.PAGE_DIR", tmp_path / "pages")
    monkeypatch.setattr("src.ocr_mistral.mistral_dir", lambda ticker: tmp_path / "ocr_mistral" / ticker.upper())

    ticker = "POWERGRID"
    pdf_page = 249
    mist_json, mist_md = page_paths(ticker, pdf_page)
    mist_json.parent.mkdir(parents=True)
    markdown = "Related Party Disclosures\nPOWERGRID Khavda RE Transmission Limited\n"
    mist_json.write_text(
        json.dumps({"ticker": ticker, "pdf_page": pdf_page, "markdown": markdown, "raw_page": {}}),
        encoding="utf-8",
    )
    mist_md.write_text(markdown, encoding="utf-8")

    page_dir = tmp_path / "pages" / ticker
    page_dir.mkdir(parents=True)
    page = {
        "ticker": ticker,
        "pdf_page": pdf_page,
        "printed_page": None,
        "page_type": ["SCANNED_IMAGE"],
        "content_status": "recoverable",
        "text_blocks": [
            text_block("garbled xx", [0, 0, 10, 10], "tesseract", "FULL", 0, selected=True),
        ],
        "tables": [],
        "captions": [],
        "images": [],
        "extraction_methods": ["tesseract"],
        "quality_metadata": {"garbled_text_layer": True, "ocr_recoverability": {"FULL": "RECOVERABLE"}},
        "warnings": [],
    }
    (page_dir / f"p{pdf_page:04d}.json").write_text(json.dumps(page), encoding="utf-8")

    result = merge_mistral_into_pages(ticker, [pdf_page])
    assert result["updated"] == 1
    updated = json.loads((page_dir / f"p{pdf_page:04d}.json").read_text(encoding="utf-8"))
    sources = {b["source"] for b in updated["text_blocks"] if b.get("selected")}
    assert sources == {"ocr_mistral"}
    assert "ocr_mistral" in updated["extraction_methods"]
    text = page_text(updated)
    assert "POWERGRID Khavda" in text


def test_folio_inference_refuses_when_neighbours_disagree(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ocr_mistral.OCR_MISTRAL_DIR", tmp_path / "ocr_mistral")
    monkeypatch.setattr("src.ocr_mistral.PAGE_DIR", tmp_path / "pages")
    monkeypatch.setattr("src.ocr_mistral.mistral_dir", lambda ticker: tmp_path / "ocr_mistral" / ticker.upper())

    ticker = "POWERGRID"
    mist_root = tmp_path / "ocr_mistral" / ticker
    page_root = tmp_path / "pages" / ticker
    mist_root.mkdir(parents=True)
    page_root.mkdir(parents=True)

    # Neighbouring pages with inconsistent offsets (printed - pdf).
    samples = {
        10: 100,  # offset 90
        11: 50,  # offset 39 — disagree
        12: 200,  # offset 188 — disagree
    }
    for pdf_page, printed in samples.items():
        md = f"Header\nBody text about subsidiaries\n{printed}\n"
        (mist_root / f"page_{pdf_page:03d}.json").write_text(
            json.dumps({"markdown": md, "pdf_page": pdf_page}),
            encoding="utf-8",
        )
        (mist_root / f"page_{pdf_page:03d}.md").write_text(md, encoding="utf-8")
        (page_root / f"p{pdf_page:04d}.json").write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "pdf_page": pdf_page,
                    "printed_page": None,
                    "text_blocks": [],
                    "quality_metadata": {},
                }
            ),
            encoding="utf-8",
        )

    report = infer_printed_folios(ticker, [10, 11, 12])
    assert report["filled"] == 0
    assert report["refused"] >= 3
    for pdf_page in samples:
        page = json.loads((page_root / f"p{pdf_page:04d}.json").read_text(encoding="utf-8"))
        assert page["printed_page"] is None


def test_folio_inference_fills_consistent_run(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ocr_mistral.OCR_MISTRAL_DIR", tmp_path / "ocr_mistral")
    monkeypatch.setattr("src.ocr_mistral.PAGE_DIR", tmp_path / "pages")
    monkeypatch.setattr("src.ocr_mistral.mistral_dir", lambda ticker: tmp_path / "ocr_mistral" / ticker.upper())

    ticker = "POWERGRID"
    mist_root = tmp_path / "ocr_mistral" / ticker
    page_root = tmp_path / "pages" / ticker
    mist_root.mkdir(parents=True)
    page_root.mkdir(parents=True)
    # offset = 5 for three consecutive pages
    for pdf_page in (20, 21, 22):
        printed = pdf_page + 5
        md = f"Title\nContent\n{printed}\n"
        (mist_root / f"page_{pdf_page:03d}.json").write_text(
            json.dumps({"markdown": md, "pdf_page": pdf_page}),
            encoding="utf-8",
        )
        (page_root / f"p{pdf_page:04d}.json").write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "pdf_page": pdf_page,
                    "printed_page": None,
                    "text_blocks": [],
                    "quality_metadata": {},
                }
            ),
            encoding="utf-8",
        )

    report = infer_printed_folios(ticker, [20, 21, 22])
    assert report["filled"] == 3
    for pdf_page in (20, 21, 22):
        page = json.loads((page_root / f"p{pdf_page:04d}.json").read_text(encoding="utf-8"))
        assert page["printed_page"] == str(pdf_page + 5)


def test_detect_folio_from_markdown_header_footer():
    assert detect_folio_from_markdown("Annual Report\nBody\n245\n") == "245"
    assert detect_folio_from_markdown("Page 12\nBody only") == "12"
    assert detect_folio_from_markdown("no numbers here") is None


def test_preferred_ocr_source_ranks_mistral_first():
    blocks = [
        text_block("a", [0, 0, 1, 1], "tesseract", "FULL", 0),
        text_block("b", [0, 0, 1, 1], "ocr_mistral", "FULL", 1),
        text_block("c", [0, 0, 1, 1], "paddleocr", "FULL", 2),
    ]
    assert preferred_ocr_source(blocks) == "ocr_mistral"


def test_check_factsheet_accepts_ocr_mistral_quotes():
    """Same grounding path as check 5: page_text from selected ocr_mistral blocks."""
    quote = "POWERGRID Unchahar Transmission Limited"
    page = {
        "text_blocks": [
            text_block(
                f"List of Subsidiaries includes {quote} as a wholly owned subsidiary.",
                [0, 0, 100, 20],
                "ocr_mistral",
                "FULL",
                0,
                selected=True,
            ),
            text_block("garbled tess", [0, 0, 10, 10], "tesseract", "FULL", 1, selected=False),
        ]
    }
    text = page_text(page)
    assert quote in text
    verdict = resolve_quote_on_page(quote, text)
    assert verdict.get("accepted") is True


def test_mistral_blocks_for_page(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ocr_mistral.OCR_MISTRAL_DIR", tmp_path)
    monkeypatch.setattr("src.ocr_mistral.mistral_dir", lambda ticker: tmp_path / ticker.upper())
    ticker = "POWERGRID"
    json_path, md_path = page_paths(ticker, 1)
    json_path.parent.mkdir(parents=True)
    md = "Para one.\n\nPara two Limited.\n"
    json_path.write_text(json.dumps({"markdown": md, "pdf_page": 1}), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    blocks = mistral_blocks_for_page(ticker, 1)
    assert len(blocks) >= 2
    assert all(b["source"] == "ocr_mistral" for b in blocks)
