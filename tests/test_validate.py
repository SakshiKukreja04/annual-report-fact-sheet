"""Quote, entity, and financial checks. The saved quote is not rewritten."""

from decimal import Decimal
from pathlib import Path

from src.validate.financial import calculate_share, check_printed_share
from src.validate.items import validate_candidate
from src.validate.quotes import fold_whitespace, locate_quote, repair_named_quote, snap_quote


def _context(text, page=10, page_count=20, ocr=None, ocr_only=None):
    return {
        "ticker": "HINDALCO",
        "page_count": page_count,
        "native_text": {page: text},
        "ocr_text": ocr or {},
        "ocr_only_pages": set(ocr_only or []),
        "page_images": {page: "page.png"},
    }


def _candidate(quote, page=10, entity="company", descriptor="aluminium rolling", share=None, name="Aluminium"):
    return {
        "candidate_id": "c1",
        "name": name,
        "kind": "product_or_service",
        "descriptor": descriptor,
        "entity": entity,
        "status": "current",
        "revenue_share_pct": share,
        "sources": [{"pdf_page": page, "printed_page": "4", "quote": quote}],
        "supporting_figures": [{"label": "Segment revenue", "value": "100 crore", "pdf_page": page, "quote": quote}],
    }


def test_exact_quote_matches_and_is_not_rewritten():
    quote = "The company produces aluminium"
    candidate = _candidate(quote)
    result = validate_candidate(candidate, _context("Intro. The company produces aluminium. End."))
    assert result["outcome"] == "validated"
    assert result["quotes"][0]["match"] == "exact_match"
    assert result["quotes"][0]["verification"] == "native_pdf"
    assert result["quotes"][0]["quote"] == quote
    assert candidate["sources"][0]["quote"] == quote


def test_line_breaks_match_without_changing_the_quote():
    quote = "The company produces aluminium"
    page = "The company\nproduces aluminium"
    assert locate_quote(quote, page) == "normalized"
    result = validate_candidate(candidate := _candidate(quote), _context(page))
    assert result["quotes"][0]["match"] == "normalized_match"
    assert candidate["sources"][0]["quote"] == quote
    assert fold_whitespace(page) != page


def test_snap_quote_uses_the_page_substring_and_drops_a_weak_match():
    page = "The company's smelter produces aluminium."
    snapped = snap_quote("The company\u2019s smelter produces aluminium.", page)
    assert snapped["ratio"] >= 0.9
    assert snapped["quote"] == page
    joined = snap_quote("aluminium sheet", "alu-\nminium sheet")
    assert joined["quote"] is not None
    assert "alu-" in joined["quote"]
    assert snap_quote("shipping schedule for the port", page)["quote"] is None


def test_a_weak_quote_is_replaced_by_the_sentence_that_names_the_item():
    page = "The Board recommended a dividend. Guthrie Recycling Plant ships ingots to Oswego by truck."
    repaired = repair_named_quote("shipping schedule for the port", page, "Guthrie Recycling Plant")
    assert repaired["outcome"] == "reanchored"
    assert repaired["quote"] is not None
    assert "Guthrie" in repaired["quote"]
    assert repaired["quote"] in page
    snapped = repair_named_quote("The Board recommended a dividend.", page, "Guthrie Recycling Plant")
    assert snapped["outcome"] == "repaired"
    assert snapped["mentions_name"] is False
    assert snapped["quote"] == "The Board recommended a dividend."


def test_incorrect_page_and_broken_quote_are_rejected():
    missing = validate_candidate(_candidate("The company produces aluminium", page=99), _context("The company produces aluminium", page_count=20))
    assert missing["outcome"] == "rejected"
    assert missing["quotes"][0]["reason"] == "invalid_page"
    broken = validate_candidate(_candidate("not in the report"), _context("The company produces aluminium"))
    assert broken["outcome"] == "rejected"
    assert broken["quotes"][0]["reason"] == "quote_not_on_page"
    assert broken["quotes"][0]["verification"] == "native_pdf"


def test_ocr_citation_is_not_labeled_as_native_text():
    quote = "Coal India Limited"
    result = validate_candidate(
        _candidate(quote, descriptor="coal india", entity="company"),
        _context(" ", ocr={10: "Letterhead Coal India Limited"}, ocr_only={10}),
    )
    assert result["quotes"][0]["match"] == "ocr_verified_exact"
    assert result["quotes"][0]["verification"] == "ocr"
    assert result["quotes"][0]["page_image"] == "page.png"
    assert result["quotes"][0]["match"] != "exact_match"


def test_unsupported_entity_is_held_for_review():
    quote = "The company produces aluminium"
    result = validate_candidate(
        _candidate(quote, entity="Novelis Inc."),
        _context(quote),
    )
    assert result["outcome"] == "review"
    assert "unsupported_entity" in result["review_reasons"]
    assert result["candidate"]["sources"][0]["quote"] == quote


def test_printed_share_is_kept_and_an_unprinted_share_is_not_calculated():
    printed = check_printed_share("12.5", ["Aluminium contributed 12.5% of revenue"])
    assert printed["status"] == "printed"
    assert Decimal(printed["revenue_share_pct"]) == Decimal("12.5")
    invented = check_printed_share(40, ["Segment revenue was 100 crore"])
    assert invented["revenue_share_pct"] is None
    assert invented["status"] == "not_printed"
    calculated = calculate_share(Decimal("25"), Decimal("200"), "crore", [180])
    assert calculated["revenue_share_pct"] == "12.50"
    assert calculated["numerator"] == "25"
    assert calculated["denominator"] == "200"
    assert calculated["unit"] == "crore"
    assert calculated["source_pages"] == [180]
    refused = calculate_share(Decimal("1"), Decimal("0"), "crore", [180])
    assert refused["revenue_share_pct"] is None
    candidate = _candidate("The company produces aluminium", share=40)
    result = validate_candidate(candidate, _context("The company produces aluminium. Segment revenue was 100 crore."))
    assert result["outcome"] == "review"
    assert result["candidate"]["revenue_share_pct"] is None
    assert result["financial"]["status"] == "not_printed"


def test_real_hindalco_quote_is_on_the_original_pdf():
    pdf = Path("data/input/HINDALCO/HINDALCO_FY25_AR.pdf")
    if not pdf.is_file():
        return
    from src.validate.quotes import read_pdf_page

    quote = "to conduct the audit of the cost records of the Company for FY 2025-26"
    text, reader = read_pdf_page(pdf, 13)
    found = locate_quote(quote, text)
    assert found in {"exact", "normalized"}
    assert reader in {"pymupdf", "pdfplumber"}
    assert quote == "to conduct the audit of the cost records of the Company for FY 2025-26"
