"""Stage 13 PDF-grounded audit rules for the Hindalco continued sheet."""

import json
from pathlib import Path

from src.factsheet.build import consistency_errors, render_markdown
from src.factsheet.schema import FactSheet
from src.factsheet.stage13_audit import audit_item, merge_survivors, run_audit


def _item(name, kind="subsidiary", entity=None, status="current", descriptor=None, page=10, quote=None):
    quote = quote or name
    return {
        "name": name,
        "kind": kind,
        "descriptor": descriptor or name,
        "entity": entity or (name if kind == "subsidiary" else "company"),
        "status": status,
        "revenue_share_pct": None,
        "sources": [{"pdf_page": page, "printed_page": "1", "quote": quote}],
    }


def _pages(*pairs):
    return {page: text for page, text in pairs}


def test_false_subsidiaries_are_rejected():
    text = "The equity shares are listed on BSE Limited and NSE. MUFG Intime India Private Limited is the registrar."
    pages = _pages((10, text))
    listed = set()
    bse = audit_item(_item("BSE Limited", quote="BSE Limited"), pages, listed, 0)
    transfer = audit_item(_item("MUFG Intime India Private Limited", quote="MUFG Intime India Private Limited"), pages, listed, 1)
    ceat = audit_item(_item("CEAT Limited", quote="CEAT Limited"), _pages((11, "CEAT Limited is a tyre company customer")), listed, 2)
    nestle = audit_item(_item("Nestle India Limited", quote="Nestle India Limited"), _pages((12, "Nestle India Limited buys foil")), listed, 3)
    assert bse["decision"] == "REJECT"
    assert transfer["decision"] == "REJECT"
    assert ceat["decision"] == "REJECT"
    assert nestle["decision"] == "REJECT"
    assert bse["reason"] == "false_subsidiary"


def test_non_company_fragments_are_rejected():
    pages = _pages((10, "Quantity Unit Cash Generating Unit Face Value Per Unit Property Plant and equipment"))
    for name in ("Quantity Unit", "Cash Generating Unit", "Face Value Per Unit", "Property Plant"):
        row = audit_item(_item(name, quote=name), pages, set(), 0)
        assert row["decision"] == "REJECT"
        assert row["reason"] == "non_company_fragment"


def test_administrative_agm_and_cost_audit_are_rejected():
    pages = _pages((10, "Members may use e-voting for the AGM. The Board approved the cost audit."))
    agm = audit_item(_item("E-voting", kind="other", quote="Members may use e-voting for the AGM"), pages, set(), 0)
    audit = audit_item(_item("Cost audit", kind="other", quote="The Board approved the cost audit"), pages, set(), 1)
    assert agm["decision"] == "REJECT"
    assert audit["decision"] == "REJECT"
    assert agm["reason"] == "administrative"


def test_training_is_not_a_business_line_and_environment_is_not_an_asset():
    pages = _pages((10, "The risk management training covered employees. The water positivity programme continued."))
    training = audit_item(
        _item("Risk management training", kind="business_line", quote="The risk management training covered employees"),
        pages,
        set(),
        0,
    )
    water = audit_item(
        _item("Water positivity", kind="asset", quote="The water positivity programme continued"),
        pages,
        set(),
        1,
    )
    assert training["decision"] in {"KEEP", "CORRECT"}
    assert training["corrected_kind"] == "other"
    assert water["corrected_kind"] == "other"


def test_operational_process_is_not_a_research_platform():
    pages = _pages((10, "The casting process methodology improved recovery."))
    row = audit_item(
        _item("Casting process methodology", kind="research_platform", quote="The casting process methodology improved recovery"),
        pages,
        set(),
        0,
    )
    assert row["corrected_kind"] == "other"


def test_unrelated_party_near_subsidiary_word_is_not_kept():
    text = (
        "Novelis Korea Ltd., a subsidiary of Novelis Inc., entered into definitive agreements "
        "with Kobe Steel Ltd. (Kobe), an unrelated party."
    )
    pages = _pages((263, text))
    row = audit_item(_item("Kobe Steel Ltd", page=263, quote="Kobe Steel Ltd"), pages, set(), 0)
    assert row["decision"] == "REVIEW"
    assert row["reason"] == "subsidiary_relationship_unproven"


def test_supported_subsidiary_and_smelter_are_kept():
    listed = {"novelis"}
    pages = _pages(
        (33, "Novelis Inc. is a wholly owned subsidiary. The Aditya aluminium smelter produced metal during the year."),
    )
    novelis = audit_item(_item("Novelis Inc.", page=33, quote="Novelis Inc. is a wholly owned subsidiary"), pages, listed, 0)
    smelter = audit_item(
        _item(
            "Aditya aluminium smelter",
            kind="asset",
            page=33,
            quote="The Aditya aluminium smelter produced metal during the year",
        ),
        pages,
        listed,
        1,
    )
    assert novelis["decision"] in {"KEEP", "CORRECT"}
    assert smelter["decision"] in {"KEEP", "CORRECT"}
    assert smelter["corrected_kind"] == "asset"


def test_invalid_quote_and_missing_page_are_rejected():
    pages = _pages((10, "Only this sentence is on the page."))
    bad = audit_item(_item("Missing quote", kind="other", quote="this quote is not present"), pages, set(), 0)
    missing = audit_item(_item("Gone page", kind="other", page=99, quote="Gone page"), pages, set(), 1)
    assert bad["decision"] == "REJECT"
    assert bad["reason"] == "invalid_or_unsupported_quote"
    assert missing["decision"] == "REJECT"
    assert missing["reason"] == "missing_pdf_page"


def test_novelis_and_csr_duplicates_merge_and_products_stay_apart():
    pages = {
        1: "Novelis Inc. is a wholly owned subsidiary",
        2: "Novelis rolls aluminium products",
        3: "Engineering Better Futures supports community education",
        4: "Engineering Better Futures supports community education",
        5: "Indigenous battery foil is produced for customers",
        6: "Battery enclosure is produced for customers",
    }
    listed = {"novelis"}
    rows = [
        audit_item(_item("Novelis Inc.", page=1, quote="Novelis Inc. is a wholly owned subsidiary"), pages, listed, 0),
        audit_item(_item("Novelis", page=2, quote="Novelis rolls aluminium products"), pages, listed, 1),
        audit_item(
            _item("Engineering Better Futures", kind="philanthropy", page=3, quote="Engineering Better Futures supports community education"),
            pages,
            listed,
            2,
        ),
        audit_item(
            _item("Engineering Better Futures", kind="philanthropy", page=4, quote="Engineering Better Futures supports community education"),
            pages,
            listed,
            3,
        ),
        audit_item(
            _item("Indigenous battery foil", kind="product_or_service", page=5, quote="Indigenous battery foil is produced for customers"),
            pages,
            listed,
            4,
        ),
        audit_item(
            _item("Battery enclosure", kind="product_or_service", page=6, quote="Battery enclosure is produced for customers"),
            pages,
            listed,
            5,
        ),
    ]
    final_items = merge_survivors(rows)
    names = {item["name"] for item in final_items}
    assert sum(row["decision"] == "MERGE" for row in rows) >= 1
    assert "Indigenous battery foil" in names
    assert "Battery enclosure" in names


def test_review_candidates_are_excluded_from_final_sheet(tmp_path: Path):
    continued = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": "x",
        "summary": "y",
        "segments": [],
        "items": [
            _item("Mystery Corp Limited", page=10, quote="Mystery Corp Limited appears once"),
            _item(
                "Aditya aluminium smelter",
                kind="asset",
                page=11,
                quote="The Aditya aluminium smelter produced metal during the year",
            ),
        ],
    }
    pages = _pages(
        (10, "Mystery Corp Limited appears once without any ownership words."),
        (11, "The Aditya aluminium smelter produced metal during the year."),
    )
    listed = set()
    mystery = audit_item(continued["items"][0], pages, listed, 0)
    smelter = audit_item(continued["items"][1], pages, listed, 1)
    assert mystery["decision"] == "REVIEW"
    rows = [mystery, smelter]
    final_items = merge_survivors(rows)
    assert all(item["name"] != "Mystery Corp Limited" for item in final_items)
    sheet = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": "Hindalco produces aluminium.",
        "summary": "Hindalco operates aluminium plants.",
        "segments": [],
        "items": final_items,
    }
    validated = FactSheet.model_validate(sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    assert consistency_errors(validated, markdown) == []
    assert sum(row["decision"] == "REVIEW" for row in rows) == 1
