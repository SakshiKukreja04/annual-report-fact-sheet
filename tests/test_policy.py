"""Extraction policy, kind, status, duplicate subsidiaries, and the company overview."""

from src.dedup.resolve import compare, resolve
from src.factsheet.build import build_fact_sheet, describe_company
from src.validate.items import validate_candidate


def _context(quote, page=10):
    return {
        "ticker": "HINDALCO",
        "page_count": 413,
        "native_text": {page: f"Hindalco. {quote}"},
        "ocr_text": {},
        "ocr_only_pages": set(),
        "page_images": {},
    }


def _candidate(name, quote, kind="other", descriptor=None, status="current", entity="company", page=10):
    return {
        "candidate_id": name,
        "name": name,
        "kind": kind,
        "descriptor": descriptor or name,
        "entity": entity,
        "status": status,
        "revenue_share_pct": None,
        "sources": [{"pdf_page": page, "printed_page": "1", "quote": quote}],
        "supporting_figures": [],
    }


def _item(name, kind, entity, descriptor, status="current", sources=None):
    return {
        "name": name,
        "kind": kind,
        "entity": entity,
        "descriptor": descriptor,
        "status": status,
        "revenue_share_pct": None,
        "sources": sources or [{"pdf_page": 1, "printed_page": "1", "quote": name}],
        "supporting_figures": [],
    }


def test_agm_speaker_registration_is_rejected():
    quote = "Members may complete speaker registration for the AGM"
    result = validate_candidate(_candidate("Speaker registration", quote), _context(quote))
    assert result["outcome"] == "rejected"
    assert "administrative" in result["rejection_reasons"]


def test_cost_audit_and_e_voting_are_rejected():
    audit = "The Board approved the cost audit of the cost records"
    voting = "Shareholders may cast votes by e-voting through InstaVote"
    audit_result = validate_candidate(_candidate("Cost audit", audit), _context(audit))
    voting_result = validate_candidate(_candidate("E-voting", voting, page=11), _context(voting, page=11))
    assert audit_result["outcome"] == "rejected"
    assert voting_result["outcome"] == "rejected"
    assert "administrative" in audit_result["rejection_reasons"]
    assert "administrative" in voting_result["rejection_reasons"]


def test_a_community_programme_is_kept():
    quote = "The Aditya Birla Centre for Community Initiatives runs a community school programme"
    result = validate_candidate(
        _candidate(
            "Aditya Birla Centre for Community Initiatives",
            quote,
            kind="philanthropy",
            descriptor="community school programme",
        ),
        _context(quote),
    )
    assert result["outcome"] == "validated"
    assert result["candidate"]["kind"] == "philanthropy"


def test_an_operating_smelter_is_kept_as_an_asset():
    quote = "The Aditya aluminium smelter produced aluminium during the year"
    result = validate_candidate(
        _candidate("Aditya aluminium smelter", quote, kind="asset", descriptor="aluminium smelter"),
        _context(quote),
    )
    assert result["outcome"] == "validated"
    assert result["candidate"]["kind"] == "asset"
    assert result["candidate"]["status"] == "current"


def test_a_reporting_capital_is_not_an_asset_and_training_is_not_a_laboratory():
    capital = "The Financial Capital section describes the reporting framework"
    training = "The risk management training covered the annual process"
    capital_result = validate_candidate(_candidate("Financial Capital", capital, kind="asset"), _context(capital))
    training_result = validate_candidate(
        _candidate("Risk management training", training, kind="research_platform", descriptor="risk management training"),
        _context(training),
    )
    footprint = "Novelis global operations span plants in several regions"
    footprint_result = validate_candidate(
        _candidate("Novelis global operations", footprint, kind="channel", descriptor="global operations"),
        _context(footprint),
    )
    assert capital_result["outcome"] == "rejected"
    assert "reporting_framework" in capital_result["rejection_reasons"]
    assert training_result["candidate"]["kind"] == "other"
    assert footprint_result["candidate"]["kind"] == "subsidiary"


def test_a_product_is_not_labeled_as_the_whole_business():
    line = "The aluminium business produces primary metal"
    product = "Battery foil is produced for customers"
    line_result = validate_candidate(
        _candidate("Aluminium business", line, kind="product_or_service", descriptor="aluminium business"),
        _context(line),
    )
    product_result = validate_candidate(
        _candidate("Battery foil", product, kind="business_line", descriptor="battery foil"),
        _context(product),
    )
    assert line_result["candidate"]["kind"] == "business_line"
    assert product_result["candidate"]["kind"] == "product_or_service"


def test_a_proposed_plant_is_planned_and_an_operating_plant_is_current():
    future = "The company plans to commission the new foil plant"
    present = "The Aditya aluminium smelter produced aluminium during the year"
    future_result = validate_candidate(
        _candidate("Foil plant", future, kind="asset", descriptor="foil plant", status="current"),
        _context(future),
    )
    present_result = validate_candidate(
        _candidate("Aditya aluminium smelter", present, kind="asset", descriptor="aluminium smelter", status="planned"),
        _context(present),
    )
    assert future_result["candidate"]["status"] == "planned"
    assert present_result["candidate"]["status"] == "current"
    underway = "The copper recycling facility works are progressing well"
    underway_result = validate_candidate(
        _candidate("Copper recycling facility", underway, kind="other", descriptor="recycling facility", status="current"),
        _context(underway),
    )
    assert underway_result["candidate"]["kind"] == "asset"
    assert underway_result["candidate"]["status"] == "planned"


def test_repeated_novelis_mentions_merge_and_keep_three_quotes():
    left = _item(
        "Novelis Inc.",
        "subsidiary",
        "Novelis Inc.",
        "wholly owned subsidiary",
        sources=[
            {"pdf_page": 1, "printed_page": "1", "quote": "Novelis Inc. is the wholly owned subsidiary"},
            {"pdf_page": 2, "printed_page": "2", "quote": "Novelis Inc. rolls aluminium"},
        ],
    )
    right = _item(
        "Novelis",
        "subsidiary",
        "Novelis",
        "wholly owned subsidiary",
        sources=[
            {"pdf_page": 3, "printed_page": "3", "quote": "Novelis operates rolling plants"},
            {"pdf_page": 4, "printed_page": "4", "quote": "Novelis is the wholly owned subsidiary for rolled products"},
        ],
    )
    assert compare(left, right, "HINDALCO")["decision"] == "merged"
    merged = resolve([left, right], "HINDALCO")["merged"][0]
    assert len(merged["sources"]) == 3


def test_similar_product_names_stay_distinct():
    foil = _item("Indigenous battery foil", "product_or_service", "company", "battery foil")
    enclosure = _item("Battery enclosure", "product_or_service", "company", "battery enclosure")
    decision = compare(foil, enclosure, "HINDALCO")
    assert decision["decision"] != "merged"
    result = resolve([foil, enclosure], "HINDALCO")
    assert result["merged_count"] == 0


def test_overview_describes_the_business_and_does_not_list_item_names():
    source = (
        "Hindalco is an aluminium and copper producer. "
        "The Novelis segment operates rolling plants. "
        "Copper operations produce cathodes."
    )
    one_liner, summary = describe_company(source)
    assert one_liner.count(".") == 1
    assert "aluminium" in one_liner.casefold()
    assert "Speaker registration" not in one_liner
    assert "segment" in summary.casefold()
    record = {
        "outcome": "validated",
        "candidate": _candidate("Speaker registration", "Members may complete speaker registration for the AGM"),
    }
    built = build_fact_sheet("HINDALCO", [record], overview_text=source)
    assert built["sheet"]["one_liner"] == one_liner
    assert "Speaker registration" not in built["sheet"]["one_liner"]
    assert "Speaker registration" not in built["sheet"]["summary"]
    assert "items kept" not in built["sheet"]["one_liner"]
