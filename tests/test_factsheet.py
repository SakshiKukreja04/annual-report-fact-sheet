"""Fact sheets contain only validated items, and the Markdown matches the JSON."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.factsheet.assemble import make_final
from src.factsheet.build import build_fact_sheet, consistency_errors, write_fact_sheet
from src.factsheet.schema import FactSheet

QUOTE = "to conduct the audit of the cost records of the Company for FY 2025-26"


def _record(name="Cost audit of cost records", kind="product_or_service", status="current",
            entity="company", share_status="unavailable", share=None, sources=None,
            outcome="validated", accepted=True, descriptor="audit of the cost records"):
    sources = sources or [{"pdf_page": 13, "printed_page": "12", "quote": QUOTE}]
    return {
        "outcome": outcome,
        "candidate": {
            "name": name,
            "kind": kind,
            "descriptor": descriptor,
            "entity": entity,
            "status": status,
            "revenue_share_pct": 40,
            "supporting_figures": [{"label": "Segment revenue", "value": "100 crore", "pdf_page": 13, "quote": QUOTE}],
            "sources": sources,
        },
        "quotes": [
            {"pdf_page": source["pdf_page"], "quote": source["quote"], "accepted": accepted}
            for source in sources
        ],
        "financial": {"status": share_status, "revenue_share_pct": share, "unit": "INR crore"},
    }


def test_validated_item_keeps_the_quote_and_drops_unprinted_share():
    built = build_fact_sheet("HINDALCO", [_record()])
    sheet = FactSheet.model_validate(built["sheet"]).model_dump(mode="json")
    assert sheet["period"] == "FY25"
    assert sheet["items"] == [{
        "name": "Cost audit of cost records",
        "kind": "product_or_service",
        "descriptor": "audit of the cost records",
        "entity": "company",
        "status": "current",
        "revenue_share_pct": None,
        "sources": [{"pdf_page": 13, "printed_page": "12", "quote": QUOTE}],
    }]
    assert "100 crore" not in json.dumps(sheet)
    assert sheet["items"][0]["sources"][0]["quote"] == QUOTE
    assert consistency_errors(sheet, built["markdown"]) == []
    assert "PDF page 13, printed page 12" in built["markdown"]


def test_rejected_quote_review_item_and_bad_page_do_not_pass():
    rejected_quote = _record(sources=[
        {"pdf_page": 13, "printed_page": "12", "quote": QUOTE},
        {"pdf_page": 99, "printed_page": "1", "quote": "not on this page"},
    ])
    rejected_quote["quotes"][1]["accepted"] = False
    review = _record(name="Novelis", entity="Novelis Inc.", outcome="review")
    bad_page = _record(name="Missing page", sources=[{"pdf_page": 0, "printed_page": "1", "quote": QUOTE}])
    built = build_fact_sheet("HINDALCO", [rejected_quote, review, bad_page])
    assert [item["name"] for item in built["sheet"]["items"]] == ["Cost audit of cost records"]
    assert built["sheet"]["items"][0]["sources"] == [{"pdf_page": 13, "printed_page": "12", "quote": QUOTE}]
    assert {entry["reason"] for entry in built["omitted"]} == {"not_validated", "no_valid_source"}


def test_more_than_three_sources_are_omitted():
    sources = [
        {"pdf_page": page, "printed_page": str(page), "quote": f"quote {page}"}
        for page in (2, 3, 4, 5)
    ]
    built = build_fact_sheet("COALINDIA", [_record(sources=sources)])
    assert built["sheet"]["items"] == []
    assert built["omitted"] == [{"name": "Cost audit of cost records", "reason": "too_many_sources"}]


def test_printed_share_and_segment_unit_come_from_validation():
    record = _record(name="Aluminium", kind="segment", share_status="printed", share="12.50")
    built = build_fact_sheet("HINDALCO", [record])
    item = built["sheet"]["items"][0]
    assert item["revenue_share_pct"] == 12.5
    assert built["sheet"]["segments"] == [{
        "name": "Aluminium",
        "revenue": None,
        "unit": None,
        "share_pct": 12.5,
    }]
    calculated = _record(
        name="Copper",
        kind="segment",
        share_status="calculated_from_printed_figures",
        share="10.00",
    )
    calculated["financial"]["unit"] = "INR crore"
    both = build_fact_sheet("HINDALCO", [record, calculated])
    copper = next(segment for segment in both["sheet"]["segments"] if segment["name"] == "Copper")
    assert copper == {"name": "Copper", "revenue": None, "unit": "INR crore", "share_pct": 10.0}


def test_empty_company_is_valid_and_markdown_matches():
    built = build_fact_sheet("POWERGRID", [])
    sheet = FactSheet.model_validate(built["sheet"])
    assert sheet.items == []
    assert sheet.segments == []
    assert "no validated items" in sheet.one_liner
    assert consistency_errors(built["sheet"], built["markdown"]) == []


def test_changed_markdown_fails_the_consistency_check():
    built = build_fact_sheet("HINDALCO", [_record()])
    changed = built["markdown"].replace("### Cost audit of cost records", "### Invented product")
    assert consistency_errors(built["sheet"], changed)


def test_a_failed_write_leaves_the_good_fact_sheet(tmp_path):
    built = build_fact_sheet("HINDALCO", [_record()])
    first = write_fact_sheet("HINDALCO", built, tmp_path)
    good = Path(first["json"]).read_bytes()
    with pytest.raises(FileExistsError):
        write_fact_sheet("HINDALCO", build_fact_sheet("HINDALCO", []), tmp_path)
    assert Path(first["json"]).read_bytes() == good
    assert not list(Path(first["json"]).parent.glob(".*.tmp"))


def test_a_continued_sheet_leaves_the_saved_files(tmp_path):
    built = build_fact_sheet("HINDALCO", [_record()])
    first = write_fact_sheet("HINDALCO", built, tmp_path)
    original = Path(first["json"]).read_bytes()
    markdown = Path(first["markdown"]).read_bytes()
    written = write_fact_sheet("HINDALCO", build_fact_sheet("HINDALCO", []), tmp_path, force=True, stem="HINDALCO.continued")
    assert Path(first["json"]).read_bytes() == original
    assert Path(first["markdown"]).read_bytes() == markdown
    assert Path(written["json"]).name == "HINDALCO.continued.json"
    with pytest.raises(FileExistsError):
        write_fact_sheet("HINDALCO", built, tmp_path, force=True, stem="HINDALCO.v0")
    with pytest.raises(FileExistsError):
        write_fact_sheet("HINDALCO", built, tmp_path, force=True, stem="HINDALCO.post")


def test_existing_files_are_not_overwritten(tmp_path):
    built = build_fact_sheet("HINDALCO", [_record()])
    first = write_fact_sheet("HINDALCO", built, tmp_path)
    original = Path(first["json"]).read_text(encoding="utf-8")
    replacement = build_fact_sheet("HINDALCO", [])
    with pytest.raises(FileExistsError):
        write_fact_sheet("HINDALCO", replacement, tmp_path)
    assert Path(first["json"]).read_text(encoding="utf-8") == original
    write_fact_sheet("HINDALCO", replacement, tmp_path, force=True)
    saved = json.loads(Path(first["json"]).read_text(encoding="utf-8"))
    assert saved["items"] == []


def test_postprocess_drops_an_agm_item_and_merges_novelis():
    from src.factsheet.postprocess import drop_items, merge_items

    items = [
        {"name": "InstaVote voting platform", "descriptor": "online voting", "kind": "channel", "sources": [{"pdf_page": 28, "quote": "InstaVote"}]},
        {"name": "Novelis Inc.", "descriptor": "rolled aluminium", "kind": "subsidiary", "sources": [{"pdf_page": 33, "quote": "Novelis Inc."}]},
        {"name": "Novelis", "descriptor": "wholly owned subsidiary Novelis", "kind": "subsidiary", "sources": [{"pdf_page": 38, "quote": "Our wholly owned subsidiary, Novelis"}]},
    ]
    kept, dropped = drop_items(items, set())
    assert [item["name"] for item in kept] == ["Novelis Inc.", "Novelis"]
    assert dropped[0]["reason"] == "excluded_topic"
    merged, folded = merge_items(kept)
    assert folded == 1
    assert len(merged) == 1
    assert len(merged[0]["sources"]) == 2


def test_make_final_writes_a_slice_and_drops_a_quote_that_is_not_on_the_page(tmp_path: Path):
    page = {
        "pdf_page": 13,
        "text_blocks": [{"text": QUOTE, "selected": True, "reading_order": 0}],
    }
    kept = {
        "name": "Cost audit",
        "kind": "product_or_service",
        "descriptor": "audit of the cost records",
        "entity": "company",
        "status": "current",
        "revenue_share_pct": None,
        "sources": [{"pdf_page": 13, "printed_page": "12", "quote": QUOTE}],
    }
    missing = {**kept, "name": "Missing quote", "sources": [{"pdf_page": 13, "printed_page": "12", "quote": "not on the page"}]}
    result = make_final("HINDALCO", [kept, missing], [page], tmp_path / "out", tmp_path / "report", gaps=[{"pdf_page": 2}])
    assert result["items"] == 1
    assert result["rejected_quotes"] == 1
    assert result["schema_ok"] is True
    saved = json.loads((tmp_path / "out" / "HINDALCO" / "HINDALCO.json").read_text(encoding="utf-8"))
    assert saved["items"][0]["sources"][0]["quote"] == QUOTE
    assert (tmp_path / "report" / "cost.json").is_file()
    assert (tmp_path / "report" / "known_gaps.json").is_file()


def test_schema_rejects_an_extra_field_and_a_fourth_source():
    built = build_fact_sheet("HINDALCO", [_record()])
    broken = dict(built["sheet"])
    broken["notes"] = "from memory"
    with pytest.raises(ValidationError):
        FactSheet.model_validate(broken)
    item = dict(built["sheet"]["items"][0])
    item["sources"] = item["sources"] * 4
    with pytest.raises(ValidationError):
        FactSheet.model_validate({**built["sheet"], "items": [item]})
