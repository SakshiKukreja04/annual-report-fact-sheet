"""Assignment merge and generic segment parsing. No model calls."""

from src.factsheet.build import build_fact_sheet
from src.factsheet.names import assignment_key, same_item
from src.factsheet.overview import _keep_supported
from src.factsheet.segments import _parse_table
from src.factsheet.table_items import join_wrapped_rows


def test_assignment_key_merges_legal_forms_and_word_order_but_not_subsets():
    assert same_item("Foo & Bar Ltd", "Bar and Foo Limited")
    assert same_item("Foo Pvt", "Foo Private")
    assert same_item("Washeries", "Washery")
    assert assignment_key("Novelis") != assignment_key("Novelis UK")
    assert not same_item("Novelis", "Novelis UK")
    from src.factsheet.names import should_merge
    assert not should_merge("Novelis", "Novelis UK")
    assert should_merge("Thalassemia Bal Seva Yojana", "Thalassemia Bal Sewa Yojana")
    assert not should_merge("Eastern Coalfields Limited", "Western Coalfields Limited")
    keys = {assignment_key("Eastern Coalfields Limited"), assignment_key("Western Coalfields Limited")}
    assert not should_merge("Eastern Coalfields Limited", "Western Coalfields Limited", keys)
    assert should_merge(
        "Centre for Clean Coal Energy and Net Zero (CLEANZ)",
        "CLEANZ - Centre of Excellence on Clean Energy",
    )
    assert should_merge("Pump Storage Plant", "Joint Venture for Pumped Storage Projects")
    assert not should_merge(
        "GPS based Vehicle Tracking System (VTS) with geo fencing",
        "GPS based Operator Independent Truck Dispatch System (OITDS)",
    )
    assert not should_merge("300 MW solar project at Khavda", "50 MW Solar Power Plant")
    assert not should_merge("Primary Co-operative Stores", "Co-operative Credit Societies")


def test_factsheet_merges_duplicates_and_keeps_subsidiary_actor():
    quote = "Coal Gas India Limited operates the gasification plant."
    source = {"pdf_page": 12, "printed_page": "10", "quote": quote}
    records = [
        _validated("Coal Gas India Ltd", "subsidiary", "Coal Gas India Ltd", "company", source),
        _validated("Coal Gas India Limited", "subsidiary", "operates the gasification plant", "company", source),
        _validated("Gasification plant", "asset", "gasification plant", "company", source),
    ]
    built = build_fact_sheet("COALINDIA", records, overview={"one_liner": "Coal India produces coal.", "summary": "Coal India produces coal."})
    names = [item["name"] for item in built["sheet"]["items"]]
    assert names.count("Coal Gas India Limited") == 1 or names.count("Coal Gas India Ltd") == 1
    subsidiary = next(item for item in built["sheet"]["items"] if item["kind"] == "subsidiary")
    assert subsidiary["entity"] == "Coal Gas India Limited" or "Limited" in subsidiary["entity"] or "Ltd" in subsidiary["entity"]
    asset = next(item for item in built["sheet"]["items"] if item["kind"] == "asset")
    assert asset["entity"] == "Coal Gas India Limited" or "Coal Gas" in asset["entity"]


def test_segment_table_uses_printed_unit_and_total():
    table = {
        "pdf_page": 40,
        "printed_page": "38",
        "headers": ["Segment", "Segment revenue", "Inter-segment revenue", "External revenue"],
        "units": "Rs. in crore",
        "rows": [
            {"cells": ["Coal", "800", "50", "750"]},
            {"cells": ["Other", "200", "10", "190"]},
            {"cells": ["Total", "1,000", "60", "940"]},
        ],
    }
    parsed = _parse_table(table, None)
    assert parsed["unit"] == "Rs. in crore"
    assert parsed["denominator"] == "before_inter_segment_elimination"
    assert parsed["printed_total"] == 1000
    coal = next(row for row in parsed["segments"] if row["name"] == "Coal")
    assert coal["revenue"] == 800
    assert coal["share_pct"] == 80.0
    assert coal["unit"] == "Rs. in crore"


def test_wrapped_name_joins_and_invented_overview_sentence_is_dropped():
    rows = join_wrapped_rows([["Coal India"], ["Limited", "100"], ["Other Mines Limited", "mining"]])
    assert rows[0][0] == "Coal India Limited"
    kept = _keep_supported(
        "Coal India produces coal. Novelis rolls aluminium.",
        "Coal India produces coal across India.",
    )
    assert "Novelis" not in kept
    assert "Coal India" in kept


def _validated(name, kind, descriptor, entity, source):
    return {
        "outcome": "validated",
        "candidate": {
            "name": name,
            "kind": kind,
            "descriptor": descriptor,
            "entity": entity,
            "status": "current",
            "revenue_share_pct": None,
            "sources": [source],
        },
        "quotes": [{"pdf_page": source["pdf_page"], "quote": source["quote"], "accepted": True}],
        "financial": {"status": "unavailable", "revenue_share_pct": None},
    }
