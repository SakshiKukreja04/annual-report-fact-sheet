"""Duplicate groups. Similar names are not enough to merge different activities."""

from src.dedup.normalize import normalize
from src.dedup.resolve import compare, resolve


def _item(name, kind, entity, descriptor, status="current", sources=None, share=None):
    return {
        "name": name,
        "kind": kind,
        "entity": entity,
        "descriptor": descriptor,
        "status": status,
        "revenue_share_pct": share,
        "sources": sources or [{"pdf_page": 1, "printed_page": "1", "quote": name}],
        "supporting_figures": [],
    }


def test_normalization_folds_case_punctuation_and_legal_forms():
    assert normalize("Hindalco Industries LTD.") == normalize("hindalco industries limited")
    assert normalize("Aluminium & Copper") == normalize("aluminium and copper")
    assert normalize("smelters") == normalize("smelter")
    ordered = _item("rolled aluminium products", "product_or_service", "company", "flat rolled metal")
    reordered = _item("aluminium rolled products", "product_or_service", "company", "flat rolled metal")
    assert compare(ordered, reordered)["decision"] == "merged"


def test_minor_name_variations_for_the_same_entity_merge():
    left = _item("Hindalco Ltd", "business_line", "company", "aluminium and copper manufacturing")
    right = _item("Hindalco Limited", "business_line", "company", "manufacturing of aluminium & copper", sources=[
        {"pdf_page": 4, "printed_page": "2", "quote": "Hindalco Limited manufactures aluminium and copper"},
    ])
    assert compare(left, right, "HINDALCO")["decision"] == "merged"
    result = resolve([left, right], "HINDALCO")
    assert result["merged_count"] == 1
    assert result["language_model_used"] is False
    assert result["merged"][0]["sources"]


def test_same_subsidiary_with_different_activities_stays_separate():
    refinery = _item("Utkal Alumina Ltd", "subsidiary", "Utkal Alumina Limited", "operates the alumina refinery")
    mine = _item("Utkal Alumina Limited", "subsidiary", "Utkal Alumina Ltd", "owns the captive bauxite mine")
    assert compare(refinery, mine)["decision"] == "distinct"
    assert compare(refinery, mine)["reason"] == "different_activity"


def test_similar_products_of_different_entities_do_not_merge():
    parent = _item("Aluminium", "product_or_service", "company", "primary metal")
    novelis = _item("Aluminium", "product_or_service", "Novelis Inc.", "rolled products")
    decision = compare(parent, novelis, "HINDALCO")
    assert decision["decision"] == "distinct"
    assert decision["reason"] == "different_entity"


def test_unrelated_assets_are_not_merged_on_a_shared_word():
    aditya = _item("Aditya aluminium smelter", "asset", "company", "smelter at Lapanga")
    mahan = _item("Mahan aluminium smelter", "asset", "company", "smelter at Bargawan")
    decision = compare(aditya, mahan)
    assert decision["decision"] == "distinct"
    assert decision["reason"] == "asset_or_project_name_not_exact"


def test_merged_record_keeps_three_sources_and_retains_the_rest():
    quotes = [
        {"pdf_page": index, "printed_page": str(index), "quote": f"short {index}"}
        for index in range(1, 4)
    ]
    quotes.append({
        "pdf_page": 9,
        "printed_page": "9",
        "quote": "Hindalco Limited manufactures aluminium and copper at its smelter",
    })
    left = _item("Hindalco Limited", "business_line", "company", "aluminium and copper", sources=quotes[:2])
    right = _item("Hindalco Ltd", "business_line", "company", "aluminium and copper", sources=quotes[2:])
    merged = resolve([left, right])["merged"][0]
    assert len(merged["sources"]) == 3
    assert len(merged["retained_sources"]) == 4
    assert merged["sources"][0]["pdf_page"] == 9


def test_ambiguous_names_stay_for_review():
    left = _item("specialty alumina", "product_or_service", "company", "alumina chemicals")
    right = _item("speciality alumina", "product_or_service", "company", "alumina chemicals")
    decision = compare(left, right)
    assert decision["decision"] == "review"
    result = resolve([left, right])
    assert result["merged_count"] == 0
    assert result["review_count"] == 1
