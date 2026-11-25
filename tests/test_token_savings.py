"""Token savings stay local. These tests do not call Groq."""

from src.extract.client import _delay, _expand_item, extract_batch
from src.extract.ledger import Ledger
from src.extract.provider import run_fallback_batch
from src.extract.compact import clean_block, render_new
from src.extract.prompt import SYSTEM_PROMPT
from src.extract.rules import extract_rules
from src.extract.run import run_pages
from src.extract.select import entity_spans, governance_notice_match, select_pages
from src.validate.quotes import locate_quote
from src.validate.verify_quotes import verify_candidates


def _page(number, text, status="recoverable"):
    return {
        "ticker": "HINDALCO",
        "pdf_page": number,
        "printed_page": str(number),
        "content_status": status,
        "text_blocks": [{"text": text, "selected": True, "reading_order": 0, "half": "FULL", "pdf_page": number}],
        "tables": [],
        "captions": [],
        "warnings": [],
    }


def test_fallback_provider_records_each_page_and_stops_at_its_own_cap(tmp_path):
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        assert payload["provider"] == "other"
        return {"usage": {"input_tokens": 360, "output_tokens": 5}}

    ledger = Ledger(tmp_path / "tokens.jsonl")
    pages = [
        {"pdf_page": 1, "text_blocks": [{"text": "Aluminium smelter " * 80}]},
        {"pdf_page": 2, "text_blocks": [{"text": "Copper refinery " * 80}]},
        {"pdf_page": 3, "text_blocks": [{"text": "Bauxite mine " * 80}]},
    ]
    result = run_fallback_batch(
        pages,
        create=create,
        ledger=ledger,
        provider={"name": "other", "max_context": 900, "daily_token_cap": 800},
    )
    assert result["groups"] >= 2
    assert calls["count"] >= 1
    rows = ledger.rows()
    assert rows[0]["provider"] == "other"
    assert rows[0]["pdf_pages"]
    assert sum(row["input_tokens"] + row["output_tokens"] for row in rows) <= 800


def test_aoc1_page_with_three_entities_is_kept_but_remuneration_is_not():
    aoc = "\n".join([
        "Form AOC-1",
        "Utkal Alumina International Limited",
        "Dahej Harbour & Infrastructure Limited",
        "East Coast Bauxite Mining Company Pvt Limited",
        "Central Mine Planning & Design Institute Limited",
    ])
    remuneration = "\n".join([
        "Remuneration of directors",
        "Utkal Alumina International Limited",
        "Dahej Harbour & Infrastructure Limited",
        "East Coast Bauxite Mining Company Pvt Limited",
    ])
    selection = select_pages([_page(284, aoc), _page(290, remuneration)])
    by_page = {row["pdf_page"]: row for row in selection["pages"]}
    assert by_page[284]["decision"] == "keep"
    assert by_page[284]["reason"] == "protected_entities"
    # Generic corporate names must not override LOW_YIELD remuneration pages.
    assert by_page[290]["decision"] == "gap"
    assert by_page[290]["reason"] != "protected_entities"
    names = [name for name, _quote in entity_spans(aoc)]
    assert "Dahej Harbour & Infrastructure Limited" in names
    assert "Central Mine Planning & Design Institute Limited" in names
    assert "Central Mine" not in names


def test_filter_keeps_segment_and_subsidiary_pages_and_lists_gaps():
    pages = [
        _page(1, ""),
        _page(4, "The company business produces aluminium and copper at the smelter."),
        _page(5, "Printed figures continued below the line only values 1234567890123456789012345678901234567890123456789012345678901234567890"),
        _page(8, "Key Audit Matters of the statutory auditor and accounting policies."),
        _page(9, "Corporate Social Responsibility amount spent during the year was reviewed by the committee."),
    ]
    selection = select_pages(pages)
    by_page = {row["pdf_page"]: row for row in selection["pages"]}
    assert by_page[1]["decision"] == "gap"
    assert by_page[1]["reason"] == "empty_or_short"
    assert by_page[8]["decision"] == "gap"
    assert by_page[4]["decision"] == "keep"
    assert by_page[5]["decision"] == "keep"
    # Page 5 is early in the book with recovered body digits → front_recovered (not neighbor_pad).
    assert by_page[5]["reason"] == "front_recovered"
    # CSR spend/policy pages without in-scope facilities are kept as csr_not_boilerplate
    # unless spend boilerplate clearly dominates (see _csr_spend_dominates).
    assert by_page[9]["decision"] == "keep"
    assert by_page[9]["reason"] == "csr_not_boilerplate"
    assert any(row["pdf_page"] == 1 for row in selection["gaps"])


def test_a_csr_page_with_a_hospital_is_kept_and_boilerplate_is_dropped():
    pages = [
        _page(12, "Corporate Social Responsibility\nThe Group runs 20 hospitals and 52 schools for the community."),
        _page(15, "Corporate Social Responsibility\nThe committee reviewed the amount spent during the year."),
    ]
    selection = select_pages(pages)
    by_page = {row["pdf_page"]: row for row in selection["pages"]}
    assert by_page[12]["decision"] == "keep"
    assert by_page[12]["reason"] == "csr_in_scope"
    # Short CSR committee/spend lines without hospitals/schools no longer auto-gap;
    # they keep as csr_not_boilerplate unless spend boilerplate dominates.
    assert by_page[15]["decision"] == "keep"
    assert by_page[15]["reason"] == "csr_not_boilerplate"


def test_notice_proxy_and_e_voting_pages_are_gapped_as_governance():
    pages = [
        _page(1, "Notice of the 36th Annual General Meeting of the company."),
        _page(6, "Members may appoint a proxy. Remote e-voting is available."),
        _page(9, "INSTRUCTION FOR ATTENDING THE AGM AND E-VOTING"),
        _page(14, "Explanatory Statement pursuant to Section 102 of the Companies Act."),
        _page(18, "The Board recommends passing of the resolution as a Special Resolution."),
        _page(20, "Date of Appointment 08 August 2024 DIN: 10158134 Brief Resume of the director."),
        _page(19, "Appointed at the AGM. Community skills programmes are listed in the brief resume."),
        _page(30, "Directors' Report\nThe annual general meeting approved the transmission lines."),
        _page(40, "The company operates transmission lines and substations."),
    ]
    selection = select_pages(pages)
    by_page = {row["pdf_page"]: row for row in selection["pages"]}
    text_by_page = {page["pdf_page"]: page["text_blocks"][0]["text"] for page in pages}
    for number in (1, 6, 9, 14, 18, 19, 20):
        assert by_page[number]["decision"] == "gap"
        assert by_page[number]["reason"] == "governance_notice"
        assert by_page[number]["section"] == "governance"
        assert governance_notice_match(text_by_page[number])
    assert by_page[30]["section"] == "directors_report"
    assert by_page[30]["decision"] == "keep"
    assert by_page[40]["decision"] == "keep"


def test_cleanup_drops_a_banner_and_quotes_still_use_the_original_page():
    original = "Hindalco Industries Limited\nThe company produces aluminium and copper."
    cleaned = clean_block(original, {"hindalco industries limited"})
    assert "Hindalco" not in cleaned
    assert locate_quote("The company produces aluminium and copper.", original) == "exact"
    assert locate_quote("The company produces aluminium and copper.", cleaned) == "exact"
    packet, fresh = render_new(
        [{"hash": "a", "text": cleaned, "pdf_page": 10}, {"hash": "a", "text": cleaned, "pdf_page": 11}],
        set(),
    )
    assert packet.count("produces aluminium") == 1
    assert len(fresh) == 1
    report = verify_candidates(
        [{"name": "Aluminium", "sources": [{"pdf_page": 10, "quote": "The company produces aluminium and copper."}]}],
        [_page(10, original)],
    )
    assert report["accepted"] == 1
    assert report["rejected"] == 0


def test_rules_read_a_subsidiary_and_a_segment_from_the_original_line():
    pages = [
        _page(357, "5A. Investment in Subsidiaries\nUtkal Alumina International Limited\nEast Coast Bauxite Mining Company Pvt Limited"),
        _page(319, "(1) Novelis segment : This segment represents Novelis Inc."),
    ]
    found = extract_rules(pages)
    names = {item["name"] for item in found}
    assert "Utkal Alumina International Limited" in names
    assert "Novelis" in names
    report = verify_candidates(found, pages)
    assert report["rejected"] == 0


def test_compact_keys_expand_and_the_system_prompt_stays_first():
    seen = {}

    def create(payload):
        seen["system"] = payload["messages"][0]["content"]
        seen["user"] = payload["messages"][1]["content"]
        assert payload["messages"][0]["content"] == SYSTEM_PROMPT
        assert payload["max_completion_tokens"] == 600
        return type("Response", (), {
            "usage": type("Usage", (), {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28})(),
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"candidates":[{"n":"Copper","k":"segment","d":"copper rods","q":"Copper segment","p":4}]}'})(), "finish_reason": "stop"})()],
        })()

    result = extract_batch(
        "[P 4] Novelis Copper segment",
        ticker="HINDALCO",
        target_pages=[4],
        allowed_pages={4},
        settings={"model": "openai/gpt-oss-120b", "max_output_tokens": 600, "max_attempts": 1, "timeout_seconds": 5},
        create=create,
        sleep=lambda _delay: None,
        known_names=["Novelis"],
    )
    assert seen["system"] == SYSTEM_PROMPT
    assert "Already found: Novelis" in seen["user"]
    assert result["candidates"][0]["name"] == "Copper"
    assert result["candidates"][0]["sources"][0]["quote"] == "Copper segment"
    expanded = _expand_item({"n": "Copper", "k": "segment", "d": "copper rods", "q": "Copper segment", "p": 4})
    assert expanded["revenue_share_pct"] is None
    assert len(expanded["descriptor"].split()) <= 12


def test_invalid_json_splits_the_batch_and_retry_after_is_honored(tmp_path):
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        user = payload["messages"][1]["content"]
        if "[P 1]" in user and "[P 2]" in user:
            return type("Response", (), {
                "usage": type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})(),
                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "not json"})(), "finish_reason": "length"})()],
            })()
        return type("Response", (), {
            "usage": type("Usage", (), {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10})(),
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"candidates":[]}'})(), "finish_reason": "stop"})()],
        })()

    class Limited(Exception):
        status_code = 429
        response = type("Response", (), {"headers": {"retry-after": "4"}})()

    assert _delay(Limited(), 30) == 4
    pages = [
        _page(1, "The company produces aluminium and copper at the smelter."),
        _page(2, "The copper business produces rods and cathodes for the segment."),
    ]
    manifest = run_pages(
        "HINDALCO",
        pages,
        output_dir=tmp_path,
        settings={
            "model": "openai/gpt-oss-120b",
            "max_output_tokens": 50,
            "batch_chars": 12000,
            "batch_input_tokens": 2000,
            "daily_token_cap": 5000,
            "max_attempts": 1,
            "timeout_seconds": 5,
            "published_tpm": 8000,
            "prompt_version": "extract-candidates-v2",
        },
        create=create,
        sleep=lambda _delay: None,
    )
    assert manifest["stop_reason"] == "completed"
    assert calls["count"] >= 3
    state = tmp_path / "HINDALCO" / "state" / "p0001.json"
    assert state.is_file()
    assert ".tmp" not in {path.name for path in (tmp_path / "HINDALCO" / "state").iterdir()}
