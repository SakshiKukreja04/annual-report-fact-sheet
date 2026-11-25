"""Candidate extraction structure. These tests do not contact Groq."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.extract.batches import make_batches
from src.extract.client import extract_batch
from src.extract.evidence import pages_for_extraction, render_page
from src.extract.known import relevant_known_names
from src.extract.ledger import estimate_tokens
from src.extract.run import run_pages
from src.extract.schema import Candidate
from src.extract.settings import text_settings
from src.vision.usage import record_attempt, usage_path


def _page(pdf_page, text, status="recoverable", warnings=None, tables=None, captions=None):
    return {
        "ticker": "HINDALCO",
        "pdf_page": pdf_page,
        "printed_page": str(pdf_page),
        "content_status": status,
        "warnings": warnings or [],
        "captions": captions or [],
        "tables": tables or [],
        "text_blocks": [
            {
                "text": text,
                "selected": True,
                "half": "FULL",
                "reading_order": 0,
                "pdf_page": pdf_page,
            }
        ],
    }


def _response(text, prompt_tokens=1000, completion_tokens=200):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)


def test_a_long_page_is_not_cut_off():
    text = "aluminium " * 2000
    batches = make_batches([_page(10, text)], max_chars=500)
    assert len(batches) == 1
    rendered = render_page(batches[0]["pages"][0], "target")
    assert "aluminium " * 2000 in rendered
    assert batches[0]["target_pages"] == [10]


def test_overlap_keeps_the_previous_page_as_context():
    pages = [_page(index, f"Business line {index} " * 30) for index in (1, 2, 3)]
    batches = make_batches(pages, max_chars=len(render_page(pages[0], "target")) + 20, overlap=1)
    assert batches[0]["target_pages"] == [1]
    assert batches[1]["context_pages"] == [1]
    assert 1 not in batches[1]["target_pages"]


def test_illegible_pages_and_unreadable_tables_stay_out():
    loaded = {
        "candidate_pages": [
            _page(100, "noise", status="illegible", warnings=["illegible_low_dpi"]),
            _page(45, "Board of Directors of the company"),
        ],
        "image_only_pages": [
            _page(26, "", status="image_only_visual", captions=[{"text": "Photo", "half": "LEFT"}]),
            _page(38, "", status="image_only_visual", captions=[{"text": "Coal India Limited integrated report", "half": "LEFT"}]),
        ],
    }
    pages = pages_for_extraction(loaded)
    assert [page["pdf_page"] for page in pages] == [38, 45]
    scanned = _page(
        200,
        "CSR",
        warnings=["scanned_table_text_is_unverified"],
        tables=[{"table_id": "t", "pdf_page": 200, "headers": ["S.No."], "rows": [{"pdf_page": 200, "cells": ["garbled"]}], "extraction_method": "paddleocr_pp_structure"}],
    )
    rendered = render_page(scanned, "target")
    assert "garbled" not in rendered
    assert "TABLE_OMITTED" in rendered


def test_schema_requires_a_quote_and_limits_sources():
    with pytest.raises(ValidationError):
        Candidate.model_validate({
            "name": "Aluminium",
            "kind": "product_or_service",
            "descriptor": "Metal",
            "entity": "company",
            "status": "current",
            "revenue_share_pct": None,
            "sources": [{"pdf_page": 10, "printed_page": "4", "quote": ""}],
        })
    with pytest.raises(ValidationError):
        Candidate.model_validate({
            "name": "Aluminium",
            "kind": "product_or_service",
            "descriptor": "Metal",
            "entity": "company",
            "status": "current",
            "revenue_share_pct": "12%",
            "sources": [{"pdf_page": index, "printed_page": None, "quote": "Aluminium"} for index in range(1, 5)],
        })
    kept = Candidate.model_validate({
        "name": "Novelis",
        "kind": "subsidiary",
        "descriptor": "Rolled products",
        "entity": "Novelis Inc.",
        "status": "current",
        "revenue_share_pct": None,
        "sources": [{"pdf_page": 180, "printed_page": "300/301", "quote": "Novelis Inc."}],
        "supporting_figures": [{"label": "Revenue", "value": "10", "pdf_page": 180, "quote": "Revenue 10"}],
    })
    assert kept.revenue_share_pct is None
    assert kept.supporting_figures[0].value == "10"
    numbered = Candidate.model_validate({
        "name": "Cost audit",
        "kind": "product_or_service",
        "descriptor": "Cost records",
        "entity": "company",
        "status": "current",
        "revenue_share_pct": None,
        "sources": [{"pdf_page": 13, "printed_page": 12, "quote": "cost records of the Company"}],
        "supporting_figures": [{"pdf_page": 13, "quote": "Rupees Twenty Lakhs only"}],
    })
    assert numbered.sources[0].printed_page == "12"
    assert numbered.supporting_figures[0].value == ""


def test_malformed_response_is_retried_and_outside_pages_are_rejected():
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        assert payload["tool_choice"] == "none"
        assert payload["tools"] == []
        if calls["count"] == 1:
            return _response("not json")
        return _response(
            '{"candidates":[{"name":"Invented","kind":"other","descriptor":"x","entity":"company","status":"current","revenue_share_pct":null,"sources":[{"pdf_page":999,"printed_page":null,"quote":"not in packet"}]}]}'
        )

    result = extract_batch(
        "[PAGE pdf_page=10]",
        ticker="HINDALCO",
        target_pages=[10],
        allowed_pages={10},
        settings={"model": "openai/gpt-oss-120b", "max_output_tokens": 200, "max_attempts": 3, "timeout_seconds": 5},
        create=create,
        sleep=lambda _delay: None,
    )
    # Malformed JSON is not retried; the caller splits the batch instead.
    assert calls["count"] == 1
    assert result["request_status"] == "malformed_response"
    assert result["candidates"] == []
    assert result["tools_enabled"] is False


def test_cached_batch_is_not_sent_again_and_the_cap_stops_the_rest(tmp_path: Path):
    page = _page(10, "The company produces aluminium and copper.")
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        return _response('{"candidates":[]}', prompt_tokens=40, completion_tokens=5)

    settings = {
        "model": "openai/gpt-oss-120b",
        "max_output_tokens": 20,
        "batch_input_tokens": 2000,
        "batch_chars": 12000,
        "daily_token_cap": 5000,
        "max_attempts": 2,
        "timeout_seconds": 5,
        "published_tpm": 8000,
        "prompt_version": "extract-candidates-v1",
    }
    first = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert first["stop_reason"] == "completed"
    assert calls["count"] == 1
    state = json.loads((tmp_path / "HINDALCO" / "state" / "p0010.json").read_text(encoding="utf-8"))
    assert "prompt_version" in state
    second = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert calls["count"] == 1
    # Covered pages are not packed again.
    assert second["batches_completed"] == 0
    assert second["pages_done"] == 1
    raw = next((tmp_path / "HINDALCO" / "raw").glob("*.json"))
    parsed = next((tmp_path / "HINDALCO" / "parsed").glob("*.json"))
    assert "raw_response" in raw.read_text(encoding="utf-8")
    assert "raw_response" not in parsed.read_text(encoding="utf-8")

    later = _page(11, "Another business line of the company.")
    settings["daily_token_cap"] = 10
    capped = run_pages("HINDALCO", [page, later], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None, force=True)
    assert capped["stop_reason"] == "daily_cap"
    assert 11 in capped["pages_not_sent"]
    assert calls["count"] == 1


def test_a_different_api_key_is_not_blocked_by_the_other_accounts_usage(tmp_path, monkeypatch):
    page = _page(10, "The company produces aluminium and copper.")
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        return _response('{"candidates":[]}', prompt_tokens=40, completion_tokens=5)

    settings = {
        "model": "openai/gpt-oss-120b",
        "max_output_tokens": 20,
        "batch_input_tokens": 2000,
        "batch_chars": 12000,
        "daily_token_cap": 5000,
        "max_attempts": 1,
        "timeout_seconds": 5,
        "published_tpm": 8000,
        "prompt_version": "extract-candidates-v1",
    }
    from src.extract.keys import key_pool

    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "account-a")
    key_pool(reset=True)
    record_attempt(usage_path(tmp_path, api_key="account-a"), {
        "ticker": "HINDALCO",
        "pdf_page": 10,
        "model": "openai/gpt-oss-120b",
        "request_status": "success",
        "retried": 0,
        "usage": {"input_tokens": 4900, "output_tokens": 0, "total_tokens": 4900},
    })
    blocked = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert blocked["stop_reason"] == "daily_cap"
    assert calls["count"] == 0

    monkeypatch.setenv("GROQ_API_KEY", "account-b")
    key_pool(reset=True)
    opened = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert opened["stop_reason"] == "completed"
    assert calls["count"] == 1


def test_model_id_is_configurable(monkeypatch):
    monkeypatch.setenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
    assert text_settings()["model"] == "openai/gpt-oss-120b"
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/extract").glob("*.py"))
    assert "gsk_" not in source
    assert "GROQ_API_KEY =" not in source


def test_tpm_budget_lowers_batch_input(monkeypatch):
    monkeypatch.setenv("GROQ_TEXT_BATCH_TOKENS", "6500")
    monkeypatch.setenv("GROQ_TEXT_MAX_OUTPUT_TOKENS", "2000")
    settings = text_settings()
    assert (
        settings["batch_input_tokens"]
        + settings["max_output_tokens"]
        + settings["prompt_reserve_tokens"]
        <= 7000
    )
    assert settings["batch_input_lowered_for_tpm"] is True


def test_already_found_keeps_only_packet_names():
    known = ["Novelis Inc.", "Unrelated Zinc Mine", "Aditya Smelter"]
    packet = "[P 10] Novelis Inc. rolled products at the Aditya Smelter."
    assert relevant_known_names(known, packet, limit=40) == ["Novelis Inc.", "Aditya Smelter"]


def test_token_estimate_applies_factor():
    assert estimate_tokens("abcd") == int(round((4 / 4) * 1.15))
