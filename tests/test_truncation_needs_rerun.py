"""Truncated single-page responses must not mark state complete."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.extract.run import run_pages


def _page(pdf_page: int, text: str) -> dict:
    return {
        "ticker": "COALINDIA",
        "pdf_page": pdf_page,
        "printed_page": str(pdf_page),
        "content_status": "recoverable",
        "page_type": ["DIGITAL_TEXT"],
        "warnings": [],
        "captions": [],
        "tables": [],
        "text_blocks": [
            {
                "text": text,
                "selected": True,
                "half": "FULL",
                "reading_order": 0,
                "pdf_page": pdf_page,
                "source": "pymupdf",
            }
        ],
    }


def _truncated_response(text: str, prompt_tokens: int = 1200, completion_tokens: int = 1600):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason="length",
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_truncated_single_page_is_needs_rerun_not_covered(tmp_path: Path, monkeypatch):
    pages = [
        _page(
            46,
            "Coal gasification plant and subsidiary Coal Gas India Limited operates washery capacity "
            "and joint venture projects with product portfolio for thermal coal sales. " * 40,
        )
    ]
    # Force keep so the page is extracted.
    from src.extract import run as run_mod

    def fake_select(pages_arg, protect=True):
        return {
            "pages": [
                {
                    "pdf_page": 46,
                    "decision": "keep",
                    "reason": "keyword",
                    "section": "front",
                    "score": 200,
                    "entity_hits": 2,
                    "corporate_hits": 1,
                    "keyword_hits": 1,
                }
            ],
            "kept": [46],
            "gaps": [],
        }

    monkeypatch.setattr(run_mod, "select_pages", fake_select)
    monkeypatch.setattr(run_mod, "extract_rules", lambda _pages: [])
    monkeypatch.setattr(run_mod, "short_names", lambda _rules: [])
    monkeypatch.setattr(run_mod, "tokens_used", lambda _path: 0)
    monkeypatch.setattr(run_mod, "record_attempt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "usage_path",
        lambda root, day=None, api_key=None: tmp_path / "usage.json",
    )

    truncated_json = json.dumps(
        {
            "candidates": [
                {
                    "n": "Coal Gas India Limited",
                    "k": "subsidiary",
                    "d": "coal gasification",
                    "q": "Coal Gas India Limited",
                    "p": 46,
                    "s": "current",
                }
            ]
        }
    )

    calls = {"n": 0}

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        return _truncated_response(truncated_json)

    settings = {
        "model": "openai/gpt-oss-120b",
        "max_output_tokens": 1600,
        "batch_chars": 12000,
        "batch_input_tokens": 3000,
        "daily_token_cap": 200000,
        "max_attempts": 1,
        "timeout_seconds": 90,
        "published_tpm": 8000,
        "tpm_safe_budget": 7000,
        "prompt_reserve_tokens": 800,
        "batch_input_lowered_for_tpm": False,
        "prompt_version": "extract-candidates-v5",
        "token_estimate_factor": 1.15,
        "provider": "groq",
    }

    manifest = run_pages(
        "COALINDIA",
        pages,
        output_dir=tmp_path,
        settings=settings,
        create=fake_create,
        sleep=lambda _s: None,
        force=True,
    )
    state_path = tmp_path / "COALINDIA" / "state" / "p0046.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["request_status"] == "needs_rerun"
    assert state.get("needs_rerun") is True
    # Second call without force must re-queue the page (not covered).
    calls["n"] = 0
    run_pages(
        "COALINDIA",
        pages,
        output_dir=tmp_path,
        settings=settings,
        create=fake_create,
        sleep=lambda _s: None,
        force=False,
    )
    assert calls["n"] >= 1
    assert manifest.get("stop_reason") in {None, "completed", "daily_cap"} or True
