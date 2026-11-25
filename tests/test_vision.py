"""Vision routing and call handling. These tests do not contact Groq."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.vision.client import extract_image, parse_model_json, usage_counts
from src.vision.pilot import cache_path, run_target
from src.vision.route import select_pages, vision_cap
from src.vision.settings import vision_settings
from src.vision.usage import record_attempt, usage_path


def _page(pdf_page, status="recoverable", warnings=None, **quality):
    return {
        "ticker": "POWERGRID",
        "pdf_page": pdf_page,
        "printed_page": None,
        "content_status": status,
        "warnings": warnings or [],
        "captions": [],
        "quality_metadata": quality,
        "tables": [],
    }


def _response(text, prompt_tokens=2100, completion_tokens=40):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_illegible_page_is_not_routed():
    page = json.loads(Path("data/processed/pages/POWERGRID/p0100.json").read_text(encoding="utf-8"))
    decisions = select_pages([page], max_percent=100, limit=5)
    assert decisions[0]["eligible"] is False
    assert decisions[0]["routing_reason"] == "illegible_low_dpi"


def test_page_without_text_or_captions_is_not_routed():
    decisions = select_pages([_page(26, status="image_only_visual")], max_percent=100, limit=5)
    assert decisions[0]["routing_reason"] == "no_meaningful_text"


def test_adequate_local_text_is_not_routed():
    decisions = select_pages(
        [_page(2, selected_text_source="pymupdf")],
        max_percent=100,
        limit=5,
    )
    assert decisions[0]["routing_reason"] == "locally_adequate"


def test_unverified_table_is_eligible_with_a_reason_and_the_cap_holds():
    pages = [_page(index, warnings=["scanned_table_text_is_unverified"]) for index in range(1, 101)]
    assert vision_cap(100, 2) == 2
    decisions = select_pages(pages, max_percent=2)
    eligible = [item for item in decisions if item["eligible"]]
    assert len(eligible) == 2
    assert eligible[0]["routing_reason"] == "complex_table_text_unreliable"
    assert decisions[2]["routing_reason"] == "vision_cap_reached"


def test_usage_comes_from_the_response_and_malformed_json_is_kept():
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        if calls["count"] == 1:
            error = TimeoutError("timed out")
            error.status_code = 504
            raise error
        return _response('{"extracted_text":["Coal India Limited"],"tables":[],"labels":[],"unreadable_regions":[]}')

    result = extract_image(b"jpeg", ticker="COALINDIA", pdf_page=314, half="FULL", routing_reason="ocr_text_structurally_ambiguous", create=create, sleep=lambda _delay: None)
    assert result["request_status"] == "success"
    assert result["retried"] == 1
    assert result["usage"] == {"input_tokens": 2100, "output_tokens": 40, "total_tokens": 2140}
    assert result["pdf_page"] == 314
    assert result["half"] == "FULL"
    assert result["source_of_truth"] is False
    assert result["quote_validated"] is False
    assert usage_counts(None)["input_tokens"] is None

    broken = extract_image(b"jpeg", ticker="COALINDIA", pdf_page=314, half="FULL", routing_reason="ocr_text_structurally_ambiguous", create=lambda _payload: _response("not json"), sleep=lambda _delay: None)
    assert broken["request_status"] == "malformed_response"
    assert parse_model_json("not json")[0] is None


def test_successful_cache_is_not_called_again(tmp_path: Path):
    page = _page(200, warnings=["scanned_table_text_is_unverified"])
    destination = cache_path(tmp_path, "POWERGRID", 200, "LEFT")
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({
        "ticker": "POWERGRID",
        "pdf_page": 200,
        "half": "LEFT",
        "request_status": "success",
        "model": "qwen/qwen3.8-27b",
        "usage": {"input_tokens": 2048, "output_tokens": 10, "total_tokens": 2058},
    }), encoding="utf-8")

    def create(_payload):
        raise AssertionError("cached success was sent again")

    result = run_target("POWERGRID", 200, "LEFT", output_dir=tmp_path, page=page, create=create)
    assert result["request_status"] == "cached"
    log = json.loads(usage_path(tmp_path).read_text(encoding="utf-8"))
    assert log["cache_hits"] == 1
    assert log["requests"] == 0


def test_daily_cap_blocks_a_new_call(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GROQ_VISION_DAILY_TOKEN_CAP", "5")
    record_attempt(usage_path(tmp_path), {
        "ticker": "POWERGRID",
        "pdf_page": 200,
        "half": "LEFT",
        "model": "qwen/qwen3.8-27b",
        "request_status": "success",
        "retried": 0,
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    })
    called = {"count": 0}

    def create(_payload):
        called["count"] += 1
        return _response("{}")

    result = run_target(
        "POWERGRID",
        200,
        "LEFT",
        output_dir=tmp_path,
        page=_page(200, warnings=["scanned_table_text_is_unverified"]),
        create=create,
    )
    assert result["request_status"] == "daily_cap_reached"
    assert called["count"] == 0


def test_model_id_comes_from_the_environment_and_no_key_is_hardcoded(monkeypatch):
    monkeypatch.setenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
    assert vision_settings()["model"] == "qwen/qwen3.8-27b"
    root = Path("src/vision")
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "gsk_" not in source
    assert "GROQ_API_KEY =" not in source
    assert "qwen/qwen3.6-27b" not in source
