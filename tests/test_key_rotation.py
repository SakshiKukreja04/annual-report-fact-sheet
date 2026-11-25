"""A daily-limit 429 rotates to the next key. The key value never appears in logs."""

from __future__ import annotations

import logging

from src.extract.client import extract_batch
from src.extract.keys import key_pool


class _DailyLimit(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("Rate limit reached: tokens per day exceeded")


def test_local_cap_switches_to_the_next_key(tmp_path, monkeypatch):
    import json

    from src.extract.keys import fit_daily_cap, key_pool
    from src.vision.usage import usage_path

    monkeypatch.setenv("GROQ_API_KEYS", "gsk_test_alpha_secret,gsk_test_beta_secret")
    key_pool(reset=True)
    full = usage_path(tmp_path, api_key="gsk_test_alpha_secret")
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps({"input_tokens": 190000, "output_tokens": 10000}), encoding="utf-8")
    fits, path = fit_daily_cap(tmp_path, 5000, 200000)
    assert fits is True
    assert key_pool().label() == "key_2"
    assert path == usage_path(tmp_path, api_key="gsk_test_beta_secret")
    other = usage_path(tmp_path, api_key="gsk_test_beta_secret")
    other.write_text(json.dumps({"input_tokens": 200000, "output_tokens": 0}), encoding="utf-8")
    key_pool(reset=True)
    fits, _path = fit_daily_cap(tmp_path, 5000, 200000)
    assert fits is False
    assert key_pool().label() == "key_2"


def test_daily_429_rotates_and_keys_stay_out_of_logs(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_test_alpha_secret,gsk_test_beta_secret")
    key_pool(reset=True)
    calls = {"n": 0}

    def create(_payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _DailyLimit()
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()
        message = type("M", (), {"content": '{"candidates":[]}'})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice], "usage": usage})()

    caplog.set_level(logging.INFO)
    result = extract_batch(
        "[PAGE pdf_page=10] The company operates a plant.",
        ticker="COALINDIA",
        target_pages=[10],
        allowed_pages={10},
        settings={"model": "openai/gpt-oss-120b", "max_output_tokens": 100, "max_attempts": 3, "timeout_seconds": 5},
        create=create,
        sleep=lambda _delay: None,
    )
    assert result["request_status"] == "success"
    assert result["key_label"] == "key_2"
    assert calls["n"] == 2
    blob = caplog.text + str(result)
    assert "gsk_test_alpha_secret" not in blob
    assert "gsk_test_beta_secret" not in blob
    assert key_pool().label() == "key_2"
