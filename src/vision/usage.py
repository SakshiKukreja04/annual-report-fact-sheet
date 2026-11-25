"""Project-local token log. Counts come from API responses, not from an estimate.

The file is per API key. Groq's daily token cap belongs to the account, so a
new key must not inherit tokens already recorded for another account.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path


def key_label(api_key: str | None = None) -> str:
    raw = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    digest = hashlib.sha256((raw or "").strip().encode("utf-8")).hexdigest()
    return digest[:12]


def usage_path(root: Path, day: str | None = None, api_key: str | None = None) -> Path:
    stamp = day or date.today().isoformat()
    return root / "usage" / f"{stamp}-{key_label(api_key)}.json"


def empty_log(day: str, model: str) -> dict:
    return {
        "date": day,
        "model": model,
        "requests": 0,
        "successful": 0,
        "failed": 0,
        "malformed": 0,
        "retried_calls": 0,
        "cache_hits": 0,
        "not_sent": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "pages": [],
    }


def load_log(path: Path, day: str, model: str) -> dict:
    if not path.is_file():
        return empty_log(day, model)
    return json.loads(path.read_text(encoding="utf-8"))


def record_attempt(path: Path, result: dict, *, cached: bool = False) -> dict:
    day = date.today().isoformat()
    log = load_log(path, day, result.get("model") or "")
    status = "cached" if cached else result.get("request_status")
    if cached:
        log["cache_hits"] += 1
    elif status in {"not_routed", "daily_cap_reached"}:
        log["not_sent"] = int(log.get("not_sent") or 0) + 1
    else:
        log["requests"] += 1
        log["retried_calls"] += int(result.get("retried") or 0)
        if status == "success":
            log["successful"] += 1
        elif status == "malformed_response":
            log["malformed"] += 1
        else:
            log["failed"] += 1
        usage = result.get("usage") or {}
        log["input_tokens"] += int(usage.get("input_tokens") or 0)
        log["output_tokens"] += int(usage.get("output_tokens") or 0)
    log["pages"].append(
        {
            "ticker": result.get("ticker"),
            "pdf_page": result.get("pdf_page"),
            "half": result.get("half"),
            "request_status": "cached" if cached else result.get("request_status"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def tokens_used(path: Path) -> int:
    if not path.is_file():
        return 0
    log = json.loads(path.read_text(encoding="utf-8"))
    return int(log.get("input_tokens") or 0) + int(log.get("output_tokens") or 0)
