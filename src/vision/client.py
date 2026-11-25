"""One Groq vision call. Successful responses are cached. Tokens come from the response."""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.vision.settings import PROMPT_VERSION, api_key, vision_settings

PROMPT = """Extract readable text from this annual-report page image.
Return ONE compact JSON object only (no markdown, no commentary).
Required keys: pdf_page, half, extracted_text, tables, labels, unreadable_regions.

Rules for staying within the token budget:
- Prefer company/entity names, legal endings (Limited/Ltd), and list/table titles.
- extracted_text: short readable lines only; skip decorative headers/footers unless they are page numbers.
- tables: headers + rows of cells; use null for unreadable cells; omit empty trailing columns.
- Do not repeat the same line. Do not invent names or numbers.
- Always close the JSON object completely even if you must truncate long tables.
- labels: short diagram labels only. unreadable_regions: brief strings, or [].

This is an extraction aid, not a source of truth.
"""


def prepare_image(path: Path, max_side: int = 1600) -> bytes:
    """Shrink a cached page image to a JPEG. The original file is left unchanged."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    data = buffer.getvalue()
    if len(data) > 8_000_000:
        raise ValueError(f"Prepared image is {len(data)} bytes, above the size kept for one request.")
    return data


def usage_counts(usage) -> dict:
    """Copy token counts from the API object. Missing fields stay null."""
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    input_tokens = _first(usage, "prompt_tokens", "input_tokens")
    output_tokens = _first(usage, "completion_tokens", "output_tokens")
    total_tokens = _first(usage, "total_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def parse_model_json(text: str) -> tuple[dict | None, str | None]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    if not isinstance(parsed, dict):
        return None, "The model returned JSON that is not an object."
    return parsed, None


def extract_image(
    image_bytes: bytes,
    *,
    ticker: str,
    pdf_page: int,
    half: str,
    routing_reason: str,
    create=None,
    sleep=time.sleep,
) -> dict:
    """Call the vision model once per attempt. create is injectable for tests."""
    settings = vision_settings()
    started = time.perf_counter()
    attempts = 0
    retried = 0
    last_error = None
    response = None
    while attempts < settings["max_attempts"]:
        attempts += 1
        try:
            response = _create(create, settings, image_bytes, pdf_page, half)
            break
        except Exception as exc:
            last_error = exc
            if attempts >= settings["max_attempts"] or not _retryable(exc):
                break
            retried += 1
            sleep(_delay(exc, 2 ** (attempts - 1)))
    duration = round(time.perf_counter() - started, 3)
    record = _base_record(ticker, pdf_page, half, routing_reason, settings, attempts, retried, duration)
    if response is None:
        record["request_status"] = "failed"
        record["error"] = f"{type(last_error).__name__}: {last_error}"
        return record
    record["usage"] = usage_counts(getattr(response, "usage", None))
    content = _message_text(response)
    parsed, error = parse_model_json(content)
    if parsed is None:
        record["request_status"] = "malformed_response"
        record["error"] = error
        record["raw_response"] = content
        return record
    record["request_status"] = "success"
    record["extracted_text"] = list(parsed.get("extracted_text") or [])
    record["tables"] = list(parsed.get("tables") or [])
    record["labels"] = list(parsed.get("labels") or [])
    record["unreadable_regions"] = list(parsed.get("unreadable_regions") or [])
    return record


def _create(create, settings: dict, image_bytes: bytes, pdf_page: int, half: str):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": settings["model"],
        "temperature": 0,
        "max_completion_tokens": settings["max_output_tokens"],
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{PROMPT}\npdf_page: {pdf_page}\nhalf: {half}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }
        ],
    }
    if create is not None:
        return create(payload)
    from groq import Groq

    from src.extract.keys import key_pool

    # Prefer GROQ_API_KEYS rotation; model always comes from GROQ_VISION_MODEL via settings.
    pool = key_pool()
    key = pool.current() if pool.keys else api_key()
    timeout = float(settings["timeout_seconds"])
    client = Groq(api_key=key, timeout=timeout)
    try:
        return client.chat.completions.create(**payload)
    except Exception as exc:
        message = str(exc).casefold()
        if pool.keys and ("429" in message or "rate" in message or "too large" in message):
            if pool.rotate():
                client = Groq(api_key=pool.current(), timeout=timeout)
                return client.chat.completions.create(**payload)
        raise


def _retryable(exc: Exception) -> bool:
    try:
        from groq import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        APIConnectionError = APITimeoutError = RateLimitError = ()
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429} or (isinstance(status, int) and status >= 500)


def _delay(exc: Exception, fallback: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return min(fallback, 30.0)


def _message_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    return content or ""


def _base_record(ticker, pdf_page, half, routing_reason, settings, attempts, retried, duration) -> dict:
    return {
        "ticker": ticker,
        "pdf_page": int(pdf_page),
        "half": half,
        "model": settings["model"],
        "prompt_version": PROMPT_VERSION,
        "routing_reason": routing_reason,
        "source_of_truth": False,
        "quote_validated": False,
        "request_status": "failed",
        "extracted_text": [],
        "tables": [],
        "labels": [],
        "unreadable_regions": [],
        "page_references": {"ticker": ticker, "pdf_page": int(pdf_page), "half": half},
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        "duration_seconds": duration,
        "attempts": attempts,
        "retried": retried,
        "error": None,
    }


def _first(usage, *names):
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None
