"""Vision settings from the environment. The API key is never written into source."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.config.paths import PROJECT_ROOT

# Checked against Groq docs on 2026-11-23. The previous 3.6 vision id is deprecated.
DOCUMENTED_VISION_MODEL = "qwen/qwen3.8-27b"
PROMPT_VERSION = "vision-extract-v2-compact"
# Groq counts every image as this many input tokens, whatever its pixel size.
IMAGE_INPUT_TOKENS = 2048
# Rate-limits page for this model. The model card's developer plan is higher.
PUBLISHED_TPM = 8000
PUBLISHED_TPD = 200_000

load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def vision_settings() -> dict:
    model = (os.environ.get("GROQ_VISION_MODEL") or DOCUMENTED_VISION_MODEL).strip()
    return {
        "model": model,
        "api_key_present": bool((os.environ.get("GROQ_API_KEY") or "").strip()),
        "max_page_percent": _float("GROQ_VISION_MAX_PAGE_PERCENT", 2.0),
        "max_output_tokens": _int("GROQ_VISION_MAX_OUTPUT_TOKENS", 4096),
        "timeout_seconds": _float("GROQ_VISION_TIMEOUT_SECONDS", 60.0),
        "max_attempts": _int("GROQ_VISION_MAX_ATTEMPTS", 3),
        "prompt_version": PROMPT_VERSION,
        "image_input_tokens": IMAGE_INPUT_TOKENS,
        "published_tpm": PUBLISHED_TPM,
        "published_tpd": PUBLISHED_TPD,
        "daily_token_cap": _int("GROQ_VISION_DAILY_TOKEN_CAP", PUBLISHED_TPD),
    }


def api_key() -> str:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env. It is not stored in source.")
    return key
