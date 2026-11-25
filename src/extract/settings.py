"""Text-model settings. The id defaults to the model checked on Groq docs."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.config.paths import PROJECT_ROOT
from src.extract.prompt import PROMPT_VERSION

load_dotenv(PROJECT_ROOT / ".env")

DOCUMENTED_TEXT_MODEL = "openai/gpt-oss-120b"
PUBLISHED_TPM = 8000
PUBLISHED_TPD = 200_000
# Leave headroom under the published 8K tokens-per-minute free-tier limit.
TPM_SAFE_BUDGET = 7000
PROMPT_RESERVE_TOKENS = 800


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def text_settings() -> dict:
    # Raise max_output: 5/18 pilot calls hit finish_reason=length at 900.
    # Headroom under TPM_SAFE_BUDGET 7000: batch_input + max_output + prompt_reserve.
    # 3000 + 1600 + 800 = 5400 (< 7000). Published TPM is 8000.
    max_output = _int("GROQ_TEXT_MAX_OUTPUT_TOKENS", 1600)
    batch_input = _int("GROQ_TEXT_BATCH_TOKENS", 3000)
    lowered = False
    if batch_input + max_output + PROMPT_RESERVE_TOKENS > TPM_SAFE_BUDGET:
        batch_input = max(400, TPM_SAFE_BUDGET - max_output - PROMPT_RESERVE_TOKENS)
        lowered = True
    return {
        "model": (os.environ.get("GROQ_TEXT_MODEL") or DOCUMENTED_TEXT_MODEL).strip(),
        "max_output_tokens": max_output,
        "batch_chars": _int("GROQ_TEXT_BATCH_CHARS", 12000),
        "batch_input_tokens": batch_input,
        "daily_token_cap": _int("GROQ_TEXT_DAILY_TOKEN_CAP", PUBLISHED_TPD),
        "max_attempts": _int("GROQ_TEXT_MAX_ATTEMPTS", 4),
        "timeout_seconds": 90,
        "published_tpm": PUBLISHED_TPM,
        "tpm_safe_budget": TPM_SAFE_BUDGET,
        "prompt_reserve_tokens": PROMPT_RESERVE_TOKENS,
        "batch_input_lowered_for_tpm": lowered,
        "prompt_version": PROMPT_VERSION,
        "token_estimate_factor": 1.15,
    }
