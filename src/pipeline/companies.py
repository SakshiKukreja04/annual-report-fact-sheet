"""The three annual reports this pipeline is allowed to read."""

from __future__ import annotations

COMPANIES = {
    "POWERGRID": {"expected_pages": 364, "period": "FY25"},
    "HINDALCO": {"expected_pages": 413, "period": "FY25"},
    "COALINDIA": {"expected_pages": 371, "period": "FY25"},
}

STAGES = (
    "discover",
    "inspect",
    "classify",
    "ocr",
    "tables",
    "pages",
    "vision",
    "extract",
    "dedup",
    "validate",
    "factsheet",
    "evaluate",
)

# Published Groq rates per 1,000,000 tokens. Cost is computed from response.usage.
TEXT_RATES = {"input": 0.15, "output": 0.60, "model": "openai/gpt-oss-120b"}
VISION_RATES = {"input": 0.80, "output": 4.00, "model": "qwen/qwen3.8-27b"}
