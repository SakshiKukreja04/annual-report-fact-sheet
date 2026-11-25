"""One text provider at a time. A second provider is used only when its own key is set."""

from __future__ import annotations

import os

from src.extract.ledger import estimate_tokens
from src.extract.settings import text_settings


def active_provider() -> dict:
    settings = text_settings()
    return {
        "name": (os.environ.get("LLM_PROVIDER") or "groq").strip(),
        "base_url": (os.environ.get("LLM_BASE_URL") or "https://api.groq.com/openai/v1").strip(),
        "model": (os.environ.get("LLM_MODEL") or settings["model"]).strip(),
        "daily_token_cap": settings["daily_token_cap"],
        "max_context": _int("LLM_MAX_CONTEXT", 131072),
        "api_key_env": "GROQ_API_KEY",
    }


def fallback_provider() -> dict | None:
    """A different provider. A Groq name or the Groq key is refused."""
    if (os.environ.get("LLM_FALLBACK_PROVIDER") or "").strip().lower() in {"", "groq"}:
        return None
    key = (os.environ.get("LLM_FALLBACK_API_KEY") or "").strip()
    if not key or key == (os.environ.get("GROQ_API_KEY") or "").strip():
        return None
    return {
        "name": os.environ["LLM_FALLBACK_PROVIDER"].strip(),
        "base_url": (os.environ.get("LLM_FALLBACK_BASE_URL") or "").strip(),
        "model": (os.environ.get("LLM_FALLBACK_MODEL") or "").strip(),
        "daily_token_cap": _int("LLM_FALLBACK_DAILY_TOKEN_CAP", 200000),
        "max_context": _int("LLM_FALLBACK_MAX_CONTEXT", 8192),
        "api_key_env": "LLM_FALLBACK_API_KEY",
    }


def split_paragraphs(paragraphs: list[dict], max_context: int, reserve: int = 800) -> list[list[dict]]:
    """Split a packet so the page text fits a smaller context window."""
    budget = max(200, max_context - reserve)
    groups: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for paragraph in paragraphs:
        cost = estimate_tokens(paragraph.get("text") or "") + 4
        if current and used + cost > budget:
            groups.append(current)
            current = []
            used = 0
        current.append(paragraph)
        used += cost
    if current:
        groups.append(current)
    return groups


def fallback_caller(provider: dict):
    """Open the fallback client. This function does not read GROQ_API_KEY."""
    key = (os.environ.get("LLM_FALLBACK_API_KEY") or "").strip()
    if not key or provider.get("name", "").lower() == "groq" or key == (os.environ.get("GROQ_API_KEY") or "").strip():
        raise RuntimeError("The fallback provider is not configured as a non-Groq account.")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=provider["base_url"] or None, timeout=90)

    def create(payload: dict):
        body = {key: value for key, value in payload.items() if key != "reasoning_effort"}
        body["model"] = provider["model"] or body.get("model")
        return client.chat.completions.create(**body)

    return create


def run_fallback_batch(pages: list[dict], *, create, ledger, provider: dict) -> dict:
    """Send one batch through the fallback provider and record that provider on each page."""
    groups = split_paragraphs(
        [{"text": (page.get("text_blocks") or [{"text": ""}])[0].get("text") or "", "pdf_page": page["pdf_page"]} for page in pages],
        provider["max_context"],
    )
    used = 0
    sent = []
    for group in groups:
        packet = "\n".join(f"[P {item['pdf_page']}] {item['text']}" for item in group)
        estimate = estimate_tokens(packet)
        if used + estimate > provider["daily_token_cap"]:
            break
        result = create({"provider": provider["name"], "pages": [item["pdf_page"] for item in group], "packet": packet})
        usage = result.get("usage") or {}
        billed_in = int(usage.get("input_tokens") or 0)
        billed_out = int(usage.get("output_tokens") or 0)
        used += billed_in + billed_out
        page_numbers = [int(item["pdf_page"]) for item in group]
        ledger.add(
            ticker="SLICE",
            step="llm",
            pdf_pages=page_numbers,
            input_tokens=billed_in,
            output_tokens=billed_out,
            billed=True,
            provider=provider["name"],
        )
        sent.append({"pdf_pages": page_numbers, "provider": provider["name"]})
        if used >= provider["daily_token_cap"]:
            break
    return {"provider": provider["name"], "groups": len(groups), "sent": sent, "tokens": used}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)
