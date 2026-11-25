"""One grounded overview call. Reruns with the same pages use the cache."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.config.factsheet_keywords import BUSINESS_TOPIC, ESG_PAGE, OPERATIONS_DENSITY, OVERVIEW_PAGE, STRATEGY_PAGE
from src.config.paths import CANDIDATE_DIR, PAGE_DIR, STAGING_DIR
from src.extract.keys import key_pool, scrub
from src.extract.ledger import Ledger, estimate_tokens
from src.extract.select import page_text
from src.factsheet.names import assignment_key
from src.vision.client import parse_model_json, usage_counts

OVERVIEW_PROMPT = (
    "You write a company overview from the annual-report text in the user message. "
    "Use only that text. Do not add a product, business, subsidiary, or customer that is not written there. "
    "The one_liner must be one sentence that says what the company produces, which subsidiaries it produces through, "
    "and who it sells to, using words from the text. "
    "The summary is one paragraph of at most four sentences on subsidiaries, operations, production, and sales channels. "
    "Do not use strategy, vision, or ESG statements. "
    "Return JSON with two keys: one_liner and summary."
)
_MAX_CALL_TOKENS = 6000
_OUTPUT_CAP = 500

FALLBACK_ONE = "{ticker} FY25 operates the businesses described in its annual report."
FALLBACK_SUMMARY = (
    "The overview is written from the business-overview text. "
    "It is not a list of extracted item names."
)


def fallback_text(ticker: str) -> tuple[str, str]:
    return FALLBACK_ONE.format(ticker=ticker), FALLBACK_SUMMARY


def build_overview(ticker: str, pages: list[dict] | None = None, *, create=None, allow_model: bool = True) -> dict:
    """Return one_liner, summary, fallback, pages, and token counts. One model call at most."""
    pages = pages if pages is not None else _load_pages(ticker)
    chosen = select_overview_pages(pages)
    packet = _packet(chosen)
    digest = hashlib.sha256((OVERVIEW_PROMPT + "\n" + packet).encode("utf-8")).hexdigest()
    cache_path = CANDIDATE_DIR / ticker / "overview_cache" / f"{digest}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        cached["billed"] = False
        return cached
    if not chosen or not allow_model:
        return _fallback(ticker, chosen, "no_overview_pages" if not chosen else "model_not_called")
    try:
        result = _call(ticker, chosen, packet, create=create)
    except Exception as exc:
        failed = _fallback(ticker, chosen, f"{type(exc).__name__}: {scrub(str(exc))}")
        _write_cache(cache_path, failed)
        return failed
    _write_cache(cache_path, result)
    return result


def write_overview_sidecar(ticker: str, overview: dict, staging_root: Path | None = None) -> Path:
    root = staging_root or (STAGING_DIR / "factsheet")
    path = root / ticker / "overview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ticker": ticker,
        "fallback": bool(overview.get("fallback")),
        "reason": overview.get("reason") or "",
        "pdf_pages": overview.get("pdf_pages") or [],
        "cache_hit": bool(overview.get("cache_hit")),
        "billed": bool(overview.get("billed")),
        "input_tokens": int(overview.get("input_tokens") or 0),
        "output_tokens": int(overview.get("output_tokens") or 0),
        "key_label": overview.get("key_label") or "",
        "one_liner": overview.get("one_liner") or "",
        "summary": overview.get("summary") or "",
    }, indent=2), encoding="utf-8")
    return path


def record_overview(ticker: str, overview: dict) -> None:
    Ledger(CANDIDATE_DIR / ticker / "ledger" / "tokens.jsonl").add(
        ticker=ticker,
        step="overview",
        pdf_pages=list(overview.get("pdf_pages") or []),
        input_tokens=int(overview.get("input_tokens") or 0),
        output_tokens=int(overview.get("output_tokens") or 0),
        billed=bool(overview.get("billed")),
        provider="groq",
        key_label=str(overview.get("key_label") or ""),
    )


def select_overview_pages(pages: list[dict], limit: int = 3) -> list[dict]:
    """Business-description pages: subsidiaries, operations, production, and sales.

    Strategy and ESG headings are skipped. The call stays inside the 6,000 token cap.
    """
    ranked = []
    for page in pages:
        text = page_text(page)
        if not text or len(text) < 400:
            continue
        head = "\n".join(text.splitlines()[:12])[:900]
        if STRATEGY_PAGE.search(head) or ESG_PAGE.search(head):
            continue
        topics = {name for name, pattern in (
            ("subsidiary", re.compile(r"subsidiar|joint venture", re.I)),
            ("operations", re.compile(r"\bmines?\b|produc", re.I)),
            ("sales", re.compile(r"\bsales?\b|e-?\s*auction|\bcustomers?\b", re.I)),
        ) if pattern.search(text)}
        hits = len(BUSINESS_TOPIC.findall(text)) + len(OPERATIONS_DENSITY.findall(text))
        if len(topics) < 1 and not OVERVIEW_PAGE.search(head):
            continue
        if hits < 2 and not OVERVIEW_PAGE.search(head):
            continue
        ranked.append((len(topics), hits, len(text), int(page["pdf_page"]), page))
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return [item[4] for item in ranked[:limit]]


def _packet(pages: list[dict]) -> str:
    parts = []
    for page in pages:
        text = page_text(page).strip()
        if text:
            parts.append(f"PDF page {int(page['pdf_page'])}:\n{text}")
    packet = "\n\n".join(parts)
    budget = _MAX_CALL_TOKENS - _OUTPUT_CAP - estimate_tokens(OVERVIEW_PROMPT) - 200
    while packet and estimate_tokens(packet) > max(budget, 800):
        packet = packet[: int(len(packet) * 0.8)]
    return packet


def _call(ticker: str, pages: list[dict], packet: str, *, create=None) -> dict:
    from src.extract.settings import text_settings

    settings = text_settings()
    user = f"Ticker: {ticker}. Write the overview from this text only.\n\n{packet}"
    if estimate_tokens(OVERVIEW_PROMPT) + estimate_tokens(user) + _OUTPUT_CAP > _MAX_CALL_TOKENS:
        raise RuntimeError("Overview packet exceeds the 6000 token call cap.")
    payload = {
        "model": settings["model"],
        "temperature": 0,
        "max_completion_tokens": _OUTPUT_CAP,
        "reasoning_effort": "low",
        "tool_choice": "none",
        "tools": [],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": OVERVIEW_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if create is not None:
        response = create(payload)
    else:
        from groq import Groq

        import time

        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = Groq(
                    api_key=key_pool().current(),
                    timeout=settings["timeout_seconds"],
                ).chat.completions.create(**payload)
                break
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                text = scrub(str(exc)).casefold()
                daily = status == 429 and ("per day" in text or "daily" in text or "tpd" in text)
                if daily or attempt == 2 or status not in {408, 429, 500, 502, 503}:
                    raise
                time.sleep(20)
        if response is None:
            raise last_error or RuntimeError("overview call failed")
    raw_usage = getattr(response, "usage", None)
    usage = usage_counts(raw_usage)
    print(json.dumps({
        "overview_response_usage": {
            "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
            "completion_tokens": getattr(raw_usage, "completion_tokens", None),
            "total_tokens": getattr(raw_usage, "total_tokens", None),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "key_label": key_pool().label(),
        }
    }))
    raw = ""
    choices = getattr(response, "choices", None) or []
    if choices:
        raw = getattr(getattr(choices[0], "message", None), "content", "") or ""
    parsed, error = parse_model_json(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(error or "overview response was not JSON")
    source = packet
    one_liner = _plain_hyphens(_keep_supported(str(parsed.get("one_liner") or ""), source))
    summary = _plain_hyphens(_keep_supported(str(parsed.get("summary") or ""), source))
    if one_liner and not _produces_and_sells(one_liner):
        one_liner = _plain_hyphens(_source_trade_sentence(source) or "")
    if not one_liner or not summary:
        failed = _fallback(ticker, pages, "sentences_failed_source_check")
        failed["input_tokens"] = int(usage.get("input_tokens") or 0)
        failed["output_tokens"] = int(usage.get("output_tokens") or 0)
        failed["billed"] = True
        failed["key_label"] = key_pool().label()
        return failed
    return {
        "ticker": ticker,
        "one_liner": one_liner,
        "summary": summary,
        "fallback": False,
        "reason": "",
        "pdf_pages": [int(page["pdf_page"]) for page in pages],
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "billed": True,
        "cache_hit": False,
        "key_label": key_pool().label(),
    }


def _plain_hyphens(text: str) -> str:
    return (
        (text or "")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )


def _produces_and_sells(text: str) -> bool:
    produces = re.search(r"\b(produc\w*|mines?|mining)\b", text, re.I)
    sells = re.search(r"\b(sells?|sales|customers?|auction)\b", text, re.I)
    through = re.search(r"\b(subsidiar\w*|through)\b", text, re.I)
    return bool(produces and sells and through)


def _source_trade_sentence(source: str) -> str:
    for sentence in _sentences(source):
        if _produces_and_sells(sentence) and _sentence_supported(sentence, source, assignment_key(source) + " " + source.casefold()):
            return sentence if sentence.endswith(".") else sentence + "."
    return ""


def _keep_supported(text: str, source: str) -> str:
    """Drop a sentence when a named span in it is absent from the source text."""
    kept = []
    source_key = assignment_key(source) + " " + source.casefold()
    for sentence in _sentences(text):
        if _sentence_supported(sentence, source, source_key):
            kept.append(sentence if sentence.endswith(".") else sentence + ".")
    return " ".join(kept).strip()


def _sentence_supported(sentence: str, source: str, source_key: str) -> bool:
    spans = re.findall(
        r"\b([A-Z][A-Za-z0-9&.'’-]{2,}(?:\s+(?:&|and|[A-Z][A-Za-z0-9&.'’-]{2,}))+)\b",
        sentence,
    )
    spans.extend(re.findall(r"\b([A-Z][A-Za-z0-9&.'’-]{2,}\s+(?:Limited|Ltd\.?|Private Limited))\b", sentence))
    source_tokens = set(source_key.split())
    folded = source.casefold()
    for span in spans:
        if span.casefold() in folded:
            continue
        tokens = set(assignment_key(span).split())
        if tokens and tokens <= source_tokens:
            continue
        return False
    # Every capitalised name in the sentence has to occur in the source.
    for word in re.findall(r"\b[A-Z][A-Za-z]{3,}\b", sentence):
        if word.casefold() in {"limited", "private"}:
            continue
        if word.casefold() not in folded:
            return False
    return True


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.])\s+", text or "") if part.strip()]


def _fallback(ticker: str, pages: list[dict], reason: str) -> dict:
    one_liner, summary = fallback_text(ticker)
    return {
        "ticker": ticker,
        "one_liner": one_liner,
        "summary": summary,
        "fallback": True,
        "reason": reason,
        "pdf_pages": [int(page["pdf_page"]) for page in pages],
        "input_tokens": 0,
        "output_tokens": 0,
        "billed": False,
        "cache_hit": False,
        "key_label": "",
    }


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_pages(ticker: str) -> list[dict]:
    folder = PAGE_DIR / ticker
    pages = []
    if not folder.is_dir():
        return pages
    for path in sorted(folder.glob("p*.json")):
        pages.append(json.loads(path.read_text(encoding="utf-8")))
    return pages
