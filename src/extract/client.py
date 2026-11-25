"""Call the text model for one batch. Tools stay off. Tokens come from the response."""

from __future__ import annotations

import json
import re
import time

from pydantic import ValidationError

from src.extract.keys import key_pool, scrub
from src.extract.known import relevant_known_names
from src.extract.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from src.extract.schema import KINDS, Candidate, ExtractionResponse
from src.validate.policy import apply_policy
from src.vision.client import parse_model_json, usage_counts


def extract_batch(
    packet: str,
    *,
    ticker: str,
    target_pages: list[int],
    allowed_pages: set[int],
    settings: dict,
    create=None,
    sleep=time.sleep,
    known_names: list[str] | None = None,
    page_records: dict[int, dict] | None = None,
) -> dict:
    attempts = 0
    retried = 0
    last_text = ""
    last_error = None
    last_finish = ""
    usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    started = time.perf_counter()
    # Malformed / truncated JSON: at most one model attempt, then split upstream.
    malformed_budget = min(1, int(settings.get("max_attempts") or 1))
    while attempts < settings["max_attempts"]:
        attempts += 1
        try:
            response = _create(
                create, settings, packet, ticker, target_pages, known_names or [], page_records=page_records
            )
        except Exception as exc:
            last_error = scrub(f"{type(exc).__name__}: {exc}")
            status_code = getattr(exc, "status_code", None)
            if status_code in {413, 429}:
                last_error = scrub(f"{status_code}: {last_error}")
            delay = _delay(exc, 2 ** (attempts - 1))
            if _daily_exhausted(exc, delay):
                pool = key_pool()
                if pool.rotate():
                    last_error = f"daily_cap_rotated_to_{pool.label()}"
                    retried += 1
                    continue
                return _record(
                    ticker, target_pages, settings, attempts, retried, started, "daily_cap",
                    usage, last_text, [], [],
                    "All Groq keys exhausted for the daily token cap. Run the same command again after the limit resets. Finished pages are not sent again.",
                    finish_reason=last_finish,
                )
            if attempts >= settings["max_attempts"] or not _retryable(exc):
                break
            retried += 1
            sleep(delay)
            continue
        usage = usage_counts(getattr(response, "usage", None))
        last_text = _message_text(response)
        parsed, error = parse_model_json(last_text)
        finish = _finish_reason(response)
        last_finish = finish
        salvaged = 0
        lost = 0
        if (parsed is None or "candidates" not in (parsed or {})) and finish == "length":
            salvaged_items, lost = salvage_truncated_candidates(last_text)
            if salvaged_items:
                parsed = {"candidates": salvaged_items}
                salvaged = len(salvaged_items)
                error = None
        if parsed is None or "candidates" not in (parsed or {}):
            last_error = error or "The response has no candidates list."
            if finish == "length":
                last_error = f"truncated: {last_error}"
            # Do not keep retrying malformed/truncated JSON; split the batch instead.
            if attempts >= malformed_budget or finish == "length":
                break
            if attempts >= settings["max_attempts"]:
                break
            retried += 1
            sleep(1)
            continue
        accepted, rejected = _split_candidates(
            parsed.get("candidates") or [],
            allowed_pages,
            page_records=page_records,
        )
        record = _record(
            ticker, target_pages, settings, attempts, retried, started, "success",
            usage, last_text, accepted, rejected, None,
            finish_reason=finish,
        )
        if salvaged or lost:
            record["salvaged_items"] = salvaged
            record["lost_items_estimate"] = lost
            record["truncated_salvaged"] = True
        # If truncated but we salvaged something, still ask the caller to split when
        # the batch had multiple pages so remaining content can be re-sent.
        if finish == "length" and len(target_pages) > 1:
            record["request_status"] = "malformed_response" if not accepted else "success"
            record["needs_split"] = True
            if not accepted:
                record["error"] = last_error or "truncated"
        return record
    status = "malformed_response" if last_text else "failed"
    if last_finish == "length" or (last_error and "truncated" in last_error.casefold()):
        # Final attempt: try salvage even if earlier parse failed without finish tag.
        salvaged_items, lost = salvage_truncated_candidates(last_text)
        if salvaged_items:
            accepted, rejected = _split_candidates(salvaged_items, allowed_pages, page_records=page_records)
            record = _record(
                ticker, target_pages, settings, attempts, retried, started,
                "success" if accepted else "malformed_response",
                usage, last_text, accepted, rejected,
                None if accepted else (last_error or "truncated"),
                finish_reason=last_finish or "length",
            )
            record["salvaged_items"] = len(salvaged_items)
            record["lost_items_estimate"] = lost
            record["truncated_salvaged"] = True
            record["needs_split"] = len(target_pages) > 1
            return record
        status = "malformed_response"
    return _record(
        ticker, target_pages, settings, attempts, retried, started, status,
        usage, last_text, [], [], last_error,
        finish_reason=last_finish,
    )


def salvage_truncated_candidates(text: str) -> tuple[list, int]:
    """Keep complete candidate objects from a truncated JSON array."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    # Prefer the candidates array body.
    match = re.search(r'"candidates"\s*:\s*\[', raw)
    if not match:
        return [], 0
    body = raw[match.end() :]
    items = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(body):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                chunk = body[start : index + 1]
                try:
                    item = json.loads(chunk)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(item, dict):
                    items.append(item)
                start = None
            continue
        if char == "]" and depth == 0:
            break
    # Estimate lost trailing material after the last complete object.
    lost = 1 if re.search(r'\{[^}]*$', body[body.rfind("}") + 1 :] if "}" in body else body) else 0
    return items, lost


def _split_candidates(
    items: list,
    allowed_pages: set[int],
    page_records: dict[int, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    for item in items:
        try:
            candidate = Candidate.model_validate(_expand_item(item, page_records=page_records))
        except ValidationError as exc:
            rejected.append({"item": item, "error": str(exc)})
            continue
        pages = {source.pdf_page for source in candidate.sources}
        if not pages.issubset(allowed_pages):
            rejected.append({"item": item, "error": f"Quote page is outside this batch: {sorted(pages - allowed_pages)}"})
            continue
        revised, reason = apply_policy(candidate.model_dump())
        if reason:
            rejected.append({"item": item, "error": reason})
            continue
        amount_reason = _reject_unverified_amount(revised, page_records)
        if amount_reason:
            rejected.append({"item": item, "error": amount_reason})
            continue
        name_reason = _reject_unverified_name(revised, page_records)
        if name_reason:
            rejected.append({"item": item, "error": name_reason})
            continue
        accepted.append(_fill_printed_pages(revised, page_records))
    return accepted, rejected


def _expand_item(item: dict, page_records: dict[int, dict] | None = None) -> dict:
    """Turn compact keys into the stored candidate shape. Full items pass through."""
    if not isinstance(item, dict):
        return item
    if "sources" in item and "name" in item:
        return _fill_printed_pages(item, page_records)
    name = str(item.get("n") or item.get("name") or "").strip()
    kind = item.get("k") or item.get("kind") or "other"
    if kind not in KINDS:
        kind = "other"
    descriptor = item.get("d") if item.get("d") is not None else item.get("descriptor")
    descriptor = " ".join(str(descriptor or name).split()[:12])
    quote = item.get("q") if item.get("q") is not None else item.get("quote")
    page = item.get("p") if item.get("p") is not None else item.get("pdf_page")
    status = item.get("s") or item.get("status") or "current"
    if status not in {"current", "planned", "aspirational"}:
        status = "current"
    try:
        page_number = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_number = None
    printed = _printed_for_page(page_records, page_number, "" if quote is None else str(quote))
    return {
        "name": name,
        "kind": kind,
        "descriptor": descriptor,
        "entity": name if kind == "subsidiary" else "company",
        "status": status,
        "revenue_share_pct": None,
        "sources": [{"pdf_page": page, "printed_page": printed, "quote": "" if quote is None else str(quote)}],
        "supporting_figures": [],
    }


def _fill_printed_pages(item: dict, page_records: dict[int, dict] | None) -> dict:
    sources = []
    for source in item.get("sources") or []:
        page = source.get("pdf_page")
        try:
            number = int(page) if page is not None else None
        except (TypeError, ValueError):
            number = None
        printed = source.get("printed_page")
        if printed in (None, "", "null"):
            printed = _printed_for_page(page_records, number, source.get("quote") or "")
        sources.append({**source, "printed_page": printed})
    return {**item, "sources": sources}


def _printed_for_page(page_records: dict[int, dict] | None, pdf_page: int | None, quote: str) -> str | None:
    if page_records is None or pdf_page is None:
        return None
    page = page_records.get(int(pdf_page))
    if not page:
        return None
    label = page.get("printed_page")
    if label in (None, "", "null"):
        return None
    text = str(label)
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if not parts:
        return text
    half = None
    for block in page.get("text_blocks") or []:
        if quote and quote in (block.get("text") or ""):
            half = block.get("half") or "FULL"
            break
    if half == "LEFT":
        return parts[0]
    if half == "RIGHT" and len(parts) > 1:
        return parts[1]
    return text


_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _page_unverified(page_records: dict[int, dict] | None, pdf_page: int | None) -> bool:
    if page_records is None or pdf_page is None:
        return False
    page = page_records.get(int(pdf_page)) or {}
    quality = page.get("quality_metadata") or {}
    return bool(quality.get("numeric_unverified"))


def _ocr_number_text(page: dict) -> str:
    """Non-vision text on the page. Vision digits are not corroboration."""
    parts = []
    for block in page.get("text_blocks") or []:
        if (block.get("source") or "") in {"vision_ocr", "vision"}:
            continue
        parts.append(block.get("text") or "")
    return re.sub(r"[, ]", "", " ".join(parts))


def _norm_amount(amount: str) -> str:
    return re.sub(r"[,\s]", "", amount or "")


def _verified_amount_norms(page: dict) -> set[str] | None:
    """None when this page has no verified_numbers list yet (older merges)."""
    quality = page.get("quality_metadata") or {}
    vision = page.get("vision_ocr") if isinstance(page.get("vision_ocr"), dict) else {}
    if "verified_numbers" not in quality and not any(
        isinstance(slot, dict) and "verified_numbers" in slot for slot in vision.values()
    ):
        return None
    rows = list(quality.get("verified_numbers") or [])
    for slot in vision.values():
        if isinstance(slot, dict):
            rows.extend(slot.get("verified_numbers") or [])
    return {_norm_amount(str(row.get("value") if isinstance(row, dict) else row)) for row in rows} - {""}


def _reject_unverified_amount(item: dict, page_records: dict[int, dict] | None) -> str | None:
    """On numeric_unverified pages, keep amounts that are in verified_numbers and skip the rest.

    Pages merged before verified_numbers existed still allow a number only when
    the non-vision OCR text contains it.
    """
    for source in item.get("sources") or []:
        try:
            pdf_page = int(source.get("pdf_page"))
        except (TypeError, ValueError):
            continue
        if not _page_unverified(page_records, pdf_page):
            continue
        quote = str(source.get("quote") or "")
        amounts = _AMOUNT.findall(quote)
        if not amounts:
            continue
        page = (page_records or {}).get(pdf_page) or {}
        verified = _verified_amount_norms(page)
        if verified is None:
            corpus = _ocr_number_text(page)
            missing = [amount for amount in amounts if _norm_amount(amount) not in corpus]
            if missing:
                return (
                    "numeric_unverified: skip amount unless it also appears in OCR text "
                    f"on pdf {pdf_page}: {', '.join(missing)}"
                )
            continue
        missing = [amount for amount in amounts if _norm_amount(amount) not in verified]
        if missing:
            return (
                "numeric_unverified: skip amount unless it is in verified_numbers "
                f"on pdf {pdf_page}: {', '.join(missing)}"
            )
    return None


def _unverified_instruction(page_records: dict[int, dict] | None, target_pages: list[int]) -> str:
    flagged = [str(int(number)) for number in target_pages if _page_unverified(page_records, int(number))]
    if not flagged:
        return ""
    pages = ", ".join(flagged)
    return (
        f"Pages {pages} are numeric_unverified. "
        "If a page lists verified_numbers, you may extract an amount only when that number is in the list. "
        "Skip every other amount. "
        "If a page has no verified_numbers list, do not extract an amount unless that same number "
        "also appears in the non-vision OCR text on that page. "
        "Names and labels with no amount are allowed.\n\n"
    )


def _name_rows(page: dict) -> tuple[list, list]:
    quality = page.get("quality_metadata") or {}
    verified = list(quality.get("verified_names") or [])
    unverified = list(quality.get("unverified_names") or [])
    vision = page.get("vision_ocr") if isinstance(page.get("vision_ocr"), dict) else {}
    for slot in vision.values():
        if isinstance(slot, dict):
            verified.extend(slot.get("verified_names") or [])
            unverified.extend(slot.get("unverified_names") or [])
    return verified, unverified


def _reject_unverified_name(item: dict, page_records: dict[int, dict] | None) -> str | None:
    """A person name is usable only when its surname was corroborated."""
    candidate = str(item.get("name") or "")
    for source in item.get("sources") or []:
        try:
            pdf_page = int(source.get("pdf_page"))
        except (TypeError, ValueError):
            continue
        page = (page_records or {}).get(pdf_page) or {}
        _verified, unverified = _name_rows(page)
        if not unverified:
            continue
        haystack = f"{candidate}\n{source.get('quote') or ''}".casefold()
        for row in unverified:
            full = str(row.get("name") or "").strip()
            if len(full) >= 5 and full.casefold() in haystack:
                return f"unverified name on pdf {pdf_page}: {full}"
    return None


def _name_instruction(page_records: dict[int, dict] | None, target_pages: list[int]) -> str:
    flagged = []
    for number in target_pages:
        page = (page_records or {}).get(int(number)) or {}
        _verified, unverified = _name_rows(page)
        if unverified:
            flagged.append(str(int(number)))
    if not flagged:
        return ""
    return (
        f"Pages {', '.join(flagged)} list unverified_names. "
        "Do not extract a person whose name is in that list. "
        "Company names are allowed.\n\n"
    )


def _create(
    create,
    settings: dict,
    packet: str,
    ticker: str,
    target_pages: list[int],
    known_names: list[str],
    page_records: dict[int, dict] | None = None,
):
    limited = relevant_known_names(known_names, packet, limit=40)
    names = ", ".join(limited) if limited else "none"
    user = (
        f"Ticker: {ticker}. Target pdf pages: {target_pages}. "
        f"Already found: {names}. Extract only new items.\n\n"
        f"{_unverified_instruction(page_records, target_pages)}"
        f"{_name_instruction(page_records, target_pages)}"
        f"{packet}"
    )
    payload = {
        "model": settings["model"],
        "temperature": 0,
        "max_completion_tokens": settings["max_output_tokens"],
        "reasoning_effort": "low",
        "tool_choice": "none",
        "tools": [],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if create is not None:
        return create(payload)
    from groq import Groq

    client = Groq(api_key=key_pool().current(), timeout=settings["timeout_seconds"])
    return client.chat.completions.create(**payload)


def _daily_exhausted(exc: Exception, delay: float) -> bool:
    """True for a daily token cap, not a short per-minute 429."""
    status = getattr(exc, "status_code", None)
    text = scrub(str(exc)).casefold()
    daily = "per day" in text or "daily" in text or "tpd" in text or "tokens per day" in text
    if status == 429 and (daily or delay > 120):
        return True
    return daily


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _delay(exc: Exception, fallback: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return min(fallback, 30.0)


def _finish_reason(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0], "finish_reason", "") or ""


def _message_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return getattr(getattr(choices[0], "message", None), "content", "") or ""


def _record(
    ticker,
    target_pages,
    settings,
    attempts,
    retried,
    started,
    status,
    usage,
    raw,
    accepted,
    rejected,
    error,
    finish_reason: str = "",
) -> dict:
    return {
        "ticker": ticker,
        "target_pages": target_pages,
        "model": settings["model"],
        "prompt_version": PROMPT_VERSION,
        "request_status": status,
        "attempts": attempts,
        "retried": retried,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
        "raw_response": raw,
        "candidates": accepted,
        "rejected": rejected,
        "error": error,
        "finish_reason": finish_reason or None,
        "tools_enabled": False,
        "key_label": key_pool().label(),
    }


def response_schema_check(payload: dict) -> ExtractionResponse:
    return ExtractionResponse.model_validate({"candidates": payload.get("candidates") or []})
