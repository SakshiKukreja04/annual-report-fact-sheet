"""Repair saved quotes in memory and write eval numbers. Fact sheets are not touched."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, PAGE_DIR, PROJECT_ROOT
from src.extract.evidence import pages_for_extraction, render_page
from src.extract.ledger import estimate_tokens
from src.extract.run import load_saved_company
from src.extract.select import select_pages
from src.validate.quotes import _slice, locate_quote, normalize_with_map, snap_quote

_ENTITY = re.compile(r"\b([A-Z][A-Za-z0-9&.'’\-]{1,40}(?:\s+[A-Z][A-Za-z0-9&.'’\-]{1,40}){0,5}\s+(?:Limited|Ltd\.?|Pvt\.?|Division|Plant|Mine|Brand))\b")


def _page_strings(ticker: str) -> dict[int, str]:
    texts = {}
    for page in pages_for_extraction(load_saved_company(ticker, PAGE_DIR)):
        blocks = page.get("text_blocks") or []
        texts[int(page["pdf_page"])] = "\n".join((block.get("text") or "") for block in blocks)
    return texts


def _candidates(ticker: str) -> list[dict]:
    items = []
    folder = CANDIDATE_DIR / ticker / "parsed"
    if not folder.is_dir():
        return items
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "rules":
            continue
        items.extend(record.get("candidates") or [])
    return items


def _known_names(items: list[dict]) -> set[str]:
    return {(item.get("name") or "").casefold() for item in items}


def repair_quotes(items: list[dict], texts: dict[int, str]) -> dict:
    norms = {number: normalize_with_map(text) for number, text in texts.items()}
    counts = {"strict": 0, "normalized": 0, "repaired": 0, "dropped": 0, "page_fixed": 0}
    rejects = []
    items_dropped = 0
    checked = 0
    for item in items:
        kept = 0
        for source in item.get("sources") or []:
            quote = source.get("quote") or ""
            cited = source.get("pdf_page")
            checked += 1
            page_text = texts.get(cited, "") if isinstance(cited, int) else ""
            kind = locate_quote(quote, page_text)
            if kind == "exact":
                counts["strict"] += 1
                kept += 1
                continue
            if kind == "normalized":
                counts["normalized"] += 1
                kept += 1
                continue
            snap = snap_quote(quote, page_text) if page_text else {"quote": None, "ratio": 0.0, "reason": "empty"}
            if snap.get("quote"):
                counts["repaired"] += 1
                kept += 1
                continue
            alt_page, alt = _other_page(quote, cited, texts, norms)
            if alt and alt.get("quote"):
                counts["repaired"] += 1
                counts["page_fixed"] += 1
                kept += 1
                continue
            counts["dropped"] += 1
            if len(rejects) < 10:
                rejects.append({
                    "name": item.get("name"),
                    "pdf_page": cited,
                    "quote": quote[:180],
                    "reason": snap.get("reason") or "below_0.9",
                    "best_ratio": snap.get("ratio"),
                    "other_page": alt_page,
                })
        if item.get("sources") and kept == 0:
            items_dropped += 1
    counts["checked"] = checked
    counts["items_with_no_quote_left"] = items_dropped
    return {"counts": counts, "reject_sample": rejects}


def _other_page(quote: str, cited, texts: dict[int, str], norms: dict[int, tuple[str, list[int]]]):
    """A quote moved by paragraph dedupe is an exact normalized hit on another page."""
    norm_quote, _mapping = normalize_with_map(quote)
    if len(norm_quote) < 12:
        return None, None
    for number, (norm_page, mapping) in norms.items():
        if number == cited:
            continue
        found = norm_page.find(norm_quote)
        if found < 0:
            continue
        snapped = _slice(texts[number], mapping, found, found + len(norm_quote))
        if snapped:
            return number, {"quote": snapped, "ratio": 1.0, "reason": "other_page"}
    return None, None


def token_gap(ticker: str) -> dict:
    texts_pages = {int(page["pdf_page"]): page for page in pages_for_extraction(load_saved_company(ticker, PAGE_DIR))}
    rows = []
    folder = CANDIDATE_DIR / ticker / "parsed"
    for path in folder.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        if record.get("request_status") != "success" or not usage.get("input_tokens"):
            continue
        estimate = 0
        for number in record.get("target_pages") or []:
            page = texts_pages.get(int(number))
            if page is not None:
                estimate += estimate_tokens(render_page(page, "target"))
        if estimate <= 0:
            continue
        rows.append((path.stat().st_mtime, estimate, int(usage["input_tokens"]), int(usage.get("output_tokens") or 0), record.get("target_pages")))
    rows.sort()
    sample = rows[:20] or rows
    fit = rows or sample
    xs = [row[1] for row in fit]
    ys = [row[2] for row in fit]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var = sum((x - mean_x) ** 2 for x in xs) or 1
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    prompt = mean_y - slope * mean_x
    body = [y - prompt for _x, y in zip(xs, ys)]
    tokenizer_gap = sum((slope - 1) * x for x in xs)
    prompt_total = prompt * len(xs)
    before = sum(row[2] for row in sample) / len(sample)
    groups = [sample[index : index + 6] for index in range(0, min(len(sample), 18), 6)]
    after = []
    for group in groups:
        page_body = sum(row[2] - prompt for row in group)
        after.append((prompt + page_body) / len(group))
    return {
        "calls": len(sample),
        "fit_calls": len(fit),
        "prompt_tokens_per_call": round(prompt, 1),
        "tokenizer_ratio_vs_chars": round(slope, 3),
        "prompt_overhead_on_sample": round(prompt_total, 1),
        "tokenizer_gap_on_sample": round(tokenizer_gap, 1),
        "tokens_per_page_before": round(before, 1),
        "tokens_per_page_after_groups_of_6": [round(value, 1) for value in after],
        "output_tokens_on_sample": sum(row[3] for row in sample),
    }


def dropped_entities(ticker: str, known: set[str]) -> list[dict]:
    pages = pages_for_extraction(load_saved_company(ticker, PAGE_DIR))
    selection = select_pages(pages)
    dropped = {row["pdf_page"] for row in selection["gaps"]}
    score = {row["pdf_page"]: row.get("score") or 0 for row in selection["pages"]}
    found = []
    for page in pages:
        number = int(page["pdf_page"])
        if number not in dropped:
            continue
        text = "\n".join((block.get("text") or "") for block in page.get("text_blocks") or [])
        for match in _ENTITY.finditer(text):
            name = " ".join(match.group(1).split())
            if name.casefold() in known or len(name) < 6:
                continue
            start = max(0, match.start() - 40)
            found.append({"ticker": ticker, "pdf_page": number, "score": score.get(number, 0), "name": name, "context": text[start : match.end() + 40].replace("\n", " ")})
    found.sort(key=lambda row: (-row["score"], row["pdf_page"], row["name"]))
    unique = []
    seen = set()
    for row in found:
        key = (row["ticker"], row["name"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique[:40]


def _page_cost(page: dict, paragraphs: list[dict], prompt: float, ratio: float) -> int:
    text = "\n".join(item["text"] for item in paragraphs)
    body = estimate_tokens(text) * max(ratio, 0)
    return max(1, round(body + prompt / 6))


def schedule(prompt: float, ratio: float, budget: int) -> dict:
    from src.extract.compact import paragraphs_for_page, repeated_lines
    from src.extract.run import _covered_pages

    rows = []
    for ticker in ("HINDALCO", "COALINDIA", "POWERGRID"):
        pages = pages_for_extraction(load_saved_company(ticker, PAGE_DIR))
        selection = select_pages(pages)
        kept = {int(number) for number in selection["kept"]}
        scores = {int(row["pdf_page"]): row.get("score") or 0 for row in selection["pages"]}
        banners = repeated_lines(pages)
        covered = _covered_pages(CANDIDATE_DIR / ticker)
        for page in pages:
            number = int(page["pdf_page"])
            if number not in kept or number in covered:
                continue
            reason = next((row.get("reason") or "" for row in selection["pages"] if int(row["pdf_page"]) == number), "")
            paragraphs = paragraphs_for_page(page, banners, scope_only=reason.endswith("_in_scope"))
            rows.append({
                "ticker": ticker,
                "pdf_page": number,
                "score": scores.get(number, 0),
                "tokens": _page_cost(page, paragraphs, prompt, ratio),
            })
    rows.sort(key=lambda row: (-row["score"], row["ticker"], row["pdf_page"]))
    left = budget
    uncovered = []
    needed = {"HINDALCO": 0, "COALINDIA": 0, "POWERGRID": 0}
    for row in rows:
        needed[row["ticker"]] += row["tokens"]
        if left >= row["tokens"]:
            left -= row["tokens"]
        else:
            uncovered.append(row)
    by_ticker = {}
    for row in uncovered:
        by_ticker.setdefault(row["ticker"], []).append(row["pdf_page"])
    ranges = {ticker: _ranges(sorted(pages)) for ticker, pages in by_ticker.items()}
    return {
        "budget_tokens": budget,
        "tokens_needed": needed,
        "tokens_left_if_highest_score_first": left if not uncovered else 0,
        "uncovered_page_counts": {ticker: len(pages) for ticker, pages in by_ticker.items()},
        "uncovered_pages": by_ticker,
        "uncovered_ranges": ranges,
    }


def _ranges(pages: list[int]) -> str:
    if not pages:
        return ""
    chunks = []
    start = previous = pages[0]
    for number in pages[1:]:
        if number == previous + 1:
            previous = number
            continue
        chunks.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = number
    chunks.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ", ".join(chunks)


def fallback_slice() -> dict:
    from src.extract.ledger import Ledger
    from src.extract.provider import fallback_provider, run_fallback_batch

    configured = fallback_provider()
    if configured is not None:
        return {"ran": False, "reason": "fallback key is set; this slice does not call it"}
    path = PROJECT_ROOT / "eval" / "fallback_ledger.jsonl"
    if path.is_file():
        path.unlink()
    ledger = Ledger(path)
    pages = [
        {"pdf_page": 2, "text_blocks": [{"text": "Aluminium smelter at Hirakud. " * 40}]},
        {"pdf_page": 3, "text_blocks": [{"text": "Copper refinery at Dahej. " * 40}]},
        {"pdf_page": 4, "text_blocks": [{"text": "Utkal alumina refinery. " * 40}]},
    ]

    def create(payload):
        return {"usage": {"input_tokens": estimate_tokens(payload["packet"]), "output_tokens": 12}}

    result = run_fallback_batch(
        pages,
        create=create,
        ledger=ledger,
        provider={"name": "other", "max_context": 900, "daily_token_cap": 500},
    )
    return {"ran": True, "live_call": False, "reason": "LLM_FALLBACK_PROVIDER is unset, so Groq was not called", **result, "ledger_rows": ledger.rows()}


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    texts = _page_strings("HINDALCO")
    items = _candidates("HINDALCO")
    repaired = repair_quotes(items, texts)
    gap = token_gap("HINDALCO")
    coal_items = _candidates("COALINDIA")
    entities = dropped_entities("HINDALCO", _known_names(items)) + dropped_entities("COALINDIA", _known_names(coal_items))
    entities = sorted(entities, key=lambda row: (-row["score"], row["ticker"], row["pdf_page"]))[:40]
    from src.vision.usage import tokens_used, usage_path

    used_today = tokens_used(usage_path(CANDIDATE_DIR))
    remaining_today = max(0, 200000 - used_today)
    prompt = gap["prompt_tokens_per_call"]
    ratio = gap["tokenizer_ratio_vs_chars"] or 1
    two_day_budget = remaining_today + 200000 * 2
    plan = schedule(prompt, ratio, two_day_budget)
    fallback = fallback_slice()
    report = {
        "quotes": repaired,
        "token_gap": gap,
        "entities": entities,
        "providers_per_day": {"groq": 200000, "fallback": 0},
        "groq_used_today": used_today,
        "groq_remaining_today": remaining_today,
        "cutoff_budget": two_day_budget,
        "schedule": plan,
        "fallback": fallback,
    }
    destination = PROJECT_ROOT / "eval" / "quote_repair.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "quotes": repaired["counts"],
        "rejects": repaired["reject_sample"],
        "token_gap": gap,
        "entity_count": len(entities),
        "entities": entities,
        "groq_used_today": used_today,
        "groq_remaining_today": remaining_today,
        "schedule": {
            "budget_tokens": plan["budget_tokens"],
            "tokens_needed": plan["tokens_needed"],
            "uncovered_page_counts": plan["uncovered_page_counts"],
            "uncovered_ranges": plan["uncovered_ranges"],
        },
        "fallback": {key: fallback[key] for key in ("ran", "live_call", "reason", "groups", "sent", "tokens")},
    }, indent=2))


if __name__ == "__main__":
    main()
