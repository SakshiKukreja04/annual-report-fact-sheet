"""Token ledger. Estimates use the same characters/4 rule as the daily cap."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# The prompt used before compact extraction. Kept so the baseline stays comparable.
LEGACY_SYSTEM_PROMPT = """You extract business-activity candidates from the annual-report text supplied in this request.
Use only that text. Do not use outside knowledge, memory of the company, or the web.
Do not invent activities to reach any count. If a page has none, return an empty list.
Every candidate needs one to three verbatim quotes copied from the supplied text.
A quote must keep the original wording and the pdf_page printed beside that text.
Do not set revenue_share_pct unless the quote itself states that percentage. Otherwise use null.
Do not calculate a percentage from other numbers.
When a printed figure supports an item, copy it into supporting_figures with its quote and pdf_page.
Use entity "company" when the reporting company does the activity.
Use the printed legal name when a subsidiary, joint venture, or other entity does it.
Status is current, planned, or aspirational, based only on the tense of the quote.
Return JSON with one key, candidates. Each candidate has name, kind, descriptor, entity, status, revenue_share_pct, sources, and supporting_figures.
kind is one of: product_or_service, brand, business_line, segment, subsidiary, channel, asset, research_platform, philanthropy, other.
sources has pdf_page, printed_page, and quote. At most three sources.
Do not call tools. Do not browse.
"""


TOKEN_ESTIMATE_FACTOR = 1.15


def estimate_tokens(text: str, factor: float = TOKEN_ESTIMATE_FACTOR) -> int:
    """chars/4 with a safety factor for the daily-cap and planning guards."""
    return int(round((len(text or "") / 4.0) * factor))


def planned_input_tokens(
    pages: list[dict],
    by_page: dict[int, list[dict]],
    ticker: str,
    max_tokens: int,
    known_names: list[str],
    *,
    filter_known: bool = True,
) -> int:
    """Input tokens after filtering, cleanup, and sending each paragraph once."""
    from src.extract.compact import pack_pages, render_new
    from src.extract.known import relevant_known_names
    from src.extract.prompt import SYSTEM_PROMPT

    batches = pack_pages(pages, by_page, max_tokens, preserve_order=True)
    seen: set[str] = set()
    total = 0
    for batch in batches:
        paragraphs = []
        for page in batch["pages"]:
            paragraphs.extend(by_page.get(int(page["pdf_page"]), []))
        packet, fresh = render_new(paragraphs, seen)
        seen.update(fresh)
        if filter_known:
            limited = relevant_known_names(known_names, packet, limit=40)
        else:
            limited = list(known_names)[:40]
        names = ", ".join(limited) if limited else "none"
        user = (
            f"Ticker: {ticker}. Target pdf pages: {batch['target_pages']}. "
            f"Already found: {names}. Extract only new items.\n\n{packet}"
        )
        total += estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user)
    return total


def legacy_input_tokens(pages: list[dict], ticker: str = "TICKER") -> dict:
    """Input tokens of the previous pipeline: full pages, one-page overlap, no filter."""
    from src.extract.batches import make_batches
    from src.extract.evidence import render_batch

    batches = make_batches(pages, max_chars=12000, overlap=1)
    total = 0
    for batch in batches:
        user = (
            f"Ticker label: {ticker}. Target pdf pages: {batch['target_pages']}. "
            "Extract candidates whose quotes appear in this packet.\n\n"
            f"{render_batch(batch)}"
        )
        total += estimate_tokens(LEGACY_SYSTEM_PROMPT) + estimate_tokens(user)
    return {
        "step": "legacy_input",
        "pages": len(pages),
        "batches": len(batches),
        "input_tokens": total,
    }


class Ledger:
    """Append-only record of billed calls and cache hits. One JSON object per line."""

    def __init__(self, path: Path):
        self.path = path

    def add(
        self,
        *,
        ticker: str,
        step: str,
        pdf_pages: list[int],
        input_tokens: int,
        output_tokens: int,
        billed: bool,
        provider: str = "groq",
        key_label: str = "",
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ticker": ticker,
            "step": step,
            "pdf_pages": pdf_pages,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "billed": billed,
            "provider": provider,
            "key_label": key_label or "",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
