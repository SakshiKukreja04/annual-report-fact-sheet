"""Extract a few filter-dropped pages into eval/. Stops before the daily cap."""

from __future__ import annotations

import json
import re

from src.config.paths import CANDIDATE_DIR, PROJECT_ROOT
from src.extract.client import extract_batch
from src.extract.evidence import render_page
from src.extract.ledger import estimate_tokens
from src.extract.run import load_saved_company
from src.extract.evidence import pages_for_extraction
from src.extract.settings import text_settings
from src.validate.quotes import locate_quote
from src.vision.usage import record_attempt, tokens_used, usage_path

_IN_SCOPE = re.compile(
    r"subsidiar|segment|aluminium|aluminum|copper|coal|capacity|plant|mine|smelter|hospital|school|community|product|brand|joint venture",
    re.I,
)
_OUT_OF_SCOPE = re.compile(
    r"e-?\s*voting|instavote|instameet|\bagm\b|scrutinizer|secretarial auditor|unclaimed dividend|remuneration",
    re.I,
)
MAX_CALLS = 4


def _original(page: dict) -> str:
    return "\n".join((block.get("text") or "") for block in page.get("text_blocks") or [])


def _in_scope(item: dict) -> bool:
    text = " ".join([item.get("name") or ""] + [source.get("quote") or "" for source in item.get("sources") or []])
    if _OUT_OF_SCOPE.search(text) and not _IN_SCOPE.search(text):
        return False
    return bool(_IN_SCOPE.search(text))


def main() -> None:
    sample = json.loads((PROJECT_ROOT / "eval" / "dropped_sample.json").read_text(encoding="utf-8"))
    settings = text_settings()
    log_path = usage_path(CANDIDATE_DIR)
    settings["provider"] = "groq"
    sent = []
    calls = 0
    for ticker, rows in sample["companies"].items():
        if ticker == "POWERGRID":
            continue
        pages = {int(page["pdf_page"]): page for page in pages_for_extraction(load_saved_company(ticker))}
        per_company = 0
        for row in rows:
            if calls >= MAX_CALLS or per_company >= 2:
                break
            page = pages.get(int(row["pdf_page"]))
            if page is None:
                continue
            packet = render_page(page, "target")
            estimate = estimate_tokens(packet) + settings["max_output_tokens"]
            if tokens_used(log_path) + estimate > settings["daily_token_cap"]:
                sent.append({"ticker": ticker, "pdf_page": row["pdf_page"], "status": "not_sent_daily_cap"})
                calls = MAX_CALLS
                break
            result = extract_batch(
                packet,
                ticker=ticker,
                target_pages=[int(row["pdf_page"])],
                allowed_pages={int(row["pdf_page"])},
                settings=settings,
            )
            record_attempt(log_path, {**result, "half": None, "pdf_page": int(row["pdf_page"])})
            original = _original(page)
            items = []
            for candidate in result.get("candidates") or []:
                quotes = []
                for source in candidate.get("sources") or []:
                    quote = source.get("quote") or ""
                    quotes.append({"quote": quote, "match": locate_quote(quote, original)})
                items.append({
                    "name": candidate.get("name"),
                    "kind": candidate.get("kind"),
                    "quotes": quotes,
                    "in_scope": _in_scope(candidate),
                    "quote_ok": all(quote["match"] for quote in quotes),
                })
            sent.append({
                "ticker": ticker,
                "pdf_page": int(row["pdf_page"]),
                "reason": row["reason"],
                "status": result.get("request_status"),
                "usage": result.get("usage"),
                "items": items,
            })
            calls += 1
            per_company += 1
    destination = PROJECT_ROOT / "eval" / "dropped_extract.json"
    destination.write_text(json.dumps({"max_calls": MAX_CALLS, "sent": sent, "tokens_used_after": tokens_used(log_path)}, indent=2), encoding="utf-8")
    print(json.dumps({"calls": calls, "tokens_used_after": tokens_used(log_path), "pages": [row.get("pdf_page") for row in sent]}, indent=2))


if __name__ == "__main__":
    main()
