"""Build eval/cost/<TICKER>.json from ledger/tokens.jsonl (+ vision usage when present)."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, PROJECT_ROOT, VISION_DIR
from src.extract.ledger import Ledger
from src.pipeline.companies import COMPANIES, TEXT_RATES, VISION_RATES


def _vision_usage(ticker: str) -> dict:
    """Sum vision usage JSON files under data/processed/vision/usage (shared) + ticker cache."""
    totals = {
        "calls": 0,
        "pages": set(),
        "input_tokens": 0,
        "output_tokens": 0,
        "billed_calls": 0,
        "cache_hits": 0,
        "provider": "groq",
        "model": VISION_RATES.get("model"),
        "ticker_filter": ticker,
    }
    # Global vision usage logs (Groq vision is shared; attribute pages mentioning ticker when present).
    usage_dir = VISION_DIR / "usage"
    if usage_dir.is_dir():
        for path in sorted(usage_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            page_rows = [row for row in (data.get("pages") or []) if isinstance(row, dict)]
            ticker_pages = [row for row in page_rows if str(row.get("ticker") or "").upper() == ticker]
            # Vision was only run for POWERGRID in this project; skip other tickers unless tagged.
            if ticker != "POWERGRID" and not ticker_pages:
                continue
            use_rows = ticker_pages or page_rows
            # File-level totals are authoritative (per-key daily logs).
            totals["input_tokens"] += int(data.get("input_tokens") or 0)
            totals["output_tokens"] += int(data.get("output_tokens") or 0)
            totals["calls"] += int(data.get("requests") or len(use_rows) or 0)
            totals["billed_calls"] += int(data.get("successful") or data.get("requests") or 0)
            totals["cache_hits"] += int(data.get("cache_hits") or 0)
            for row in use_rows:
                if row.get("pdf_page"):
                    totals["pages"].add(int(row["pdf_page"]))
            if data.get("model"):
                totals["model"] = data["model"]
    # Fallback: count OCR halves with vision transcripts for page set if empty.
    if not totals["pages"]:
        ocr = PROJECT_ROOT / "data" / "processed" / "ocr" / ticker
        if ocr.is_dir():
            for path in ocr.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (data.get("vision_ocr") or {}).get("transcript"):
                    totals["pages"].add(int(data["pdf_page"]))
                    totals["calls"] = max(totals["calls"], 0) + 1
                    totals["billed_calls"] = totals["calls"]
    return {
        **totals,
        "pages": sorted(totals["pages"]),
        "page_count": len(totals["pages"]),
    }


def build_cost(ticker: str) -> dict:
    ticker = ticker.upper()
    ledger = Ledger(CANDIDATE_DIR / ticker / "ledger" / "tokens.jsonl")
    rows = ledger.rows()
    steps: dict[str, dict] = {}
    for row in rows:
        if row.get("ticker") not in (None, "", ticker):
            continue
        step = str(row.get("step") or "unknown")
        bucket = steps.setdefault(
            step,
            {
                "step": step,
                "calls": 0,
                "pages": set(),
                "input_tokens": 0,
                "output_tokens": 0,
                "billed_input_tokens": 0,
                "billed_output_tokens": 0,
                "billed_calls": 0,
                "cache_hit_calls": 0,
                "provider": row.get("provider") or "unknown",
            },
        )
        incoming = int(row.get("input_tokens") or 0)
        outgoing = int(row.get("output_tokens") or 0)
        bucket["calls"] += 1
        bucket["input_tokens"] += incoming
        bucket["output_tokens"] += outgoing
        for page in row.get("pdf_pages") or []:
            bucket["pages"].add(int(page))
        if row.get("billed"):
            bucket["billed_calls"] += 1
            bucket["billed_input_tokens"] += incoming
            bucket["billed_output_tokens"] += outgoing
            bucket["provider"] = row.get("provider") or bucket["provider"]
        else:
            bucket["cache_hit_calls"] += 1
            if not row.get("provider"):
                bucket["provider"] = "local"

    # Map ledger steps to report categories
    parsing_steps = {"rules", "tables", "baseline", "planned_input"}
    extraction_steps = {"llm", "overview"}
    parsing = _merge([steps[s] for s in parsing_steps if s in steps], "parsing")
    extraction = _merge([steps[s] for s in extraction_steps if s in steps], "extraction")
    vision = _vision_usage(ticker)
    vision_row = {
        "step": "vision",
        "calls": vision["calls"],
        "pages": vision["pages"],
        "page_count": vision["page_count"],
        "pdf_pages": vision["pages"],
        "input_tokens": vision["input_tokens"],
        "output_tokens": vision["output_tokens"],
        "billed_calls": vision["billed_calls"],
        "cache_hit_calls": vision["cache_hits"],
        "provider": vision["provider"],
        "model": vision.get("model"),
    }

    def usd(inp: int, out: int, rates: dict) -> float:
        return round((inp * rates["input"] + out * rates["output"]) / 1_000_000, 4)

    payload = {
        "ticker": ticker,
        "period": COMPANIES.get(ticker, {}).get("period"),
        "ledger_path": str(CANDIDATE_DIR / ticker / "ledger" / "tokens.jsonl"),
        "steps": {
            key: {
                **val,
                "pages": sorted(val["pages"]) if isinstance(val.get("pages"), set) else val.get("pages"),
                "page_count": len(val["pages"]) if isinstance(val.get("pages"), set) else val.get("page_count") or val.get("pages") or 0,
            }
            for key, val in steps.items()
        },
        "by_category": {
            "parsing": parsing,
            "vision": vision_row,
            "extraction": extraction,
        },
        "rates": {"text": TEXT_RATES, "vision": VISION_RATES},
        "estimated_usd": {
            "extraction": usd(extraction["billed_input_tokens"], extraction["billed_output_tokens"], TEXT_RATES),
            "vision": usd(vision_row["input_tokens"], vision_row["output_tokens"], VISION_RATES),
        },
    }
    # normalize parsing/extraction page fields
    for cat in ("parsing", "extraction"):
        row = payload["by_category"][cat]
        pages = row.get("pages")
        if isinstance(pages, set):
            row["pages"] = sorted(pages)
            row["page_count"] = len(row["pages"])
        elif isinstance(pages, list):
            row["page_count"] = len(pages)
    return payload


def _merge(buckets: list[dict], name: str) -> dict:
    out = {
        "step": name,
        "calls": 0,
        "pages": set(),
        "input_tokens": 0,
        "output_tokens": 0,
        "billed_input_tokens": 0,
        "billed_output_tokens": 0,
        "billed_calls": 0,
        "cache_hit_calls": 0,
        "provider": "mixed",
    }
    providers = set()
    for bucket in buckets:
        out["calls"] += bucket["calls"]
        out["input_tokens"] += bucket["input_tokens"]
        out["output_tokens"] += bucket["output_tokens"]
        out["billed_input_tokens"] += bucket["billed_input_tokens"]
        out["billed_output_tokens"] += bucket["billed_output_tokens"]
        out["billed_calls"] += bucket["billed_calls"]
        out["cache_hit_calls"] += bucket["cache_hit_calls"]
        out["pages"] |= set(bucket["pages"])
        providers.add(bucket.get("provider") or "unknown")
    out["provider"] = ",".join(sorted(providers)) if providers else "unknown"
    out["page_count"] = len(out["pages"])
    out["pages"] = sorted(out["pages"])
    return out


def write_all(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (PROJECT_ROOT / "eval" / "cost")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ticker in COMPANIES:
        payload = build_cost(ticker)
        path = out_dir / f"{ticker}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        results[ticker] = str(path)
        print(f"WROTE {path}")
        cat = payload["by_category"]
        for key in ("parsing", "vision", "extraction"):
            row = cat[key]
            print(
                f"  {ticker} {key}: calls={row.get('calls')} pages={row.get('page_count')} "
                f"in={row.get('input_tokens')} out={row.get('output_tokens')} "
                f"billed_calls={row.get('billed_calls')} cache={row.get('cache_hit_calls')} "
                f"provider={row.get('provider')}"
            )
    return results


if __name__ == "__main__":
    write_all()
