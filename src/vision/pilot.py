"""Run vision on an explicit small set of pages. This does not scan the three reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.paths import LOG_DIR, PAGE_DIR, PREPROCESS_CACHE_DIR, VISION_DIR
from src.utils.logging_config import configure_logging
from src.vision.client import extract_image, prepare_image
from src.vision.route import route_page
from src.vision.settings import vision_settings
from src.vision.usage import record_attempt, tokens_used, usage_path

# One readable-but-garbled table half. Page 100 is illegible and is not included.
PILOT_TARGETS = (("POWERGRID", 200, "LEFT"),)


def cache_path(root: Path, ticker: str, pdf_page: int, half: str) -> Path:
    return root / "cache" / f"{ticker}_p{pdf_page:04d}_{half}.json"


def image_path(ticker: str, pdf_page: int, half: str) -> Path:
    return PREPROCESS_CACHE_DIR / ticker / f"p{pdf_page:04d}_{half}" / "full.png"


def load_page(ticker: str, pdf_page: int, page_dir: Path | None = None) -> dict:
    path = (page_dir or PAGE_DIR) / ticker / f"p{pdf_page:04d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_target(
    ticker: str,
    pdf_page: int,
    half: str,
    *,
    output_dir: Path | None = None,
    page: dict | None = None,
    table_statuses: list[str] | None = None,
    create=None,
    sleep=None,
    force: bool = False,
) -> dict:
    """Route one half, then call vision only when it is eligible and not already cached."""
    output_dir = output_dir or VISION_DIR
    page = page if page is not None else load_page(ticker, pdf_page)
    decision = route_page(page, table_statuses=table_statuses)
    destination = cache_path(output_dir, ticker, pdf_page, half)
    if not decision["eligible"]:
        refused = {
            "ticker": ticker,
            "pdf_page": pdf_page,
            "half": half,
            "request_status": "not_routed",
            "routing_reason": decision["routing_reason"],
            "source_of_truth": False,
            "quote_validated": False,
            "model": vision_settings()["model"],
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        }
        record_attempt(usage_path(output_dir), refused)
        return refused
    if destination.is_file() and not force:
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if cached.get("request_status") == "success":
            cached["request_status"] = "cached"
            record_attempt(usage_path(output_dir), cached, cached=True)
            return cached
    settings = vision_settings()
    log_path = usage_path(output_dir)
    if tokens_used(log_path) >= settings["daily_token_cap"]:
        blocked = {
            "ticker": ticker,
            "pdf_page": pdf_page,
            "half": half,
            "request_status": "daily_cap_reached",
            "routing_reason": decision["routing_reason"],
            "source_of_truth": False,
            "quote_validated": False,
            "model": settings["model"],
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        }
        record_attempt(log_path, blocked)
        return blocked
    source = image_path(ticker, pdf_page, half)
    image = prepare_image(source)
    kwargs = {"create": create}
    if sleep is not None:
        kwargs["sleep"] = sleep
    result = extract_image(
        image,
        ticker=ticker,
        pdf_page=pdf_page,
        half=half,
        routing_reason=decision["routing_reason"],
        **kwargs,
    )
    result["source_image"] = str(source)
    result["printed_page"] = page.get("printed_page")
    result["page_references"]["printed_page"] = page.get("printed_page")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if result["request_status"] == "success":
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        failed = destination.with_name(destination.stem + ".failed.json")
        failed.write_text(json.dumps(result, indent=2), encoding="utf-8")
    record_attempt(log_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision pilot for one explicit page half.")
    parser.add_argument("--ticker", default="POWERGRID")
    parser.add_argument("--page", type=int, default=200)
    parser.add_argument("--half", default="LEFT")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_logging(LOG_DIR / "vision.log")
    target = (args.ticker.upper(), args.page, args.half.upper())
    if target not in PILOT_TARGETS:
        raise SystemExit("The pilot only sends POWERGRID PDF page 200 LEFT. It does not scan the reports.")
    result = run_target(*target, force=args.force)
    public = {
        "ticker": result.get("ticker"),
        "pdf_page": result.get("pdf_page"),
        "half": result.get("half"),
        "request_status": result.get("request_status"),
        "routing_reason": result.get("routing_reason"),
        "model": result.get("model"),
        "prompt_version": result.get("prompt_version"),
        "usage": result.get("usage"),
        "duration_seconds": result.get("duration_seconds"),
        "attempts": result.get("attempts"),
        "retried": result.get("retried"),
        "source_of_truth": False,
        "quote_validated": False,
    }
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
