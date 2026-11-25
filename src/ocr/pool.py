"""OCR scanned halves with a worker pool. Finished pages are skipped. Writes are atomic."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config.paths import INPUT_DIR, PROJECT_ROOT
from src.ingestion.audit import locate_pdfs
from src.pipeline.runner import RunContext, _ocr_one, _ocr_work

LOGGER = logging.getLogger("factsheet.ocr")


def pending(ticker: str, limit: int | None = None) -> list[dict]:
    context = RunContext(ticker=ticker)
    rows = _ocr_work(context)
    if limit is not None:
        return rows[:limit]
    return rows


def _failure(item: dict, exc: BaseException) -> dict:
    message = f"{type(exc).__name__}: {exc}"
    LOGGER.error("OCR half failed pdf %s %s: %s", item.get("pdf_page"), item.get("half"), message)
    print(f"OCR half failed pdf {item.get('pdf_page')} {item.get('half')}: {message}", flush=True)
    return {"pdf_page": item.get("pdf_page"), "half": item.get("half"), "status": "failed", "error": message}


def _write_record(ticker: str, item: dict) -> dict:
    try:
        context = RunContext(ticker=ticker, pdf_path=locate_pdfs(INPUT_DIR)[ticker])
        destination = context.ocr_dir / ticker / f"{ticker}_p{item['pdf_page']:04d}_{item['half']}.json"
        if destination.is_file():
            return {"pdf_page": item["pdf_page"], "half": item["half"], "status": "skipped"}
        _ocr_one(context, item)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if destination.is_file() and temporary.is_file():
            temporary.unlink()
        return {"pdf_page": item["pdf_page"], "half": item["half"], "status": "written"}
    except Exception as exc:
        return _failure(item, exc)


def _attempt(ticker: str, rows: list[dict], workers: int) -> list[dict]:
    """Run one pass. A failed half is recorded and the rest continue."""
    if workers <= 0:
        results = []
        for item in rows:
            try:
                results.append(_write_record(ticker, item))
            except Exception as exc:
                results.append(_failure(item, exc))
        return results
    pool = ProcessPoolExecutor(max_workers=max(workers, 1))
    results = []
    try:
        futures = {pool.submit(_write_record, ticker, item): item for item in rows}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(_failure(item, exc))
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            LOGGER.error("OCR pool shutdown: %s", exc)
    seen = {(result.get("pdf_page"), result.get("half")) for result in results}
    for item in rows:
        if (item.get("pdf_page"), item.get("half")) not in seen:
            results.append(_failure(item, RuntimeError("OCR worker did not return a result")))
    return results


def run_pool(ticker: str, limit: int | None = None, workers: int | None = None) -> dict:
    import os
    import time

    # Keep Tesseract single-threaded per process; avoid oversubscribing with Paddle.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    rows = pending(ticker, limit)
    if not rows:
        return {"ticker": ticker, "pending": 0, "results": [], "seconds": 0.0}
    pool_size = 2 if workers is None else workers
    started = time.perf_counter()
    results = _attempt(ticker, rows, pool_size)
    failed_rows = [
        row for row in rows
        if any(result.get("pdf_page") == row["pdf_page"] and result.get("half") == row["half"] and result.get("status") == "failed" for result in results)
    ]
    if failed_rows:
        LOGGER.error("retrying %s failed OCR halves once", len(failed_rows))
        retried = _attempt(ticker, failed_rows, pool_size)
        by_key = {(result.get("pdf_page"), result.get("half")): result for result in results}
        for result in retried:
            by_key[(result.get("pdf_page"), result.get("half"))] = result
        results = list(by_key.values())
    elapsed = round(time.perf_counter() - started, 2)
    gaps = [
        {"pdf_page": item["pdf_page"], "half": item["half"], "reason": "pending_ocr"}
        for item in pending(ticker)
    ]
    written = sum(1 for row in results if row.get("status") == "written")
    skipped = sum(1 for row in results if row.get("status") == "skipped")
    failed = [row for row in results if row.get("status") == "failed"]
    sec_per_half = round(elapsed / max(written, 1), 2) if written else None
    remaining = len(gaps)
    hours_left = round((remaining * (sec_per_half or 0)) / max(pool_size, 1) / 3600.0, 2) if sec_per_half else None
    report = {
        "ticker": ticker,
        "workers": pool_size,
        "submitted": len(rows),
        "written": written,
        "skipped": skipped,
        "failed": [{"pdf_page": row.get("pdf_page"), "half": row.get("half"), "error": row.get("error")} for row in failed],
        "seconds": elapsed,
        "seconds_per_written_half": sec_per_half,
        "gaps_remaining": gaps,
        "gaps_remaining_count": remaining,
        "estimated_hours_remaining": hours_left,
        "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT"),
        "path": "fast",
    }
    destination = PROJECT_ROOT / "eval" / f"{ticker.lower()}_ocr_pool.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR pending scanned halves. Existing files are skipped.")
    parser.add_argument("--company", default="POWERGRID")
    parser.add_argument("--limit", type=int, default=0, help="How many halves to run. 0 lists the gap count only.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel halves. Two is the default (Paddle-light).")
    args = parser.parse_args()
    ticker = args.company.upper()
    if args.limit <= 0:
        rows = pending(ticker)
        print(json.dumps({"ticker": ticker, "pending_halves": len(rows), "pages": sorted({row['pdf_page'] for row in rows})[:20]}, indent=2))
        return
    report = run_pool(ticker, args.limit, args.workers)
    print(json.dumps({key: report[key] for key in (
        "ticker", "workers", "submitted", "written", "skipped", "failed", "seconds",
        "seconds_per_written_half", "gaps_remaining_count", "estimated_hours_remaining", "path",
    ) if key in report}, indent=2))


if __name__ == "__main__":
    main()
