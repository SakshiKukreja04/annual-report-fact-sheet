"""Run Qwen vision OCR on all weak POWERGRID pages, merge into page JSON, then stage3."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.config.paths import LOG_DIR, PAGE_DIR, STAGING_DIR, SUBMISSION_DIR, VISION_DIR
from src.extract.select import ocr_quality
from src.pages.assemble import text_block
from src.utils.logging_config import configure_logging
from src.vision.coverage_pilot import DEFAULT_PAGES, _call_vision_half, _log, _vision_text
from src.vision.pilot import cache_path, image_path, load_page
from src.vision.route import route_page

_ENTITY_HINT = re.compile(
    r"\b(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH)\b",
    re.I,
)
_FOLIO_LINE = re.compile(
    r"^\s*(?:page\s*)?(\d{1,4}[A-Za-z]?)(?:\s*/\s*(\d{1,4}[A-Za-z]?))?\s*$",
    re.I,
)


def discover_weak_pages(ticker: str = "POWERGRID") -> list[int]:
    """Weak OCR pages + known subsidiary/list ranges."""
    root = PAGE_DIR / ticker.upper()
    weak: set[int] = set()
    for path in sorted(root.glob("p*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        pdf_page = int(page["pdf_page"])
        qm = page.get("quality_metadata") or {}
        selected = [
            b for b in page.get("text_blocks") or [] if b.get("selected") and (b.get("text") or "").strip()
        ]
        text = "\n".join(b.get("text") or "" for b in selected)
        chars = len(text)
        src = str(qm.get("selected_text_source") or "")
        garbled = bool(qm.get("garbled_text_layer"))
        quality = ocr_quality(text) if text else {"low": True}
        is_ocr = src == "ocr" or any((b.get("source") or "") in {"tesseract", "paddleocr"} for b in selected)
        if garbled or (is_ocr and (chars < 120 or quality.get("low"))):
            weak.add(pdf_page)
    # Always include subsidiary / AOC / related-party list bands.
    weak.update(range(243, 251))
    weak.update(range(355, 363))
    weak.update(DEFAULT_PAGES)
    return sorted(weak)


def merge_vision_into_page(ticker: str, pdf_page: int) -> bool:
    """Attach ocr_vision blocks from cache; prefer them when longer than Tesseract."""
    from src.pages.assemble import assemble_page

    path = PAGE_DIR / ticker.upper() / f"p{pdf_page:04d}.json"
    if not path.is_file():
        return False
    page = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for half in ("LEFT", "RIGHT", "FULL"):
        dest = cache_path(VISION_DIR, ticker.upper(), pdf_page, half)
        if not dest.is_file():
            continue
        record = json.loads(dest.read_text(encoding="utf-8"))
        if record.get("request_status") != "success":
            continue
        text = _vision_text(record).strip()
        if text:
            parts.append((half, text))
    if not parts:
        return False
    digital = [b for b in (page.get("text_blocks") or []) if (b.get("source") or "") == "pymupdf"]
    other_ocr = [
        b
        for b in (page.get("text_blocks") or [])
        if (b.get("source") or "") in {"tesseract", "paddleocr", "ocr_mistral"}
    ]
    vision_blocks = []
    for index, (half, text) in enumerate(parts):
        # Split on blank lines into blocks for reading order.
        chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()] or [text]
        for chunk in chunks:
            vision_blocks.append(
                text_block(
                    chunk,
                    [0, float(index * 20), 100, float(index * 20 + 10)],
                    "ocr_vision",
                    half if half in {"LEFT", "RIGHT", "FULL"} else "FULL",
                    0,
                    ocr_confidence=None,
                    selected=False,
                    bbox_space="vision_hint",
                )
            )
    ocr = other_ocr + vision_blocks
    recoverability = dict((page.get("quality_metadata") or {}).get("ocr_recoverability") or {})
    recoverability.setdefault("FULL", "RECOVERABLE")
    for half, _ in parts:
        recoverability[half] = "RECOVERABLE"
    rebuilt = assemble_page(
        ticker=ticker.upper(),
        pdf_page=pdf_page,
        printed_page=page.get("printed_page"),
        page_type=list(page.get("page_type") or []),
        digital_blocks=digital,
        ocr_blocks=ocr,
        captions=list(page.get("captions") or []),
        images=list(page.get("images") or []),
        tables=list(page.get("tables") or []),
        ocr_recoverability=recoverability,
        garbled=bool((page.get("quality_metadata") or {}).get("garbled_text_layer")),
        classification_confidence=(page.get("quality_metadata") or {}).get("classification_confidence"),
        recommended_method=(page.get("quality_metadata") or {}).get("recommended_extraction_method"),
    )
    meta = rebuilt.setdefault("quality_metadata", {})
    vision_chars = sum(len(b.get("text") or "") for b in vision_blocks)
    meta["vision_char_count"] = vision_chars
    meta["vision_weak"] = vision_chars < 80
    # Folio from vision lines when page still null.
    if rebuilt.get("printed_page") in (None, ""):
        folio = _folio_from_vision_parts(parts)
        if folio:
            rebuilt["printed_page"] = folio
            meta["folio_source"] = "ocr_vision"
    path.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
    return True


def _folio_from_vision_parts(parts: list[tuple[str, str]]) -> str | None:
    candidates = []
    for _half, text in parts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            match = _FOLIO_LINE.match(line)
            if not match:
                continue
            left, right = match.group(1), match.group(2)
            candidates.append(f"{left}/{right}" if right else left)
    if not candidates:
        return None
    return max(set(candidates), key=candidates.count)


def run_vision_jobs(ticker: str, pages: list[int], *, force: bool = False, pause_s: float = 25.0) -> dict:
    ticker = ticker.upper()
    jobs: list[tuple[int, str]] = []
    for pdf_page in pages:
        halves = [h for h in ("LEFT", "RIGHT") if image_path(ticker, pdf_page, h).is_file()]
        if not halves and image_path(ticker, pdf_page, "FULL").is_file():
            halves = ["FULL"]
        for half in halves or ["LEFT", "RIGHT"]:
            jobs.append((pdf_page, half))
    total = len(jobs)
    done = ok = failed = cached = 0
    _log(f"[weak-ocr] {ticker} pages={len(pages)} halves={total} force={force} pause={pause_s}s")
    for pdf_page, half in jobs:
        done += 1
        left = total - done
        page = load_page(ticker, pdf_page)
        decision = route_page(page)
        reason = decision["routing_reason"] if decision["eligible"] else "ocr_text_weak_or_garbled"
        destination = cache_path(VISION_DIR, ticker, pdf_page, half)
        _log(f"[weak-ocr] ({done}/{total}, {left} left) START pdf={pdf_page} half={half}")
        started = time.perf_counter()
        if destination.is_file() and not force:
            result = json.loads(destination.read_text(encoding="utf-8"))
            if result.get("request_status") == "success":
                result["request_status"] = "cached"
            else:
                result = _call_vision_half(ticker, pdf_page, half, page, reason, destination)
        else:
            result = _call_vision_half(ticker, pdf_page, half, page, reason, destination)
        status = result.get("request_status")
        elapsed = round(time.perf_counter() - started, 1)
        usage = result.get("usage") or {}
        err = (result.get("error") or "")[:140]
        if status == "cached":
            cached += 1
        elif status == "success":
            ok += 1
        else:
            failed += 1
        _log(
            f"[weak-ocr] ({done}/{total}, {left} left) DONE  pdf={pdf_page} half={half} "
            f"status={status} {elapsed}s tokens={usage.get('total_tokens')} "
            f"ok={ok} cached={cached} failed={failed}"
            + (f" err={err}" if err else "")
        )
        if pause_s > 0 and done < total and status != "cached":
            # Longer pause after rate-limit style failures.
            err_l = (result.get("error") or "").casefold()
            wait = pause_s
            # Only real rate-limit / TPM errors get the long cooldown (avoid matching "geneRATE").
            if status == "failed" and (
                "429" in err_l or "rate_limit" in err_l or "too large" in err_l or "tokens per minute" in err_l
            ):
                wait = max(pause_s, 70.0)
            elif status == "failed":
                wait = min(pause_s, 8.0)  # JSON/400 failures: brief pause, then continue
            _log(f"[weak-ocr] pause {wait}s…")
            time.sleep(wait)
    # Merge all pages that have any vision success.
    merged = 0
    for pdf_page in pages:
        if merge_vision_into_page(ticker, pdf_page):
            merged += 1
    summary = {
        "pages": pages,
        "halves": total,
        "ok": ok,
        "cached": cached,
        "failed": failed,
        "merged_pages": merged,
    }
    out = VISION_DIR / "pilot" / f"{ticker}_weak_ocr_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"[weak-ocr] summary {summary}")
    return summary


def run_extraction(ticker: str = "POWERGRID") -> dict:
    """Rebuild factsheet via stage3 (page_text grounding, including ocr_vision when selected)."""
    import shutil

    from src.config.paths import OUTPUT_DIR
    from src.eval.check_factsheet import check_factsheet
    from src.factsheet.powergrid_stage3 import run as stage3_run

    # Clear page-layer caches inside stage3 by reimporting fresh run state.
    import src.factsheet.powergrid_stage3 as stage3

    stage3._PAGE_CACHE.clear()
    stage3._LAYER_CACHE.clear()
    stage3._PDF_CACHE.clear()
    report = stage3_run()
    staging = STAGING_DIR / "factsheet" / ticker
    results = check_factsheet(ticker, path=staging)
    failed = [row for row in results if not row.get("passed")]
    report["check_failed"] = len(failed)
    report["checks"] = len(results)
    for row in results:
        _log(f"{'PASS' if row['passed'] else 'FAIL'} {row['check']}. {row['detail']}")
    if not failed:
        for dest_root in (OUTPUT_DIR, SUBMISSION_DIR):
            dest = dest_root / ticker
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / f"{ticker}.json", dest / f"{ticker}.json")
            shutil.copy2(staging / f"{ticker}.md", dest / f"{ticker}.md")
            shortfall = staging / "SHORTFALL.md"
            if shortfall.is_file():
                shutil.copy2(shortfall, dest / "SHORTFALL.md")
        report["promoted"] = True
        _log(f"[weak-ocr] promoted {SUBMISSION_DIR / ticker / f'{ticker}.json'}")
    else:
        report["promoted"] = False
        # Still write staging sheet; copy to outputs only if user wants — keep staging.
        _log(f"[weak-ocr] checks failed={len(failed)}; left in staging only")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision-OCR all weak pages then extract POWERGRID.")
    parser.add_argument("--company", default="POWERGRID")
    parser.add_argument("--pages", default="", help="Override page list; default = discover weak + list bands.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause", type=float, default=25.0)
    parser.add_argument("--skip-vision", action="store_true", help="Only merge + extract from existing cache.")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(LOG_DIR / "vision.log")
    ticker = args.company.upper()
    if args.pages.strip():
        pages = []
        seen: set[int] = set()
        for part in args.pages.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "-" in piece:
                a, b = piece.split("-", 1)
                for n in range(int(a), int(b) + 1):
                    if n not in seen:
                        seen.add(n)
                        pages.append(n)
            else:
                n = int(piece)
                if n not in seen:
                    seen.add(n)
                    pages.append(n)
    else:
        pages = discover_weak_pages(ticker)
    _log(f"[weak-ocr] target pages ({len(pages)}): {pages}")
    if not args.extract_only and not args.skip_vision:
        run_vision_jobs(ticker, pages, force=args.force, pause_s=args.pause)
    elif args.skip_vision:
        merged = sum(1 for p in pages if merge_vision_into_page(ticker, p))
        _log(f"[weak-ocr] skip-vision merged={merged}")
    report = run_extraction(ticker)
    summary = {
        "final_items": report.get("final_items"),
        "shortfall_vs_190": report.get("shortfall_vs_190"),
        "promoted": report.get("promoted"),
        "check_failed": report.get("check_failed"),
    }
    _log(f"[weak-ocr] extraction {json.dumps(summary)}")
    return 0 if report.get("promoted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
