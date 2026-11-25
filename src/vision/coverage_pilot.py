"""Pilot Qwen vision vs Tesseract on weak POWERGRID pages. Vision is hint-only."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.config.paths import LOG_DIR, VISION_DIR
from src.extract.select import page_text
from src.utils.logging_config import configure_logging
from src.vision.client import extract_image, prepare_image
from src.vision.pilot import cache_path, image_path, load_page
from src.vision.route import route_page
from src.vision.settings import vision_settings
from src.vision.usage import record_attempt, tokens_used, usage_path

_ENTITY_HINT = re.compile(
    r"\b(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH)\b",
    re.I,
)

DEFAULT_PAGES = [361, 362, 146, 249, 281, 282, 283, 284, 243, 245, 246, 320]


def _vision_text(record: dict) -> str:
    lines = record.get("extracted_text") or []
    if isinstance(lines, list):
        parts = [str(line).strip() for line in lines if line]
    else:
        parts = [str(lines).strip()] if lines else []
    for table in record.get("tables") or []:
        if not isinstance(table, dict):
            continue
        headers = table.get("headers") or []
        if headers:
            parts.append(" | ".join(str(h) for h in headers if h is not None))
        for row in table.get("rows") or []:
            if isinstance(row, list):
                parts.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(p for p in parts if p)


def _tesseract_text(page: dict) -> str:
    parts = [
        block.get("text") or ""
        for block in page.get("text_blocks") or []
        if (block.get("source") or "") in {"tesseract", "paddleocr"} and (block.get("text") or "").strip()
    ]
    if parts:
        return "\n".join(parts)
    return page_text(page) or ""


def _sample_lines(text: str, limit: int = 3) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()][:limit]


def _log(message: str) -> None:
    print(message, flush=True)


def _call_vision_half(
    ticker: str,
    pdf_page: int,
    half: str,
    page: dict,
    routing_reason: str,
    destination: Path,
) -> dict:
    """Smaller JPEG + direct call to reduce Groq TPM 'request too large' errors."""
    settings = vision_settings()
    log_path = usage_path(VISION_DIR)
    if tokens_used(log_path) >= settings["daily_token_cap"]:
        blocked = {
            "ticker": ticker,
            "pdf_page": pdf_page,
            "half": half,
            "request_status": "daily_cap_reached",
            "routing_reason": routing_reason,
            "source_of_truth": False,
            "quote_validated": False,
            "model": settings["model"],
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            "error": "daily_cap_reached",
            "extracted_text": [],
            "tables": [],
        }
        record_attempt(log_path, blocked)
        return blocked
    source = image_path(ticker, pdf_page, half)
    if not source.is_file():
        return {
            "ticker": ticker,
            "pdf_page": pdf_page,
            "half": half,
            "request_status": "no_image",
            "routing_reason": routing_reason,
            "source_of_truth": False,
            "extracted_text": [],
            "tables": [],
            "error": f"missing {source}",
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        }
    image = prepare_image(source, max_side=900)
    result = extract_image(
        image,
        ticker=ticker,
        pdf_page=pdf_page,
        half=half,
        routing_reason=routing_reason,
    )
    result["source_image"] = str(source)
    result["printed_page"] = page.get("printed_page")
    result.setdefault("page_references", {})["printed_page"] = page.get("printed_page")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if result["request_status"] == "success":
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        failed = destination.with_name(destination.stem + ".failed.json")
        if failed.is_file():
            failed.unlink()
    else:
        failed = destination.with_name(destination.stem + ".failed.json")
        failed.write_text(json.dumps(result, indent=2), encoding="utf-8")
    record_attempt(log_path, result)
    return result


def run_pilot(ticker: str, pages: list[int], *, force: bool = False, pause_s: float = 2.0) -> dict:
    ticker = ticker.upper()
    rows = []
    jobs: list[tuple[int, str]] = []
    for pdf_page in pages:
        usable = [half for half in ("LEFT", "RIGHT", "FULL") if image_path(ticker, pdf_page, half).is_file()]
        if not usable:
            usable = ["LEFT", "RIGHT"]
        for half in usable:
            jobs.append((pdf_page, half))
    total = len(jobs)
    done = 0
    ok = 0
    failed = 0
    cached = 0
    _log(f"[vision-pilot] {ticker} pages={pages} halves={total} force={force} pause={pause_s}s")
    by_page: dict[int, dict] = {}
    for pdf_page, half in jobs:
        done += 1
        left = total - done
        page = load_page(ticker, pdf_page)
        decision = route_page(page)
        if pdf_page not in by_page:
            by_page[pdf_page] = {
                "decision": decision,
                "tess": _tesseract_text(page),
                "vision_parts": [],
                "call_statuses": [],
            }
        slot = by_page[pdf_page]
        reason = decision["routing_reason"] if decision["eligible"] else "ocr_text_weak_or_garbled"
        destination = cache_path(VISION_DIR, ticker, pdf_page, half)
        _log(f"[vision-pilot] ({done}/{total}, {left} left) START pdf={pdf_page} half={half}")
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
        err = (result.get("error") or "")[:160]
        if status == "cached":
            cached += 1
        elif status == "success":
            ok += 1
        else:
            failed += 1
        _log(
            f"[vision-pilot] ({done}/{total}, {left} left) DONE  pdf={pdf_page} half={half} "
            f"status={status} {elapsed}s tokens={usage.get('total_tokens')} "
            f"ok={ok} cached={cached} failed={failed}"
            + (f" err={err}" if err else "")
        )
        slot["call_statuses"].append({"half": half, "status": status, "reason": reason})
        if status in {"success", "cached"}:
            slot["vision_parts"].append(_vision_text(result))
        if pause_s > 0 and done < total and status != "cached":
            _log(f"[vision-pilot] pause {pause_s}s before next half…")
            time.sleep(pause_s)

    for pdf_page in pages:
        slot = by_page.get(pdf_page)
        if not slot:
            continue
        tess = slot["tess"]
        vision = "\n".join(p for p in slot["vision_parts"] if p)
        rows.append(
            {
                "pdf_page": pdf_page,
                "route": slot["decision"],
                "tesseract_chars": len(tess),
                "vision_chars": len(vision),
                "tesseract_entities": len(_ENTITY_HINT.findall(tess)),
                "vision_entities": len(_ENTITY_HINT.findall(vision)),
                "tesseract_sample": _sample_lines(tess),
                "vision_sample": _sample_lines(vision),
                "calls": slot["call_statuses"],
                "vision_is_grounding": False,
            }
        )
    _log(f"[vision-pilot] finished ok={ok} cached={cached} failed={failed} total={total}")
    report = {
        "ticker": ticker,
        "model_hint": "GROQ_VISION_MODEL / qwen",
        "note": "Vision is extraction hint only; quotes must still verify on page_text or PDF.",
        "pages": rows,
        "totals": {
            "tesseract_chars": sum(r["tesseract_chars"] for r in rows),
            "vision_chars": sum(r["vision_chars"] for r in rows),
            "tesseract_entities": sum(r["tesseract_entities"] for r in rows),
            "vision_entities": sum(r["vision_entities"] for r in rows),
            "vision_better_chars": sum(1 for r in rows if r["vision_chars"] > r["tesseract_chars"]),
            "vision_better_entities": sum(1 for r in rows if r["vision_entities"] > r["tesseract_entities"]),
        },
    }
    out = VISION_DIR / "pilot" / f"{ticker}_coverage_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen vision vs Tesseract pilot (hint-only).")
    parser.add_argument("--company", default="POWERGRID")
    parser.add_argument("--pages", default="", help="Comma/ranges; default is the 12 garbled pilot pages.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause", type=float, default=2.0)
    args = parser.parse_args(argv)
    configure_logging(LOG_DIR / "vision.log")
    if args.pages.strip():
        pages = []
        seen: set[int] = set()
        for part in args.pages.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "-" in piece:
                start, end = piece.split("-", 1)
                for number in range(int(start), int(end) + 1):
                    if number not in seen:
                        seen.add(number)
                        pages.append(number)
            else:
                number = int(piece)
                if number not in seen:
                    seen.add(number)
                    pages.append(number)
    else:
        pages = list(DEFAULT_PAGES)
    report = run_pilot(args.company.upper(), pages, force=args.force, pause_s=args.pause)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
