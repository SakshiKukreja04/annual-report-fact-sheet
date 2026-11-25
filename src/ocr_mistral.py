"""Mistral Document OCR layer. Cached under data/processed/ocr_mistral/<TICKER>/."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

from src.config.paths import INPUT_DIR, OCR_MISTRAL_DIR, PAGE_DIR, PROJECT_ROOT
from src.ingestion.audit import locate_pdfs
from src.pages.assemble import text_block

LOGGER = logging.getLogger("factsheet")
MODEL = "mistral-ocr-latest"
_FOLIO_LINE = re.compile(
    r"^\s*(?:page\s*)?(\d{1,4}[A-Za-z]?)(?:\s*/\s*(\d{1,4}[A-Za-z]?))?\s*$",
    re.I,
)
_ENTITY_HINT = re.compile(
    r"\b(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH)\b",
    re.I,
)


def mistral_dir(ticker: str) -> Path:
    return OCR_MISTRAL_DIR / ticker.upper()


def page_paths(ticker: str, pdf_page: int) -> tuple[Path, Path]:
    folder = mistral_dir(ticker)
    stem = f"page_{pdf_page:03d}"
    return folder / f"{stem}.json", folder / f"{stem}.md"


def parse_page_spec(text: str) -> list[int]:
    wanted: list[int] = []
    seen: set[int] = set()
    for part in (text or "").split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            for number in range(int(start), int(end) + 1):
                if number not in seen:
                    seen.add(number)
                    wanted.append(number)
        else:
            number = int(piece)
            if number not in seen:
                seen.add(number)
                wanted.append(number)
    return wanted


def _client():
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    try:
        from mistralai import Mistral
    except ImportError:  # mistralai 3.x layout
        from mistralai.client import Mistral
    return Mistral(api_key=key)


def _backoff_sleep(attempt: int, *, rate_limited: bool = False) -> None:
    if rate_limited:
        time.sleep(min(180.0, 30.0 * (attempt + 1)))
    else:
        time.sleep(min(60.0, (2**attempt) + 0.25))


def _is_rate_limited(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "429" in message or "rate" in message or "rate_limited" in message


def _call_with_retry(fn, *, attempts: int = 8):
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — SDK raises varied HTTP errors
            last = exc
            message = str(exc).casefold()
            rate_limited = _is_rate_limited(exc)
            retryable = rate_limited or any(
                token in message for token in ("500", "502", "503", "504", "timeout")
            )
            if not retryable or attempt >= attempts - 1:
                raise
            LOGGER.warning(
                "Mistral OCR retry %s after %s rate_limited=%s",
                attempt + 1,
                type(exc).__name__,
                rate_limited,
            )
            _backoff_sleep(attempt, rate_limited=rate_limited)
    raise last  # type: ignore[misc]


def _response_to_dict(response) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") else dict(response)


def ocr_pdf_pages(
    ticker: str,
    pages: list[int],
    *,
    force: bool = False,
    pdf_path: Path | None = None,
    halves: bool = False,
) -> dict:
    """OCR selected 1-based PDF pages. Cached pages are skipped unless force."""
    ticker = ticker.upper()
    pdf_path = pdf_path or locate_pdfs(INPUT_DIR)[ticker]
    folder = mistral_dir(ticker)
    folder.mkdir(parents=True, exist_ok=True)
    todo = []
    cached = []
    for page in pages:
        json_path, md_path = page_paths(ticker, page)
        if not force and json_path.is_file() and md_path.is_file():
            cached.append(page)
            continue
        todo.append(page)
    if not todo:
        return {"ticker": ticker, "cached": cached, "written": [], "halves": halves}

    client = _client()
    # Prefer pages= 0-based API when supported; fall back to whole-file then slice.
    written = []
    if halves:
        written.extend(_ocr_halves(client, ticker, pdf_path, todo, force=force))
    else:
        written.extend(_ocr_document_pages(client, ticker, pdf_path, todo, force=force))
    return {"ticker": ticker, "cached": cached, "written": written, "halves": halves}


def _upload_pdf(client, pdf_path: Path) -> str:
    from mistralai.client.models.file import File

    uploaded = _call_with_retry(
        lambda: client.files.upload(
            file=File(file_name=pdf_path.name, content=pdf_path.read_bytes()),
            purpose="ocr",
        )
    )
    file_id = uploaded.id if hasattr(uploaded, "id") else uploaded["id"]
    signed = _call_with_retry(lambda: client.files.get_signed_url(file_id=file_id))
    return signed.url if hasattr(signed, "url") else signed["url"]


def _ocr_document_pages(client, ticker: str, pdf_path: Path, pages: list[int], *, force: bool) -> list[int]:
    written = []
    url = _upload_pdf(client, pdf_path)
    # Chunk to stay under API comfort (≤50 pages per request; docs allow 1000).
    for start in range(0, len(pages), 40):
        chunk = pages[start : start + 40]
        zero_based = [p - 1 for p in chunk]

        def _run(pages_arg=zero_based, document_url=url):
            return client.ocr.process(
                model=MODEL,
                document={"type": "document_url", "document_url": document_url},
                pages=pages_arg,
                include_image_base64=False,
                extract_header=True,
                extract_footer=True,
            )

        try:
            raw = _response_to_dict(_call_with_retry(_run))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "document OCR failed (%s); falling back to per-page images with backoff",
                type(exc).__name__,
            )
            for pdf_page in chunk:
                if _is_rate_limited(exc):
                    time.sleep(20)
                entry = _ocr_page_as_image(client, pdf_path, pdf_page)
                _write_page_artifacts(ticker, pdf_page, entry, mode="full")
                written.append(pdf_page)
            continue
        by_index = {}
        for entry in raw.get("pages") or []:
            index = int(entry.get("index", entry.get("page_index", -1)))
            by_index[index] = entry
        for pdf_page in chunk:
            entry = by_index.get(pdf_page - 1)
            if entry is None and len(by_index) == 1:
                entry = next(iter(by_index.values()))
            if entry is None:
                entry = _ocr_page_as_image(client, pdf_path, pdf_page)
            _write_page_artifacts(ticker, pdf_page, entry, mode="full")
            written.append(pdf_page)
    return written


def _ocr_page_as_image(client, pdf_path: Path, pdf_page: int, *, clip=None) -> dict:
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        page = document[pdf_page - 1]
        matrix = pymupdf.Matrix(2, 2)
        if clip is not None:
            pix = page.get_pixmap(matrix=matrix, clip=clip)
        else:
            pix = page.get_pixmap(matrix=matrix)
        png = pix.tobytes("png")
    encoded = base64.b64encode(png).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"

    def _run():
        return client.ocr.process(
            model=MODEL,
            document={"type": "image_url", "image_url": data_url},
            include_image_base64=False,
            extract_header=True,
            extract_footer=True,
        )

    raw = _response_to_dict(_call_with_retry(_run))
    pages = raw.get("pages") or []
    if pages:
        return pages[0]
    return {"index": pdf_page - 1, "markdown": raw.get("markdown") or "", "raw": raw}


def _ocr_halves(client, ticker: str, pdf_path: Path, pages: list[int], *, force: bool) -> list[int]:
    import pymupdf

    written = []
    with pymupdf.open(pdf_path) as document:
        for pdf_page in pages:
            page = document[pdf_page - 1]
            rect = page.rect
            mid = (rect.x0 + rect.x1) / 2
            left = pymupdf.Rect(rect.x0, rect.y0, mid, rect.y1)
            right = pymupdf.Rect(mid, rect.y0, rect.x1, rect.y1)
            left_entry = _ocr_page_as_image(client, pdf_path, pdf_page, clip=left)
            right_entry = _ocr_page_as_image(client, pdf_path, pdf_page, clip=right)
            combined = {
                "index": pdf_page - 1,
                "markdown": (left_entry.get("markdown") or "") + "\n\n" + (right_entry.get("markdown") or ""),
                "halves": {"LEFT": left_entry, "RIGHT": right_entry},
                "mode": "halves",
            }
            _write_page_artifacts(ticker, pdf_page, combined, mode="halves")
            written.append(pdf_page)
    return written


def _write_page_artifacts(ticker: str, pdf_page: int, entry: dict, *, mode: str) -> None:
    json_path, md_path = page_paths(ticker, pdf_page)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = entry.get("markdown") or ""
    payload = {
        "ticker": ticker.upper(),
        "pdf_page": int(pdf_page),
        "model": MODEL,
        "mode": mode,
        "markdown": markdown,
        "raw_page": entry,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown + ("\n" if markdown and not markdown.endswith("\n") else ""), encoding="utf-8")


def load_mistral_page(ticker: str, pdf_page: int) -> dict | None:
    json_path, _md = page_paths(ticker, pdf_page)
    if not json_path.is_file():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def mistral_blocks_for_page(ticker: str, pdf_page: int) -> list[dict]:
    """Convert cached Mistral markdown into page text_blocks (source=ocr_mistral)."""
    record = load_mistral_page(ticker, pdf_page)
    if not record:
        return []
    markdown = (record.get("markdown") or "").strip()
    if not markdown:
        return []
    # Prefer paragraph chunks for reading order.
    chunks = [part.strip() for part in re.split(r"\n{2,}", markdown) if part.strip()]
    if not chunks:
        chunks = [markdown]
    blocks = []
    for index, chunk in enumerate(chunks):
        blocks.append(
            text_block(
                chunk,
                [0, float(index * 12), 100, float(index * 12 + 10)],
                "ocr_mistral",
                "FULL",
                index,
                ocr_confidence=None,
                selected=False,
                bbox_space="mistral_markdown",
            )
        )
    return blocks


def mistral_char_count(ticker: str, pdf_page: int) -> int:
    record = load_mistral_page(ticker, pdf_page)
    if not record:
        return 0
    return len((record.get("markdown") or "").strip())


def mistral_is_weak(ticker: str, pdf_page: int, *, min_chars: int = 80) -> bool:
    return mistral_char_count(ticker, pdf_page) < min_chars


def detect_folio_from_markdown(markdown: str) -> str | None:
    """Return a printed folio from header/footer-ish lines, or None."""
    lines = [line.strip() for line in (markdown or "").splitlines() if line.strip()]
    if not lines:
        return None
    candidates = []
    for line in lines[:4] + lines[-4:]:
        match = _FOLIO_LINE.match(line)
        if not match:
            continue
        left = match.group(1)
        right = match.group(2)
        if right:
            candidates.append(f"{left}/{right}")
        else:
            candidates.append(left)
    if not candidates:
        return None
    # Prefer the most common candidate in header/footer window
    return max(set(candidates), key=candidates.count)


def infer_printed_folios(ticker: str, pages: list[int] | None = None) -> dict:
    """Fill printed_page only in runs where ≥3 neighbouring pages agree on pdf−printed offset."""
    ticker = ticker.upper()
    folder = PAGE_DIR / ticker
    if not folder.is_dir():
        return {"filled": 0, "refused": 0, "offsets": []}
    targets = pages or sorted(int(p.name[1:5]) for p in folder.glob("p*.json"))
    detections: list[tuple[int, int]] = []  # (pdf_page, printed_int)
    for pdf_page in targets:
        record = load_mistral_page(ticker, pdf_page)
        if not record:
            continue
        folio = detect_folio_from_markdown(record.get("markdown") or "")
        if not folio or "/" in folio:
            # spreads skipped for offset math unless single int
            if folio and "/" not in folio and folio.isdigit():
                detections.append((pdf_page, int(re.sub(r"\D", "", folio) or "0")))
            continue
        digits = re.sub(r"\D", "", folio)
        if digits:
            detections.append((pdf_page, int(digits)))

    # Group by offset = printed - pdf
    by_offset: dict[int, list[tuple[int, int]]] = {}
    for pdf_page, printed in detections:
        offset = printed - pdf_page
        by_offset.setdefault(offset, []).append((pdf_page, printed))

    filled = 0
    refused = 0
    accepted_offsets = []
    fill_map: dict[int, str] = {}
    for offset, rows in by_offset.items():
        rows = sorted(rows)
        # Find contiguous runs of length ≥ 3
        run: list[tuple[int, int]] = []
        for row in rows:
            if not run or row[0] == run[-1][0] + 1:
                run.append(row)
            else:
                if len(run) >= 3:
                    for pdf_page, printed in run:
                        fill_map[pdf_page] = str(printed)
                    accepted_offsets.append({"offset": offset, "pages": [p for p, _ in run]})
                else:
                    refused += len(run)
                run = [row]
        if len(run) >= 3:
            for pdf_page, printed in run:
                fill_map[pdf_page] = str(printed)
            accepted_offsets.append({"offset": offset, "pages": [p for p, _ in run]})
        else:
            refused += len(run)

    for pdf_page, printed in fill_map.items():
        path = folder / f"p{pdf_page:04d}.json"
        if not path.is_file():
            continue
        page = json.loads(path.read_text(encoding="utf-8"))
        if page.get("printed_page") not in (None, ""):
            continue
        page["printed_page"] = printed
        meta = page.setdefault("quality_metadata", {})
        meta["folio_source"] = "mistral_offset"
        meta["folio_offset"] = next(
            (row["offset"] for row in accepted_offsets if pdf_page in row["pages"]),
            None,
        )
        path.write_text(json.dumps(page, indent=2) + "\n", encoding="utf-8")
        filled += 1
    LOGGER.info("mistral folio filled=%s refused=%s", filled, refused)
    return {"filled": filled, "refused": refused, "offsets": accepted_offsets, "detections": len(detections)}


def merge_mistral_into_pages(ticker: str, pages: list[int], *, prefer: bool = True) -> dict:
    """Attach ocr_mistral blocks and re-select text source (pymupdf → mistral → tess/paddle)."""
    from src.pages.assemble import assemble_page

    ticker = ticker.upper()
    updated = 0
    for pdf_page in pages:
        path = PAGE_DIR / ticker / f"p{pdf_page:04d}.json"
        if not path.is_file():
            continue
        page = json.loads(path.read_text(encoding="utf-8"))
        mistral_blocks = mistral_blocks_for_page(ticker, pdf_page)
        if not mistral_blocks:
            continue
        digital = [b for b in (page.get("text_blocks") or []) if (b.get("source") or "") == "pymupdf"]
        other_ocr = [
            b
            for b in (page.get("text_blocks") or [])
            if (b.get("source") or "") in {"tesseract", "paddleocr"}
        ]
        ocr = other_ocr + mistral_blocks
        recoverability = dict((page.get("quality_metadata") or {}).get("ocr_recoverability") or {})
        recoverability.setdefault("FULL", "RECOVERABLE")
        garbled = bool((page.get("quality_metadata") or {}).get("garbled_text_layer"))
        rebuilt = assemble_page(
            ticker=ticker,
            pdf_page=pdf_page,
            printed_page=page.get("printed_page"),
            page_type=list(page.get("page_type") or []),
            digital_blocks=digital,
            ocr_blocks=ocr,
            captions=list(page.get("captions") or []),
            images=list(page.get("images") or []),
            tables=list(page.get("tables") or []),
            ocr_recoverability=recoverability,
            garbled=garbled,
            classification_confidence=(page.get("quality_metadata") or {}).get("classification_confidence"),
            recommended_method=(page.get("quality_metadata") or {}).get("recommended_extraction_method"),
        )
        meta = rebuilt.setdefault("quality_metadata", {})
        meta["mistral_char_count"] = sum(len(b.get("text") or "") for b in mistral_blocks)
        meta["mistral_weak"] = meta["mistral_char_count"] < 80
        # Preserve prior folio_source metadata when present.
        for key in ("folio_source", "folio_confidence", "folio_half", "folio_offset"):
            if key in (page.get("quality_metadata") or {}) and key not in meta:
                meta[key] = page["quality_metadata"][key]
        path.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
        updated += 1
    return {"updated": updated, "pages": pages}


def compare_full_vs_halves(ticker: str, sample_pages: list[int], *, pdf_path: Path | None = None) -> dict:
    """OCR 5 sample pages both ways; report which yields more text/entities."""
    ticker = ticker.upper()
    pdf_path = pdf_path or locate_pdfs(INPUT_DIR)[ticker]
    client = _client()
    rows = []
    for pdf_page in sample_pages[:5]:
        full_entry = _ocr_page_as_image(client, pdf_path, pdf_page)
        # halves
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            page = document[pdf_page - 1]
            rect = page.rect
            mid = (rect.x0 + rect.x1) / 2
            left = pymupdf.Rect(rect.x0, rect.y0, mid, rect.y1)
            right = pymupdf.Rect(mid, rect.y0, rect.x1, rect.y1)
        left_entry = _ocr_page_as_image(client, pdf_path, pdf_page, clip=left)
        right_entry = _ocr_page_as_image(client, pdf_path, pdf_page, clip=right)
        full_md = full_entry.get("markdown") or ""
        half_md = (left_entry.get("markdown") or "") + "\n" + (right_entry.get("markdown") or "")
        row = {
            "pdf_page": pdf_page,
            "full_chars": len(full_md),
            "halves_chars": len(half_md),
            "full_entities": len(_ENTITY_HINT.findall(full_md)),
            "halves_entities": len(_ENTITY_HINT.findall(half_md)),
            "winner": "halves" if len(half_md) > len(full_md) * 1.05 else "full",
        }
        rows.append(row)
    halves_wins = sum(1 for row in rows if row["winner"] == "halves")
    return {
        "rows": rows,
        "prefer_halves": halves_wins >= 3,
        "halves_wins": halves_wins,
        "full_wins": len(rows) - halves_wins,
    }


def tesseract_text_for_page(ticker: str, pdf_page: int) -> str:
    path = PAGE_DIR / ticker.upper() / f"p{pdf_page:04d}.json"
    if not path.is_file():
        return ""
    page = json.loads(path.read_text(encoding="utf-8"))
    parts = [
        b.get("text") or ""
        for b in (page.get("text_blocks") or [])
        if (b.get("source") or "") in {"tesseract", "paddleocr"} and (b.get("text") or "").strip()
    ]
    return "\n".join(parts)


def pilot_report(ticker: str, pages: list[int]) -> dict:
    """Side-by-side Tesseract vs Mistral for pilot pages."""
    ticker = ticker.upper()
    rows = []
    for pdf_page in pages:
        tess = tesseract_text_for_page(ticker, pdf_page)
        mist = load_mistral_page(ticker, pdf_page) or {}
        md = mist.get("markdown") or ""
        tess_lines = [ln for ln in tess.splitlines() if ln.strip()][:3]
        mist_lines = [ln for ln in md.splitlines() if ln.strip()][:3]
        rows.append(
            {
                "pdf_page": pdf_page,
                "tesseract_chars": len(tess),
                "mistral_chars": len(md),
                "tesseract_entities": len(_ENTITY_HINT.findall(tess)),
                "mistral_entities": len(_ENTITY_HINT.findall(md)),
                "tesseract_sample": tess_lines,
                "mistral_sample": mist_lines,
            }
        )
    return {"ticker": ticker, "pages": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mistral OCR on selected PDF pages.")
    parser.add_argument("--company", required=True)
    parser.add_argument("--pages", default="", help="1-based pages/ranges, e.g. 361-362,243-247")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mistral-halves", action="store_true")
    parser.add_argument("--compare-halves", action="store_true", help="Compare full vs halves on up to 5 pages.")
    parser.add_argument("--merge", action="store_true", help="Merge cached Mistral output into page JSON.")
    parser.add_argument("--folios", action="store_true", help="Infer printed folios from Mistral markdown.")
    parser.add_argument("--pilot", action="store_true", help="Print Tesseract vs Mistral pilot report.")
    args = parser.parse_args(argv)
    ticker = args.company.upper()
    pages = parse_page_spec(args.pages) if args.pages else []
    if args.compare_halves:
        sample = pages[:5] or [243, 245, 246, 361, 362]
        report = compare_full_vs_halves(ticker, sample)
        print(json.dumps(report, indent=2))
        return 0
    if pages:
        summary = ocr_pdf_pages(ticker, pages, force=args.force, halves=args.mistral_halves)
        print(json.dumps({"ocr": summary}, indent=2))
    if args.merge and pages:
        print(json.dumps({"merge": merge_mistral_into_pages(ticker, pages)}, indent=2))
    if args.folios:
        print(json.dumps({"folios": infer_printed_folios(ticker, pages or None)}, indent=2))
    if args.pilot and pages:
        print(json.dumps(pilot_report(ticker, pages), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
