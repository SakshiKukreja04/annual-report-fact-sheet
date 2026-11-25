"""Read PyMuPDF text, cached OCR, and cached tables for one page."""

from __future__ import annotations

import json
from pathlib import Path

import struct

from src.classification.classify import SPREAD_MIN_WIDTH_PT
from src.pages.assemble import ocr_xywh_to_bbox, text_block
from src.preprocessing.pipeline import HALF_FULL, HALF_LEFT, HALF_RIGHT, is_two_page_spread
from src.tables.provenance import OFFSET_MIN_CONFIDENCE, combine_half_folios, printed_page, printed_page_detail


def half_for_bbox(bbox: list[float], page_width: float) -> str:
    if page_width < SPREAD_MIN_WIDTH_PT:
        return HALF_FULL
    center = (float(bbox[0]) + float(bbox[2])) / 2
    if center < page_width / 2:
        return HALF_LEFT
    return HALF_RIGHT


def digital_content(page: pymupdf.Page) -> dict:
    """Text blocks and images in PDF points. Halves share this page's number."""
    width = float(page.rect.width)
    height = float(page.rect.height)
    words = [{"text": word[4], "x0": word[0], "top": word[1]} for word in page.get_text("words")]
    printed = printed_page(
        words,
        height,
        page_width=width,
        allow_pair=is_two_page_spread(width, height),
    )
    blocks = []
    images = []
    for block in page.get_text("dict").get("blocks", []):
        bbox = [float(value) for value in block.get("bbox", (0, 0, 0, 0))]
        half = half_for_bbox(bbox, width)
        if block.get("type") == 1:
            images.append(
                {
                    "pdf_page": page.number + 1,
                    "bbox": [round(value, 2) for value in bbox],
                    "half": half,
                    "source": "pymupdf",
                }
            )
            continue
        parts = []
        for line in block.get("lines") or []:
            line_text = " ".join(span.get("text", "") for span in line.get("spans") or []).strip()
            if line_text:
                parts.append(line_text)
        text = " ".join(parts).strip()
        if not text:
            continue
        blocks.append(text_block(text, bbox, "pymupdf", half, 0))
    return {"printed_page": printed, "text_blocks": blocks, "images": images}


def ocr_blocks_from_record(record: dict) -> tuple[list[dict], list[dict], str | None]:
    """Kept Tesseract lines and Paddle lines. They are not merged into one string."""
    half = record.get("half") or HALF_FULL
    blocks = []
    tesseract = record.get("tesseract") or {}
    for line in tesseract.get("kept_lines") or []:
        blocks.append(
            text_block(
                line.get("text") or "",
                ocr_xywh_to_bbox(line.get("bbox")),
                "tesseract",
                half,
                0,
                ocr_confidence=line.get("line_confidence"),
                bbox_space="preprocessed_image_px",
            )
        )
    paddle = record.get("paddleocr") or {}
    for line in paddle.get("lines") or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            text_block(
                text,
                ocr_xywh_to_bbox(line.get("bbox")),
                "paddleocr",
                half,
                0,
                ocr_confidence=line.get("confidence"),
                bbox_space="preprocessed_image_px",
            )
        )
    captions = []
    caption_text = (record.get("caption_text") or "").strip()
    if caption_text:
        for line in caption_text.splitlines():
            line = line.strip()
            if line:
                captions.append(
                    {
                        "text": line,
                        "half": half,
                        "source": "tesseract",
                        "pdf_page": record.get("pdf_page"),
                        "ocr_confidence": None,
                    }
                )
    status = record.get("recoverability")
    return blocks, captions, status


def png_size(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        handle.read(4)
        if handle.read(4) != b"IHDR":
            return None
        width, height = struct.unpack(">II", handle.read(8))
    return int(width), int(height)


def _ocr_image(record: dict) -> Path | None:
    source = record.get("source_image")
    if source and Path(source).is_file():
        return Path(source)
    ticker = record.get("ticker")
    half = record.get("half") or HALF_FULL
    pdf_page = record.get("pdf_page")
    if not ticker or not pdf_page:
        return None
    from src.config.paths import PREPROCESS_CACHE_DIR

    candidate = PREPROCESS_CACHE_DIR / str(ticker) / f"p{int(pdf_page):04d}_{half}" / "full.png"
    return candidate if candidate.is_file() else None


def ocr_printed_folio(records: list[dict], *, spread: bool) -> dict | None:
    """Folio from raw OCR words. One half never invents a left/right pair."""
    choices = []
    for record in records:
        half = str(record.get("half") or HALF_FULL)
        words = []
        for variant in (record.get("tesseract") or {}).get("raw_variants") or []:
            words.extend(variant.get("words") or [])
        if not words:
            continue
        size = png_size(_ocr_image(record))
        if not size:
            continue
        width, height = size
        detail = printed_page_detail(words, height, page_width=width, allow_pair=False)
        if not detail or detail.get("confidence") is None:
            continue
        if float(detail["confidence"]) < OFFSET_MIN_CONFIDENCE:
            continue
        choices.append({**detail, "half": half})
    if spread:
        return combine_half_folios(choices)
    singles = [item for item in choices if not item.get("pair") and "/" not in str(item.get("printed_page"))]
    if len(singles) != 1:
        return None
    return singles[0]


def load_jsonl_by_page(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["pdf_page"])] = row
    return rows


def load_ocr_records(ocr_dir: Path, ticker: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    if not ocr_dir.is_dir():
        return grouped
    for path in ocr_dir.glob(f"{ticker}_p*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(int(record["pdf_page"]), []).append(record)
    return grouped


def load_tables(table_dir: Path, ticker: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    if not table_dir.is_dir():
        return grouped
    for path in table_dir.rglob(f"{ticker}-p*.json"):
        if path.name == "summary.json":
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        if "pdf_page" not in table or "rows" not in table:
            continue
        grouped.setdefault(int(table["pdf_page"]), []).append(table)
    return grouped
