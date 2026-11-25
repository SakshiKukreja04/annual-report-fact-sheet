"""Stage 2: classify every PDF page and choose an extraction method.

Labels may overlap. A spread can also be table-heavy. This stage does not
run OCR and does not call a model.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.config.paths import CLASSIFICATION_DIR, LOG_DIR, PROJECT_ROOT
from src.ingestion.audit import locate_pdfs
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")

# Wider than A4 landscape (~842 pt) and at least two portrait pages side by side.
SPREAD_MIN_WIDTH_PT = 1050.0
# Between landscape-single and double-page width. Do not force a spread label.
AMBIGUOUS_SPREAD_MIN_WIDTH_PT = 900.0

DIGITAL_TEXT_CHARS = 400
SHORT_TEXT_CHARS = 200
CAPTION_MAX_CHARS = 400
SCANNED_MAX_CHARS = 8

LABELS = (
    "DIGITAL_TEXT",
    "SCANNED_IMAGE",
    "MIXED_TEXT_IMAGE",
    "TABLE_HEAVY",
    "TWO_PAGE_SPREAD",
    "IMAGE_ONLY_VISUAL",
    "GARBLED_OR_LOW_QUALITY_TEXT",
    "UNKNOWN",
)

_CID_PATTERN = re.compile(r"\(cid:\d+\)")
# Non-breaking spaces are normal PDF spacing, not corruption.
_REPEAT_PATTERN = re.compile(r"([^\s.\-_|=])\1{7,}")
_COMMON_PUNCT = set(".,;:'\"!?()[]{}-/&%+*#@")
_ORDINARY_NON_ASCII = set("\u00a0\u2010\u2011\u2012\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u00b7")


def _ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part / whole, 4)


def text_quality(text: str) -> dict:
    """Preliminary quality stats. A low letter ratio alone is not garbled text."""
    raw_count = len(text)
    stripped = text.strip()
    stripped_count = len(stripped)
    non_space = [char for char in text if not char.isspace()]
    non_space_count = len(non_space)
    alpha = sum(char.isalpha() for char in non_space)
    replacement = text.count("\ufffd") + len(_CID_PATTERN.findall(text))
    symbols = sum(
        1
        for char in non_space
        if not char.isalnum() and char not in _COMMON_PUNCT
    )
    odd_counts = Counter(
        char for char in text if ord(char) > 127 and char not in _ORDINARY_NON_ASCII
    )
    dominant_odd = odd_counts.most_common(1)[0][1] if odd_counts else 0
    # Ignore non-breaking spaces when looking for a repeated corrupted glyph.
    run_source = text.replace("\u00a0", " ")
    repeated_match = _REPEAT_PATTERN.search(run_source)
    repeated_ratio = 0.0
    if repeated_match:
        repeated_char = repeated_match.group(1)
        repeated_ratio = _ratio(run_source.count(repeated_char), raw_count)
    dominant_ratio = _ratio(dominant_odd, raw_count)
    latin = sum(1 for char in non_space if ("a" <= char <= "z") or ("A" <= char <= "Z"))
    latin_ratio = _ratio(latin, non_space_count)
    # Long layers with almost no Latin letters are usually the wrong script/encoding
    # for these English annual reports and should fall back to OCR when available.
    low_latin = stripped_count >= 200 and latin_ratio < 0.35
    garbled = stripped_count >= 80 and (
        _ratio(replacement, non_space_count) >= 0.005
        or dominant_ratio >= 0.08
        or repeated_ratio >= 0.05
        or low_latin
    )
    return {
        "text_char_count": raw_count,
        "text_stripped_char_count": stripped_count,
        "alpha_ratio": _ratio(alpha, non_space_count),
        "latin_ratio": latin_ratio,
        "replacement_char_ratio": _ratio(replacement, non_space_count),
        "symbol_ratio": _ratio(symbols, non_space_count),
        "dominant_non_ascii_ratio": dominant_ratio,
        "repeated_corruption": repeated_ratio >= 0.05 or dominant_ratio >= 0.08,
        "garbled_or_low_quality": garbled,
    }


def _image_coverage(page: pymupdf.Page, image_info: list[dict]) -> float:
    rect = page.rect
    page_area = float(rect.width * rect.height) or 1.0
    covered = 0.0
    for image in image_info:
        bbox = image.get("bbox")
        if not bbox:
            continue
        x0 = max(float(bbox[0]), float(rect.x0))
        y0 = max(float(bbox[1]), float(rect.y0))
        x1 = min(float(bbox[2]), float(rect.x1))
        y1 = min(float(bbox[3]), float(rect.y1))
        covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return round(min(covered / page_area, 1.0), 4)


def _table_signals(page: pymupdf.Page, words: list) -> dict:
    """Detect row/column structure from word gaps and horizontal rules.

    Body text in columns has small gaps between words. Table cells are
    separated by much wider gaps and are usually short.
    """
    width = float(page.rect.width)
    horizontal_rules = 0
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is not None and rect.width > width * 0.15 and rect.height < 2.2:
            horizontal_rules += 1

    ordered = sorted(words, key=lambda word: (word[1], word[0]))
    rows: list[list] = []
    for word in ordered:
        if rows and abs(word[1] - rows[-1][0][1]) <= 3.0:
            rows[-1].append(word)
        else:
            rows.append([word])

    table_rows = 0
    for row in rows:
        row = sorted(row, key=lambda word: word[0])
        cells: list[list] = []
        current: list = []
        for word in row:
            if current and word[0] - current[-1][2] > 22:
                cells.append(current)
                current = [word]
            else:
                current.append(word)
        if current:
            cells.append(current)
        if len(cells) < 3:
            continue
        texts = [" ".join(item[4] for item in cell) for cell in cells]
        short_share = sum(len(value) <= 22 for value in texts) / len(texts)
        numeric_cells = sum(any(char.isdigit() for char in value) for value in texts)
        if short_share >= 0.6 and numeric_cells >= 1:
            table_rows += 1

    table_heavy = table_rows >= 6 or (horizontal_rules >= 12 and table_rows >= 3)
    if table_rows >= 8:
        score = 0.9
    elif table_heavy:
        score = 0.75
    elif table_rows >= 3:
        score = 0.45
    else:
        score = 0.0
    return {
        "table_row_count": table_rows,
        "horizontal_rule_count": horizontal_rules,
        "table_score": score,
        "table_heavy": table_heavy,
    }


def _gutter_from_words(words: list, width: float) -> dict:
    midpoint = width / 2
    band = width * 0.025
    left = 0
    right = 0
    center = 0
    for word in words:
        chars = len(word[4])
        if word[2] < midpoint - band:
            left += chars
        elif word[0] > midpoint + band:
            right += chars
        else:
            center += chars
    both_sides = left >= 40 and right >= 40
    quiet_center = center / max(left + right, 1) < 0.06
    return {
        "gutter_left_chars": left,
        "gutter_right_chars": right,
        "gutter_center_chars": center,
        "text_gutter_found": both_sides and quiet_center,
    }


def _image_gutter_found(page: pymupdf.Page) -> bool:
    """Look for a pale vertical band at the center. This is not OCR."""
    import pymupdf
    rect = page.rect
    width = float(rect.width)
    height = float(rect.height)
    midpoint = width / 2

    def luminance(clip: pymupdf.Rect) -> float:
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(0.2, 0.2),
            clip=clip,
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )
        samples = pixmap.samples
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    center = luminance(pymupdf.Rect(midpoint - 5, height * 0.2, midpoint + 5, height * 0.8))
    left = luminance(pymupdf.Rect(width * 0.22, height * 0.2, width * 0.22 + 10, height * 0.8))
    right = luminance(pymupdf.Rect(width * 0.78, height * 0.2, width * 0.78 + 10, height * 0.8))
    return center > left + 15 and center > right + 15


def measure_page(page: pymupdf.Page) -> dict:
    """Collect rule inputs for one page."""
    text = page.get_text("text") or ""
    words = page.get_text("words") or []
    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = sum(1 for block in blocks if block.get("type") == 0)
    image_info = page.get_image_info()
    width = round(float(page.rect.width), 2)
    height = round(float(page.rect.height), 2)
    coverage = _image_coverage(page, image_info)
    quality = text_quality(text)
    tables = _table_signals(page, words)
    gutter = _gutter_from_words(words, width)
    aspect = round(width / height, 4) if height else 0.0

    if width >= SPREAD_MIN_WIDTH_PT and aspect >= 1.25:
        spread_band = "spread"
    elif AMBIGUOUS_SPREAD_MIN_WIDTH_PT <= width < SPREAD_MIN_WIDTH_PT and aspect >= 1.2:
        spread_band = "ambiguous"
    else:
        spread_band = "not_spread"

    image_gutter = False
    if spread_band == "spread" and not gutter["text_gutter_found"]:
        image_gutter = _image_gutter_found(page)

    stripped = quality["text_stripped_char_count"]
    if coverage >= 0.75 and stripped < SCANNED_MAX_CHARS:
        scanned_likelihood = 0.95
    elif coverage >= 0.45 and stripped < CAPTION_MAX_CHARS:
        scanned_likelihood = 0.35
    else:
        scanned_likelihood = 0.05

    return {
        "width_pt": width,
        "height_pt": height,
        "aspect_ratio": aspect,
        "rotation": int(page.rotation),
        "text_block_count": text_blocks,
        "word_count": len(words),
        "image_count": len(image_info),
        "image_coverage_ratio": coverage,
        "scanned_likelihood": scanned_likelihood,
        "spread_band": spread_band,
        "image_gutter_found": image_gutter,
        "gutter_found": gutter["text_gutter_found"] or image_gutter,
        **quality,
        **tables,
        **gutter,
    }


def decide(ticker: str, pdf_page: int, indicators: dict) -> dict:
    """Turn measurements into labels and one extraction method."""
    stripped = indicators["text_stripped_char_count"]
    coverage = indicators["image_coverage_ratio"]
    labels: list[str] = []
    reasons: list[str] = []
    confidences: list[float] = []

    digital = stripped >= DIGITAL_TEXT_CHARS or (
        stripped >= SHORT_TEXT_CHARS and coverage < 0.25 and not indicators["garbled_or_low_quality"]
    )
    scanned = coverage >= 0.75 and stripped < SCANNED_MAX_CHARS
    visual = (
        coverage >= 0.40
        and SCANNED_MAX_CHARS <= stripped < CAPTION_MAX_CHARS
        and indicators["word_count"] < 80
    )
    mixed = digital and coverage >= 0.25
    spread = indicators["spread_band"] == "spread"
    ambiguous_spread = indicators["spread_band"] == "ambiguous"
    table = indicators["table_heavy"]
    garbled = indicators["garbled_or_low_quality"] and digital

    if digital:
        labels.append("DIGITAL_TEXT")
        confidences.append(0.9 if indicators["alpha_ratio"] >= 0.4 else 0.74)
        reasons.append(
            f"Text layer has {stripped} characters, so digital extraction is the starting point."
        )
    if scanned:
        labels.append("SCANNED_IMAGE")
        confidences.append(0.92 if coverage >= 0.9 else 0.7)
        reasons.append(
            "A page-covering image has almost no text layer, so this is a scanned page for later OCR."
        )
    if mixed:
        labels.append("MIXED_TEXT_IMAGE")
        confidences.append(0.8)
        reasons.append(
            f"Image coverage is {coverage:.0%} and the text layer is still long enough to use."
        )
    if table:
        labels.append("TABLE_HEAVY")
        confidences.append(indicators["table_score"] or 0.75)
        reasons.append(
            f"{indicators['table_row_count']} aligned short rows "
            f"and {indicators['horizontal_rule_count']} horizontal rules suggest a table."
        )
    if spread:
        labels.append("TWO_PAGE_SPREAD")
        confidences.append(0.95 if indicators["width_pt"] >= 1100 else 0.8)
        gutter_note = "A central gutter was found." if indicators["gutter_found"] else (
            "No ink gutter was required because the width is about two portrait pages."
        )
        reasons.append(f"Width {indicators['width_pt']} pt is a two-page spread. {gutter_note}")
    if visual:
        labels.append("IMAGE_ONLY_VISUAL")
        confidences.append(0.78)
        reasons.append(
            "A large image has only a short caption. This is a visual page, not an OCR failure."
        )
    if garbled:
        labels.append("GARBLED_OR_LOW_QUALITY_TEXT")
        confidences.append(0.82)
        reasons.append(
            "The text layer contains repeated corrupted characters or replacement glyphs."
        )
    if ambiguous_spread:
        reasons.append(
            f"Width {indicators['width_pt']} pt is between a landscape page and a clear spread. "
            "Spread was not assigned; the page needs visual inspection."
        )

    content_labels = [label for label in labels if label != "TWO_PAGE_SPREAD"]
    if not content_labels:
        labels.append("UNKNOWN")
        confidences.append(0.4)
        reasons.append(
            "The layout was measured, but the page did not match a text, scan, table, or visual rule."
        )

    if scanned:
        method = "ocr"
    elif visual and not digital:
        method = "caption_only"
    elif garbled:
        method = "digital_text_review_quality"
    elif table and digital:
        method = "digital_text_and_tables"
    elif digital:
        method = "digital_text"
    else:
        method = "manual_review"

    uncertain = ambiguous_spread or "UNKNOWN" in labels or method == "manual_review" or min(confidences) < 0.6
    confidence = round(min(confidences), 2)
    return {
        "ticker": ticker,
        "pdf_page": pdf_page,
        "classifications": labels,
        "indicators": indicators,
        "confidence": confidence,
        "recommended_extraction_method": method,
        "classification_reason": " ".join(reasons),
        "uncertain": uncertain,
        "needs_visual_inspection": ambiguous_spread,
    }


def classify_page(page: pymupdf.Page, ticker: str) -> dict:
    return decide(ticker, page.number + 1, measure_page(page))


def classify_document(pdf_path: Path, ticker: str) -> list[dict]:
    import pymupdf

    rows: list[dict] = []
    with pymupdf.open(pdf_path) as document:
        LOGGER.info("%s classifying %s pages", ticker, document.page_count)
        for index, page in enumerate(document):
            rows.append(classify_page(page, ticker))
            if (index + 1) % 100 == 0:
                LOGGER.info("%s classified %s/%s pages", ticker, index + 1, document.page_count)
    return rows


def summarize(ticker: str, rows: list[dict]) -> dict:
    label_counts = {label: 0 for label in LABELS}
    method_counts: Counter[str] = Counter()
    uncertain_pages = []
    visual_inspection_pages = []
    for row in rows:
        for label in row["classifications"]:
            label_counts[label] = label_counts.get(label, 0) + 1
        method_counts[row["recommended_extraction_method"]] += 1
        if row["uncertain"]:
            uncertain_pages.append(
                {
                    "pdf_page": row["pdf_page"],
                    "classifications": row["classifications"],
                    "reason": row["classification_reason"],
                }
            )
        if row["needs_visual_inspection"]:
            visual_inspection_pages.append(row["pdf_page"])
    return {
        "ticker": ticker,
        "page_count": len(rows),
        "label_counts": label_counts,
        "method_counts": dict(method_counts),
        "uncertain_page_count": len(uncertain_pages),
        "uncertain_pages": uncertain_pages,
        "visual_inspection_pages": visual_inspection_pages,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_classification(
    output_dir: Path | None = None,
    *,
    company: str | None = None,
) -> dict:
    """Classify one company when `company` is set. Never classifies all tickers by default."""
    destination = output_dir or CLASSIFICATION_DIR
    located = locate_pdfs()
    if company:
        ticker = company.upper().strip()
        if ticker not in located:
            raise SystemExit(f"Unknown company {ticker}. Expected one of {', '.join(sorted(located))}.")
        targets = {ticker: located[ticker]}
    else:
        targets = located
    companies = []
    for ticker, pdf_path in targets.items():
        rows = classify_document(pdf_path, ticker)
        page_path = destination / f"{ticker}_pages.jsonl"
        write_jsonl(page_path, rows)
        company_summary = summarize(ticker, rows)
        try:
            company_summary["page_classification_path"] = page_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            company_summary["page_classification_path"] = page_path.resolve().as_posix()
        companies.append(company_summary)
        LOGGER.info(
            "%s labels %s uncertain=%s visual_inspection=%s",
            ticker,
            company_summary["label_counts"],
            company_summary["uncertain_page_count"],
            len(company_summary["visual_inspection_pages"]),
        )
    summary = {
        "stage": "page_classification",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": [
            "A page may have more than one label.",
            "TWO_PAGE_SPREAD requires width of at least 1050 pt. Ordinary landscape pages are not spreads.",
            "Widths from 900 pt up to 1050 pt are withheld and marked for visual inspection.",
            "GARBLED_OR_LOW_QUALITY_TEXT is a text-layer defect. Numeric tables are not garbled only because they contain digits.",
            "IMAGE_ONLY_VISUAL keeps a caption and is not sent to OCR as a failure.",
            "SCANNED_IMAGE is a full-page image with almost no text layer. OCR is recommended later and is not run here.",
        ],
        "companies": companies,
    }
    destination.mkdir(parents=True, exist_ok=True)
    # When classifying a single company, merge into existing summary instead of wiping others.
    summary_path = destination / "summary.json"
    if company and summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            prior = {
                item.get("ticker"): item
                for item in (previous.get("companies") or [])
                if isinstance(item, dict) and item.get("ticker")
            }
            for item in companies:
                prior[item["ticker"]] = item
            summary["companies"] = [prior[key] for key in sorted(prior)]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify PDF pages for one company. Does not run OCR or call a model.",
    )
    parser.add_argument(
        "--company",
        required=True,
        help="Ticker to classify (required). Example: --company COALINDIA",
    )
    args = parser.parse_args(argv)
    configure_logging(LOG_DIR / "classification.log")
    run_classification(company=args.company)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
