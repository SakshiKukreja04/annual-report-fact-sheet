"""Turn one PDF page's extracted pieces into a single record.

Digital text and OCR are both kept. Only one of them is marked selected.
"""

from __future__ import annotations

RECOVERABLE = "recoverable"
ILLEGIBLE = "illegible"
IMAGE_ONLY = "image_only_visual"
PENDING_OCR = "pending_ocr"

REQUIRED_KEYS = (
    "ticker",
    "pdf_page",
    "printed_page",
    "page_type",
    "content_status",
    "text_blocks",
    "tables",
    "captions",
    "images",
    "extraction_methods",
    "quality_metadata",
    "warnings",
)


def empty_page(ticker: str, pdf_page: int, printed_page: str | None = None) -> dict:
    return {
        "ticker": ticker,
        "pdf_page": int(pdf_page),
        "printed_page": printed_page,
        "page_type": [],
        "content_status": PENDING_OCR,
        "text_blocks": [],
        "tables": [],
        "captions": [],
        "images": [],
        "extraction_methods": [],
        "quality_metadata": {},
        "warnings": [],
    }


def text_block(
    text: str,
    bbox: list[float],
    source: str,
    half: str,
    reading_order: int,
    ocr_confidence: float | None = None,
    selected: bool = False,
    bbox_space: str = "pdf_points",
) -> dict:
    return {
        "text": text,
        "bbox": [round(float(value), 2) for value in bbox],
        "bbox_space": bbox_space,
        "source": source,
        "ocr_confidence": ocr_confidence,
        "half": half,
        "reading_order": reading_order,
        "selected": selected,
    }


def assign_reading_order(blocks: list[dict]) -> list[dict]:
    """Left half, then right half. Within a half, top to bottom and left to right."""
    rank = {"LEFT": 0, "FULL": 0, "RIGHT": 1}
    ordered = sorted(
        blocks,
        key=lambda block: (rank.get(block["half"], 0), block["bbox"][1], block["bbox"][0]),
    )
    for index, block in enumerate(ordered):
        block["reading_order"] = index
    return ordered


def _char_count(blocks: list[dict]) -> int:
    return sum(len(block["text"]) for block in blocks)


def _half_name(block: dict) -> str:
    return str(block.get("half") or "FULL")


def _blocks_on(blocks: list[dict], halves: set[str] | None) -> list[dict]:
    if halves is None:
        return list(blocks)
    return [block for block in blocks if _half_name(block) in halves]


def choose_text_source(digital_blocks: list[dict], ocr_blocks: list[dict], ocr_recoverable: bool, garbled: bool) -> tuple[str | None, str]:
    """Pick a source. The other source stays in the record for audit."""
    digital_chars = _char_count(digital_blocks)
    ocr_chars = _char_count(ocr_blocks)
    if digital_chars >= 80 and not garbled:
        return "pymupdf", "The text layer is long enough. OCR is retained and not mixed into it."
    if ocr_recoverable and ocr_chars > digital_chars:
        return "ocr", "OCR kept more readable text than the text layer."
    if digital_chars >= 40:
        note = " The text layer is flagged as low quality." if garbled else ""
        return "pymupdf", "The text layer is the better of the two sources." + note
    if ocr_recoverable and ocr_chars > 0:
        return "ocr", "The text layer is short, so the recoverable OCR lines are selected."
    return None, "No source produced text worth selecting."


def preferred_ocr_source(ocr_blocks: list[dict]) -> str | None:
    """Prefer the strongest OCR layer: Mistral, then Qwen vision OCR, then Tess/Paddle."""
    sources = {block["source"] for block in ocr_blocks}
    if "ocr_mistral" in sources:
        return "ocr_mistral"
    if "ocr_vision" in sources:
        return "ocr_vision"
    if "tesseract" in sources:
        return "tesseract"
    if "paddleocr" in sources:
        return "paddleocr"
    return None


def _mark_selected(blocks: list[dict], source_names: set[str], halves: set[str] | None = None) -> None:
    for block in blocks:
        if halves is not None and _half_name(block) not in halves:
            block["selected"] = False
            continue
        block["selected"] = block["source"] in source_names


def ocr_xywh_to_bbox(box: list[float] | None) -> list[float]:
    if not box or len(box) < 4:
        return [0, 0, 0, 0]
    x, y, width, height = [float(value) for value in box[:4]]
    return [x, y, x + width, y + height]


def table_view(table: dict) -> dict:
    """Copy the table fields the page record needs, including each row's page."""
    rows = []
    for row in table.get("rows") or []:
        rows.append(
            {
                "pdf_page": int(row.get("pdf_page", table["pdf_page"])),
                "printed_page": row.get("printed_page", table.get("printed_page")),
                "cells": list(row.get("cells") or []),
            }
        )
    return {
        "table_id": table.get("table_id"),
        "pdf_page": int(table["pdf_page"]),
        "printed_page": table.get("printed_page"),
        "headers": list(table.get("headers") or []),
        "rows": rows,
        "units": table.get("units"),
        "footnotes": list(table.get("footnotes") or []),
        "extraction_method": table.get("extraction_method"),
        "confidence": table.get("confidence"),
        "continuation_from": table.get("continuation_from"),
        "continuation_to": table.get("continuation_to"),
    }


def assemble_page(
    ticker: str,
    pdf_page: int,
    printed_page: str | None,
    page_type: list[str],
    digital_blocks: list[dict],
    ocr_blocks: list[dict],
    captions: list[dict],
    images: list[dict],
    tables: list[dict],
    ocr_recoverability: dict[str, str] | None = None,
    garbled: bool = False,
    classification_confidence: float | None = None,
    recommended_method: str | None = None,
) -> dict:
    """Build one page. Spread halves stay on this same pdf_page."""
    page = empty_page(ticker, pdf_page, printed_page)
    page["page_type"] = list(page_type)
    labels = set(page_type)
    recoverability = {str(half): status for half, status in (ocr_recoverability or {}).items()}
    bad_halves = {half for half, status in recoverability.items() if status == "ILLEGIBLE"}
    good_halves = {half for half, status in recoverability.items() if status != "ILLEGIBLE"}
    scoped_digital = _blocks_on(digital_blocks, good_halves)
    scoped_ocr = _blocks_on(ocr_blocks, good_halves)
    scoped_status = {half: recoverability[half] for half in good_halves}
    # One illegible half must not throw away the other half when that half has text.
    usable_partial = bool(bad_halves) and bool(good_halves) and (
        (_char_count(scoped_digital) >= 40 and not garbled)
        or (any(status == "RECOVERABLE" for status in scoped_status.values()) and _char_count(scoped_ocr) > 0)
        or _char_count(scoped_ocr) >= 80
    )
    if usable_partial:
        illegible = False
        select_digital = scoped_digital
        select_ocr = scoped_ocr
        select_rec = scoped_status
        select_halves: set[str] | None = good_halves
    else:
        illegible = bool(bad_halves)
        select_digital = list(digital_blocks)
        select_ocr = list(ocr_blocks)
        select_rec = recoverability
        select_halves = None
    text_page = "DIGITAL_TEXT" in labels or "MIXED_TEXT_IMAGE" in labels
    visual = not text_page and (
        "IMAGE_ONLY_VISUAL" in labels
        or any(status == "IMAGE_ONLY_VISUAL" for status in select_rec.values())
    )
    digital_usable = _char_count(select_digital) >= 40 and not garbled
    # OCR assessor may label a half IMAGE_ONLY_VISUAL even after returning kept lines.
    # Treat substantial OCR text as recoverable so image-only pages can fall back.
    ocr_recoverable = (
        any(status == "RECOVERABLE" for status in select_rec.values())
        or _char_count(select_ocr) >= 80
    ) and not illegible

    # Image-only pages: prefer OCR body text over short digital captions.
    if visual and not tables and ocr_recoverable and _char_count(select_ocr) >= 80 and _char_count(select_ocr) > _char_count(select_digital):
        source, reason = "ocr", "OCR recovered body text on an image-only page."
    else:
        allow_ocr = ocr_recoverable and (not visual or _char_count(select_ocr) >= 80)
        source, reason = choose_text_source(select_digital, select_ocr, allow_ocr, garbled)
        if visual and not tables and source != "ocr":
            source = None
            reason = "This is an image-only page. Captions are kept. Body text is not selected."
    if illegible and not digital_usable:
        source = None
        reason = "The scan is illegible. OCR text is stored for audit and is not selected."
    elif usable_partial and source:
        reason = "One half is illegible. The other half has usable text and is selected. " + reason

    selected_sources = set()
    if source == "pymupdf":
        selected_sources.add("pymupdf")
    elif source == "ocr":
        chosen_ocr = preferred_ocr_source(select_ocr if usable_partial else ocr_blocks)
        if chosen_ocr:
            selected_sources.add(chosen_ocr)
    blocks = assign_reading_order(list(digital_blocks) + list(ocr_blocks))
    _mark_selected(blocks, selected_sources, halves=select_halves)
    for block in blocks:
        block["pdf_page"] = int(pdf_page)
    page["text_blocks"] = blocks
    if visual:
        known = {caption.get("text") for caption in captions}
        for block in digital_blocks:
            if block["text"] not in known:
                captions.append(
                    {
                        "text": block["text"],
                        "half": block["half"],
                        "source": "pymupdf",
                        "pdf_page": int(pdf_page),
                        "ocr_confidence": None,
                    }
                )
                known.add(block["text"])
    for caption in captions:
        caption["pdf_page"] = int(pdf_page)
    page["captions"] = captions
    for image in images:
        image["pdf_page"] = int(pdf_page)
    page["images"] = images
    page["tables"] = [table_view(table) for table in tables]

    methods = []
    for block in blocks:
        if block["source"] not in methods:
            methods.append(block["source"])
    for table in page["tables"]:
        method = table.get("extraction_method")
        if method and method not in methods:
            methods.append(method)
    page["extraction_methods"] = methods

    if illegible and not digital_usable:
        page["content_status"] = ILLEGIBLE
        page["warnings"].append("illegible_low_dpi")
    elif any(block["selected"] for block in blocks) or page["tables"]:
        if usable_partial:
            page["warnings"].append("partial_half_illegible")
        # OCR-selected image-only pages become recoverable once body text is chosen.
        page["content_status"] = RECOVERABLE
        if visual and source == "ocr":
            page["warnings"].append("image_only_ocr_recovered")
    elif visual and not page["tables"]:
        page["content_status"] = IMAGE_ONLY
        page["warnings"].append("image_only_visual")
    elif "SCANNED_IMAGE" in labels and not recoverability:
        page["content_status"] = PENDING_OCR
        page["warnings"].append("ocr_not_run")
    else:
        page["content_status"] = PENDING_OCR
        page["warnings"].append("missing_text")

    if digital_blocks and ocr_blocks:
        page["warnings"].append("digital_and_ocr_both_present")
    ocr_sources = {block["source"] for block in ocr_blocks}
    if len(ocr_sources) > 1:
        page["warnings"].append("ocr_engines_kept_separate")
    if garbled:
        page["warnings"].append("garbled_text_layer")
    if any(table.get("extraction_method") == "paddleocr_pp_structure" for table in page["tables"]):
        page["warnings"].append("scanned_table_text_is_unverified")

    page["quality_metadata"] = {
        "classification_confidence": classification_confidence,
        "recommended_extraction_method": recommended_method,
        "selected_text_source": source,
        "selection_reason": reason,
        "ocr_recoverability": recoverability,
        "digital_char_count": _char_count(digital_blocks),
        "ocr_char_count": _char_count(ocr_blocks),
        "garbled_text_layer": garbled,
        "candidate_extraction": page["content_status"] == RECOVERABLE,
    }
    return page


def is_candidate(page: dict) -> bool:
    """Illegible and not-yet-read scans do not enter candidate extraction."""
    return page["content_status"] == RECOVERABLE and page["quality_metadata"].get("candidate_extraction") is True
