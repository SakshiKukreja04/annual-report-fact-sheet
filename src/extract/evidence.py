"""Turn page records into prompt text. Unreadable scans and OCR tables are left out."""

from __future__ import annotations


def pages_for_extraction(loaded: dict) -> list[dict]:
    """Candidate pages, plus image pages that have a readable caption. Illegible pages stay out."""
    chosen = []
    for page in loaded.get("candidate_pages") or []:
        if page.get("content_status") == "illegible":
            continue
        chosen.append(page)
    for page in loaded.get("image_only_pages") or []:
        if _captions(page):
            chosen.append(page)
    chosen.sort(key=lambda page: int(page["pdf_page"]))
    return chosen


def render_page(page: dict, role: str) -> str:
    """Render one page without cutting its text. Role is target or context."""
    lines = [
        f"[PAGE pdf_page={page['pdf_page']} printed_page={page.get('printed_page')} role={role}]"
    ]
    blocks = [block for block in page.get("text_blocks") or [] if block.get("selected") and (block.get("text") or "").strip()]
    for block in sorted(blocks, key=lambda item: item.get("reading_order", 0)):
        lines.append(f"[TEXT pdf_page={page['pdf_page']} half={block.get('half')}] {block['text']}")
    if "scanned_table_text_is_unverified" not in set(page.get("warnings") or []):
        for table in page.get("tables") or []:
            if not _table_text_usable(table):
                lines.append(f"[TABLE_OMITTED pdf_page={page['pdf_page']} table_id={table.get('table_id')} reason=unreadable]")
                continue
            lines.append(
                f"[TABLE pdf_page={table.get('pdf_page', page['pdf_page'])} printed_page={table.get('printed_page')} "
                f"table_id={table.get('table_id')}]"
            )
            if table.get("headers"):
                lines.append("headers: " + " | ".join(str(cell) for cell in table["headers"]))
            for row in table.get("rows") or []:
                lines.append(
                    f"row pdf_page={row.get('pdf_page', page['pdf_page'])}: "
                    + " | ".join(str(cell) for cell in row.get("cells") or [])
                )
    else:
        for table in page.get("tables") or []:
            lines.append(f"[TABLE_OMITTED pdf_page={page['pdf_page']} table_id={table.get('table_id')} reason=unverified_scan]")
    for caption in _captions(page):
        lines.append(f"[CAPTION pdf_page={page['pdf_page']} half={caption.get('half')}] {caption['text']}")
    if len(lines) == 1:
        lines.append("[NO_READABLE_EVIDENCE]")
    return "\n".join(lines)


def render_batch(batch: dict) -> str:
    parts = [render_page(page, "context" if page["pdf_page"] in batch["context_pages"] else "target") for page in batch["pages"]]
    return "\n\n".join(parts)


def _captions(page: dict) -> list[dict]:
    kept = []
    for caption in page.get("captions") or []:
        text = (caption.get("text") or "").strip()
        letters = sum(character.isalpha() for character in text)
        if letters >= 12:
            kept.append({**caption, "text": text})
    return kept


def _table_text_usable(table: dict) -> bool:
    method = (table.get("extraction_method") or "").casefold()
    if "paddle" in method or "tesseract" in method:
        return False
    return bool(table.get("headers") or table.get("rows"))
