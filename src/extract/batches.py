"""Split pages into batches. A long page is sent whole rather than cut off."""

from __future__ import annotations

from src.extract.evidence import render_page


def make_batches(pages: list[dict], max_chars: int = 12000, overlap: int = 1) -> list[dict]:
    """Group consecutive pages. Overlap is context only and is not required to fit by dropping text."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    ordered = sorted(pages, key=lambda page: int(page["pdf_page"]))
    batches = []
    start = 0
    while start < len(ordered):
        context = []
        if batches and overlap > 0:
            previous = batches[-1]["target_pages"]
            context = [page for page in ordered if page["pdf_page"] in previous[-overlap:]]
        chosen = []
        size = sum(len(render_page(page, "context")) for page in context)
        index = start
        while index < len(ordered):
            rendered = render_page(ordered[index], "target")
            if chosen and size + len(rendered) > max_chars:
                break
            chosen.append(ordered[index])
            size += len(rendered)
            index += 1
            if len(rendered) > max_chars:
                break
        target_ids = [page["pdf_page"] for page in chosen]
        context_ids = [page["pdf_page"] for page in context if page["pdf_page"] not in target_ids]
        packets = [page for page in context if page["pdf_page"] in context_ids] + chosen
        batches.append(
            {
                "pages": packets,
                "target_pages": target_ids,
                "context_pages": context_ids,
                "char_count": sum(len(render_page(page, "target")) for page in packets),
            }
        )
        start = index
    return batches
