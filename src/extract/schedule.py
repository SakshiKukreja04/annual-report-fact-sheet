"""Assign ranked pages to providers that still have daily tokens left."""

from __future__ import annotations


def allocate(rows: list[dict], providers: list[dict]) -> dict:
    """Spend each provider's remaining tokens on the highest-scoring pages first."""
    remaining = {provider["name"]: int(provider.get("remaining") or 0) for provider in providers}
    daily = {provider["name"]: int(provider.get("daily_tokens") or 0) for provider in providers}
    assigned: dict[str, list[dict]] = {name: [] for name in remaining}
    uncovered = []
    for row in sorted(rows, key=lambda item: (-int(item.get("score") or 0), item.get("ticker") or "", int(item["pdf_page"]))):
        if row.get("done"):
            continue
        cost = int(row.get("tokens") or 0)
        choice = next((name for name, left in remaining.items() if left >= cost and cost > 0), None)
        if choice is None:
            uncovered.append(row)
            continue
        remaining[choice] -= cost
        assigned[choice].append(row)
    by_company: dict[str, dict] = {}
    for row in rows:
        company = by_company.setdefault(
            row["ticker"],
            {"pages_done": 0, "pages_queued": 0, "tokens_needed": 0, "pages_uncovered": []},
        )
        if row.get("done"):
            company["pages_done"] += 1
            continue
        company["tokens_needed"] += int(row.get("tokens") or 0)
        company["pages_queued"] += 1
    for row in uncovered:
        by_company[row["ticker"]]["pages_uncovered"].append(int(row["pdf_page"]))
        by_company[row["ticker"]]["pages_queued"] -= 1
    return {
        "providers": [
            {"name": name, "daily_tokens": daily[name], "remaining": remaining[name], "pages": len(assigned[name])}
            for name in remaining
        ],
        "companies": by_company,
        "uncovered": uncovered,
    }
