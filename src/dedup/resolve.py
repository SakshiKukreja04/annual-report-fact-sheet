"""Group repeated candidates. Only a transparent high-confidence rule merges them."""

from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz

from src.dedup.normalize import content_tokens, entity_key, normalize

AUTO_NAME = 94
REVIEW_NAME = 85
DESCRIPTOR_OK = 75
DESCRIPTOR_SPLIT = 55
EXACT_KINDS = {"asset", "research_platform"}


def compare(left: dict, right: dict, ticker: str = "") -> dict:
    """Decide whether two candidates are the same activity."""
    if left.get("kind") != right.get("kind"):
        return _decision("distinct", 0.0, "different_kind")
    if entity_key(left.get("entity", ""), ticker) != entity_key(right.get("entity", ""), ticker):
        return _decision("distinct", 0.2, "different_entity")
    if left.get("status") != right.get("status"):
        return _decision("review", 0.6, "status_conflict")
    left_name = normalize(left.get("name", ""))
    right_name = normalize(right.get("name", ""))
    name_score = 100.0 if left_name == right_name and left_name else float(fuzz.token_set_ratio(left_name, right_name))
    same_subsidiary = (
        left.get("kind") == "subsidiary"
        and content_tokens(left.get("name", ""))
        and content_tokens(left.get("name", "")) == content_tokens(right.get("name", ""))
    )
    if same_subsidiary:
        name_score = 100.0
    descriptor_score = _descriptor_score(left.get("descriptor", ""), right.get("descriptor", ""))
    unique = content_tokens(left.get("name", "")).symmetric_difference(content_tokens(right.get("name", "")))
    if left.get("kind") in EXACT_KINDS and left_name != right_name:
        return _decision("distinct", round(name_score / 100, 2), "asset_or_project_name_not_exact")
    if unique and name_score < 100:
        if name_score >= REVIEW_NAME:
            return _decision("review", round(name_score / 100, 2), "unique_name_tokens")
        return _decision("distinct", round(name_score / 100, 2), "different_name")
    if descriptor_score < DESCRIPTOR_SPLIT:
        return _decision("distinct", round(name_score / 100, 2), "different_activity")
    if name_score >= AUTO_NAME and descriptor_score >= DESCRIPTOR_OK:
        reason = "same_entity_same_activity"
        confidence = 0.98 if left_name == right_name else round(name_score / 100, 2)
        return _decision("merged", confidence, reason)
    if name_score >= REVIEW_NAME:
        return _decision("review", round(name_score / 100, 2), "ambiguous_name")
    return _decision("distinct", round(name_score / 100, 2), "different_name")


def resolve(candidates: list[dict], ticker: str = "") -> dict:
    """Merge high-confidence groups. Ambiguous pairs stay in the review list."""
    indexed = [{**item, "candidate_id": item.get("candidate_id") or f"c{index}"} for index, item in enumerate(candidates)]
    parent = list(range(len(indexed)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    reviews = []
    for left_index, left in enumerate(indexed):
        for right_index in range(left_index + 1, len(indexed)):
            decision = compare(left, indexed[right_index], ticker)
            if decision["decision"] == "merged":
                parent[find(right_index)] = find(left_index)
            elif decision["decision"] == "review":
                reviews.append({"members": [left["candidate_id"], indexed[right_index]["candidate_id"]], **decision})

    groups: dict[int, list[dict]] = {}
    for index, item in enumerate(indexed):
        groups.setdefault(find(index), []).append(item)

    merged = []
    unchanged = []
    for members in groups.values():
        if len(members) == 1:
            unchanged.append(members[0])
            continue
        merged.append(_merge_group(members))
    return {
        "ticker": ticker,
        "language_model_used": False,
        "merged": merged,
        "review": reviews,
        "unchanged": unchanged,
        "merged_count": len(merged),
        "review_count": len(reviews),
    }


def _merge_group(members: list[dict]) -> dict:
    base = dict(members[0])
    all_sources = []
    figures = []
    shares = []
    for member in members:
        for source in member.get("sources") or []:
            if source not in all_sources:
                all_sources.append(source)
        for figure in member.get("supporting_figures") or []:
            if figure not in figures:
                figures.append(figure)
        if member.get("revenue_share_pct") is not None:
            shares.append(member["revenue_share_pct"])
    chosen = _prefer_sources(all_sources, base.get("name", ""))
    share = shares[0] if shares and all(value == shares[0] for value in shares) else None
    descriptors = [member.get("descriptor") or "" for member in members]
    base.update(
        {
            "descriptor": max(descriptors, key=len),
            "sources": chosen,
            "retained_sources": all_sources,
            "supporting_figures": figures,
            "revenue_share_pct": share,
            "member_ids": [member["candidate_id"] for member in members],
            "confidence": 0.98,
            "reason": "same_entity_same_activity",
        }
    )
    return base


def _prefer_sources(sources: list[dict], name: str) -> list[dict]:
    """Keep three quotes. Longer quotes that use the activity name rank first."""
    tokens = content_tokens(name)

    def score(source: dict) -> tuple:
        quote = source.get("quote") or ""
        present = sum(token in normalize(quote) for token in tokens)
        return (present, min(len(quote), 280), -int(source.get("pdf_page") or 0))

    ordered = sorted(sources, key=score, reverse=True)
    return ordered[:3]


def _descriptor_score(left: str, right: str) -> float:
    left_key = normalize(left)
    right_key = normalize(right)
    if not left_key or not right_key:
        return 100.0
    return float(fuzz.token_set_ratio(left_key, right_key))


def _decision(decision: str, confidence: float, reason: str) -> dict:
    return {"decision": decision, "confidence": confidence, "reason": reason}


def render_markdown(result: dict) -> str:
    lines = [
        f"# Duplicate resolution for {result.get('ticker') or 'candidates'}",
        "",
        "High-confidence matches were merged with deterministic rules. Ambiguous pairs were not sent to a language model.",
        "",
        f"- Merged groups: {result['merged_count']}",
        f"- Review pairs: {result['review_count']}",
        f"- Unchanged candidates: {len(result['unchanged'])}",
        "",
    ]
    for group in result["merged"]:
        lines.append(f"## Merged: {group['name']}")
        lines.append("")
        lines.append(f"- Entity: {group.get('entity')}")
        lines.append(f"- Kind: {group.get('kind')}")
        lines.append(f"- Confidence: {group.get('confidence')}")
        lines.append(f"- Reason: {group.get('reason')}")
        lines.append(f"- Members: {', '.join(group.get('member_ids') or [])}")
        lines.append(f"- Sources kept: {len(group.get('sources') or [])}")
        lines.append(f"- Sources retained: {len(group.get('retained_sources') or [])}")
        lines.append("")
    if result["review"]:
        lines.append("## Manual review")
        lines.append("")
        for pair in result["review"]:
            lines.append(f"- {pair['members']}: {pair['reason']} ({pair['confidence']})")
        lines.append("")
    return "\n".join(lines)


def load_parsed_candidates(folder: Path) -> list[dict]:
    items = []
    parsed = folder / "parsed"
    if not parsed.is_dir():
        return items
    for path in sorted(parsed.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for index, candidate in enumerate(record.get("candidates") or []):
            items.append({**candidate, "candidate_id": f"{path.stem}:{index}", "batch_id": record.get("batch_id")})
    return items


def write_report(ticker: str, folder: Path) -> dict:
    result = resolve(load_parsed_candidates(folder), ticker)
    destination = folder / "dedup"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (destination / "resolution.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    from src.config.paths import CANDIDATE_DIR

    parser = argparse.ArgumentParser(description="Resolve duplicate candidates for one company.")
    parser.add_argument("ticker")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    result = write_report(ticker, CANDIDATE_DIR / ticker)
    print(json.dumps({
        "ticker": ticker,
        "merged_count": result["merged_count"],
        "review_count": result["review_count"],
        "unchanged": len(result["unchanged"]),
        "language_model_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
