"""Build JSON and Markdown from the same validated candidates."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.config.paths import PAGE_DIR, STAGING_DIR
from src.extract.schema import KINDS, STATUSES
from src.extract.select import page_text
from src.factsheet.names import actor_entity, assignment_key
from src.factsheet.overview import fallback_text
from src.factsheet.precision import folio_for_quote
from src.factsheet.schema import FactSheet

COMPANIES = ("POWERGRID", "HINDALCO", "COALINDIA")
SECTIONS = (
    ("Business segments", {"segment"}),
    ("Products and services", {"product_or_service", "business_line"}),
    ("Brands", {"brand"}),
    ("Subsidiaries and joint ventures", {"subsidiary"}),
    ("Channels and assets", {"channel", "asset"}),
    ("Research and other activities", {"research_platform", "philanthropy", "other"}),
)
_KIND_RANK = {
    "subsidiary": 9,
    "segment": 8,
    "business_line": 7,
    "product_or_service": 6,
    "brand": 5,
    "channel": 5,
    "asset": 4,
    "research_platform": 4,
    "philanthropy": 3,
    "other": 1,
}
_KEPT_SHARE = {"printed", "calculated_from_printed_figures"}


def load_validated(path: Path) -> list[dict]:
    """Read the validation file. A missing file is an empty validated set."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("validated") or []
    return list(payload)


def build_fact_sheet(
    ticker: str,
    records: list[dict],
    overview_text: str = "",
    *,
    segments: list[dict] | None = None,
    overview: dict | None = None,
    apply_overrides: bool = False,
) -> dict:
    """Return the sheet, the markdown, and any records that were left out."""
    items = []
    omitted = []
    for record in records:
        item, reason = _item_from_record(record)
        if reason:
            omitted.append({"name": _record_name(record), "reason": reason})
            continue
        items.append(item)
    items, merge_groups = merge_spec_items(items, _subsidiary_list_keys(ticker))
    pages = _load_pages(ticker)
    from src.factsheet.overrides import apply_manual_overrides

    if apply_overrides:
        items = apply_manual_overrides(ticker, items, pages)
    _apply_actors(items)
    _repair_list_entities(items, pages)
    _restamp_printed(items, pages)
    items.sort(key=lambda item: (min(source["pdf_page"] for source in item["sources"]), item["name"]))
    if overview and overview.get("one_liner") and overview.get("summary"):
        one_liner, summary = overview["one_liner"], overview["summary"]
    else:
        one_liner, summary = _overview(ticker, items, overview_text)
    sheet = {
        "ticker": ticker,
        "period": "FY25",
        "one_liner": one_liner,
        "summary": summary,
        "segments": list(segments) if segments is not None else _segments(records, items),
        "items": items,
    }
    validated = FactSheet.model_validate(sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    if errors:
        raise RuntimeError("Markdown and JSON items differ: " + "; ".join(errors))
    return {"sheet": validated, "markdown": markdown, "omitted": omitted, "merge_groups": merge_groups}


def render_markdown(sheet: dict) -> str:
    """Render every JSON item. Planned items are repeated in their own section."""
    lines = [
        f"# {sheet['ticker']} FY25",
        "",
        "## Company overview",
        "",
        sheet["one_liner"],
        "",
        sheet["summary"],
        "",
    ]
    for title, kinds in SECTIONS:
        chosen = [item for item in sheet["items"] if item["kind"] in kinds]
        lines.extend(_section(title, chosen))
    planned = [item for item in sheet["items"] if item["status"] in {"planned", "aspirational"}]
    lines.extend(_section("Planned and aspirational initiatives", planned))
    supported = [item for item in sheet["items"] if item["revenue_share_pct"] is not None]
    lines.extend(["## Revenue shares", ""])
    if not supported:
        lines.append("None.")
    else:
        for item in supported:
            lines.append(f"- {item['name']}: {json.dumps(item['revenue_share_pct'])}%")
    lines.append("")
    return "\n".join(lines)


def consistency_errors(sheet: dict, markdown: str) -> list[str]:
    """Compare the item set in JSON with the item set rendered in Markdown."""
    json_keys = {_item_key(item) for item in sheet["items"]}
    markdown_keys = {_item_key(item) for item in extract_markdown_items(markdown)}
    errors = []
    missing = json_keys - markdown_keys
    extra = markdown_keys - json_keys
    if missing:
        errors.append("missing from markdown: " + ", ".join(sorted(name for name, *_ in missing)))
    if extra:
        errors.append("missing from json: " + ", ".join(sorted(name for name, *_ in extra)))
    return errors


def extract_markdown_items(markdown: str) -> list[dict]:
    items = []
    for chunk in re.split(r"(?m)^### ", markdown)[1:]:
        rows = chunk.splitlines()
        fields = {}
        sources = []
        for line in rows[1:]:
            if line.startswith("#"):
                break
            for label in ("Kind", "Entity", "Status", "Descriptor", "Revenue share"):
                prefix = f"- {label}: "
                if line.startswith(prefix):
                    fields[label] = line[len(prefix):]
            matched = re.match(r"  - PDF page (\d+), printed page (.*?): (.*)", line)
            if matched:
                printed = matched.group(2)
                sources.append({
                    "pdf_page": int(matched.group(1)),
                    "printed_page": None if printed == "null" else printed,
                    "quote": json.loads(matched.group(3)),
                })
        share = fields["Revenue share"]
        items.append({
            "name": rows[0],
            "kind": fields["Kind"],
            "descriptor": json.loads(fields["Descriptor"]),
            "entity": fields["Entity"],
            "status": fields["Status"],
            "revenue_share_pct": None if share == "null" else float(share),
            "sources": sources,
        })
    return items


def write_fact_sheet(
    ticker: str,
    built: dict,
    output_dir: Path,
    force: bool = False,
    stem: str | None = None,
) -> dict:
    """Write both files. An existing file is left unchanged unless force is set."""
    folder = output_dir / ticker
    name = stem or ticker
    if name.endswith(".v0") or name.endswith(".post"):
        raise FileExistsError(f"{name} is a saved copy and is not rewritten.")
    json_path = folder / f"{name}.json"
    markdown_path = folder / f"{name}.md"
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"{names} already exists in {folder}. Refusing to overwrite.")
    errors = consistency_errors(built["sheet"], built["markdown"])
    if errors:
        raise RuntimeError("Markdown and JSON items differ: " + "; ".join(errors))
    payload = FactSheet.model_validate(built["sheet"]).model_dump(mode="json")
    _atomic_write(json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(markdown_path, built["markdown"])
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    reread = consistency_errors(saved, markdown_path.read_text(encoding="utf-8"))
    if reread:
        raise RuntimeError("Saved Markdown and JSON items differ: " + "; ".join(reread))
    return {
        "ticker": ticker,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "item_count": len(payload["items"]),
        "segment_count": len(payload["segments"]),
        "omitted": built["omitted"],
    }


def _atomic_write(path: Path, text: str) -> None:
    """Replace the final file only after the full text is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _item_from_record(record: dict) -> tuple[dict | None, str | None]:
    if record.get("outcome") not in (None, "validated"):
        return None, "not_validated"
    candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
    name = candidate.get("name") or ""
    if not str(name).strip():
        return None, "empty_name"
    if candidate.get("kind") not in KINDS:
        return None, "invalid_kind"
    if candidate.get("status") not in STATUSES:
        return None, "invalid_status"
    if not str(candidate.get("entity") or "").strip():
        return None, "missing_text"
    descriptor = str(candidate.get("descriptor") or "")
    if not descriptor.strip() and not candidate.get("list_subsidiary"):
        return None, "missing_text"
    sources = _sources(record, candidate)
    if not sources:
        return None, "no_valid_source"
    if len(sources) > 3:
        return None, "too_many_sources"
    return {
        "name": name,
        "kind": candidate["kind"],
        "descriptor": descriptor,
        "entity": candidate["entity"],
        "status": candidate["status"],
        "revenue_share_pct": _share(record),
        "sources": sources,
    }, None


def _sources(record: dict, candidate: dict) -> list[dict]:
    accepted = {
        (quote.get("pdf_page"), quote.get("quote"))
        for quote in record.get("quotes") or []
        if quote.get("accepted") is True
    }
    require_acceptance = bool(record.get("quotes"))
    sources = []
    for source in candidate.get("sources") or []:
        page = source.get("pdf_page")
        quote = source.get("quote")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            continue
        if not isinstance(quote, str) or not quote.strip():
            continue
        if require_acceptance and (page, quote) not in accepted:
            continue
        printed = source.get("printed_page")
        sources.append({
            "pdf_page": page,
            "printed_page": None if printed in (None, "") else str(printed),
            "quote": quote,
        })
    return sources


def _share(record: dict) -> float | None:
    financial = record.get("financial") or {}
    if financial.get("status") not in _KEPT_SHARE:
        return None
    value = financial.get("revenue_share_pct")
    if value is None or value == "":
        return None
    return float(value)


def _segments(records: list[dict], items: list[dict]) -> list[dict]:
    units = {}
    for record in records:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
        financial = record.get("financial") or {}
        if financial.get("status") == "calculated_from_printed_figures" and financial.get("unit"):
            units[candidate.get("name")] = financial["unit"]
    segments = []
    for item in items:
        if item["kind"] != "segment":
            continue
        segments.append({
            "name": item["name"],
            "revenue": None,
            "unit": units.get(item["name"]),
            "share_pct": item["revenue_share_pct"],
        })
    return segments


def describe_company(source_text: str) -> tuple[str, str] | None:
    """One sentence and one paragraph taken from business-overview text, not from item names."""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.])\s+", source_text or "") if sentence.strip()]
    business = [sentence if sentence.endswith(".") else sentence + "." for sentence in sentences if _BUSINESS.search(sentence)]
    if not business:
        return None
    one_liner = business[0]
    summary = " ".join(business[:4])
    return one_liner, summary


_LEGAL_NAME = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|[A-Z][A-Za-z0-9&.'’\-]*)){0,12}"
    r"\s+(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?))\b"
)


def _subsidiary_list_keys(ticker: str) -> set[str]:
    path = STAGING_DIR / "factsheet" / ticker / "list_subsidiaries.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key) for key in payload.get("keys") or []}


def _load_pages(ticker: str) -> dict[int, dict]:
    folder = PAGE_DIR / ticker
    pages = {}
    if not folder.is_dir():
        return pages
    for path in folder.glob("p*.json"):
        page = json.loads(path.read_text(encoding="utf-8"))
        pages[int(page["pdf_page"])] = page
    return pages


def _own_pages(item: dict, pages: dict[int, dict]) -> dict[int, dict]:
    """Pages cited by this item. Entity text is never taken from a different page."""
    chosen = {}
    for source in item.get("sources") or []:
        number = int(source.get("pdf_page") or 0)
        if number in pages:
            chosen[number] = pages[number]
    return chosen


def _legal_names(text: str) -> list[str]:
    return [match.group(1) for match in _LEGAL_NAME.finditer(text or "")]


def _name_in(text: str, name: str) -> bool:
    folded = (text or "").casefold().replace("&", " and ")
    target = (name or "").casefold().replace("&", " and ")
    return bool(target) and target in folded


def _repair_list_entities(items: list[dict], pages: dict[int, dict]) -> None:
    """A quote that lists several legal names keeps this item's own name as the entity."""
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        for source in item.get("sources") or []:
            found = _legal_names(source.get("quote") or "")
            if len(found) < 2 or not _name_in(source.get("quote") or "", name):
                continue
            item["entity"] = name
            own = _own_pages(item, pages)
            solo, page_no = _solo_quote(own, name, found, int(source.get("pdf_page") or 0))
            if solo:
                source["quote"] = solo
                if page_no:
                    source["pdf_page"] = page_no
        quotes = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
        entity = str(item.get("entity") or "")
        if entity.casefold() not in {"company", "the company", "parent"} and not _name_in(quotes, entity):
            if _name_in(quotes, name):
                item["entity"] = name
            else:
                _attach_printed_name(item, pages)


def _solo_quote(pages: dict[int, dict], name: str, others: list[str], prefer: int) -> tuple[str, int]:
    blocked = [
        other for other in others
        if not _name_in(other, name) and not _name_in(name, other)
    ]
    order = [prefer] + [number for number in pages if number != prefer]
    for number in order:
        page = pages.get(number)
        if not page:
            continue
        text = page_text(page) or ""
        for part in re.split(r"\n|(?<=[.!?])\s+", text):
            compact = " ".join(part.split())
            if not _name_in(compact, name) or any(_name_in(compact, other) for other in blocked):
                continue
            if len(compact) < len(name):
                continue
            verbatim = _verbatim_slice(text, name) if len(compact) > 240 else compact
            if verbatim and _name_in(verbatim, name) and not any(_name_in(verbatim, other) for other in blocked):
                return verbatim[:240], number
        sliced = _verbatim_slice(text, name)
        if sliced and not any(_name_in(sliced, other) for other in blocked):
            return sliced[:240], number
    return "", 0


def _attach_printed_name(item: dict, pages: dict[int, dict]) -> None:
    """Keep the entity only when that exact name is printed. Otherwise use a verbatim page quote or company."""
    name = str(item.get("name") or "").strip()
    entity = str(item.get("entity") or "").strip()
    own = _own_pages(item, pages)
    for label in (name, entity):
        quote, page_no = _solo_quote(own, label, _legal_names(label), int((item.get("sources") or [{}])[0].get("pdf_page") or 0))
        if not quote or not _name_in(quote, label):
            continue
        sources = item.get("sources") or []
        if sources:
            sources[0]["quote"] = quote
            sources[0]["pdf_page"] = page_no or sources[0].get("pdf_page")
        item["entity"] = label if _name_in(quote, label) else "company"
        return
    item["entity"] = "company"


def _verbatim_slice(text: str, name: str) -> str:
    compact = " ".join((text or "").split())
    folded = compact.casefold()
    candidates = [name]
    if re.search(r"\band\b", name, re.I):
        candidates.append(re.sub(r"\band\b", "&", name, flags=re.I))
    if "&" in name:
        candidates.append(name.replace("&", "and"))
    for candidate in candidates:
        index = folded.find(candidate.casefold())
        if index >= 0:
            return compact[index: index + len(candidate)]
    return ""


def _restamp_printed(items: list[dict], pages: dict[int, dict]) -> None:
    """Printed page comes from this source's own PDF page, never from a merged sibling."""
    for item in items:
        for source in item.get("sources") or []:
            page = pages.get(int(source.get("pdf_page") or 0))
            if not page:
                continue
            printed, _rule = folio_for_quote(page, source.get("quote") or "")
            if printed:
                source["printed_page"] = printed


def merge_spec_items(items: list[dict], list_keys: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Merge close names across kinds. A proper subset stays its own item."""
    from src.factsheet.names import should_merge

    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if should_merge(items[left].get("name") or "", items[right].get("name") or "", list_keys):
                parent[find(right)] = find(left)
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for index, item in enumerate(items):
        root = find(index)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(item)
    merged = []
    merge_groups = []
    for root in order:
        members = groups[root]
        if len(members) == 1:
            merged.append(members[0])
            continue
        merged.append(_merge_members(members))
        merge_groups.append([member.get("name") or "" for member in members])
    return merged, merge_groups


def _quote_has_percent(quote: str, share: float | None) -> bool:
    if share is None:
        return False
    return bool(re.search(rf"(?<!\d){re.escape(f'{share:g}')}\s*%", quote or ""))


def _merge_members(members: list[dict]) -> dict:
    base = dict(max(members, key=lambda member: (_KIND_RANK.get(member.get("kind"), 0), len(member.get("descriptor") or ""))))
    base["name"] = max((member["name"] for member in members), key=len)
    descriptors = [member.get("descriptor") or "" for member in members]
    informative = [text for text in descriptors if text.strip() and text.strip() != base["name"].strip()]
    base["descriptor"] = max(informative or descriptors, key=len)
    entities = [member.get("entity") or "" for member in members]
    named = [entity for entity in entities if entity.casefold() not in {"company", "the company", "parent"}]
    base["entity"] = named[0] if named else entities[0]
    shares = [member.get("revenue_share_pct") for member in members if member.get("revenue_share_pct") is not None]
    share = shares[0] if shares and all(value == shares[0] for value in shares) else None
    sources = []
    for member in members:
        for source in member.get("sources") or []:
            identity = (source.get("pdf_page"), source.get("quote"))
            if identity not in {(item.get("pdf_page"), item.get("quote")) for item in sources}:
                sources.append(source)
    sources.sort(key=lambda source: (_quote_has_percent(source.get("quote") or "", share), len(source.get("quote") or "")), reverse=True)
    base["sources"] = sources[:3]
    if share is not None and not any(_quote_has_percent(source.get("quote") or "", share) for source in base["sources"]):
        share = None
    base["revenue_share_pct"] = share
    return base


def _apply_actors(items: list[dict]) -> None:
    names = [item["name"] for item in items if item.get("kind") == "subsidiary" and item.get("name")]
    if not names:
        return
    for item in items:
        for source in item.get("sources") or []:
            actor = actor_entity(source.get("quote") or "", item.get("entity") or "", names)
            if actor.casefold() not in {"company", "the company", "parent"}:
                item["entity"] = actor
                break


def _overview(ticker: str, items: list[dict], source_text: str = "") -> tuple[str, str]:
    if not items:
        return (
            f"{ticker} FY25 has no validated items.",
            "Items appear here only after a quote from the cited PDF page has been verified. "
            "This company has no validated candidates in the dataset.",
        )
    described = describe_company(source_text)
    if described:
        return described
    return fallback_text(ticker)


_BUSINESS = re.compile(
    r"\b(produces|producer|manufactur\w*|operat\w*|aluminium|aluminum|copper|segment|subsidiary|plant|mine)\b",
    re.I,
)


def _citation(source: dict) -> str:
    if source["printed_page"]:
        return f"PDF page {source['pdf_page']}, printed page {source['printed_page']}"
    return f"PDF page {source['pdf_page']}"


def _section(title: str, items: list[dict]) -> list[str]:
    lines = [f"## {title}", ""]
    if not items:
        lines.extend(["None.", ""])
        return lines
    for item in items:
        lines.extend([_render_item(item), ""])
    return lines


def _render_item(item: dict) -> str:
    share = "null" if item["revenue_share_pct"] is None else json.dumps(item["revenue_share_pct"])
    lines = [
        f"### {item['name']}",
        f"- Kind: {item['kind']}",
        f"- Entity: {item['entity']}",
        f"- Status: {item['status']}",
        f"- Descriptor: {json.dumps(item['descriptor'], ensure_ascii=False)}",
        f"- Revenue share: {share}",
        "- Sources:",
    ]
    for source in item["sources"]:
        printed = "null" if source["printed_page"] is None else source["printed_page"]
        quote = json.dumps(source["quote"], ensure_ascii=False)
        lines.append(f"  - PDF page {source['pdf_page']}, printed page {printed}: {quote}")
    return "\n".join(lines)


def _item_key(item: dict) -> tuple:
    return (
        item["name"],
        item["kind"],
        item["descriptor"],
        item["entity"],
        item["status"],
        item["revenue_share_pct"],
        tuple((source["pdf_page"], source["printed_page"], source["quote"]) for source in item["sources"]),
    )


def _record_name(record: dict) -> str:
    candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
    return str(candidate.get("name") or "")
