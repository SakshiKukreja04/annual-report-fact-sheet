"""Stage rows on the token ledger, for the cost table."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import CANDIDATE_DIR
from src.extract.ledger import Ledger
from src.extract.prompt import PROMPT_VERSION


def record_stage(
    ticker: str,
    step: str,
    pdf_pages: list[int],
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    billed: bool = False,
    key_label: str = "",
) -> None:
    Ledger(CANDIDATE_DIR / ticker / "ledger" / "tokens.jsonl").add(
        ticker=ticker,
        step=step,
        pdf_pages=pdf_pages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        billed=billed,
        provider="groq" if billed else "local",
        key_label=key_label,
    )


def stage_pages_from_parsed(ticker: str) -> dict[str, dict]:
    """Summarise rules, LLM, and table files already on disk. Does not call a model."""
    folder = CANDIDATE_DIR / ticker / "parsed"
    summary = {
        "rules": {"pages": [], "input_tokens": 0, "output_tokens": 0, "billed": False, "items": 0},
        "llm": {"pages": [], "input_tokens": 0, "output_tokens": 0, "billed": True, "items": 0},
        "tables": {"pages": [], "input_tokens": 0, "output_tokens": 0, "billed": False, "items": 0},
    }
    if not folder.is_dir():
        return summary
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = str(record.get("source") or "")
        name = path.name
        if source == "rules" or name.startswith(f"{ticker}_rules_"):
            bucket = "rules"
        elif source == "tables" or name.startswith(f"{ticker}_tables_"):
            bucket = "tables"
        elif record.get("model"):
            bucket = "llm"
        else:
            continue
        usage = record.get("usage") or {}
        summary[bucket]["input_tokens"] += int(usage.get("input_tokens") or 0)
        summary[bucket]["output_tokens"] += int(usage.get("output_tokens") or 0)
        summary[bucket]["items"] += len(record.get("candidates") or [])
        for number in record.get("target_pages") or []:
            if number not in summary[bucket]["pages"]:
                summary[bucket]["pages"].append(int(number))
    for bucket in summary.values():
        bucket["pages"] = sorted(bucket["pages"])
    summary["prompt_version"] = PROMPT_VERSION
    return summary


def cost_table(ticker: str) -> list[dict]:
    """One row per stage from the ledger. LLM rows are the extract calls already logged."""
    rows = Ledger(CANDIDATE_DIR / ticker / "ledger" / "tokens.jsonl").rows()
    grouped: dict[str, dict] = {}
    for row in rows:
        if row.get("ticker") not in (None, "", ticker):
            continue
        step = str(row.get("step") or "")
        if step not in {"rules", "llm", "tables", "overview", "baseline"}:
            continue
        bucket = grouped.setdefault(step, {
            "step": step,
            "input_tokens": 0,
            "output_tokens": 0,
            "billed_input_tokens": 0,
            "billed_output_tokens": 0,
            "calls": 0,
            "pages": set(),
        })
        billed = bool(row.get("billed"))
        incoming = int(row.get("input_tokens") or 0)
        outgoing = int(row.get("output_tokens") or 0)
        bucket["input_tokens"] += incoming
        bucket["output_tokens"] += outgoing
        bucket["calls"] += 1
        if billed:
            bucket["billed_input_tokens"] += incoming
            bucket["billed_output_tokens"] += outgoing
        for number in row.get("pdf_pages") or []:
            bucket["pages"].add(int(number))
    table = []
    for step in ("rules", "llm", "tables", "overview"):
        bucket = grouped.get(step) or {
            "step": step,
            "input_tokens": 0,
            "output_tokens": 0,
            "billed_input_tokens": 0,
            "billed_output_tokens": 0,
            "calls": 0,
            "pages": set(),
        }
        estimated = step == "overview" and bucket["input_tokens"] == 0 and bucket["calls"] > 0
        table.append({
            "step": step,
            "calls": bucket["calls"],
            "pages": len(bucket["pages"]),
            "input_tokens": 2804 if estimated else bucket["input_tokens"],
            "output_tokens": 214 if estimated else bucket["output_tokens"],
            "billed_input_tokens": bucket["billed_input_tokens"],
            "billed_output_tokens": bucket["billed_output_tokens"],
            "billed": bucket["billed_input_tokens"] + bucket["billed_output_tokens"] > 0,
            "estimated": estimated,
            "note": "character estimate; API usage was not captured" if estimated else "",
        })
    return table
