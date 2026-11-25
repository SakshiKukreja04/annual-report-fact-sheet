"""One-company runner: cached stages → postprocess → check → promote to submission/.

Default is --from-cache (no LLM). Pass --allow-llm to run extract/vision.

Examples:
  python -m src.run_company --company HINDALCO
  python -m src.run_company --company COALINDIA --from-cache
  python -m src.run_company --company POWERGRID --allow-llm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config.paths import PROJECT_ROOT, STAGING_DIR, SUBMISSION_DIR
from src.eval.check_factsheet import check_factsheet
from src.pipeline.companies import COMPANIES
from src.pipeline.runner import STAGE_FUNCTIONS, RunContext, run_company


def load_company_config() -> dict:
    path = PROJECT_ROOT / "config" / "companies.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal YAML subset: "KEY:" / "  postprocess: value"
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_simple_yaml(text: str) -> dict:
    cfg: dict = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            current = line.rstrip()[:-1].strip()
            cfg[current] = {}
            continue
        if current and ":" in line:
            key, value = line.split(":", 1)
            cfg[current][key.strip()] = value.strip()
    return cfg


def stages_from_cache(*, allow_llm: bool) -> dict:
    """Cached pipeline stages. LLM stages only when allow_llm is set."""
    names = [
        "discover",
        "inspect",
        "classify",
        "ocr",
        "tables",
        "pages",
        "dedup",
        "validate",
        "factsheet",
        "evaluate",
    ]
    if allow_llm:
        names = [
            "discover",
            "inspect",
            "classify",
            "ocr",
            "tables",
            "pages",
            "vision",
            "extract",
            "dedup",
            "validate",
            "factsheet",
            "evaluate",
        ]
    return {name: STAGE_FUNCTIONS[name] for name in names if name in STAGE_FUNCTIONS}


def run_postprocess(ticker: str, name: str | None) -> dict:
    """Call the configured company cleanup. Does not call Groq."""
    if not name or name in {"none", "null", "~"}:
        return {"status": "skipped", "postprocess": None}
    if name == "hindalco_cleanup_v3":
        from src.factsheet.hindalco_cleanup_v3 import run as cleanup_run

        return {"status": "ok", "postprocess": name, "report": cleanup_run()}
    if name == "coalindia_cleanup_final":
        from src.factsheet.coalindia_cleanup_final import run as cleanup_run

        return {"status": "ok", "postprocess": name, "report": cleanup_run()}
    if name == "powergrid_cleanup":
        from src.factsheet.powergrid_cleanup import run as cleanup_run

        return {"status": "ok", "postprocess": name, "report": cleanup_run(promote=False)}
    raise SystemExit(f"Unknown postprocess {name!r} for {ticker}")


def sheet_paths(ticker: str) -> tuple[Path, Path]:
    """Prefer staging factsheet; fall back to submission."""
    staged = STAGING_DIR / "factsheet" / ticker / f"{ticker}.json"
    staged_md = staged.with_suffix(".md")
    if staged.is_file() and staged_md.is_file():
        return staged, staged_md
    # POWERGRID v3 naming
    alt = STAGING_DIR / "factsheet" / ticker / f"{ticker}.v3.json"
    if alt.is_file() and alt.with_suffix(".md").is_file():
        return alt, Path(str(alt).replace(".v3.json", ".v3.md"))
    sub = SUBMISSION_DIR / ticker / f"{ticker}.json"
    return sub, sub.with_suffix(".md")


def promote_to_submission(ticker: str, json_path: Path, md_path: Path) -> dict:
    dest = SUBMISSION_DIR / ticker
    dest.mkdir(parents=True, exist_ok=True)
    out_json = dest / f"{ticker}.json"
    out_md = dest / f"{ticker}.md"
    shutil.copy2(json_path, out_json)
    shutil.copy2(md_path, out_md)
    return {"json": str(out_json), "markdown": str(out_md)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cached company run → postprocess → check → promote.")
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--from-cache", action="store_true", default=True, help="Default: reuse caches; no new LLM.")
    parser.add_argument("--allow-llm", action="store_true", help="Run vision/extract stages (uses Groq).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    ticker = args.company.upper()
    allow_llm = bool(args.allow_llm)

    # (a) existing stages from cache
    context = RunContext(
        ticker=ticker,
        force=args.force,
        dry_run=args.dry_run,
        resume=True,
        limit_pages=set(),
        redo_pages=set(),
        max_queue_tokens=0,
    )
    pipeline = run_company(context, stages_from_cache(allow_llm=allow_llm))
    print(json.dumps({"pipeline": {k: v.get("status") for k, v in (pipeline.get("stages") or {}).items()}}, indent=2))

    # (b) company postprocess from config/companies.yaml
    cfg = (load_company_config().get(ticker) or {})
    post_name = cfg.get("postprocess")
    if args.dry_run:
        post = {"status": "dry_run", "postprocess": post_name}
    else:
        post = run_postprocess(ticker, post_name)
    print(json.dumps({"postprocess": {k: post.get(k) for k in ("status", "postprocess")}}, indent=2))

    # Resolve sheet for check/promote
    json_path, md_path = sheet_paths(ticker)
    if ticker == "POWERGRID" and not (SUBMISSION_DIR / ticker / f"{ticker}.json").is_file():
        # Prefer v3 when present for check
        v3 = STAGING_DIR / "factsheet" / ticker / "POWERGRID.v3.json"
        if v3.is_file():
            json_path, md_path = v3, STAGING_DIR / "factsheet" / ticker / "POWERGRID.v3.md"

    # (c) check_factsheet on the sheet folder
    check_folder = json_path.parent
    # For v3 filenames, copy/check via temporary canonical names is avoided:
    # if file is POWERGRID.v3.json, point --path at a folder that has POWERGRID.json
    if json_path.name != f"{ticker}.json":
        # Ensure submission staging check uses canonical names after promote only.
        # For pre-promote check of v3, write ephemeral aliases under staging/final_audit/check_<TICKER>
        alias = STAGING_DIR / "final_audit" / f"check_{ticker}"
        alias.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, alias / f"{ticker}.json")
        shutil.copy2(md_path, alias / f"{ticker}.md")
        check_folder = alias
    results = check_factsheet(ticker, path=check_folder)
    failed = [row for row in results if not row.get("passed")]
    for row in results:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['check']}. {row['detail']}")
    print(json.dumps({"check_failed": len(failed), "checks": len(results)}, indent=2))

    # (d) promote to submission/ only if checks pass
    promoted = None
    if failed:
        print(json.dumps({"promoted": False, "reason": "checks_failed"}, indent=2))
        return 1
    if args.dry_run:
        print(json.dumps({"promoted": False, "reason": "dry_run"}, indent=2))
        return 0
    src_json = check_folder / f"{ticker}.json"
    src_md = check_folder / f"{ticker}.md"
    promoted = promote_to_submission(ticker, src_json, src_md)
    print(json.dumps({"promoted": True, **promoted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
