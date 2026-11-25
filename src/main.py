"""Run the full company pipeline.

Examples:
    python -m src.main --company POWERGRID
    python -m src.main --company HINDALCO
    python -m src.main --company COALINDIA
    python -m src.main --company POWERGRID --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from src.config.paths import LOG_DIR, OUTPUT_DIR, PROJECT_ROOT, STAGING_DIR, SUBMISSION_DIR
from src.pipeline.companies import COMPANIES
from src.pipeline.runner import STAGE_FUNCTIONS, RunContext, run_company
from src.utils.logging_config import configure_logging

# Default path: discover → … → pages → extract (select+rules inside) → dedup → validate → factsheet.
# Vision only when --allow-llm. Cleanup + check + promote happen after stages.
_DEFAULT_STAGES = (
    "discover",
    "inspect",
    "classify",
    "ocr",
    "tables",
    "pages",
    "extract",
    "dedup",
    "validate",
    "factsheet",
    "evaluate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one annual report from the PDF to the fact sheet.")
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--force", action="store_true", help="Rebuild cached stages and replace an unchanged fact sheet.")
    parser.add_argument("--stage", default="", help="Run only this stage.")
    parser.add_argument("--limit-pages", default="", help="Comma-separated PDF pages or ranges, such as 13,60-62.")
    parser.add_argument(
        "--redo-pages",
        default="",
        help="Comma-separated PDF pages to re-extract even if state marks them complete.",
    )
    parser.add_argument(
        "--max-queue-tokens",
        type=int,
        default=0,
        help="Stop the ranked extract queue once cumulative estimated input tokens reach N. 0 means no cut.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the work without writing or calling an API.")
    parser.add_argument("--resume", action="store_true", help="Continue after the last completed stage. This is the default.")
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Include vision + allow extract to call Groq. Default reuses cached extractions only.",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Run dedup, validate, and factsheet on cached candidates, then the factsheet check. Does not promote.",
    )
    parser.add_argument(
        "--to-staging",
        action="store_true",
        help="Keep the fact sheet in staging/factsheet. Skip promote even if checks pass.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip company postprocess from config/companies.yaml.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=("tesseract", "paddle", "mistral"),
        default="tesseract",
        help="OCR engine for the ocr stage. Default keeps Tesseract/Paddle path unchanged.",
    )
    parser.add_argument(
        "--ocr-pages",
        default="",
        help="Limit OCR to these 1-based PDF pages/ranges (e.g. 361-362,243). Applies to Mistral especially.",
    )
    parser.add_argument(
        "--mistral-halves",
        action="store_true",
        help="When --ocr-engine mistral, OCR LEFT/RIGHT crops instead of the full page.",
    )
    return parser


def _load_company_config() -> dict:
    path = PROJECT_ROOT / "config" / "companies.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg: dict = {}
        current = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not line.startswith(" ") and line.rstrip().endswith(":"):
                current = line.rstrip()[:-1].strip()
                cfg[current] = {}
            elif current and ":" in line:
                key, value = line.split(":", 1)
                cfg[current][key.strip()] = value.strip()
        return cfg


def _run_cleanup(ticker: str, *, dry_run: bool) -> dict:
    cfg = (_load_company_config().get(ticker) or {})
    name = cfg.get("postprocess")
    if not name or name in {"none", "null", "~"}:
        return {"status": "skipped", "postprocess": None}
    if dry_run:
        return {"status": "dry_run", "postprocess": name}
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


def _promote(ticker: str) -> dict:
    src = STAGING_DIR / "factsheet" / ticker
    json_path = src / f"{ticker}.json"
    md_path = src / f"{ticker}.md"
    if not json_path.is_file() or not md_path.is_file():
        return {"promoted": False, "reason": "staging_missing"}
    for dest_root in (OUTPUT_DIR, SUBMISSION_DIR):
        dest = dest_root / ticker
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dest / f"{ticker}.json")
        shutil.copy2(md_path, dest / f"{ticker}.md")
    return {
        "promoted": True,
        "outputs": str(OUTPUT_DIR / ticker / f"{ticker}.json"),
        "submission": str(SUBMISSION_DIR / ticker / f"{ticker}.json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(LOG_DIR / "pipeline.log")
    logging.getLogger("factsheet").info(
        "pipeline %s dry_run=%s allow_llm=%s", args.company, args.dry_run, args.allow_llm
    )
    ocr_pages = _parse_pages(args.ocr_pages) if args.ocr_pages else set()
    context = RunContext(
        ticker=args.company,
        force=args.force,
        dry_run=args.dry_run,
        resume=True,
        limit_pages=_parse_pages(args.limit_pages),
        redo_pages=_parse_pages(args.redo_pages),
        max_queue_tokens=int(args.max_queue_tokens or 0),
        ocr_engine=args.ocr_engine,
        ocr_pages=ocr_pages,
        mistral_halves=bool(args.mistral_halves),
    )

    if args.all_stages:
        stages = {name: STAGE_FUNCTIONS[name] for name in ("dedup", "validate", "factsheet")}
        context.resume = False
    elif args.stage:
        if args.stage not in STAGE_FUNCTIONS:
            raise SystemExit(f"Unknown stage {args.stage}. Expected one of {', '.join(STAGE_FUNCTIONS)}.")
        stages = {args.stage: STAGE_FUNCTIONS[args.stage]}
        context.resume = False
    else:
        names = list(_DEFAULT_STAGES)
        if args.allow_llm:
            # Insert vision before extract.
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
        stages = {name: STAGE_FUNCTIONS[name] for name in names if name in STAGE_FUNCTIONS}

    if args.resume:
        context.resume = True

    report = run_company(context, stages)
    print(json.dumps({"pipeline": report}, indent=2))

    # Cleanup (offline) — writes staging only.
    cleanup = {"status": "skipped"}
    if not args.skip_cleanup and not report.get("failed_stage"):
        cleanup = _run_cleanup(args.company.upper(), dry_run=args.dry_run)
        print(json.dumps({"cleanup": {k: cleanup.get(k) for k in ("status", "postprocess")}}, indent=2))

    # Validate / check against staging (or submission path override later).
    from src.eval.check_factsheet import check_factsheet

    ticker = args.company.upper()
    check_path = STAGING_DIR / "factsheet" / ticker
    if not (check_path / f"{ticker}.json").is_file():
        check_path = SUBMISSION_DIR / ticker
    if args.dry_run:
        print(json.dumps({"check": "dry_run", "promoted": False}, indent=2))
        return 0 if not report.get("failed_stage") else 1

    results = check_factsheet(ticker, path=check_path)
    failed = [row for row in results if not row.get("passed")]
    for row in results:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['check']}. {row['detail']}")
    print(json.dumps({"check_failed": len(failed), "checks": len(results)}, indent=2))

    if failed or args.to_staging:
        print(json.dumps({"promoted": False, "reason": "checks_failed" if failed else "to_staging"}, indent=2))
        return 1 if failed or report.get("failed_stage") else 0

    promoted = _promote(ticker)
    print(json.dumps(promoted, indent=2))
    return 0 if not report.get("failed_stage") else 1


def _parse_pages(text: str) -> set[int]:
    wanted = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(piece))
    return wanted


if __name__ == "__main__":
    raise SystemExit(main())
