"""Write one fact sheet per company from its validated candidates."""

from __future__ import annotations

import argparse
import json

from src.config.paths import CANDIDATE_DIR, STAGING_DIR
from src.factsheet.build import COMPANIES, build_fact_sheet, load_validated, write_fact_sheet


def generate(ticker: str, force: bool = False, candidate_dir=None, output_dir=None) -> dict:
    """Write to staging by default. Promote to outputs/submission only from final assembly."""
    candidate_dir = candidate_dir or CANDIDATE_DIR
    output_dir = output_dir or (STAGING_DIR / "factsheet")
    records = load_validated(candidate_dir / ticker / "validation" / "validated.json")
    built = build_fact_sheet(ticker, records)
    written = write_fact_sheet(ticker, built, output_dir, force=force)
    written["staging"] = True
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Write fact sheets from validated candidates.")
    parser.add_argument("ticker", nargs="?", help="One company. Omit to write all three.")
    parser.add_argument("--force", action="store_true", help="Replace an existing fact sheet.")
    args = parser.parse_args()
    tickers = (args.ticker.upper(),) if args.ticker else COMPANIES
    results = [generate(ticker, force=args.force) for ticker in tickers]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
