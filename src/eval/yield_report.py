"""Print new candidates per 50k input tokens from the yield log."""

from __future__ import annotations

import argparse
import json

from src.config.paths import STAGING_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show extraction yield per 50k input tokens.")
    parser.add_argument("--company", required=True)
    args = parser.parse_args(argv)
    path = STAGING_DIR / "pilot" / f"{args.company.upper()}_yield.jsonl"
    if not path.is_file():
        print(f"No yield log at {path}")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = 0
    print(f"{'tokens':>10} {'pages':>8} {'candidates':>12} {'new':>8}")
    for row in rows:
        candidates = int(row.get("candidates") or row.get("validated_estimate") or 0)
        new_items = candidates - previous
        print(
            f"{int(row.get('cumulative_tokens') or 0):>10} "
            f"{int(row.get('pages_done') or 0):>8} "
            f"{candidates:>12} "
            f"{new_items:>8}"
        )
        previous = candidates
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
