"""Check quotes against original page text. Cleaned prompt text is not used."""

from __future__ import annotations

import argparse
import json

from src.extract.select import page_text
from src.validate.quotes import locate_quote


def verify_candidates(candidates: list[dict], pages: list[dict]) -> dict:
    by_page = {int(page["pdf_page"]): page_text(page) for page in pages}
    checked = []
    for candidate in candidates:
        for source in candidate.get("sources") or []:
            page = source.get("pdf_page")
            quote = source.get("quote") or ""
            original = by_page.get(page, "")
            match = locate_quote(quote, original)
            checked.append(
                {
                    "name": candidate.get("name"),
                    "pdf_page": page,
                    "quote": quote,
                    "match": match,
                    "accepted": match is not None,
                }
            )
    accepted = sum(1 for row in checked if row["accepted"])
    return {
        "checked": len(checked),
        "accepted": accepted,
        "rejected": len(checked) - accepted,
        "rows": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify quotes against original page text.")
    parser.add_argument("candidates")
    parser.add_argument("pages")
    args = parser.parse_args()
    candidates = json.loads(open(args.candidates, encoding="utf-8").read())
    pages = json.loads(open(args.pages, encoding="utf-8").read())
    report = verify_candidates(candidates, pages)
    print(json.dumps({key: report[key] for key in ("checked", "accepted", "rejected")}, indent=2))


if __name__ == "__main__":
    main()
