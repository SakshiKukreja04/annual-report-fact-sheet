# Annual report → product fact sheet

End-to-end pipeline that turns a company annual report PDF into a source-backed product fact sheet: products, services, brands, business lines, segments, and subsidiaries with a stated activity.

Built and run on three hard FY25 reports: POWERGRID, HINDALCO, and COALINDIA.

## Pipeline

PDF → page audit → classify → OCR → unified page JSON → optional vision OCR on weak pages → page selection / rules → LLM extract → dedup → validate → company cleanup → automated checks → `submission/<TICKER>/`.

| Stage | Choice | Why |
|---|---|---|
| Digital text | PyMuPDF | Fast when a text layer exists |
| OCR fallback | Tesseract | Local, no API cost on scanned / weak pages |
| Vision OCR | Groq `qwen/qwen3.8-27b` | Used when Tesseract text is too weak to extract from (mainly POWERGRID spreads) |
| Extraction | Groq `openai/gpt-oss-120b` | Candidate extraction with quotes constrained to page text |
| Postprocess | Per-ticker cleanup in `config/companies.yaml` | Merge near-duplicates, drop unsupported shares, company-specific polish |

Assumptions used throughout:

- An item is kept only if at least one quote can be located on the cited page (page text and/or PDF layer).
- `revenue_share_pct` is set only when the report prints the figure, or when it can be computed from printed figures; otherwise `null`.
- Descriptors and activities come only from the document, not from outside knowledge.
- Names that differ only by &/and, Ltd/Limited, plurals, or word order are treated as one item.
- For POWERGRID, vision OCR was applied to weak halves that completed successfully within free-tier rate limits; remaining weak pages stay on Tesseract / digital text already selected for those pages.

## Run (one command per company)

```bash
python -m src.main --company POWERGRID
python -m src.main --company HINDALCO
python -m src.main --company COALINDIA
```

Default runs reuse local caches (no new API calls). Use `--allow-llm` for vision and fresh extract calls. Use `--dry-run` to plan without writes.

Setup: Python 3.12, `pip install -r requirements.txt`, copy `.env.example` to `.env` and set `GROQ_API_KEY` or `GROQ_API_KEYS`, install Tesseract on PATH, place each PDF under `data/input/<TICKER>/`.

Outputs for evaluation:

- `submission/POWERGRID/POWERGRID.json` + `POWERGRID.md`
- `submission/HINDALCO/HINDALCO.json` + `HINDALCO.md`
- `submission/COALINDIA/COALINDIA.json` + `COALINDIA.md`

## Cost

Figures from `eval/cost/<TICKER>.json`. Paid-equivalent USD uses the rates coded in the pipeline (text $0.15 / $0.60 per 1M in/out; vision $0.80 / $4.00 per 1M). This submission was run on Groq free tier, so billed spend was $0; the USD column is the equivalent if the same tokens were charged at those rates.

| Company | Step | Billed calls | Pages | Input tok | Output tok | Est. USD |
|---|---|---:|---:|---:|---:|---:|
| POWERGRID | parsing (local / OCR prep) | 0 | 275 | 503,029 | 0 | ~0 |
| POWERGRID | vision OCR | 90 | 42 | 183,437 | 46,231 | ~0.33 |
| POWERGRID | extraction | 82 | 277 | 140,261 | 16,360 | ~0.03 |
| HINDALCO | parsing | 0 | 395 | 2,876,956 | 0 | ~0 |
| HINDALCO | vision | 0 | 0 | 0 | 0 | 0 |
| HINDALCO | extraction | 46 | 125 | 174,963 | 21,589 | ~0.04 |
| COALINDIA | parsing | 0 | 157 | 6,274,101 | 0 | ~0 |
| COALINDIA | vision | 0 | 0 | 0 | 0 | 0 |
| COALINDIA | extraction | 107 | 127 | 287,092 | 64,987 | ~0.08 |

## Outcomes

| | HINDALCO | COALINDIA | POWERGRID |
|---|---|---|---|
| Items in sheet | 190 | 203 | 152 |
| Assignment band (190–300) | In band | In band | Below band |
| Automated sheet checks | Pass | Pass | Pass except item-count floor |
| Quote checks | Pass | Pass | Pass |

How accuracy was checked:

- Schema and consistency against the assignment fact-sheet shape.
- Verbatim quote lookup on the cited page (page text / selected OCR; PDF text layer when present).
- Duplicate, share, entity, and descriptor shape checks in `src.eval.check_factsheet`.
- No padding of items to hit the count floor.

What the sheets emphasize:

- HINDALCO and COALINDIA: mixed kinds (products, assets, subsidiaries, business lines, etc.) with grounded quotes.
- POWERGRID: fully scanned two-page spreads; sheet combines digital text, completed vision-OCR pages, and Tesseract fallbacks. Coverage is incomplete relative to the 190-item floor because free-tier vision throughput limited how many weak halves could be finished in time. Items that could not be quote-verified were not added.

## Scaling to 50 or 500 companies

1. Keep OCR and extract caches keyed by PDF content; do not re-bill unchanged reports.
2. Rotate API keys and resume cleanly after daily or TPM caps.
3. Prefer rules and digital tables before LLM; use vision only on weak pages.
4. Keep cleanup and promote gates config-driven per ticker.
5. For hundreds of companies, use a paid or self-hosted OCR path so scanned books are not gated by free-tier vision limits.
