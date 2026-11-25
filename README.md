# Annual report → product fact sheet

Pipeline for three hard FY25 Nifty-50 reports (POWERGRID, HINDALCO, COALINDIA): PDF → page audit → classify → OCR → page JSON → optional vision OCR → select/rules → LLM extract → dedup → validate → company cleanup → `check_factsheet` → promote to `submission/`.

**Models (Groq free tier):** text extract `openai/gpt-oss-120b`; scanned/weak pages use vision `qwen/qwen3.8-27b` as OCR, then the same text model. Digital text prefers PyMuPDF; weak pages fall back Tesseract (then vision when budget allows). Chosen for cost/TPM on free tier while keeping quotes grounded to page text or PDF.

## One command per company

```bash
python -m src.main --company POWERGRID
python -m src.main --company HINDALCO
python -m src.main --company COALINDIA
```

Defaults reuse caches (no new API calls). Add `--allow-llm` for vision + fresh extract. `--dry-run` plans only.

**Setup:** Python 3.12, `pip install -r requirements.txt`, copy `.env.example` → `.env` (`GROQ_API_KEY` or `GROQ_API_KEYS`), Tesseract on PATH, place each PDF under `data/input/<TICKER>/`.

Deliverables live in `submission/<TICKER>/<TICKER>.json` and `.md`.

## Cost (tokens / pages billed)

From `eval/cost/<TICKER>.json`. Paid-equivalent uses published Groq-style rates in code (text $0.15/$0.60 per 1M in/out; vision $0.80/$4.00). **Actual spend this run: $0** (free tier).

| Company | Step | Billed calls | Pages | Input tok | Output tok | Est. USD |
|---|---|---:|---:|---:|---:|---:|
| POWERGRID | parsing (local/OCR prep) | 0 | 275 | 503,029 | 0 | ~0 |
| POWERGRID | vision OCR | 90 | 42 | 183,437 | 46,231 | ~0.33 |
| POWERGRID | extraction | 82 | 277 | 140,261 | 16,360 | ~0.03 |
| HINDALCO | parsing | 0 | 395 | 2,876,956 | 0 | ~0 |
| HINDALCO | vision | 0 | 0 | 0 | 0 | 0 |
| HINDALCO | extraction | 46 | 125 | 174,963 | 21,589 | ~0.04 |
| COALINDIA | parsing | 0 | 157 | 6,274,101 | 0 | ~0 |
| COALINDIA | vision | 0 | 0 | 0 | 0 | 0 |
| COALINDIA | extraction | 107 | 127 | 287,092 | 64,987 | ~0.08 |

## Accuracy (self-check)

| | HINDALCO | COALINDIA | POWERGRID |
|---|---|---|---|
| Items | **190** | **203** | **152** (target 190–300) |
| `check_factsheet` | all PASS | all PASS | FAIL only check 2 (shortfall 38); quotes **0** failures |
| Quote validity | grounded | grounded | grounded on page_text / selected OCR |

**Missing / tradeoffs**

- **POWERGRID:** fully scanned + spreads. Vision OCR stopped early on free-tier **429 TPM / request-too-large**. Sheet merges digital + completed Groq vision pages + Tesseract; no invented rows. Coverage ~80% of the 190 floor; kinds/descriptors thinner than COALINDIA/HINDALCO.
- **HINDALCO / COALINDIA:** in the 190–300 band; cleanup postprocess in `config/companies.yaml`.

## Scaling to 50 / 500

1. Cache OCR + extract; never re-bill unchanged PDFs.  
2. Rotate `GROQ_API_KEYS`; stop on daily cap and resume.  
3. Vision only on weak halves; rules/tables before LLM.  
4. Config-driven cleanup per ticker; promote only after checks pass.  
5. For 500: batch queue + paid tier (or self-hosted OCR) to avoid free-tier TPM walls.
