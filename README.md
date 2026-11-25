# Annual Report to Product Fact Sheet

**An end-to-end document intelligence pipeline that transforms corporate annual reports into structured, source-backed product fact sheets.**

The system processes company annual report PDFs and extracts information about products, services, brands, business lines, operating segments, subsidiaries, joint ventures and other relevant business entities.

It combines document parsing, OCR, selective vision-based text recovery, LLM-assisted extraction, deterministic post-processing and automated validation to produce structured JSON and human-readable Markdown fact sheets.

The pipeline was implemented and evaluated on three FY25 annual reports:

- POWERGRID
- HINDALCO
- COALINDIA

---

## 1. Problem Statement

Corporate annual reports contain large volumes of information distributed across hundreds of pages, including digitally generated text, scanned documents, tables, diagrams and multi-column layouts.

Extracting a reliable product and business fact sheet from these documents presents several challenges:

- **Mixed document formats:** Reports contain both digitally accessible text and scanned pages.
- **Complex layouts:** Tables, two-page spreads, columns and graphical representations can make text extraction unreliable.
- **Information fragmentation:** Product and business information is distributed across multiple sections.
- **Entity duplication:** The same entity may appear under slightly different names throughout a report.
- **Source traceability:** Extracted facts must be supported by the original document rather than generated from model knowledge.
- **Quantitative accuracy:** Revenue shares and other numerical attributes must not be inferred without supporting figures.

The objective is to automate this process while maintaining document-level traceability and conservative extraction standards.

## 2. Solution Overview

The pipeline converts an annual report PDF into a structured company fact sheet through a sequence of processing and validation stages.

**Input:** Company annual report PDF.

**Output:** A source-backed fact sheet containing extracted entities, descriptions, activities, classifications and supporting quotations.

### High-level architecture

```text
             Annual Report PDF
                     |
                     v
             PDF Discovery & Audit
                     |
                     v
               Page Classification
                     |
                     v
           Digital Text Extraction
                     |
              +------+------+
              |             |
              v             v
          Good Text     Weak / Scanned
              |             |
              |             v
              |        Tesseract OCR
              |             |
              |             v
              |      Quality Assessment
              |             |
              |             v
              |       Selective Vision OCR
              |             |
              +------+------+
                     |
                     v
              Unified Page JSON
                     |
                     v
             Page Selection & Rules
                     |
                     v
              LLM Fact Extraction
                     |
                     v
              Entity Deduplication
                     |
                     v
              Schema Validation
                     |
                     v
             Company-Specific Cleanup
                     |
                     v
              Automated Evaluation
                     |
                     v
             Final Fact Sheet Output
                  /       \
                 v         v
              JSON       Markdown
```

The pipeline is designed to avoid unnecessary API calls by reusing cached results and prioritizing local document processing wherever possible.

## 3. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| PDF processing | PyMuPDF | PDF inspection, rendering and digital text extraction |
| Image preprocessing | OpenCV | Spread split, grayscale, denoising, contrast and deskew before OCR |
| Local OCR | Tesseract | Text recovery from scanned or weak pages |
| Vision OCR | Groq – Qwen3.8 27B | Recovery of difficult text from selected page images |
| LLM extraction | Groq – GPT-OSS 120B | Structured fact and entity extraction |
| Configuration | YAML | Company-specific extraction and cleanup rules |
| Data representation | JSON | Intermediate page data and final structured outputs |
| Validation | Python | Schema, quotation, entity and consistency checks |

### Model selection

**Qwen3.8 27B – Vision OCR**

Used selectively when local OCR produces insufficient text, particularly for difficult scanned layouts and two-page spreads.

**GPT-OSS 120B – Fact Extraction**

Used to identify relevant entities, descriptions, activities and supporting quotations from the selected document content.

The extraction process constrains supporting quotations to the available page text to reduce unsupported generation.

## 4. Pipeline Implementation

### Stage 1: PDF Discovery and Inspection

The pipeline locates the input annual report and performs an initial document audit.

It identifies document characteristics such as:

- Total page count.
- Available digital text.
- Pages requiring OCR.
- Potential scanned pages.
- Document layout characteristics.

This provides the foundation for selecting appropriate processing methods.

### Stage 2: Page Classification

Each page is classified according to its content and layout characteristics.

The classification helps distinguish between:

- Digitally readable pages.
- Scanned or image-based pages.
- Pages with weak or incomplete text extraction.
- Complex layouts requiring additional processing.

Page classification enables selective processing rather than applying expensive OCR operations indiscriminately.

### Stage 3: Digital Text Extraction

PyMuPDF is used as the primary text extraction mechanism.

When a PDF contains a usable text layer, its content is extracted directly without requiring OCR.

This approach is computationally efficient and preserves the original text more reliably than OCR-based reconstruction.

### Stage 4: Image Preprocessing, OCR and Text Recovery

Before OCR, scanned and image-heavy pages (especially POWERGRID’s fully scanned two-page spreads) pass through an OpenCV preprocessing path in `src/preprocessing/pipeline.py`.

That path:

- Detects wide two-page spreads and splits them on the page gutter into LEFT / RIGHT halves.
- Converts each half to grayscale.
- Scales the image toward a target OCR DPI (~300).
- Measures speckle noise and, when the page is noisy enough, applies fast non-local-means denoising plus a light median filter.
- Applies CLAHE contrast enhancement and unsharp masking.
- Estimates skew and deskews the half when the measured angle is meaningful.

Binarization is intentionally deferred so Tesseract can choose its own threshold on the cleaned grayscale image. Preprocessed halves are cached under `data/cache/preprocessing/` and reused by OCR and vision stages.

Tesseract then provides the local OCR fallback for pages where digital text extraction is unavailable or insufficient.

For difficult pages, selective vision-based OCR is used to recover text that local OCR cannot reliably interpret.

The vision OCR path is particularly useful for:

- Dense scanned pages.
- Complex page layouts.
- Multi-column content.
- Difficult text regions.
- Two-page spreads.

Vision OCR is not applied indiscriminately. It is reserved for pages where additional text recovery can improve extraction quality.

### Stage 5: Unified Page Representation

Text recovered through different processing paths is consolidated into a unified page-level JSON representation.

This provides a consistent input format for downstream processing, regardless of whether the original content came from digital extraction, Tesseract or vision OCR.

Maintaining page-level information also supports quotation verification and source traceability.

### Stage 6: Page Selection and Rule-Based Filtering

Not every page in an annual report is equally relevant to the fact sheet.

The pipeline applies page selection and configured rules to prioritize content likely to contain:

- Products and services.
- Brands.
- Business lines.
- Operating segments.
- Subsidiaries and joint ventures.
- Business activities.
- Assets and operations.

This reduces unnecessary LLM processing and focuses extraction on relevant document content.

### Stage 7: LLM-Based Fact Extraction

Selected page content is passed to GPT-OSS 120B for structured candidate extraction.

The extraction focuses on identifying:

- Entity names.
- Entity classifications.
- Product and service descriptions.
- Business activities.
- Supporting quotations.
- Page references.
- Revenue share, where supported by reported figures.

The model is used for candidate identification and structuring, not as an independent source of company information.

Extracted facts must be grounded in the supplied annual report content.

### Stage 8: Entity Deduplication

Annual reports may refer to the same entity using different naming conventions.

The post-processing layer identifies common naming variations, including:

- Ampersand versus the word "and".
- Ltd versus Limited.
- Singular versus plural forms.
- Minor word-order differences.
- Near-duplicate entity names.

This reduces duplicate entries and improves consistency across the final fact sheet.

### Stage 9: Validation and Company-Specific Cleanup

Extracted candidates pass through validation and configurable cleanup rules.

The validation process checks:

- Required output fields.
- Entity consistency.
- Supporting quotation availability.
- Page-reference validity.
- Duplicate entities.
- Revenue-share consistency.
- Unsupported or irrelevant entries.
- Description and activity formatting.

Company-specific cleanup rules are maintained in:

`config/companies.yaml`

This allows company-specific corrections and normalization without changing the shared extraction pipeline.

### Stage 10: Evaluation and Output Generation

The final fact sheets are checked using automated evaluation utilities.

The pipeline generates both machine-readable and human-readable outputs:

- JSON for structured consumption and downstream processing.
- Markdown for reviewing extracted information.

Final outputs are organized by company ticker under the `submission/` directory.

## 5. Data Grounding and Accuracy Principles

Source traceability is a central design requirement.

The following rules are applied throughout extraction and validation:

**1. Document-grounded entities**

An entity is retained only when it can be supported by the annual report.

**2. Verifiable quotations**

Every retained item must have at least one supporting quotation that can be located on its cited page using the available page text and/or PDF text layer.

**3. Conservative revenue attribution**

`revenue_share_pct` is populated only when the annual report explicitly provides the figure or when it can be calculated from reported figures.

Otherwise, the value remains `null`.

**4. No external knowledge injection**

Descriptions and activities are derived from the annual report rather than external company information or model assumptions.

**5. Duplicate control**

Naming variations are normalized to prevent the same entity from being unnecessarily represented multiple times.

**6. No artificial count padding**

The pipeline does not add unsupported entities simply to satisfy a minimum item-count requirement.

These principles prioritize evidence-backed extraction over maximizing the number of generated entries.

## 6. Evaluation Results

The pipeline was executed on three FY25 annual reports with different document characteristics.

### Final fact sheet results

| Metric | HINDALCO | COALINDIA | POWERGRID |
|---|---:|---:|---:|
| Items extracted | 190 | 203 | 152 |
| Assignment target (190–300) | Achieved | Achieved | Below target |
| Automated sheet checks | Pass | Pass | Pass, except count floor |
| Supporting quotation checks | Pass | Pass | Pass |

The outputs demonstrate that the shared pipeline can process reports with different document structures and produce structured, source-backed fact sheets.

### Report-specific observations

**HINDALCO**

- 395-page annual report.
- Primarily processed using digital text extraction and local processing.
- 46 LLM extraction calls.
- 190 final fact sheet items.
- All automated sheet and quotation checks passed.

**COALINDIA**

- 157-page annual report.
- 107 LLM extraction calls.
- 203 final fact sheet items.
- All automated sheet and quotation checks passed.

**POWERGRID**

- 275-page annual report.
- Contains extensive scanned content and two-page spreads.
- Required additional vision-based OCR alongside local text extraction.
- 90 vision OCR calls and 82 extraction calls.
- 152 final fact sheet items.
- All quotation checks passed; the item-count minimum was not achieved.

POWERGRID's output is therefore a partially covered fact sheet rather than a complete representation of all relevant entities in the report.

## 7. API Usage and Cost Analysis

The pipeline uses local processing wherever possible and selectively invokes Groq models for vision OCR and fact extraction.

The following figures are derived from the generated evaluation cost reports.

### Usage summary

| Company | Processing pages | Vision calls | Extraction calls |
|---|---:|---:|---:|
| POWERGRID | 275 | 90 | 82 |
| HINDALCO | 395 | 0 | 46 |
| COALINDIA | 157 | 0 | 107 |

### Token usage and estimated cost

| Company | Processing tokens | Vision tokens (in/out) | Extraction tokens (in/out) |
|---|---:|---:|---:|
| POWERGRID | 503,029 | 183,437 / 46,231 | 140,261 / 16,360 |
| HINDALCO | 2,876,956 | 0 / 0 | 174,963 / 21,589 |
| COALINDIA | 6,274,101 | 0 / 0 | 287,092 / 64,987 |

### Estimated API cost

The submission was executed using the Groq free tier.

Consequently, the billed API spend was **$0**.

For cost comparison, the pipeline calculates paid-equivalent estimates using the following configured rates:

| Model usage | Input per 1M tokens | Output per 1M tokens |
|---|---:|---:|
| Text extraction | $0.15 | $0.60 |
| Vision OCR | $0.80 | $4.00 |

| Company | Estimated paid-equivalent cost |
|---|---:|
| POWERGRID | $0.36 |
| HINDALCO | $0.04 |
| COALINDIA | $0.08 |
| **Total** | **$0.48** |

These are estimates based on the configured pricing assumptions, not actual billed expenditure. Local parsing and OCR preparation are not included as paid API expenditure.

## 8. Reproducibility and Execution

The pipeline supports execution through a single company-level command.

### Prerequisites

- Python 3.12
- Tesseract OCR installed and available on PATH.
- Groq API key for vision OCR and fresh extraction.
- Input annual report PDF.

### Installation

Clone the repository and navigate to the project directory.

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
copy .env.example .env
```

Configure the required Groq credentials:

```env
GROQ_API_KEY=your_groq_api_key
```

Alternatively, configure multiple API keys through `GROQ_API_KEYS` if supported by the project configuration.

### Input directory structure

Place annual report PDFs in the corresponding company directory:

```text
data/
└── input/
    ├── POWERGRID/
    ├── HINDALCO/
    └── COALINDIA/
```

### Run the pipeline

Execute the following commands independently:

```bash
python -m src.main --company POWERGRID

python -m src.main --company HINDALCO

python -m src.main --company COALINDIA
```

### Execution options

| Option | Behaviour |
|---|---|
| Default execution | Reuses available local caches and avoids unnecessary new API calls |
| `--allow-llm` | Enables vision OCR and fresh LLM extraction calls |
| `--dry-run` | Plans pipeline execution without writing outputs |

Cached execution is useful for reproducibility and incremental processing.

## 9. Output Structure

The final deliverables are organized by company ticker.

```text
submission/
├── POWERGRID/
│   ├── POWERGRID.json
│   └── POWERGRID.md
│
├── HINDALCO/
│   ├── HINDALCO.json
│   └── HINDALCO.md
│
└── COALINDIA/
    ├── COALINDIA.json
    └── COALINDIA.md
```

**JSON output**

Provides structured fact sheet data suitable for programmatic evaluation, downstream applications and further processing.

**Markdown output**

Provides a human-readable representation for reviewing extracted entities, descriptions, classifications and source quotations.

## 10. Scalability and Future Improvements

The pipeline follows a modular architecture that can be extended to additional companies without redesigning the extraction workflow.

Potential improvements for processing 50–500 annual reports include:

**1. Content-addressed caching**

Cache OCR and extraction results using PDF content hashes so unchanged reports are not processed or billed again.

**2. Resumable processing**

Persist intermediate results and resume processing from the last completed stage when API quotas or execution limits are reached.

**3. Improved OCR infrastructure**

Introduce higher-throughput OCR infrastructure for large collections of scanned annual reports.

**4. Selective vision processing**

Continue prioritizing digital extraction and local OCR, reserving vision models for pages where additional recovery is necessary.

**5. Configuration-driven processing**

Extend company-specific cleanup, filtering and validation through centralized configuration rather than introducing company-specific pipeline implementations.

**6. Expanded evaluation**

Introduce more comprehensive coverage auditing, entity-level quality assessment and cross-company benchmarking.

## 11. Current Scope and Limitations

- Fact sheet completeness depends on the quality of the source document and the effectiveness of text recovery.
- Vision OCR throughput can constrain processing of reports containing extensive scanned content when operating under free-tier API limits.
- POWERGRID currently has 152 verified items and does not meet the 190-item minimum.
- Quotation validation establishes that supporting text exists on the cited page; it does not independently establish that every extracted entity is semantically complete or that the entire report has been covered.
- Revenue shares are intentionally left null when supporting quantitative information is unavailable.

The system prioritizes traceability and evidence-backed extraction over unsupported completion of the target item count.

---

## 12. Conclusion

This project implements a reusable annual-report intelligence pipeline that combines local document processing, OCR, selective vision-based recovery, LLM-assisted extraction and deterministic validation.

Its central design principle is that extracted facts must remain traceable to the source document.

The three FY25 company reports demonstrate the pipeline's ability to handle different report structures, reduce unnecessary model usage through selective processing and generate structured fact sheets with verifiable supporting quotations.

The resulting architecture provides a foundation for scaling annual-report processing across larger collections of companies while retaining reproducibility, cost awareness and source-grounded extraction.