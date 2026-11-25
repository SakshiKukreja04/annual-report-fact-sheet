"""Write the POWERGRID table-recovery audit without changing sample JSON."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import PREPROCESS_CACHE_DIR, TABLE_DIR
from src.tables.recovery import audit_half

# Expected counts come from inspection of the cached page images, not from the detector.
LEFT_EXPECTATIONS = [
    {
        "label": "section_e_spent_or_unspent",
        "ticker": "POWERGRID",
        "matched_table_id": "POWERGRID-p0200-s02-LEFT",
        "expected_columns": 6,
        "expected_body_rows": 1,
        "note": "Visible spent-or-unspent grid above the set-off table. Header labels were not transcribed.",
    },
    {
        "label": "set_off",
        "ticker": "POWERGRID",
        "matched_table_id": "POWERGRID-p0200-s01-LEFT",
        "expected_columns": ["S.No.", "Particulars", "Amount"],
        "expected_body_rows": 6,
        "note": "Set-off table. Expected columns are S.No., Particulars, and Amount.",
    },
    {
        "label": "three_year_unspent_csr",
        "ticker": "POWERGRID",
        "matched_table_id": None,
        "note": (
            "The three-year unspent CSR table is visible and was not returned. "
            "Its header is nested, so the exact column count and expected row count were not locked. "
            "No cell values were filled in."
        ),
    },
]
RIGHT_EXPECTATIONS = [
    {
        "label": "capital_assets",
        "ticker": "POWERGRID",
        "matched_table_id": None,
        "note": (
            "A ruled capital-assets table is visible on the right half and was not returned. "
            "The header is nested, so the exact column count was not locked. No cell values were filled in."
        ),
    },
]

LEFT_NOTES = [
    "Compared data/cache/preprocessing/POWERGRID/p0200_LEFT/A_original.png with full.png.",
    "The set-off table is visible with three columns: S.No., Particulars, Amount. The extractor returned 3 columns and 6 body rows.",
    "Those cells are garbled. The status is STRUCTURE_ONLY. The text was not repaired.",
    "The spent-or-unspent grid above it was returned as POWERGRID-p0200-s02-LEFT. Its cell text is also garbled.",
    "The three-year unspent table below the set-off table was not returned.",
    "Table bounding boxes were null in the sample JSON, so the boxes could not be overlaid. cell_boxes_present was true, but the boxes were not stored.",
]
RIGHT_NOTES = [
    "Compared data/cache/preprocessing/POWERGRID/p0200_RIGHT/A_original.png with full.png.",
    "The extractor returned no table. The cached image still shows one ruled table under the capital-assets question.",
    "That absence is a missed table, not an empty page. Item 9 is a sentence, not a second table.",
    "Exact columns and rows were not transcribed.",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_powergrid_audit(sample_dir: Path | None = None) -> dict:
    """Read the existing sample records and attach recovery status. Files are not rewritten."""
    sample_dir = sample_dir or (TABLE_DIR / "sample")
    left_tables = [
        _load_json(sample_dir / "POWERGRID-p0200-s02-LEFT.json"),
        _load_json(sample_dir / "POWERGRID-p0200-s01-LEFT.json"),
    ]
    summary = _load_json(sample_dir / "summary.json")
    right_count = next(
        item["table_count"]
        for item in summary["scanned"]
        if item["ticker"] == "POWERGRID" and item["pdf_page"] == 200 and item["half"] == "RIGHT"
    )
    if right_count != 0:
        raise RuntimeError(f"Expected the stored right-half count to stay 0, found {right_count}")
    cache = PREPROCESS_CACHE_DIR / "POWERGRID"
    left = audit_half(
        "POWERGRID",
        200,
        left_tables[0].get("printed_page"),
        "LEFT",
        left_tables,
        LEFT_EXPECTATIONS,
        str(cache / "p0200_LEFT" / "full.png"),
        "paddleocr_pp_structure",
        LEFT_NOTES,
    )
    right = audit_half(
        "POWERGRID",
        200,
        None,
        "RIGHT",
        [],
        RIGHT_EXPECTATIONS,
        str(cache / "p0200_RIGHT" / "full.png"),
        "paddleocr_pp_structure",
        RIGHT_NOTES,
    )
    return {
        "company": "POWERGRID",
        "financial_shares_calculated": False,
        "language_model_called": False,
        "sample_outputs_modified": False,
        "pages_not_manually_inspected": (
            "Every POWERGRID page except PDF page 200. Expected table counts there are unknown, "
            "so unreturned tables on those pages are not marked missing."
        ),
        "halves": [left, right],
    }


def render_markdown(audit: dict) -> str:
    lines = [
        "# POWERGRID table recovery",
        "",
        "Only PDF page 200 was inspected. Other pages keep an unknown expected count and are not listed as missing tables.",
        "",
        "A missing table means a person saw a table that the extractor did not return. Failed text means a grid came back but the cells are not reliable. An empty inspected page is not a failure.",
        "",
    ]
    for half in audit["halves"]:
        lines.append(f"## PDF page {half['pdf_page']} {half['half']}")
        lines.append("")
        lines.append(f"- Printed page: {half['printed_page']}")
        lines.append(f"- Expected tables: {half['expected_table_count']}")
        lines.append(f"- Detected tables: {half['detected_table_count']}")
        lines.append(f"- Recovered with reliable text: {half['recovered_table_count']}")
        lines.append(f"- Missing tables: {half['missing_table_count']}")
        lines.append(f"- Detection status: {half['table_detection_status']}")
        lines.append(f"- Manual review: {half['manual_review_required']}")
        lines.append(f"- Source image: `{half['source_image']}`")
        lines.append(f"- Method: {half['extraction_method']}")
        lines.append("")
        for note in half["recovery_notes"]:
            lines.append(f"- {note}")
        lines.append("")
        for table in half["tables"]:
            label = table.get("label") or table.get("table_id")
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"- Status: {table['recovery_status']}")
            lines.append(f"- PDF page: {table['pdf_page']}")
            lines.append(f"- Half: {table['half']}")
            lines.append(f"- Rows: {table['detected_row_count']}")
            lines.append(f"- Columns: {table['detected_column_count']}")
            lines.append(f"- Reason: {table['rejection_reason']}")
            if table["recovery_status"] == "NOT_DETECTED":
                lines.append("- No cell values were added.")
            lines.append("")
    lines.append("## Not inspected")
    lines.append("")
    lines.append(audit["pages_not_manually_inspected"])
    lines.append("")
    return "\n".join(lines)


def write_powergrid_audit(output_dir: Path | None = None, sample_dir: Path | None = None) -> dict:
    output_dir = output_dir or (TABLE_DIR / "recovery_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_powergrid_audit(sample_dir)
    (output_dir / "POWERGRID_recovery.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "POWERGRID_recovery.md").write_text(render_markdown(audit), encoding="utf-8")
    return audit


def main() -> None:
    audit = write_powergrid_audit()
    print(json.dumps({
        "halves": [
            {
                "half": half["half"],
                "expected_table_count": half["expected_table_count"],
                "detected_table_count": half["detected_table_count"],
                "recovered_table_count": half["recovered_table_count"],
                "missing_table_count": half["missing_table_count"],
                "table_detection_status": half["table_detection_status"],
            }
            for half in audit["halves"]
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
