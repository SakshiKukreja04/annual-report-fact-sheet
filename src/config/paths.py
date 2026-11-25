"""Paths resolved from the project root, independent of the shell cwd."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
AUDIT_DIR = PROJECT_ROOT / "data" / "processed" / "audit"
CLASSIFICATION_DIR = PROJECT_ROOT / "data" / "processed" / "classification"
PREPROCESS_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "preprocessing"
OCR_DIR = PROJECT_ROOT / "data" / "processed" / "ocr"
OCR_MISTRAL_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_mistral"
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
PAGE_DIR = PROJECT_ROOT / "data" / "processed" / "pages"
VISION_DIR = PROJECT_ROOT / "data" / "processed" / "vision"
CANDIDATE_DIR = PROJECT_ROOT / "data" / "processed" / "candidates"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Intermediate factsheet/validate writes land here; promote to outputs/submission only from final assembly.
STAGING_DIR = PROJECT_ROOT / "staging"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
LOG_DIR = PROJECT_ROOT / "logs"
