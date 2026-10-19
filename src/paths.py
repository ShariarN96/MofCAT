"""Single source of truth for every directory and file location in the project.

Nothing else in the package should build paths from ``__file__``. Import from here:

    from mofcat.paths import FULLTEXT_DIR, CACHE_DIR
"""

from __future__ import annotations

from pathlib import Path

# <repo>/src/mofcat/paths.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

# ---- data (inputs and intermediate artefacts; large and git-ignored) ------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SCOPUS_DIR = RAW_DIR / "scopus"                  # stage 1 output: scopus_results_*.csv / .xlsx
SNAPSHOT_DIR = RAW_DIR / "snapshots"             # frozen copies of earlier inputs
XML_DIR = RAW_DIR / "elsevier_xml"               # standalone article XML downloads
PROCESSED_DIR = DATA_DIR / "processed"
MARKDOWN_DIR = PROCESSED_DIR / "elsevier_md"     # XML -> Markdown conversions
FULLTEXT_DIR = DATA_DIR / "fulltext"             # stage 2+ per-reaction cache: <RXN>/{manifest,verification,extractions}
CACHE_DIR = DATA_DIR / "cache"                   # SQLite caches for LLM calls

# The workbook the full-text stages read. Written by ``scopus_search`` (stage 1).
SCOPUS_XLSX = SCOPUS_DIR / "scopus_results_ALL.xlsx"

# ---- outputs (results you read and share) ----------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXCEL_DIR = OUTPUT_DIR / "excel"                 # per-reaction metrics + MOF verification workbooks
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ---- cache files ------------------------------------------------------------------
EXTRACT_CACHE = CACHE_DIR / "metrics_extract_cache.db"
VERIFY_CACHE = CACHE_DIR / "mof_verification_cache.db"
LINKER_CACHE = CACHE_DIR / "linker_resolver_cache.db"
