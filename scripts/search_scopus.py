#!/usr/bin/env python3
"""Stage 1: search Scopus for every reaction query and save the records.

    python scripts/01_search_scopus.py

Queries live in src/mofcat/ingestion/queries.py. Output goes to data/raw/scopus/:
scopus_results_<RXN>.csv, scopus_results_ALL.csv/.xlsx and two summary tables.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from mofcat.ingestion.scopus_search import main  # noqa: E402

if __name__ == "__main__":
    main()
