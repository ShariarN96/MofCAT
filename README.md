# MofCAT: MOF electrocatalysis literature mining

Finds papers on metal–organic frameworks (MOFs) for five electrocatalytic reactions (HER, OER, ORR, N2RR, NO3RR), downloads their full text, and uses an LLM to extract per-catalyst performance data into Excel workbooks and figures.

```
Scopus search ─► full-text download ─► MOF verification ─► metric extraction ─► linker resolution ─► Excel + figures
  (ingestion)         (ingestion)          (extraction)        (extraction)         (extraction)       (reporting)
```

## Layout

```
mof_pipeline/
├── src/mofcat/                  the package: all reusable code
│   ├── paths.py                 every directory/file location (edit paths here only)
│   ├── env.py                   loads API keys from .env
│   ├── ingestion/               find and save articles
│   │   ├── queries.py           Scopus query per reaction, year/open-access filters
│   │   ├── scopus_search.py     threaded, rate-limited Scopus search
│   │   ├── fulltext.py          Elsevier full-text XML download + parsing
│   │   ├── xml_to_markdown.py   XML -> Markdown
│   │   ├── xml_to_text.py       XML -> plain text
│   │   └── wiley_tdm.py         DOI helpers (+ optional Wiley PDF download)
│   ├── extraction/              LLM + dictionary extraction
│   │   ├── vocabulary.py        metal / linker / MOF-name normalisation
│   │   ├── reaction_schemas.py  per-reaction metric schemas and prompts
│   │   ├── mof_verification.py  is this paper really about a MOF catalyst?
│   │   ├── metrics_extract.py   performance metrics per catalyst
│   │   ├── linker_resolver.py   second-pass linker / metal fact-check
│   │   └── data_checks.py       range checks and flags
│   └── reporting/               outputs
│       ├── excel_export.py      verification + per-reaction workbooks
│       ├── figures.py           shared loaders + per-reaction / cross-reaction figures
│       ├── science_figures.py
│       └── panel_figures.py     six-panel figures
├── scripts/                     command-line entry points, one per stage
│   ├── 01_search_scopus.py … 07_make_figures.py
│   ├── run_all.py               runs stages 2-7 (resumable)
│   ├── convert_xml.py           XML -> Markdown for a folder of files
│   └── fetch_single_article.py  test one DOI (key, entitlement, converters)
├── data/                        inputs and intermediate artefacts (git-ignored except raw/scopus)
│   ├── raw/scopus/              stage 1 output: scopus_results_*.csv / .xlsx
│   ├── raw/snapshots/           frozen earlier inputs
│   ├── raw/elsevier_xml/        standalone XML downloads
│   ├── processed/elsevier_md/   Markdown conversions
│   ├── fulltext/<RXN>/          per-reaction cache: xml/, text/, manifest.csv, verification/, extractions/
│   └── cache/                   SQLite caches for LLM calls (reruns are free)
├── outputs/                     results
│   ├── excel/                   <RXN>_performance_metrics.xlsx, MOF_verification_all_reactions.xlsx
│   ├── figures/
│   └── reports/                 e.g. scopus_overview/ (publication-count charts)
├── tests/                       smoke tests (no network or keys needed)
├── docs/
├── pyproject.toml · requirements.txt · .env.example · .gitignore
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt
copy .env.example .env         # then fill in your keys
pytest                         # optional: check everything imports
```

`.env` keys: `ELSEVIER_API_KEY` (required), `ELSEVIER_INST_TOKEN` (optional, full text off-campus), `OPENAI_API_KEY` (verification, extraction, linker resolution), plus `ANTHROPIC_API_KEY` / `RSC_API_KEY` if you use them. The scripts also run without `pip install -e .`.

## Usage

```powershell
python scripts/01_search_scopus.py        # 1. Scopus search -> data/raw/scopus/
python scripts/run_all.py                 # 2-7. fetch -> verify -> extract -> resolve -> excel -> figures
python scripts/run_all.py --from verify   #      resume from any stage
```

Or run the stages one at a time (each has `--help`):

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_search_scopus.py` | `ingestion/queries.py` | `data/raw/scopus/` |
| 2 | `02_fetch_fulltext.py` | Scopus workbook | `data/fulltext/<RXN>/{xml,text}`, `manifest.csv` |
| 3 | `03_verify_mof.py` | full texts | `…/verification/*.csv` |
| 4 | `04_extract_metrics.py --only-verified` | full texts | `…/extractions/*.csv` |
| 5 | `05_resolve_linkers.py` | extractions | updated extractions |
| 6 | `06_export_excel.py` | extractions | `outputs/excel/*.xlsx` |
| 7 | `07_make_figures.py` | extractions | `outputs/figures/` |

Tuning: Scopus queries, year window and open-access filter are in `src/mofcat/ingestion/queries.py`; per-reaction metrics are in `src/mofcat/extraction/reaction_schemas.py`; `TARGET` / `BUFFER_SHEET` (papers per reaction) are at the top of `scripts/run_all.py`.

## Notes

- **Stage 2 input.** It reads `data/raw/scopus/scopus_results_ALL.xlsx`, the file stage 1 writes. The earlier 504-record input behind the first Excel results is kept in `data/raw/snapshots/`; pass `--xlsx` to `02_fetch_fulltext.py` to use it.
- **Full text is licensed.** Downloaded Elsevier text is for your own text-mining under Elsevier's TDM terms. It lives in git-ignored folders; do not redistribute it.
- **Never commit `.env`.**
