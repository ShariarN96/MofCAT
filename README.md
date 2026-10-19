# MofCAT: MOF Electrocatalysis Literature Mining

## Overview

Metal–organic frameworks (MOFs) are a leading class of electrocatalyst because their metal nodes and organic linkers can be tuned almost freely. The research is spread across thousands of papers, and each paper reports its results in its own format. That makes it hard to answer basic questions: which MOFs have been tried for which reaction, how well did they perform, and what has never been tried?

MofCAT is an automated pipeline that answers those questions. It searches the literature, downloads full texts, uses an LLM to read each paper, and turns the results into structured, comparable data. It covers five electrocatalytic reactions:

| Reaction | Application |
|---|---|
| **HER**: hydrogen evolution | Hydrogen production |
| **OER**: oxygen evolution | Metal–air batteries |
| **ORR**: oxygen reduction | Metal–air batteries |
| **N2RR**: nitrogen reduction | Ammonia synthesis |
| **NO3RR**: nitrate reduction | Ammonia synthesis |

## Aims and goals

1. **Build a reproducible corpus.** Retrieve MOF electrocatalysis papers (2016–2025) from Scopus with documented, versioned queries, so the dataset can be rebuilt or extended.
2. **Extract data from full text, not just abstracts.** Abstracts rarely name linkers or report numbers. Reading the full text recovers the metals, organic linkers and performance metrics.
3. **Verify before extracting.** Check that each paper is genuinely about a MOF catalyst (pristine, MOF-derived or composite), so off-topic papers don't contaminate the data.
4. **Capture reaction-specific performance.** Record the metrics that matter for each reaction, such as overpotential and Tafel slope for HER/OER, half-wave potential for ORR, and NH₃ yield and Faradaic efficiency for N2RR/NO3RR.
5. **Map the chemical space.** Show which metals and linkers dominate each reaction and how architecture type differs across reactions.
6. **Find gaps.** Highlight metal–linker–reaction combinations with little or no published work.

**Scope.** MofCAT measures what has been *reported*. A gap in the literature is not a prediction that a material will work. Extracted values come from an LLM, so spot-check them against the source papers before using them in a publication.

## Architecture

```
 Scopus search
      │
      ▼
 Full-text download
      │
      ▼
 MOF verification   (LLM)
      │
      ▼
 Metric extraction  (LLM)
      │
      ▼
 Linker resolution  (LLM)
      │
      ▼
 Excel workbooks + Figures
```

Each step reads the files written by the step before it, so any step can be run on its own or resumed after a failure. LLM calls are cached, so reruns are free.

## Layout

```
mof_pipeline/
├── src/                         reusable code
│   ├── paths.py
│   ├── env.py
│   ├── ingestion/               find and save articles
│   │   ├── queries.py
│   │   ├── scopus_search.py
│   │   ├── fulltext.py
│   │   ├── xml_to_markdown.py
│   │   ├── xml_to_text.py
│   │   └── wiley_tdm.py
│   ├── extraction/              LLM extraction
│   │   ├── vocabulary.py
│   │   ├── reaction_schemas.py
│   │   ├── mof_verification.py
│   │   ├── metrics_extract.py
│   │   ├── linker_resolver.py
│   │   └── data_checks.py
│   └── reporting/               outputs
│       ├── excel_export.py
│       ├── figures.py
│       ├── science_figures.py
│       └── panel_figures.py
├── utils/                       command-line entry points, one per step
├── data/                        inputs and intermediate files
├── outputs/                     Excel workbooks, figures, reports
├── tests/
├── docs/
└── pyproject.toml · requirements.txt · .env.example · .gitignore