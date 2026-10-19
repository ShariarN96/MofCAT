#!/usr/bin/env python3
"""Extract reaction-specific performance metrics from saved Elsevier full texts.

Run scripts/02_fetch_fulltext.py first. Reads data/fulltext/<REACTION>/manifest.csv
(papers with status 'ok') and writes, per reaction, to <REACTION>/extractions/:
  <rxn>_records.csv / .json   one row per catalyst x condition
  <rxn>_papers.csv            one row per paper (flags, confidence, notes)

Usage:
    python scripts/04_extract_metrics.py --dry-run
    python scripts/04_extract_metrics.py --limit 10                 # pilot: 10 per reaction
    python scripts/04_extract_metrics.py --reactions HER --limit 2 --model gpt-4o
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mofcat.extraction.metrics_extract import coverage_summary, run_reaction  # noqa: E402
from mofcat.ingestion.queries import REACTIONS as REACTION_SHEETS  # noqa: E402
from mofcat.env import ensure_loaded  # noqa: E402

from mofcat.paths import FULLTEXT_DIR  # noqa: E402

DEFAULT_ROOT = FULLTEXT_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--reactions", nargs="+", default=REACTION_SHEETS)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None, help="papers per reaction")
    ap.add_argument("--only-verified", action="store_true",
                    help="skip papers not marked 'verified' by scripts/03_verify_mof.py")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_loaded()
    root = Path(args.root)

    if args.dry_run:
        for rxn in args.reactions:
            mp = root / rxn / "manifest.csv"
            if not mp.exists():
                print(f"{rxn}: no manifest")
                continue
            m = pd.read_csv(mp)
            n = int((m["status"] == "ok").sum())
            print(f"{rxn}: {n} full-text papers ready ({m['status'].value_counts().to_dict()})")
        print(f"Model: {args.model}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in .env")
        return 1

    for rxn in args.reactions:
        mp = root / rxn / "manifest.csv"
        if not mp.exists():
            print(f"{rxn}: no manifest, run scripts/02_fetch_fulltext.py first")
            continue
        ver_path = None
        if args.only_verified:
            ver_path = root / rxn / "verification" / f"{rxn.lower()}_mof_verification.csv"
            if not ver_path.exists():
                print(f"{rxn}: no verification file; run scripts/03_verify_mof.py first")
                continue
        papers, records = run_reaction(mp, rxn, model=args.model, limit=args.limit,
                                       verification_path=ver_path, workers=args.workers)
        out_dir = root / rxn / "extractions"
        out_dir.mkdir(parents=True, exist_ok=True)
        low = rxn.lower()
        papers.to_csv(out_dir / f"{low}_papers.csv", index=False)
        records.to_csv(out_dir / f"{low}_records.csv", index=False)
        records.to_json(out_dir / f"{low}_records.json", orient="records", indent=2, force_ascii=False)
        print(f"\n{rxn}: {len(papers)} papers, {len(records)} records -> {out_dir}")
        print("  metric coverage (records with a value):")
        for k, v in coverage_summary(records, rxn).items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

