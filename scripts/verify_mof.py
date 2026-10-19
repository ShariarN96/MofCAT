"""Fact-check, from the saved full text, that each paper is genuinely about a MOF electrocatalyst.

    python scripts/03_verify_mof.py --limit 10          # the 10 pilot papers per reaction
    python scripts/03_verify_mof.py                      # everything downloaded
    python scripts/03_verify_mof.py --reactions NO3RR --model gpt-4o

Writes data/fulltext/<RXN>/verification/<rxn>_mof_verification.csv and prints a summary.
Then run: python scripts/04_extract_metrics.py --only-verified
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

from mofcat.ingestion.queries import REACTIONS as REACTION_SHEETS  # noqa: E402
from mofcat.env import ensure_loaded  # noqa: E402
from mofcat.extraction.mof_verification import verify_reaction  # noqa: E402

from mofcat.paths import FULLTEXT_DIR  # noqa: E402

DEFAULT_ROOT = FULLTEXT_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--reactions", nargs="+", default=REACTION_SHEETS)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None, help="papers per reaction")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ensure_loaded()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in .env")
        return 1
    root = Path(args.root)

    for rx in args.reactions:
        mp = root / rx / "manifest.csv"
        if not mp.exists():
            print(f"{rx}: no manifest")
            continue
        df = verify_reaction(mp, rx, model=args.model, limit=args.limit, workers=args.workers)
        out = root / rx / "verification"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{rx.lower()}_mof_verification.csv"
        df.to_csv(path, index=False)
        print(f"\n{rx}: {df['verification_status'].value_counts().to_dict()}  -> {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

