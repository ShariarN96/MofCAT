"""Second-pass linker/metal resolution with text fact-check (run after extract_metrics.py).

    python utils/resolve_linkers.py --workers 5

Reads  data/fulltext/<RXN>/extractions/<rxn>_records.csv
Writes data/fulltext/<RXN>/extractions/<rxn>_records_resolved.csv   (used by the figures/Excel export)
"""

from __future__ import annotations

import argparse
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
from mofcat.extraction.linker_resolver import resolve_reaction  # noqa: E402

from mofcat.paths import FULLTEXT_DIR  # noqa: E402

DEFAULT_ROOT = FULLTEXT_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--reactions", nargs="+", default=REACTION_SHEETS)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ensure_loaded()
    root = Path(args.root)
    for rx in args.reactions:
        ext = root / rx / "extractions"
        src = ext / f"{rx.lower()}_records.csv"
        if not src.exists():
            print(f"{rx}: no records")
            continue
        out = resolve_reaction(rx, src, root / rx / "manifest.csv", model=args.model, workers=args.workers)
        dst = ext / f"{rx.lower()}_records_resolved.csv"
        out.to_csv(dst, index=False)
        if "linker_check" in out:
            print(f"\n{rx}: linker_check = {out['linker_check'].value_counts(dropna=False).to_dict()}  -> {dst}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

