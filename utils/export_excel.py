"""Write the Excel deliverables.

    python utils/export_excel.py

outputs/excel/MOF_verification_all_reactions.xlsx   which papers are genuinely MOF papers
outputs/excel/<RXN>_performance_metrics.xlsx        metrics + metals + organic linkers per catalyst
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from mofcat.reporting.excel_export import export_reaction, export_verification  # noqa: E402
from mofcat.reporting.figures import REACTIONS  # noqa: E402


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default outputs/excel)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    print(export_verification(out / "MOF_verification_all_reactions.xlsx" if out else None))
    for rx in REACTIONS:
        print(export_reaction(rx, out / f"{rx}_performance_metrics.xlsx" if out else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

