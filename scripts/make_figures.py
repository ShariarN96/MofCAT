#!/usr/bin/env python3
"""Six-panel catalyst / linker / metal figure per reaction.

    python scripts/07_make_figures.py
    python scripts/07_make_figures.py --reactions HER OER

Writes outputs/figures/<RXN>_6panel.png and individual panels to outputs/figures/panels/<RXN>/.
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

from mofcat.reporting.panel_figures import REACTIONS, make_panel_figures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactions", nargs="+", default=REACTIONS)
    ap.add_argument("--out", default=None, help="output folder (default outputs/figures)")
    args = ap.parse_args()
    for rx in args.reactions:
        paths = make_panel_figures(rx, Path(args.out)) if args.out else make_panel_figures(rx)
        print(f"{rx}: {len(paths)} files; main figure -> {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

