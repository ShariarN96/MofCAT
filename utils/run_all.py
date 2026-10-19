"""Run the whole Elsevier pipeline on every eligible paper (resumable; everything is cached).

    python scripts/run_all.py                    # fetch -> verify -> extract -> resolve -> excel -> figures
    python scripts/run_all.py --from verify      # resume from a stage
    python scripts/run_all.py --from search      # also re-run the Scopus search first (re-queries the API)

Stages: search -> fetch -> verify -> extract -> resolve -> excel -> figures
Outputs: data/raw/scopus/, data/fulltext/<RXN>/..., outputs/excel/*.xlsx, outputs/figures/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
S = ROOT / "scripts"

TARGET = 50              # MOF-verified papers wanted per reaction (net, after filtering)
BUFFER_SHEET = 60        # full texts fetched per reaction (buffer for papers rejected by MOF verification)
SMALL_SHEET = 100        # N2RR / NO3RR have few Elsevier papers -> fetch every available one
FETCH = [PY, "-u", str(S / "02_fetch_fulltext.py"), "--interval", "2.0"]  # gentle 2 s between API calls

STAGES = {
    "search": [PY, "-u", str(S / "01_search_scopus.py")],
    "fetch": [FETCH + ["--reactions", "HER", "OER", "ORR", "--per-sheet", str(BUFFER_SHEET)],
              FETCH + ["--reactions", "N2RR", "NO3RR", "--per-sheet", str(SMALL_SHEET)]],
    "verify": [PY, "-u", str(S / "03_verify_mof.py"), "--workers", "2"],
    "extract": [PY, "-u", str(S / "04_extract_metrics.py"), "--only-verified", "--workers", "2",
                "--limit", str(TARGET)],
    "resolve": [PY, "-u", str(S / "05_resolve_linkers.py"), "--workers", "2"],
    "excel": [PY, "-u", str(S / "06_export_excel.py"), "--out", str(ROOT / "outputs" / "excel")],
    "figures": [PY, "-u", str(S / "07_make_figures.py"), "--out", str(ROOT / "outputs" / "figures")],
}
RETRY = {"fetch", "verify", "extract", "resolve"}  # rerun once to pick up transient (rate-limit) failures


def run(name: str) -> int:
    for attempt in range(2 if name in RETRY else 1):
        t0 = time.time()
        print(f"\n===== {name} (attempt {attempt + 1}) =====", flush=True)
        cmds = STAGES[name] if isinstance(STAGES[name][0], list) else [STAGES[name]]
        rc = 0
        for cmd in cmds:
            rc = subprocess.call(cmd, cwd=ROOT) or rc
        print(f"===== {name} done in {(time.time() - t0) / 60:.1f} min, exit {rc} =====", flush=True)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="fetch", choices=list(STAGES))
    args = ap.parse_args()
    names = list(STAGES)
    for name in names[names.index(args.start):]:
        rc = run(name)
        if rc != 0:
            print(f"stage {name} failed", flush=True)
            return rc
    print("\nALL STAGES COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
