"""Physical-plausibility checks for extracted electrocatalysis metrics.

Values outside these ranges are almost always extraction slips (a potential read as an overpotential,
a unit mix-up, a percent read as a fraction). They are flagged in the Excel export and excluded from figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# metric -> (low, high, exclusive_low)
RANGES: dict[str, tuple[float, float, bool]] = {
    "overpotential_10_ma_cm2_mv": (0, 1500, True),
    "overpotential_100_ma_cm2_mv": (0, 2000, True),
    "tafel_slope_mv_dec": (5, 500, False),
    "half_wave_potential_v": (0.0, 1.3, False),
    "onset_potential_v": (0.0, 1.5, False),
    "electron_transfer_number": (1.0, 4.3, False),
    "h2o2_yield_pct": (0, 100, False),
    "nh3_faradaic_efficiency_pct": (0, 100.5, False),
    "nh3_selectivity_pct": (0, 100.5, False),
    "nitrate_conversion_pct": (0, 100.5, False),
    "nh3_yield_norm": (0, 1e7, True),
}


def out_of_range(rec: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame (same index) marking implausible values per metric column."""
    flags = {}
    for col, (lo, hi, excl) in RANGES.items():
        if col in rec:
            v = pd.to_numeric(rec[col], errors="coerce")
            bad = (v <= lo) if excl else (v < lo)
            flags[col] = (bad | (v > hi)) & v.notna()
    return pd.DataFrame(flags, index=rec.index)


def flag_text(rec: pd.DataFrame) -> pd.Series:
    bad = out_of_range(rec)
    return bad.apply(lambda r: "; ".join(f"{c} out of plausible range" for c in bad.columns[r.values]) or np.nan, axis=1)


def clean_metrics(rec: pd.DataFrame) -> pd.DataFrame:
    """Copy of rec with implausible metric values set to NaN."""
    out = rec.copy()
    bad = out_of_range(out)
    for c in bad.columns:
        out.loc[bad[c], c] = np.nan
    return out
