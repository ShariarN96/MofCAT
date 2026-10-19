"""EDA figures for the Elsevier full-text extractions, one set per reaction.

Reads  data/elsevier_fulltext/<RXN>/extractions/<rxn>_records.csv and <rxn>_papers.csv
Writes figures/elsevier_pilot/<RXN>/*.png, figures/elsevier_pilot/cross_reaction/*.png
and data/elsevier_fulltext/<RXN>/extractions/<rxn>_records_normalized.csv.

Plots depend on each reaction's metrics:
  HER / OER : overpotential, Tafel slope (mechanism bands), Cdl, Rct, benchmark deltas
  ORR       : E1/2 vs onset, n vs H2O2 yield (RRDE consistency line), Pt/C deltas, Zn-air
  N2RR      : NH3 yield (unit-normalised) vs FE, potential dependence, control-experiment rigor
  NO3RR     : FE vs potential, selectivity vs conversion, yield by basis, rigor
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..extraction.reaction_schemas import METRICS  # noqa: E402
from ..ingestion.queries import REACTIONS  # noqa: E402
from ..paths import FIGURES_DIR, FULLTEXT_DIR  # noqa: E402

DATA_ROOT = FULLTEXT_DIR
FIG_ROOT = FIGURES_DIR

RXN_COLORS = {"HER": "#2196F3", "OER": "#F44336", "ORR": "#4CAF50", "N2RR": "#9C27B0", "NO3RR": "#FF9800"}
CLASS_COLORS = {
    "pristine_mof": "#4CAF50", "mof_derived": "#FF9800", "mof_composite": "#3F51B5",
    "non_mof": "#9E9E9E", "unclear": "#CFD8DC",
}
CLASS_LABELS = {
    "pristine_mof": "Pristine MOF", "mof_derived": "MOF-derived", "mof_composite": "MOF composite",
    "non_mof": "Non-MOF", "unclear": "Unclear",
}
ROLE_MARKERS = {"main": "o", "variant": "s", "benchmark": "*"}
ROLE_ORDER = ["main", "variant", "benchmark"]
MEDIUM_COLORS = {"acidic": "#E53935", "neutral": "#78909C", "alkaline": "#1E88E5", "other": "#BDBDBD"}

MW_NH3 = 17.031  # g/mol, for umol <-> ug conversion

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "axes.labelsize": 9.5})


# --------------------------------------------------------------------------- #
# Loading and derived columns
# --------------------------------------------------------------------------- #
def normalize_yield(value: float, unit: str) -> tuple[float, str]:
    """Convert an NH3 yield to ug h-1 mg-1 (mass basis) or ug h-1 cm-2 (area basis).

    Returns (nan, 'other') for units that cannot be converted unambiguously
    (per litre, per minute, ...). Only a mass conversion is applied (umol -> ug using
    M = 17.031 g/mol for NH3); NH4+ reporting would differ by ~6%.
    """
    if value is None or pd.isna(value) or not isinstance(unit, str):
        return np.nan, "other"
    u = unit.lower().replace("μ", "u").replace("µ", "u")
    u = re.sub(r"catalyst|cat", "", u)
    u = re.sub(r"[\s^_\-−–·*/()\.]", "", u)
    if "min" in u or re.search(r"l\d", u):  # per minute or per litre: not comparable
        return np.nan, "other"
    m = re.match(r"(ug|umol|mmol|nmol|mg)", u)
    if not m:
        return np.nan, "other"
    num = m.group(1)
    rest = u[len(num):]
    to_ug = {"ug": 1.0, "mg": 1e3, "umol": MW_NH3, "mmol": MW_NH3 * 1e3, "nmol": MW_NH3 / 1e3}[num]
    if "cm" in rest:
        return float(value) * to_ug, "area"
    if "mg" in rest:  # per mg catalyst -> already the target basis
        return float(value) * to_ug, "mass"
    if "g" in rest:  # per g catalyst -> divide by 1000 to get per mg
        return float(value) * to_ug / 1e3, "mass"
    return np.nan, "other"


def _to_bool(series: pd.Series) -> pd.Series:
    return series.map({True: True, "True": True, False: False, "False": False})


def load(reaction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext = DATA_ROOT / reaction / "extractions"
    base = ext / f"{reaction.lower()}_records.csv"
    resolved = ext / f"{reaction.lower()}_records_resolved.csv"
    # prefer the linker-resolved table when it is at least as new as the first-pass table
    use = resolved if resolved.exists() and resolved.stat().st_mtime >= base.stat().st_mtime else base
    rec = pd.read_csv(use)
    pap = pd.read_csv(ext / f"{reaction.lower()}_papers.csv")
    order = {doi: f"P{i}" for i, doi in enumerate(pap["doi"], start=1)}
    rec["paper"] = rec["doi"].map(order)
    pap["paper"] = pap["doi"].map(order)
    for c in rec.columns:
        if rec[c].dtype == object and rec[c].dropna().isin([True, False, "True", "False"]).all() and rec[c].notna().any():
            rec[c] = _to_bool(rec[c])
    for col in ("material_class", "role", "medium"):
        if col in rec:
            rec[col] = rec[col].fillna("unclear" if col == "material_class" else "other")
    if "nh3_yield_rate" in rec:
        res = [normalize_yield(v, u) for v, u in zip(rec["nh3_yield_rate"], rec["nh3_yield_rate_unit"])]
        rec["nh3_yield_norm"] = [r[0] for r in res]
        rec["nh3_yield_basis"] = [r[1] for r in res]
        rec.to_csv(ext / f"{reaction.lower()}_records_normalized.csv", index=False)
    return rec, pap


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _footer(fig, rec: pd.DataFrame, reaction: str) -> None:
    fig.text(0.995, 0.004,
             f"{reaction} pilot: {len(rec)} records from {rec['doi'].nunique()} Elsevier papers (LLM extraction from full text)",
             ha="right", va="bottom", fontsize=6.5, color="#757575")


def _save(fig, reaction: str, name: str, rec: pd.DataFrame) -> Path:
    out = FIG_ROOT / reaction
    out.mkdir(parents=True, exist_ok=True)
    _footer(fig, rec, reaction)
    path = out / f"{reaction}_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty(ax, msg: str = "No values reported in this sample") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#9E9E9E", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_class_role(ax, d: pd.DataFrame, loc="best") -> None:
    handles = []
    for cls in [c for c in CLASS_COLORS if c in set(d["material_class"])]:
        handles.append(Line2D([], [], marker="o", ls="", color=CLASS_COLORS[cls], label=CLASS_LABELS[cls]))
    for role in [r for r in ROLE_ORDER if r in set(d["role"])]:
        handles.append(Line2D([], [], marker=ROLE_MARKERS[role], ls="", color="#455A64",
                              markerfacecolor="none", label=role))
    ax.legend(handles=handles, fontsize=7, loc=loc, frameon=True, framealpha=0.85)


def scatter(ax, d, x, y, xlabel, ylabel, *, logx=False, logy=False, title=None, diag=None, subtitle=None):
    d = d.dropna(subset=[x, y])
    if logx:
        d = d[d[x] > 0]
    if logy:
        d = d[d[y] > 0]
    if d.empty:
        _empty(ax)
        ax.set_title(title or "")
        return
    for role in ROLE_ORDER:
        sub = d[d["role"] == role]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], c=[CLASS_COLORS.get(c, "#9E9E9E") for c in sub["material_class"]],
                   marker=ROLE_MARKERS[role], s={"main": 70, "variant": 42, "benchmark": 150}[role],
                   edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title((title or f"{ylabel.split('(')[0].strip()} vs {xlabel.split('(')[0].strip()}  (n={len(d)})")
                 + (f"\n{subtitle}" if subtitle else ""))
    if diag:
        diag(ax)
    _legend_class_role(ax, d)


def group_strip(ax, d, col, ylabel, *, by="material_class", logy=False, title=None, order=None, subtitle=None):
    d = d.dropna(subset=[col])
    if logy:
        d = d[d[col] > 0]
    if d.empty:
        _empty(ax)
        ax.set_title(title or "")
        return
    order = [o for o in (order or list(CLASS_COLORS)) if o in set(d[by])]
    palette = CLASS_COLORS if by == "material_class" else MEDIUM_COLORS
    sns.boxplot(data=d, x=by, y=col, order=order, color="white", fliersize=0, width=0.55,
                linewidth=1.0, ax=ax, boxprops={"alpha": 0.7})
    for role in ROLE_ORDER:
        sub = d[d["role"] == role]
        if sub.empty:
            continue
        xs = [order.index(v) + np.random.default_rng(1).uniform(-0.18, 0.18) for v in sub[by]]
        ax.scatter(xs, sub[col], c=[palette.get(v, "#9E9E9E") for v in sub[by]],
                   marker=ROLE_MARKERS[role], s={"main": 62, "variant": 38, "benchmark": 140}[role],
                   edgecolor="black", linewidth=0.6, zorder=4)
    ax.set_xticks(range(len(order)))
    labels = CLASS_LABELS if by == "material_class" else {k: k for k in order}
    ax.set_xticklabels([f"{labels.get(o, o)}\n(n={int((d[by] == o).sum())})" for o in order], fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title((title or f"{ylabel.split('(')[0].strip()} by {by.replace('_', ' ')}") + (f"\n{subtitle}" if subtitle else ""))
    ax.legend(handles=[Line2D([], [], marker=ROLE_MARKERS[r], ls="", color="#455A64", markerfacecolor="none", label=r)
                       for r in ROLE_ORDER if r in set(d["role"])], fontsize=7, loc="best")


def dumbbell(ax, d, col, xlabel, *, better="low", title=None, subtitle=None):
    """Per paper: best 'main' catalyst vs benchmark (e.g. Pt/C), connected by a line."""
    rows = []
    for paper, g in d.dropna(subset=[col]).groupby("paper"):
        main = g[g["role"] == "main"][col]
        bench = g[g["role"] == "benchmark"][col]
        if main.empty or bench.empty:
            continue
        m = main.min() if better == "low" else main.max()
        b = bench.min() if better == "low" else bench.max()
        rows.append((paper, m, b, g[g["role"] == "benchmark"]["catalyst_name"].iloc[0]))
    if not rows:
        _empty(ax, "No paper reports both a main catalyst\nand a benchmark value for this metric")
        ax.set_title(title or "")
        return
    rows.sort(key=lambda r: r[1] - r[2])
    for i, (paper, m, b, name) in enumerate(rows):
        win = (m < b) if better == "low" else (m > b)
        ax.plot([m, b], [i, i], color="#2E7D32" if win else "#C62828", lw=2, zorder=1)
        ax.scatter(m, i, s=70, color="#1565C0", edgecolor="black", zorder=3, label="This work (main)" if i == 0 else None)
        ax.scatter(b, i, s=110, marker="*", color="#FFB300", edgecolor="black", zorder=3, label="Benchmark" if i == 0 else None)
        ax.text(max(m, b), i, f"  {name[:18]}", va="center", fontsize=6.5, color="#546E7A")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title((title or f"Main catalyst vs benchmark (n={len(rows)} papers)") + (f"\n{subtitle}" if subtitle else ""))
    ax.legend(fontsize=7, loc="best")
    ax.margins(x=0.15)


def composition(ax, d, p):
    ct = pd.crosstab(d["role"], d["material_class"]).reindex(
        index=[r for r in ROLE_ORDER if r in set(d["role"])],
        columns=[c for c in CLASS_COLORS if c in set(d["material_class"])], fill_value=0)
    ct.plot(kind="barh", stacked=True, ax=ax, color=[CLASS_COLORS[c] for c in ct.columns], edgecolor="white")
    for i, tot in enumerate(ct.sum(axis=1)):
        ax.text(tot + 0.3, i, str(int(tot)), va="center", fontsize=8)
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    ax.set_title(f"Records by role and material class (n={len(d)})")
    ax.legend([CLASS_LABELS[c] for c in ct.columns], fontsize=7, loc="lower right")
    ax.invert_yaxis()


def completeness(ax, d, p, reaction):
    names = [n for n, k, _ in METRICS[reaction] if k != "bool" and not n.endswith("_unit")]
    names = [n for n in names if n in d.columns]
    pct = pd.Series({n: 100 * d[n].notna().mean() for n in names}).sort_values()
    ax.barh(range(len(pct)), pct.values, color=RXN_COLORS[reaction], alpha=0.85)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels([n.replace("_", " ") for n in pct.index], fontsize=7)
    for i, v in enumerate(pct.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of records with a reported value")
    ax.set_title("Metric reporting completeness")


def metals(ax, d, p):
    m = d[d["role"] != "benchmark"].dropna(subset=["node_metals"]).copy()
    m["metal"] = m["node_metals"].str.split("|")
    m = m.explode("metal")[["doi", "metal", "material_class"]].drop_duplicates(["doi", "metal"])
    m["metal"] = m["metal"].str.strip()
    ct = m.groupby("metal").size().sort_values(ascending=False).head(10)
    if ct.empty:
        _empty(ax)
        return
    ax.bar(ct.index, ct.values, color="#546E7A")
    ax.set_ylabel("Papers")
    ax.set_title("Framework node metals (papers per metal)")


def medium_bar(ax, d, p):
    ct = pd.crosstab(d["medium"], d["role"].where(d["role"] != "benchmark", "variant"))
    ct = ct.reindex(columns=[c for c in ["main", "variant"] if c in ct.columns], fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#1976D2", "#90CAF9"][: len(ct.columns)], rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("Records")
    ax.set_title("Electrolyte medium")
    ax.legend(fontsize=7)


def electrolytes(ax, d, p):
    e = d["electrolyte"].dropna().astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 34)
    vc = e.value_counts().head(8)[::-1]
    if vc.empty:
        _empty(ax)
        return
    ax.barh(vc.index, vc.values, color="#78909C")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Records")
    ax.set_title("Most common electrolytes")


def stability(ax, d, p):
    s = d.dropna(subset=["stability_duration_h"]).copy()
    s = s[s["role"] != "benchmark"].sort_values("stability_duration_h").tail(14)
    if s.empty:
        _empty(ax)
        ax.set_title("Reported stability test duration")
        return
    ax.barh([f"{n[:22]} ({pp})" for n, pp in zip(s["catalyst_name"], s["paper"])], s["stability_duration_h"],
            color=[CLASS_COLORS.get(c, "#9E9E9E") for c in s["material_class"]], edgecolor="black", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Stability test duration (h)")
    ax.set_title(f"Reported stability test duration (n={len(s)})")


def rigor(ax, d, p, items: list[tuple[str, str, str]], title="Experimental rigor (share of papers)"):
    """items: (label, column, mode) where mode is 'true' (any True) or 'notna'."""
    n_p = d["doi"].nunique()
    vals, labels = [], []
    for label, col, mode in items:
        if col not in d.columns:
            continue
        flag = d.groupby("doi")[col].apply(lambda s: bool((s == True).any()) if mode == "true" else bool(s.notna().any()))  # noqa: E712
        vals.append(100 * flag.mean())
        labels.append(label)
    ax.barh(labels[::-1], vals[::-1], color="#00897B")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"% of papers (n={n_p})")
    ax.set_title(title)


def method_counts(ax, d, p):
    m = d.drop_duplicates("doi")["nh3_quantification_method"].dropna().astype(str).str.lower().str.slice(0, 28)
    vc = m.value_counts().head(7)[::-1]
    if vc.empty:
        _empty(ax)
        return
    ax.barh(vc.index, vc.values, color="#8E24AA")
    ax.set_xlabel("Papers")
    ax.set_title("NH$_3$ quantification method")


# --------------------------------------------------------------------------- #
# Reaction-specific plot specs: name -> (callable(ax, d, p), figsize)
# --------------------------------------------------------------------------- #
TAFEL_BANDS = [(0, 40, "Tafel-limited (<40)", "#E8F5E9"), (40, 120, "Heyrovsky-limited (40-120)", "#FFF8E1"),
               (120, 400, "Volmer-limited (>120)", "#FFEBEE")]


def _tafel_bands(ax):
    lo, hi = ax.get_ylim()
    x0 = ax.get_xlim()[0]
    for a, b, label, color in TAFEL_BANDS:
        ax.axhspan(a, b, color=color, alpha=0.6, zorder=0)
        ax.text(x0, (max(a, lo) + min(b, hi)) / 2, " " + label, ha="left", va="center",
                fontsize=6.5, color="#546E7A", zorder=1)
    ax.set_ylim(lo, hi)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_loc("upper right")


def rhe_view(d: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Keep RHE-referenced records for potential-valued metrics; never convert scales silently."""
    has = d[cols].notna().any(axis=1)
    is_rhe = d["reference_electrode"].astype(str).str.upper().str.contains("RHE")
    k = int((has & ~is_rhe).sum())
    note = f"RHE scale only ({k} records on SCE/Ag-AgCl/NHE omitted)" if k else "RHE scale"
    return d[is_rhe | ~has], note


def _rhe(fn, ax, d, cols, *args, **kw):
    sub, note = rhe_view(d, cols)
    return fn(ax, sub, *args, subtitle=note, **kw)


def _her_oer_specs(rx: str) -> dict[str, tuple[Callable, tuple[float, float]]]:
    eta = r"Overpotential @ 10 mA cm$^{-2}$ (mV)"
    return {
        "01_overpotential_by_class": (lambda ax, d, p: group_strip(ax, d, "overpotential_10_ma_cm2_mv", eta), (6.5, 4.4)),
        "02_tafel_vs_overpotential": (lambda ax, d, p: scatter(
            ax, d, "overpotential_10_ma_cm2_mv", "tafel_slope_mv_dec", eta, "Tafel slope (mV dec$^{-1}$)", diag=_tafel_bands),
            (6.8, 5.0)),
        "03_benchmark_deltas": (lambda ax, d, p: dumbbell(ax, d, "overpotential_10_ma_cm2_mv", eta, better="low"), (6.8, 4.4)),
        "04_overpotential_by_medium": (lambda ax, d, p: group_strip(
            ax, d, "overpotential_10_ma_cm2_mv", eta, by="medium", order=["acidic", "neutral", "alkaline", "other"]), (6.0, 4.4)),
        "05_cdl_vs_overpotential": (lambda ax, d, p: scatter(
            ax, d, "cdl_mf_cm2", "overpotential_10_ma_cm2_mv", r"C$_{dl}$ (mF cm$^{-2}$)", eta, logx=True), (6.0, 4.6)),
        "06_rct_vs_overpotential": (lambda ax, d, p: scatter(
            ax, d, "rct_ohm", "overpotential_10_ma_cm2_mv", r"R$_{ct}$ ($\Omega$)", eta, logx=True), (6.0, 4.6)),
    }


def specs_for(reaction: str, d: pd.DataFrame) -> dict[str, tuple[Callable, tuple[float, float]]]:
    if reaction == "HER":
        s = _her_oer_specs("HER")
        s["07_overall_water_splitting"] = (lambda ax, d, p: group_strip(
            ax, d, "overall_water_splitting_voltage_v", "Cell voltage @ 10 mA cm$^{-2}$ (V)"), (6.0, 4.2))
        return s
    if reaction == "OER":
        s = _her_oer_specs("OER")
        eta10, eta100 = "overpotential_10_ma_cm2_mv", "overpotential_100_ma_cm2_mv"

        def eta_pair(ax, dd, p):
            scatter(ax, dd, eta10, eta100, r"$\eta_{10}$ (mV)", r"$\eta_{100}$ (mV)",
                    title="Current-density scaling: $\\eta_{100}$ vs $\\eta_{10}$")

        s["07_eta100_vs_eta10"] = (eta_pair, (6.0, 4.6))

        def reconstruction(ax, dd, p):
            dd = dd.dropna(subset=[eta10, "surface_reconstruction_observed"])
            if dd.empty:
                _empty(ax)
                return
            dd = dd.assign(flag=dd["surface_reconstruction_observed"].map({True: "Reconstruction\nreported", False: "No reconstruction\nreported"}))
            sns.boxplot(data=dd, x="flag", y=eta10, color="white", fliersize=0, ax=ax, width=0.5)
            sns.stripplot(data=dd, x="flag", y=eta10, hue="material_class", palette=CLASS_COLORS, size=7,
                          edgecolor="black", linewidth=0.5, ax=ax, legend=False)
            ax.set_xlabel("")
            ax.set_ylabel(r"Overpotential @ 10 mA cm$^{-2}$ (mV)")
            ax.set_title(f"Overpotential vs reported surface reconstruction (n={len(dd)})")

        s["08_reconstruction_vs_overpotential"] = (reconstruction, (6.0, 4.6))
        return s
    if reaction == "ORR":
        def e12_onset(ax, dd, p):
            def diag(a):
                lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
                a.plot(lim, lim, "--", color="#9E9E9E", lw=1, zorder=1)
            _rhe(scatter, ax, dd, ["onset_potential_v", "half_wave_potential_v"], "onset_potential_v",
                 "half_wave_potential_v", "Onset potential (V vs RHE)", r"Half-wave potential E$_{1/2}$ (V vs RHE)", diag=diag)

        def n_vs_h2o2(ax, dd, p):
            scatter(ax, dd, "electron_transfer_number", "h2o2_yield_pct", "Electron transfer number n",
                    r"H$_2$O$_2$ yield (%)", title="n vs peroxide yield (line: RRDE relation)",
                    diag=lambda a: a.plot(np.linspace(2, 4, 50), 50 * (4 - np.linspace(2, 4, 50)), "--", color="#9E9E9E", lw=1, zorder=1))

        def zn_air(ax, dd, p):
            z = dd.dropna(subset=["zn_air_peak_power_density_mw_cm2"]).sort_values("zn_air_peak_power_density_mw_cm2")
            if z.empty:
                _empty(ax)
                ax.set_title("Zn-air battery peak power density")
                return
            ax.barh([f"{n[:22]} ({pp})" for n, pp in zip(z["catalyst_name"], z["paper"])],
                    z["zn_air_peak_power_density_mw_cm2"],
                    color=[CLASS_COLORS.get(c, "#9E9E9E") for c in z["material_class"]], edgecolor="black", linewidth=0.5)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel(r"Peak power density (mW cm$^{-2}$)")
            ax.set_title(f"Zn-air battery peak power density (n={len(z)})")

        return {
            "01_e12_vs_onset": (e12_onset, (6.0, 5.0)),
            "02_e12_by_class": (lambda ax, d, p: _rhe(
                group_strip, ax, d, ["half_wave_potential_v"], "half_wave_potential_v",
                r"Half-wave potential E$_{1/2}$ (V vs RHE)"), (6.4, 4.6)),
            "03_n_vs_h2o2_consistency": (n_vs_h2o2, (6.0, 4.8)),
            "04_benchmark_deltas": (lambda ax, d, p: _rhe(
                dumbbell, ax, d, ["half_wave_potential_v"], "half_wave_potential_v",
                r"E$_{1/2}$ (V vs RHE)", better="high"), (6.8, 4.6)),
            "05_limiting_current_vs_e12": (lambda ax, d, p: _rhe(
                scatter, ax, d, ["half_wave_potential_v"], "limiting_current_density_ma_cm2", "half_wave_potential_v",
                r"Limiting current density (mA cm$^{-2}$)", r"E$_{1/2}$ (V vs RHE)"), (6.0, 4.8)),
            "06_tafel_vs_e12": (lambda ax, d, p: _rhe(
                scatter, ax, d, ["half_wave_potential_v"], "tafel_slope_mv_dec", "half_wave_potential_v",
                "Tafel slope (mV dec$^{-1}$)", r"E$_{1/2}$ (V vs RHE)"), (6.0, 4.8)),
            "07_zn_air_power": (zn_air, (6.8, 4.4)),
        }
    if reaction == "N2RR":
        return {
            "01_yield_by_class": (lambda ax, d, p: group_strip(
                ax, d.query("nh3_yield_basis == 'mass'"), "nh3_yield_norm",
                r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$, normalised)", logy=True,
                title="NH$_3$ yield by material class (mass-normalised units)"), (6.4, 4.6)),
            "02_fe_vs_yield": (lambda ax, d, p: scatter(
                ax, d.query("nh3_yield_basis == 'mass'"), "nh3_yield_norm", "nh3_faradaic_efficiency_pct",
                r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$)", "NH$_3$ Faradaic efficiency (%)", logx=True,
                title="Activity-selectivity trade-off: FE vs yield"), (6.4, 4.8)),
            "03_fe_vs_potential": (lambda ax, d, p: _rhe(
                scatter, ax, d, ["potential_v"], "potential_v", "nh3_faradaic_efficiency_pct", "Potential (V vs RHE)",
                "NH$_3$ Faradaic efficiency (%)"), (6.0, 4.8)),
            "04_yield_vs_potential": (lambda ax, d, p: _rhe(
                scatter, ax, d.query("nh3_yield_basis == 'mass'"), ["potential_v"], "potential_v", "nh3_yield_norm",
                "Potential (V vs RHE)", r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$)", logy=True), (6.0, 4.8)),
            "05_experimental_rigor": (lambda ax, d, p: rigor(ax, d, p, [
                ("15N2 isotope labeling", "isotope_15n2_labeling", "true"),
                ("Blank / control experiments", "blank_controls_reported", "true"),
                ("N2H4 by-product tested", "n2h4_detected", "notna"),
                ("NH3 method stated", "nh3_quantification_method", "notna"),
                ("FE reported", "nh3_faradaic_efficiency_pct", "notna")]), (6.2, 3.8)),
            "06_quantification_methods": (method_counts, (6.0, 3.8)),
        }
    if reaction == "NO3RR":
        return {
            "01_fe_vs_potential": (lambda ax, d, p: _rhe(
                scatter, ax, d, ["potential_v"], "potential_v", "nh3_faradaic_efficiency_pct", "Potential (V vs RHE)",
                "NH$_3$ Faradaic efficiency (%)"), (6.0, 4.8)),
            "02_selectivity_vs_conversion": (lambda ax, d, p: scatter(
                ax, d, "nitrate_conversion_pct", "nh3_selectivity_pct", "Nitrate conversion (%)", "NH$_3$ selectivity (%)"), (6.0, 4.6)),
            "03_yield_by_catalyst": (_no3rr_yield, (7.0, 4.8)),
            "04_fe_by_medium": (lambda ax, d, p: group_strip(
                ax, d, "nh3_faradaic_efficiency_pct", "NH$_3$ Faradaic efficiency (%)", by="medium",
                order=["acidic", "neutral", "alkaline", "other"]), (5.8, 4.4)),
            "05_experimental_rigor": (lambda ax, d, p: rigor(ax, d, p, [
                ("15NO3- isotope labeling", "isotope_15n_labeling", "true"),
                ("NH3 method stated", "nh3_quantification_method", "notna"),
                ("Nitrate conversion reported", "nitrate_conversion_pct", "notna"),
                ("FE reported", "nh3_faradaic_efficiency_pct", "notna"),
                ("By-products (NO2-/N2) reported", "nitrite_selectivity_or_fe_pct", "notna")]), (6.2, 3.8)),
            "06_quantification_methods": (method_counts, (6.0, 3.8)),
        }
    return {}


def _no3rr_yield(ax, d, p):
    y = d.dropna(subset=["nh3_yield_norm"]).copy()
    y = y[y["nh3_yield_basis"].isin(["mass", "area"])]
    if y.empty:
        _empty(ax)
        return
    y["label"] = [f"{n[:20]} ({pp})" for n, pp in zip(y["catalyst_name"], y["paper"])]
    y = y.sort_values(["nh3_yield_basis", "nh3_yield_norm"])
    colors = y["nh3_yield_basis"].map({"mass": "#1976D2", "area": "#F57C00"})
    ax.barh(y["label"], y["nh3_yield_norm"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel(r"NH$_3$ yield (log; $\mu$g h$^{-1}$ mg$^{-1}$ or $\mu$g h$^{-1}$ cm$^{-2}$)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color="#1976D2", label="mass basis ($\\mu$g h$^{-1}$ mg$^{-1}$)"),
                       Line2D([], [], marker="s", ls="", color="#F57C00", label="area basis ($\\mu$g h$^{-1}$ cm$^{-2}$)")],
              fontsize=7, loc="lower right")
    ax.set_title(f"NH$_3$ yield rate, unit-normalised (n={len(y)})")


# --------------------------------------------------------------------------- #
# Per-reaction driver
# --------------------------------------------------------------------------- #
def common_specs(reaction: str) -> dict[str, tuple[Callable, tuple[float, float]]]:
    return {
        "C1_composition": (composition, (6.4, 2.8)),
        "C2_metric_completeness": (lambda ax, d, p: completeness(ax, d, p, reaction), (6.4, 4.2)),
        "C3_node_metals": (metals, (6.0, 3.8)),
        "C4_medium": (medium_bar, (4.8, 3.8)),
        "C5_electrolytes": (electrolytes, (6.4, 3.8)),
        "C6_stability": (stability, (6.4, 4.4)),
    }


def make_reaction_figures(reaction: str) -> list[Path]:
    rec, pap = load(reaction)
    specs = specs_for(reaction, rec)
    common = common_specs(reaction)
    paths: list[Path] = []

    for name, (fn, size) in {**specs, **common}.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, rec, pap)
        fig.tight_layout()
        paths.append(_save(fig, reaction, name, rec))

    # Six-panel overview: composition, completeness + first four reaction-specific plots.
    panel_fns = [common["C1_composition"][0], common["C2_metric_completeness"][0]] + [v[0] for v in list(specs.values())[:4]]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, fn in zip(axes.ravel(), panel_fns):
        fn(ax, rec, pap)
    fig.suptitle(f"{reaction}: full-text extraction overview", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    paths.append(_save(fig, reaction, "00_overview", rec))
    return paths


# --------------------------------------------------------------------------- #
# Cross-reaction EDA
# --------------------------------------------------------------------------- #
def make_cross_reaction_figures() -> list[Path]:
    data = {rx: load(rx) for rx in REACTIONS}
    rec_all = pd.concat([d[0].assign(reaction=rx) for rx, d in data.items()], ignore_index=True)
    pap_all = pd.concat([d[1].assign(reaction=rx) for rx, d in data.items()], ignore_index=True)
    out = FIG_ROOT / "cross_reaction"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name):
        fig.text(0.995, 0.004, f"Pilot: {len(rec_all)} records from {rec_all['doi'].nunique()} Elsevier papers",
                 ha="right", va="bottom", fontsize=6.5, color="#757575")
        p = out / f"X_{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 1. records per reaction by role
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ct = pd.crosstab(rec_all["reaction"], rec_all["role"]).reindex(index=REACTIONS, columns=ROLE_ORDER, fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#1976D2", "#90CAF9", "#FFB300"], rot=0, edgecolor="white")
    for i, tot in enumerate(ct.sum(axis=1)):
        ax.text(i, tot + 0.5, int(tot), ha="center", fontsize=9)
    ax.set_ylabel("Records")
    ax.set_xlabel("")
    ax.set_title("Extracted records per reaction (10 papers each)")
    fig.tight_layout()
    save(fig, "01_records_per_reaction")

    # 2-3. material class and medium by reaction (100% stacked)
    for col, palette, labels, name, title in [
        ("material_class", CLASS_COLORS, CLASS_LABELS, "02_material_class_by_reaction", "Material class by reaction"),
        ("medium", MEDIUM_COLORS, {k: k for k in MEDIUM_COLORS}, "03_medium_by_reaction", "Electrolyte medium by reaction"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4))
        sub = rec_all[rec_all["role"] != "benchmark"]
        ct = pd.crosstab(sub["reaction"], sub[col], normalize="index").reindex(index=REACTIONS) * 100
        ct = ct[[c for c in palette if c in ct.columns]]
        ct.plot(kind="bar", stacked=True, ax=ax, color=[palette[c] for c in ct.columns], rot=0, edgecolor="white")
        ax.set_ylabel("% of records")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.legend([labels[c] for c in ct.columns], fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save(fig, name)

    # 4. node metals x reaction heatmap (papers)
    m = rec_all[rec_all["role"] != "benchmark"].dropna(subset=["node_metals"]).copy()
    m["metal"] = m["node_metals"].str.split("|")
    m = m.explode("metal").reset_index(drop=True)
    m["metal"] = m["metal"].str.strip()
    m = m.drop_duplicates(["doi", "metal"])
    hm = pd.crosstab(m["metal"], m["reaction"]).reindex(columns=REACTIONS, fill_value=0)
    hm = hm.loc[hm.sum(axis=1).sort_values(ascending=False).head(14).index]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sns.heatmap(hm, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Papers"}, ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Framework node metals by reaction")
    fig.tight_layout()
    save(fig, "04_metals_by_reaction")

    # 5. reporting of common fields (% of non-benchmark records)
    fields = ["electrolyte", "reference_electrode", "substrate_or_electrode", "cell_type",
              "stability_duration_h", "stability_retention_pct", "evidence_quote"]
    sub = rec_all[rec_all["role"] != "benchmark"]
    cov = pd.DataFrame({rx: [100 * sub.loc[sub["reaction"] == rx, f].notna().mean() for f in fields] for rx in REACTIONS},
                       index=[f.replace("_", " ") for f in fields])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.heatmap(cov, annot=True, fmt=".0f", cmap="Greens", vmin=0, vmax=100, linewidths=0.5,
                cbar_kws={"label": "% of records"}, ax=ax)
    ax.set_title("Reporting of experimental context")
    ax.set_xlabel("")
    fig.tight_layout()
    save(fig, "05_context_reporting")

    # 6. where numbers came from
    fig, ax = plt.subplots(figsize=(6.4, 4))
    src = rec_all["source_location"].fillna("unknown").str.lower().str.extract(r"(text|table|figure|abstract)")[0].fillna("other")
    ct = pd.crosstab(rec_all["reaction"], src, normalize="index").reindex(index=REACTIONS) * 100
    ct.plot(kind="bar", stacked=True, ax=ax, rot=0, colormap="Set2", edgecolor="white")
    ax.set_ylabel("% of records")
    ax.set_xlabel("")
    ax.set_title("Source of extracted values")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    save(fig, "06_value_source")

    # 7. text volume and tables per paper
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=pap_all, x="reaction", y="n_chars_full", order=REACTIONS, hue="reaction", palette=RXN_COLORS,
                legend=False, ax=axes[0], fliersize=0)
    sns.stripplot(data=pap_all, x="reaction", y="n_chars_full", order=REACTIONS, color="black", size=4, ax=axes[0])
    axes[0].set_ylabel("Full-text characters")
    axes[0].set_xlabel("")
    axes[0].set_title("Full-text length per paper")
    tab = pap_all.groupby("reaction")["n_tables"].apply(lambda s: s.fillna(0).gt(0).mean() * 100).reindex(REACTIONS)
    axes[1].bar(tab.index, tab.values, color=[RXN_COLORS[r] for r in tab.index])
    axes[1].set_ylabel("% of papers with parsed tables")
    axes[1].set_title("Papers with at least one table in the text")
    fig.tight_layout()
    save(fig, "07_text_volume_and_tables")

    # 8. papers flagged non-experimental / no records
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    no_rec = pap_all.assign(empty=pap_all["n_records"].fillna(0) == 0).groupby("reaction")["empty"].sum().reindex(REACTIONS)
    ax.bar(no_rec.index, no_rec.values, color=[RXN_COLORS[r] for r in no_rec.index])
    ax.set_ylabel("Papers with no extractable records")
    ax.set_title("Off-topic / theory-only papers returned by the search")
    ax.set_yticks(range(0, int(no_rec.max()) + 2))
    fig.tight_layout()
    save(fig, "08_papers_without_records")
    return paths

