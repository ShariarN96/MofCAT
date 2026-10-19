"""Scientific EDA figures (catalyst comparisons) from the Elsevier full-text extractions.

Per reaction (written to figures/elsevier_pilot/science/<RXN>/):
  S1  catalyst ranking on the headline metric (with benchmark reference)
  S2  activity trade-off / kinetics scatter with labelled leaders
  S3  structure-activity: metal count, noble vs earth-abundant, material class, medium
  S4  within-paper controls (ablation) and, for N2RR/NO3RR, potential dependence
  S5  multi-metric scorecard of the featured catalysts
  Fig_composite  2 x 2 summary panel
plus <RXN>_catalyst_table.csv with the exact values plotted.

Conventions
  * Values are as reported by the authors (LLM-extracted from full text). Nothing is
    converted between potential scales; potential-dependent plots use RHE-referenced records.
  * One point per catalyst: the best value per (paper, catalyst, medium) for HER/OER/ORR,
    per (paper, catalyst) for N2RR/NO3RR.
  * Papers are cited as "Author 'YY" using the Scopus creator field.
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
from adjustText import adjust_text  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..paths import SCOPUS_XLSX  # noqa: E402
from .figures import FIG_ROOT, REACTIONS, load  # noqa: E402

SCI_ROOT = FIG_ROOT / "science"
XLSX = SCOPUS_XLSX

CLASS_C = {"pristine_mof": "#009E73", "mof_derived": "#E69F00", "mof_composite": "#0072B2",
           "non_mof": "#9A9A9A", "unclear": "#CCCCCC"}
CLASS_L = {"pristine_mof": "Pristine MOF", "mof_derived": "MOF-derived", "mof_composite": "MOF composite",
           "non_mof": "Non-MOF", "unclear": "Unclear"}
MED_MARK = {"acidic": "^", "alkaline": "o", "neutral": "s", "other": "D"}
MED_ABBR = {"acidic": "acid", "alkaline": "alk", "neutral": "neut", "other": ""}
NOBLE = {"Pt", "Pd", "Ru", "Ir", "Rh", "Au", "Ag", "Os"}

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E9E9E9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "figure.dpi": 110, "mathtext.default": "regular",
})

_SUB = str.maketrans("0123456789+-x", "₀₁₂₃₄₅₆₇₈₉₊₋ₓ")
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


# --------------------------------------------------------------------------- #
# Labels and data preparation
# --------------------------------------------------------------------------- #
def pretty(name: str, maxlen: int = 30) -> str:
    s = str(name)
    s = re.sub(r"_([0-9x]+)", lambda m: m.group(1).translate(_SUB), s)
    s = re.sub(r"_\{?([A-Za-z]+)\}?", r"\1", s)
    s = re.sub(r"\^([0-9+\-−]+)", lambda m: m.group(1).replace("−", "-").translate(_SUP), s)
    s = s.replace("^", "").replace("  ", " ")
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


_TAGS: dict[str, str] | None = None


def paper_tags() -> dict[str, str]:
    """doi -> "Surname 'YY" from the Scopus workbook (suffix a/b if ambiguous)."""
    global _TAGS
    if _TAGS is None:
        df = pd.read_excel(XLSX, sheet_name="ALL", usecols=["doi", "creator", "year"]).dropna(subset=["doi"])
        tags: dict[str, str] = {}
        seen: dict[str, int] = {}
        for doi, creator, year in zip(df["doi"].str.lower(), df["creator"].fillna(""), df["year"]):
            sur = str(creator).split()[0].rstrip(",") if str(creator).strip() else "n.a."
            base = f"{sur} ’{str(int(year))[-2:]}"
            n = seen.get(base, 0)
            seen[base] = n + 1
            tags.setdefault(doi, base if n == 0 else f"{base}{chr(96 + n + 1)}")
        _TAGS = tags
    return _TAGS


def prep(reaction: str) -> pd.DataFrame:
    rec, _ = load(reaction)
    tags = paper_tags()
    rec["tag"] = rec["doi"].str.lower().map(tags).fillna(rec["paper"])
    rec["name"] = rec["catalyst_name"].map(pretty)
    metals = rec["node_metals"].fillna("").map(lambda s: [m for m in s.split("|") if m])
    rec["n_metals"] = metals.map(len)
    rec["metal_group"] = rec["n_metals"].map(lambda n: "1 metal" if n == 1 else "2 metals" if n == 2 else "≥3 metals" if n >= 3 else "unspecified")
    rec["noble"] = metals.map(lambda ms: "Contains noble metal" if any(m in NOBLE for m in ms) else "Earth-abundant only")
    rec.loc[rec["n_metals"] == 0, "noble"] = "unspecified"
    rec["medium_abbr"] = rec["medium"].map(MED_ABBR).fillna("")
    if "surface_reconstruction_observed" in rec:
        rec["recon"] = rec["surface_reconstruction_observed"].map({True: "Reconstruction reported", False: "None reported"})
    return rec


def is_rhe(rec: pd.DataFrame) -> pd.Series:
    return rec["reference_electrode"].astype(str).str.upper().str.contains("RHE")


def best_per(rec: pd.DataFrame, metric: str, better: str, keys=("doi", "catalyst_name", "medium")) -> pd.DataFrame:
    d = rec[rec["role"] != "benchmark"].dropna(subset=[metric])
    if d.empty:
        return d.copy()
    grp = d.groupby(list(keys))[metric]
    idx = grp.idxmin() if better == "low" else grp.idxmax()
    out = d.loc[idx].copy()
    multi_med = out["medium"].nunique() > 1
    out["ylabel"] = [f"{n} ({t})" + (f" [{m}]" if multi_med and m else "") for n, t, m in zip(out["name"], out["tag"], out["medium_abbr"])]
    return out


def benchmarks(rec: pd.DataFrame, metric: str) -> pd.DataFrame:
    return rec[(rec["role"] == "benchmark")].dropna(subset=[metric])


# --------------------------------------------------------------------------- #
# Generic panels (each draws on one Axes)
# --------------------------------------------------------------------------- #
def _nodata(ax, msg="Insufficient reported data\nin this pilot sample"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#9A9A9A", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _class_legend(ax, df, extra=()):
    """Legend below the axes so it never covers bars."""
    h = [Patch(facecolor=CLASS_C[c], edgecolor="black", linewidth=0.5, label=CLASS_L[c])
         for c in CLASS_C if c in set(df["material_class"])]
    h += list(extra)
    if h:
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=len(h), frameon=False,
                  columnspacing=1.2, handlelength=1.2)


def _panel_label(ax, letter):
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def ranking(ax, best, bench, metric, xlabel, better="low", annotate: Callable | None = None,
            logx=False, maxn=14, title=None, ref_name="Pt/C"):
    d = best.sort_values(metric, ascending=(better == "low")).head(maxn)
    if d.empty:
        _nodata(ax)
        ax.set_title(title or "")
        return d
    y = np.arange(len(d))
    bars = ax.barh(y, d[metric], color=[CLASS_C.get(c, "#999") for c in d["material_class"]],
                   edgecolor="black", linewidth=0.5, height=0.72)
    for b, role in zip(bars, d["role"]):
        b.set_alpha(1.0 if role == "main" else 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["ylabel"], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(left=d[metric].min() / 2.5, right=d[metric].max() * 6)
    else:
        ax.margins(x=0.22)
    if annotate:
        for yi, (_, r) in zip(y, d.iterrows()):
            t = annotate(r)
            if t:
                ax.annotate(t, (r[metric], yi), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=6.3, color="#444444")
    if bench is not None and logx:
        bench = bench[bench[metric] > 0]
    if bench is not None and len(bench):
        lo, hi = bench[metric].min(), bench[metric].max()
        if hi > lo:
            ax.axvspan(lo, hi, color="#BDBDBD", alpha=0.35, zorder=0)
        ax.axvline(bench[metric].median(), color="#555555", ls="--", lw=1, zorder=1)
        ax.text(bench[metric].median(), 1.01, f"{ref_name} (n={len(bench)})", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#444444")
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"Catalyst ranking (n={len(d)} of {len(best)})", pad=14 if bench is not None and len(bench) else 6)
    _class_legend(ax, d, extra=[Patch(facecolor="white", edgecolor="black", alpha=0.5, label="lighter = variant / control")])
    return d


def scatter_labeled(ax, best, bench, x, y, xlabel, ylabel, *, logx=False, logy=False, label_n=7, score=None,
                    bands=None, pareto=None, title=None, ref_name="Pt/C"):
    d = best.dropna(subset=[x, y])
    if logx:
        d = d[d[x] > 0]
    if logy:
        d = d[d[y] > 0]
    if len(d) < 2:
        _nodata(ax)
        ax.set_title(title or "")
        return
    if bands:
        for lo, hi, lab, col in bands:
            ax.axhspan(lo, hi, color=col, alpha=0.55, zorder=0)
            ax.text(0.99, (lo + hi) / 2, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.3, color="#6B7B83")
    for med, g in d.groupby("medium"):
        ax.scatter(g[x], g[y], s=[85 if r == "main" else 42 for r in g["role"]], marker=MED_MARK.get(med, "o"),
                   c=[CLASS_C.get(c, "#999") for c in g["material_class"]], edgecolor="black", linewidth=0.7,
                   alpha=0.92, zorder=3)
    if bench is not None and len(bench.dropna(subset=[x, y])):
        b = bench.dropna(subset=[x, y])
        ax.scatter(b[x], b[y], marker="*", s=170, c="#FFD54F", edgecolor="black", linewidth=0.8, zorder=4)
    if pareto:
        px, py = pareto  # "max"/"min" directions
        pts = d[[x, y]].sort_values(x, ascending=(px == "min")).values
        front, best_y = [], None
        for xv, yv in pts:
            if best_y is None or (yv > best_y if py == "max" else yv < best_y):
                front.append((xv, yv))
                best_y = yv
        if len(front) >= 2:
            fx, fy = zip(*front)
            ax.step(fx, fy, where="post", color="#B71C1C", lw=1.1, ls="--", zorder=2, label="Pareto front")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # label the leaders (and benchmarks) with collision avoidance
    key = score if score is not None else (lambda r: -r[y])
    lead = d.assign(_s=d.apply(key, axis=1)).sort_values("_s").drop_duplicates("name").head(label_n)
    texts = [ax.text(r[x], r[y], r["name"], fontsize=6.5, zorder=5) for _, r in lead.iterrows()]
    if bench is not None and len(bench.dropna(subset=[x, y])):
        b = bench.dropna(subset=[x, y]).drop_duplicates("name")
        texts += [ax.text(r[x], r[y], r["name"], fontsize=6.5, style="italic", color="#6D5800", zorder=5) for _, r in b.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5), expand=(1.2, 1.4))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel.split('(')[0].strip()} vs {xlabel.split('(')[0].strip()} (n={len(d)})")
    handles = [Line2D([], [], marker=MED_MARK[m], ls="", color="#444444", markerfacecolor="white", label=m)
               for m in MED_MARK if m in set(d["medium"])]
    handles += [Patch(facecolor=CLASS_C[c], edgecolor="black", linewidth=0.5, label=CLASS_L[c]) for c in CLASS_C if c in set(d["material_class"])]
    if bench is not None and len(bench.dropna(subset=[x, y])):
        handles.append(Line2D([], [], marker="*", ls="", color="#FFD54F", markeredgecolor="black", markersize=10, label=ref_name))
    ax.legend(handles=handles, loc="best", framealpha=0.9, edgecolor="#DDDDDD", ncol=1)


def compare_groups(ax, best, metric, group, order, ylabel, better="low", logy=False, title=None):
    d = best.dropna(subset=[metric])
    d = d[d[group].isin(order)]
    if logy:
        d = d[d[metric] > 0]
    order = [o for o in order if o in set(d[group])]
    if len(d) < 3 or len(order) < 2:
        _nodata(ax)
        ax.set_title(title or "")
        return
    rng = np.random.default_rng(3)
    for i, o in enumerate(order):
        g = d[d[group] == o]
        v = g[metric].values
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#F4F4F4", edgecolor="#555555", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.8), whiskerprops=dict(color="#555555"),
                   capprops=dict(color="#555555"), zorder=1)
        xs = i + rng.uniform(-0.17, 0.17, len(g))
        ax.scatter(xs, v, s=[62 if r == "main" else 34 for r in g["role"]],
                   c=[CLASS_C.get(c, "#999") for c in g["material_class"]], edgecolor="black", linewidth=0.6, zorder=3,
                   marker="o")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{o}\n(n={int((d[group] == o).sum())})" for o in order])
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_title(title or "")
    ax.text(0.02, 0.97, "median = bold line", transform=ax.transAxes, fontsize=6.5, color="#777777", va="top")


def ablation(ax, rec, metric, ylabel, better="low", logy=False, title=None, mask=None):
    d = rec.dropna(subset=[metric]).copy()
    if mask is not None:
        d = d[mask.reindex(d.index, fill_value=False)]
    if logy:
        d = d[d[metric] > 0]
    groups = [(k, g) for k, g in d.groupby(["doi", "medium"]) if g["catalyst_name"].nunique() >= 2]
    if not groups:
        _nodata(ax, "No paper reports the metric for\nboth the featured catalyst and a control")
        ax.set_title(title or "")
        return
    groups.sort(key=lambda kg: (kg[1].loc[kg[1]["role"] == "main", metric].min() if (kg[1]["role"] == "main").any() else np.inf)
                * (1 if better == "low" else -1))
    for i, (_, g) in enumerate(groups):
        g = g.drop_duplicates("catalyst_name")
        ax.plot([i, i], [g[metric].min(), g[metric].max()], color="#BBBBBB", lw=1.4, zorder=1)
        for _, r in g.iterrows():
            if r["role"] == "benchmark":
                ax.scatter(i, r[metric], marker="*", s=150, c="#FFD54F", edgecolor="black", linewidth=0.7, zorder=4)
            elif r["role"] == "main":
                ax.scatter(i, r[metric], s=75, c=CLASS_C.get(r["material_class"], "#999"), edgecolor="black", linewidth=1.0, zorder=5)
            else:
                ax.scatter(i, r[metric], s=40, facecolor="white", edgecolor="#555555", linewidth=1.0, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g['tag'].iloc[0]}" + (f"\n{MED_ABBR.get(k[1], '')}" if d['medium'].nunique() > 1 else "")
                        for k, g in groups], rotation=45, ha="right", fontsize=7)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_title(title or f"Featured catalyst vs controls within each paper (n={len(groups)} papers)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="#0072B2", markeredgecolor="black", label="featured (colour = class)"),
                       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", label="variant / control"),
                       Line2D([], [], marker="*", ls="", color="#FFD54F", markeredgecolor="black", markersize=10, label="benchmark")],
              loc="best", framealpha=0.9, edgecolor="#DDDDDD")


def potential_lines(ax, rec, ycol, ylabel, logy=False, title=None, min_points=2):
    d = rec[(rec["role"] != "benchmark")].dropna(subset=["potential_v", ycol])
    n_all = d["catalyst_name"].nunique()
    d = d[is_rhe(d)]
    if logy:
        d = d[d[ycol] > 0]
    series = [(k, g.sort_values("potential_v")) for k, g in d.groupby(["doi", "catalyst_name"]) if g["potential_v"].nunique() >= min_points]
    if not series:
        _nodata(ax, "No catalyst reported at ≥2 RHE potentials")
        ax.set_title(title or "")
        return
    cmap = plt.get_cmap("tab10")
    texts = []
    for i, ((doi, _), g) in enumerate(series):
        g = g.drop_duplicates("potential_v")
        ax.plot(g["potential_v"], g[ycol], "-o", color=cmap(i % 10), lw=1.4, ms=5, mec="black", mew=0.5)
        r = g.iloc[-1]
        texts.append(ax.text(r["potential_v"], r[ycol], f"{r['name']} ({r['tag']})", fontsize=6.5, color=cmap(i % 10)))
    if logy:
        ax.set_yscale("log")
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4))
    ax.set_xlabel("Potential (V vs RHE)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Potential dependence (RHE only; n={len(series)} catalysts)")


def scorecard(ax, df, specs, title=None, maxrows=14):
    """specs: (col, header, better, fmt) with better in {'low','high','flag'}."""
    cols = [s for s in specs if s[0] in df.columns]
    d = df.copy()
    d = d[d[[c[0] for c in cols]].notna().sum(axis=1) >= 2]
    if d.empty:
        _nodata(ax)
        ax.set_title(title or "")
        return
    first = cols[0]
    d = d.sort_values(first[0], ascending=(first[2] == "low")).head(maxrows)
    M = np.full((len(d), len(cols)), np.nan)
    for j, (col, _, better, _) in enumerate(cols):
        if better == "flag":
            continue
        v = d[col].astype(float)
        n = v.notna().sum()
        if n == 0:
            continue
        r = v.rank(method="average")
        sc = (r - 1) / (n - 1) if n > 1 else pd.Series(1.0, index=v.index)
        M[:, j] = (1 - sc if better == "low" else sc).where(v.notna()).values
    ax.imshow(np.ma.masked_invalid(M), cmap="YlGnBu", vmin=-0.1, vmax=1.15, aspect="auto")
    for i, (_, r) in enumerate(d.iterrows()):
        for j, (col, _, better, fmt) in enumerate(cols):
            val = r[col]
            if pd.isna(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F2F2F2", edgecolor="white"))
                ax.text(j, i, "–", ha="center", va="center", color="#B0B0B0", fontsize=8)
            elif better == "flag":
                ok = bool(val)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#D5EBDD" if ok else "#F3D6D6", edgecolor="white"))
                ax.text(j, i, "✓" if ok else "✗", ha="center", va="center", fontsize=9, color="#2E7D32" if ok else "#C62828")
            else:
                shade = M[i, j]
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7,
                        color="white" if shade > 0.62 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["ylabel"] if "ylabel" in d else d["name"], fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title((title or "Performance scorecard") + "\n(darker = better within column; values as reported)", pad=34, fontsize=9)


# --------------------------------------------------------------------------- #
# Reaction definitions
# --------------------------------------------------------------------------- #
def _footer(fig, rec, extra=""):
    fig.text(0.995, 0.003, f"Pilot sample: {rec['doi'].nunique()} Elsevier papers; values as reported by the authors "
                           f"(LLM extraction from full text). {extra}", ha="right", va="bottom", fontsize=6, color="#8A8A8A")


def _save(fig, rx, name, rec, extra=""):
    out = SCI_ROOT / rx
    out.mkdir(parents=True, exist_ok=True)
    _footer(fig, rec, extra)
    p = out / f"{rx}_{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _rhe_note(rec, cols):
    has = rec[cols].notna().any(axis=1) & (rec["role"] != "benchmark")
    k = int((has & ~is_rhe(rec)).sum())
    return f"RHE-referenced records only; {k} on other scales omitted." if k else "RHE scale."


TAFEL_BANDS = [(0, 40, "Tafel step (<40)", "#E3F2E7"), (40, 120, "Heyrovsky step (40–120)", "#FFF4D6"),
               (120, 300, "Volmer step (>120)", "#FCE4E4")]


def _hero_overpotential(rx: str, rec: pd.DataFrame):
    m = "overpotential_10_ma_cm2_mv"
    best = best_per(rec, m, "low")
    bench = benchmarks(rec, m)
    ref = "Pt/C" if rx == "HER" else "RuO\u2082/IrO\u2082"
    eta = "η$_{10}$ (mV)"
    xl = f"Overpotential at 10 mA cm$^{{-2}}$, {eta} — lower is better"

    def ann(r):
        return f"b = {r['tafel_slope_mv_dec']:.0f}" if pd.notna(r["tafel_slope_mv_dec"]) else ""

    panels = {
        "rank": lambda ax: ranking(ax, best, bench, m, xl, "low", ann, ref_name=ref, title=f"{rx} catalyst ranking (b = Tafel slope, mV dec$^{{-1}}$)"),
        "trade": lambda ax: scatter_labeled(
            ax, best, bench, m, "tafel_slope_mv_dec", "Overpotential at 10 mA cm$^{-2}$ (mV)", "Tafel slope (mV dec$^{-1}$)",
            label_n=8, score=lambda r: r[m], bands=TAFEL_BANDS if rx == "HER" else None, pareto=("min", "min"),
            title="Activity–kinetics map (lower-left is better)", ref_name=ref),
        "metals": lambda ax: compare_groups(ax, best, m, "metal_group", ["1 metal", "2 metals", "≥3 metals"],
                                            "η$_{10}$ (mV)", title="Number of metals"),
        "noble": lambda ax: compare_groups(ax, best, m, "noble", ["Earth-abundant only", "Contains noble metal"],
                                           "η$_{10}$ (mV)", title="Noble vs earth-abundant"),
        "class": lambda ax: compare_groups(ax, best, m, "material_class", ["pristine_mof", "mof_derived", "mof_composite"],
                                           "η$_{10}$ (mV)", title="Material class"),
        "medium": lambda ax: compare_groups(ax, best, m, "medium", ["acidic", "neutral", "alkaline"],
                                            "η$_{10}$ (mV)", title="Electrolyte medium"),
        "cdl": lambda ax: scatter_labeled(ax, best, None, "cdl_mf_cm2", m, "C$_{dl}$ (mF cm$^{-2}$)", "η$_{10}$ (mV)", logx=True,
                                          label_n=6, score=lambda r: r[m], title="Activity vs double-layer capacitance"),
        "rct": lambda ax: scatter_labeled(ax, best, None, "rct_ohm", m, "R$_{ct}$ (Ω)", "η$_{10}$ (mV)", logx=True,
                                          label_n=6, score=lambda r: r[m], title="Activity vs charge-transfer resistance"),
        "abl": lambda ax: ablation(ax, rec, m, "η$_{10}$ (mV)", "low"),
        "score": lambda ax: scorecard(ax, best[best["role"] == "main"], [
            (m, "η$_{10}$\n(mV)", "low", "{:.0f}"), ("tafel_slope_mv_dec", "Tafel\n(mV/dec)", "low", "{:.0f}"),
            ("cdl_mf_cm2", "C$_{dl}$\n(mF/cm²)", "high", "{:.1f}"), ("rct_ohm", "R$_{ct}$\n(Ω)", "low", "{:.1f}"),
            ("stability_duration_h", "Stability\n(h)", "high", "{:.0f}")], title=f"{rx}: featured catalysts"),
    }
    if rx == "OER":
        panels["recon"] = lambda ax: compare_groups(ax, best, m, "recon", ["None reported", "Reconstruction reported"],
                                                    "η$_{10}$ (mV)", title="Surface reconstruction")
    table = best.drop(columns=[c for c in ("paper",) if c in best])
    return panels, table


def _hero_orr(rec: pd.DataFrame):
    rhe = rec[is_rhe(rec)]
    m = "half_wave_potential_v"
    best = best_per(rhe, m, "high")
    bench = benchmarks(rhe, m)
    note = _rhe_note(rec, ["half_wave_potential_v", "onset_potential_v"])

    def rank_ax(ax):
        d = best.sort_values(m, ascending=False).head(14)
        if d.empty:
            _nodata(ax)
            return
        y = np.arange(len(d))
        ax.hlines(y, d["onset_potential_v"].fillna(d[m]), d[m], color="#B0B0B0", lw=2, zorder=1)
        ax.scatter(d["onset_potential_v"], y, marker="D", s=26, c="white", edgecolor="#555555", zorder=3, label="Onset potential")
        ax.scatter(d[m], y, s=[80 if r == "main" else 46 for r in d["role"]], c=[CLASS_C.get(c, "#999") for c in d["material_class"]],
                   edgecolor="black", linewidth=0.7, zorder=4, label="Half-wave potential E$_{1/2}$")
        ax.set_yticks(y)
        ax.set_yticklabels(d["ylabel"], fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for yi, (_, r) in zip(y, d.iterrows()):
            if pd.notna(r["electron_transfer_number"]):
                ax.annotate(f"n = {r['electron_transfer_number']:.2f}", (max(r[m], r["onset_potential_v"] if pd.notna(r["onset_potential_v"]) else r[m]), yi),
                            xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.3, color="#444444")
        if len(bench):
            ax.axvline(bench[m].median(), color="#555555", ls="--", lw=1)
            ax.text(bench[m].median(), 1.01, f"Pt/C (n={len(bench)})", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="#444444")
        ax.margins(x=0.12)
        ax.set_xlabel("Potential (V vs RHE) — higher is better")
        ax.set_title(f"ORR catalyst ranking (n={len(d)})", pad=14)
        h = [Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="#555555", label="Onset"),
             Line2D([], [], marker="o", ls="", markerfacecolor="#999999", markeredgecolor="black", label="E$_{1/2}$ (colour = class)")]
        hh = h + [Patch(facecolor=CLASS_C[c], edgecolor="black", label=CLASS_L[c]) for c in CLASS_C if c in set(d["material_class"])]
        ax.legend(handles=hh, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=len(hh), frameon=False, columnspacing=1.2)

    def zn_ax(ax):
        z = best_per(rec, "zn_air_peak_power_density_mw_cm2", "high")
        z = z.sort_values("zn_air_peak_power_density_mw_cm2", ascending=False)
        ranking(ax, z, None, "zn_air_peak_power_density_mw_cm2", "Peak power density (mW cm$^{-2}$) — higher is better", "high",
                lambda r: f"{r['zn_air_specific_capacity_mah_g']:.0f} mAh g⁻¹" if pd.notna(r["zn_air_specific_capacity_mah_g"]) else "",
                title="Zn–air battery performance")

    panels = {
        "rank": rank_ax,
        "trade": lambda ax: scatter_labeled(ax, best, bench, "limiting_current_density_ma_cm2", m, "Limiting current density (mA cm$^{-2}$)",
                                            "E$_{1/2}$ (V vs RHE)", label_n=7, score=lambda r: -r[m], pareto=("max", "max"),
                                            title="Half-wave potential vs limiting current (upper-right is better)"),
        "kin": lambda ax: scatter_labeled(ax, best, bench, "tafel_slope_mv_dec", m, "Tafel slope (mV dec$^{-1}$)", "E$_{1/2}$ (V vs RHE)",
                                          label_n=6, score=lambda r: -r[m], title="Kinetics vs half-wave potential"),
        "metals": lambda ax: compare_groups(ax, best, m, "metal_group", ["1 metal", "2 metals", "≥3 metals"], "E$_{1/2}$ (V vs RHE)",
                                            better="high", title="Number of metals"),
        "noble": lambda ax: compare_groups(ax, best, m, "noble", ["Earth-abundant only", "Contains noble metal"], "E$_{1/2}$ (V vs RHE)",
                                           better="high", title="Noble vs earth-abundant"),
        "class": lambda ax: compare_groups(ax, best, m, "material_class", ["mof_derived", "mof_composite", "pristine_mof"], "E$_{1/2}$ (V vs RHE)",
                                           better="high", title="Material class"),
        "n_h2o2": lambda ax: scatter_labeled(ax, best, None, "electron_transfer_number", "h2o2_yield_pct", "Electron transfer number n",
                                             "H$_2$O$_2$ yield (%)", label_n=5, score=lambda r: r["h2o2_yield_pct"],
                                             title="Four-electron selectivity"),
        "zn": zn_ax,
        "abl": lambda ax: ablation(ax, rec, m, "E$_{1/2}$ (V vs RHE)", "high", mask=is_rhe(rec)),
        "score": lambda ax: scorecard(ax, best[best["role"] == "main"], [
            (m, "E$_{1/2}$\n(V)", "high", "{:.3f}"), ("onset_potential_v", "Onset\n(V)", "high", "{:.3f}"),
            ("electron_transfer_number", "n", "high", "{:.2f}"), ("limiting_current_density_ma_cm2", "j$_L$\n(mA/cm²)", "high", "{:.1f}"),
            ("tafel_slope_mv_dec", "Tafel\n(mV/dec)", "low", "{:.0f}"), ("zn_air_peak_power_density_mw_cm2", "Zn–air\n(mW/cm²)", "high", "{:.0f}")],
            title="ORR: featured catalysts (RHE)"),
    }
    return panels, best, note


def _hero_ammonia(rx: str, rec: pd.DataFrame):
    mass = rec[(rec["nh3_yield_basis"] == "mass")]
    area = rec[(rec["nh3_yield_basis"] == "area")]
    ymass = best_per(mass, "nh3_yield_norm", "high", keys=("doi", "catalyst_name"))
    yarea = best_per(area, "nh3_yield_norm", "high", keys=("doi", "catalyst_name"))
    fe = best_per(rec, "nh3_faradaic_efficiency_pct", "high", keys=("doi", "catalyst_name"))
    bm = benchmarks(mass, "nh3_yield_norm")
    ylab_mass = r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$) — higher is better"
    ylab_area = r"NH$_3$ yield ($\mu$g h$^{-1}$ cm$^{-2}$) — higher is better"

    def at(r):
        pot = f" @ {r['potential_v']:.2f} V" if pd.notna(r["potential_v"]) else ""
        ref = f" {r['reference_electrode']}" if pd.notna(r.get("reference_electrode")) and pot else ""
        return pot.strip() + ref

    def ann_yield(r):
        parts = []
        if pd.notna(r["nh3_faradaic_efficiency_pct"]):
            parts.append(f"FE {r['nh3_faradaic_efficiency_pct']:.0f}%")
        a = at(r)
        if a:
            parts.append(a)
        return "  ".join(parts)

    def ann_fe(r):
        parts = []
        if pd.notna(r["nh3_yield_norm"]):
            unit = "µg h⁻¹ mg⁻¹" if r["nh3_yield_basis"] == "mass" else "µg h⁻¹ cm⁻²"
            parts.append(f"{r['nh3_yield_norm']:.3g} {unit}")
        a = at(r)
        if a:
            parts.append(a)
        return "  ".join(parts)

    both = rec.dropna(subset=["nh3_yield_norm", "nh3_faradaic_efficiency_pct"])
    both = both[both["nh3_yield_basis"] == "mass"]
    both_best = both[both["role"] != "benchmark"].copy()
    both_best["ylabel"] = both_best["name"] + " (" + both_best["tag"] + ")"

    def fe_rank(ax):
        ranking(ax, fe, None, "nh3_faradaic_efficiency_pct", "NH$_3$ Faradaic efficiency (%) — higher is better", "high", ann_fe,
                title=f"{rx}: peak Faradaic efficiency")
        ax.set_xlim(0, 125)

    panels = {
        "rank": lambda ax: ranking(ax, ymass, bm, "nh3_yield_norm", ylab_mass, "high", ann_yield, logx=True,
                                   title=f"{rx} NH$_3$ yield ranking (mass-normalised; FE at the same potential)", ref_name="Control/benchmark"),
        "rank_area": lambda ax: ranking(ax, yarea, None, "nh3_yield_norm", ylab_area, "high", ann_yield, logx=True,
                                        title="Area-normalised NH$_3$ yield"),
        "rank_fe": fe_rank,
        "trade": lambda ax: scatter_labeled(ax, both_best, benchmarks(both, "nh3_yield_norm"), "nh3_yield_norm", "nh3_faradaic_efficiency_pct",
                                            r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$)", "NH$_3$ Faradaic efficiency (%)",
                                            logx=True, label_n=8, score=lambda r: -(r["nh3_faradaic_efficiency_pct"] + 10 * np.log10(r["nh3_yield_norm"])),
                                            pareto=("max", "max"), title="Activity–selectivity trade-off (same potential; upper-right is better)",
                                            ref_name="Control"),
        "pot_fe": lambda ax: potential_lines(ax, rec, "nh3_faradaic_efficiency_pct", "NH$_3$ Faradaic efficiency (%)"),
        "pot_yield": lambda ax: potential_lines(ax, mass, "nh3_yield_norm", r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$)", logy=True),
        "metals": lambda ax: compare_groups(ax, ymass, "nh3_yield_norm", "metal_group", ["1 metal", "2 metals", "≥3 metals"],
                                            r"Peak yield ($\mu$g h$^{-1}$ mg$^{-1}$)", better="high", logy=True, title="Number of metals"),
        "noble": lambda ax: compare_groups(ax, ymass, "nh3_yield_norm", "noble", ["Earth-abundant only", "Contains noble metal"],
                                           r"Peak yield ($\mu$g h$^{-1}$ mg$^{-1}$)", better="high", logy=True, title="Noble vs earth-abundant"),
        "class": lambda ax: compare_groups(ax, ymass, "nh3_yield_norm", "material_class", ["pristine_mof", "mof_derived", "mof_composite", "non_mof"],
                                           r"Peak yield ($\mu$g h$^{-1}$ mg$^{-1}$)", better="high", logy=True, title="Material class"),
        "medium": lambda ax: compare_groups(ax, ymass, "nh3_yield_norm", "medium", ["acidic", "neutral", "alkaline", "other"],
                                            r"Peak yield ($\mu$g h$^{-1}$ mg$^{-1}$)", better="high", logy=True, title="Electrolyte medium"),
        "fe_metals": lambda ax: compare_groups(ax, fe, "nh3_faradaic_efficiency_pct", "metal_group", ["1 metal", "2 metals", "≥3 metals"],
                                               "Peak FE (%)", better="high", title="Number of metals"),
        "fe_class": lambda ax: compare_groups(ax, fe, "nh3_faradaic_efficiency_pct", "material_class", ["pristine_mof", "mof_derived", "mof_composite"],
                                              "Peak FE (%)", better="high", title="Material class"),
        "fe_medium": lambda ax: compare_groups(ax, fe, "nh3_faradaic_efficiency_pct", "medium", ["acidic", "neutral", "alkaline"],
                                               "Peak FE (%)", better="high", title="Electrolyte medium"),
        "abl": lambda ax: ablation(ax, mass, "nh3_yield_norm", r"NH$_3$ yield ($\mu$g h$^{-1}$ mg$_{cat}^{-1}$)", "high", logy=True),
    }
    iso = "isotope_15n2_labeling" if rx == "N2RR" else "isotope_15n_labeling"
    score_df = pd.concat([ymass, fe[~fe.index.isin(ymass.index)]]).drop_duplicates(["doi", "catalyst_name"])
    score_df = score_df[score_df["role"] == "main"]
    specs = [("nh3_yield_norm", "Yield\n(µg/h/mg)", "high", "{:.3g}"), ("nh3_faradaic_efficiency_pct", "FE\n(%)", "high", "{:.0f}")]
    if rx == "NO3RR":
        specs += [("nh3_selectivity_pct", "NH$_3$ sel.\n(%)", "high", "{:.0f}"), ("nitrate_conversion_pct", "NO$_3^-$ conv.\n(%)", "high", "{:.0f}")]
    specs += [(iso, "¹⁵N\nlabel", "flag", ""), ("nh3_quantification_method", "", "skip", "")]
    specs = [s for s in specs if s[2] != "skip"]
    if rx == "N2RR":
        specs.insert(-1, ("blank_controls_reported", "Blank\ncontrol", "flag", ""))
    score_df = score_df.assign(**{iso: score_df[iso]})
    panels["score"] = lambda ax: scorecard(ax, score_df, specs, title=f"{rx}: featured catalysts")
    if rx == "NO3RR":
        panels["sel_conv"] = lambda ax: scatter_labeled(
            ax, best_per(rec, "nh3_selectivity_pct", "high", keys=("doi", "catalyst_name", "potential_v")), None, "nitrate_conversion_pct",
            "nh3_selectivity_pct", "Nitrate conversion (%)", "NH$_3$ selectivity (%)", label_n=6, score=lambda r: -r["nh3_selectivity_pct"],
            title="Selectivity vs conversion")
    table = pd.concat([ymass, yarea, fe]).drop_duplicates(["doi", "catalyst_name", "potential_v"])
    return panels, table


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def _grid(panels, keys, fig_size, ratios=None, share_y=None):
    cols = len(keys)
    fig, axes = plt.subplots(1, cols, figsize=fig_size, gridspec_kw={"width_ratios": ratios} if ratios else None)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        panels[k](ax)
    if cols > 1:
        for ax, letter in zip(axes, "abcdefgh"):
            _panel_label(ax, letter)
    fig.tight_layout()
    return fig, axes


def make_science_figures(rx: str) -> list[Path]:
    rec = prep(rx)
    out: list[Path] = []
    note = ""
    if rx in ("HER", "OER"):
        panels, table = _hero_overpotential(rx, rec)
        figs = {
            "S1_catalyst_ranking": (["rank"], (7.4, 6.2), None),
            "S2_activity_kinetics": (["trade", "cdl", "rct"], (17, 5.2), [1.3, 1, 1]),
            "S3_structure_activity": ([k for k in ("metals", "noble", "class", "medium", "recon") if k in panels], (4.0 * (5 if rx == "OER" else 4), 4.4), None),
            "S4_within_paper_controls": (["abl"], (8.4, 4.8), None),
            "S5_scorecard": (["score"], (6.6, 5.6), None),
        }
        composite = ["rank", "trade", "noble", "abl"]
    elif rx == "ORR":
        panels, table, note = _hero_orr(rec)
        figs = {
            "S1_catalyst_ranking": (["rank", "zn"], (14, 6.0), [1.25, 1]),
            "S2_activity_kinetics": (["trade", "kin", "n_h2o2"], (16, 5.0), None),
            "S3_structure_activity": (["metals", "noble", "class"], (12, 4.4), None),
            "S4_within_paper_controls": (["abl"], (8.4, 4.8), None),
            "S5_scorecard": (["score"], (7.0, 5.2), None),
        }
        composite = ["rank", "trade", "zn", "abl"]
    else:
        panels, table = _hero_ammonia(rx, rec)
        figs = {
            "S1_catalyst_ranking": (["rank", "rank_fe"], (15, 6.6), [1.3, 1]) if rx == "N2RR" else (["rank", "rank_area", "rank_fe"], (20, 5.4), [1, 1, 1]),
            "S2_activity_selectivity": (["trade"] + (["sel_conv"] if rx == "NO3RR" else []), (7.2 * (2 if rx == "NO3RR" else 1), 5.4), None),
            "S3_structure_activity": (["metals", "noble", "class", "medium"], (16, 4.4), None),
            "S3b_structure_selectivity": (["fe_metals", "fe_class", "fe_medium"], (12, 4.4), None),
            "S4_potential_dependence": (["pot_fe", "pot_yield"], (13, 4.8), None),
            "S5_within_paper_controls": (["abl"], (8.4, 4.8), None),
            "S6_scorecard": (["score"], (7.4, 5.2), None),
        }
        composite = ["rank", "trade", "pot_fe", "abl"] if rx == "N2RR" else ["rank_fe", "rank_area", "rank", "score"]

    for name, (keys, size, ratios) in figs.items():
        fig, _ = _grid(panels, keys, size, ratios)
        out.append(_save(fig, rx, name, rec, note))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    for ax, k, letter in zip(axes.ravel(), composite, "abcd"):
        panels[k](ax)
        _panel_label(ax, letter)
    fig.suptitle(f"{rx}: MOF-based electrocatalyst performance (Elsevier pilot sample)", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    out.append(_save(fig, rx, "Fig_composite", rec, note))

    (SCI_ROOT / rx).mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["doi", "tag", "catalyst_name", "role", "material_class", "node_metals", "medium", "reference_electrode", "potential_v"]
            + [c for c in table.columns if c not in ("doi", "tag", "catalyst_name", "role", "material_class", "node_metals", "medium", "reference_electrode", "potential_v")]
            if c in table.columns]
    table[cols].to_csv(SCI_ROOT / rx / f"{rx}_catalyst_table.csv", index=False)
    return out
