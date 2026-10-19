"""Six-panel figure per reaction, centred on catalysts, organic linkers and metals.

  (a) Top-15 catalysts: headline metric (bars) + secondary metric (diamonds)
  (b) Organic linkers ranked by the best catalyst they produced
  (c) Metals ranked the same way
  (d) Metal x linker map (best value per combination)
  (e) Morphology classes
  (f) Activity vs BET surface area (falls back to metal-count when BET is sparse)

Headline metrics
  HER / OER : eta10 (mV, lower better)      secondary: Tafel slope
  ORR       : E1/2 (V vs RHE, higher better) secondary: onset potential
  N2RR/NO3RR: NH3 yield (normalised; log)   secondary: Faradaic efficiency

Only papers that passed the full-text MOF verification are included (the extraction step
already enforces this). Panels (b), (c), (e) are bar charts: one bar per group = its best catalyst
(colour = material class); n in the row label = number of catalysts in that group.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from ..extraction.data_checks import clean_metrics, out_of_range  # noqa: E402
from .figures import FIG_ROOT, REACTIONS  # noqa: E402
from .science_figures import CLASS_C, CLASS_L, MED_ABBR, is_rhe, pretty, prep  # noqa: E402

PANEL_ROOT = FIG_ROOT / "panels"

# reaction -> headline spec
SPEC = {
    "HER": dict(metric="overpotential_10_ma_cm2_mv", better="low", label="Overpotential at 10 mA cm$^{-2}$ (mV)",
                sec="tafel_slope_mv_dec", sec_label="Tafel slope (mV dec$^{-1}$)", short="η$_{10}$", log=False, rhe=True),
    "OER": dict(metric="overpotential_10_ma_cm2_mv", better="low", label="Overpotential at 10 mA cm$^{-2}$ (mV)",
                sec="tafel_slope_mv_dec", sec_label="Tafel slope (mV dec$^{-1}$)", short="η$_{10}$", log=False, rhe=True),
    "ORR": dict(metric="half_wave_potential_v", better="high", label="Half-wave potential E$_{1/2}$ (V vs RHE)",
                sec="onset_potential_v", sec_label="Onset potential (V vs RHE)", short="E$_{1/2}$", log=False, rhe=True),
    "N2RR": dict(metric="nh3_yield_norm", better="high", label="NH$_3$ yield", sec="nh3_faradaic_efficiency_pct",
                 sec_label="Faradaic efficiency (%)", short="yield", log=True, rhe=False),
    "NO3RR": dict(metric="nh3_yield_norm", better="high", label="NH$_3$ yield", sec="nh3_faradaic_efficiency_pct",
                  sec_label="Faradaic efficiency (%)", short="yield", log=True, rhe=False),
}
BASIS_LABEL = {"mass": "NH$_3$ yield (µg h$^{-1}$ mg$^{-1}$)", "area": "NH$_3$ yield (µg h$^{-1}$ cm$^{-2}$)"}
BASIS_SHORT = {"mass": "µg h⁻¹ mg⁻¹", "area": "µg h⁻¹ cm⁻²"}

SEC_COLOR = "#222222"
GRID_CMAP = LinearSegmentedColormap.from_list("good", ["#F7FBFF", "#9ECAE1", "#08519C"])
GRID_CMAP_R = GRID_CMAP.reversed()


# --------------------------------------------------------------------------- #
# Descriptor handling
# --------------------------------------------------------------------------- #
def linker_label(s: str, maxlen: int = 22) -> str:
    """Short display label: 'BDC (terephthalate)' -> 'BDC'; 'Imidazolate (2-MIM)' -> '2-MIM'."""
    s = str(s).strip()
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", s)
    if m:
        lead, paren = m.group(1).strip(), m.group(2).strip()
        s = lead if re.fullmatch(r"[A-Z0-9][A-Za-z0-9\-]{1,8}", lead) else paren
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


_MORPH_RULES = [
    ("Hollow / cage / yolk-shell", r"hollow|cage|yolk|core.?shell|box"),
    ("Flower / urchin / hierarchical", r"flower|urchin|cauliflower|hierarch|dendrit|sea.?urchin|chrysanth|nest"),
    ("Nanosheet / 2D / lamellar", r"sheet|2d|lamell|layer|plate|flake|film|membrane|rectangular"),
    ("Rod / wire / tube / needle", r"rod|wire|tube|needle|fib|array|belt|ribbon|pillar|setaria"),
    ("Polyhedron / cube / octahedron", r"polyhed|octahed|cub|dodecahed|crystal|prism|rhombic|diamond"),
    ("Sphere / particle / dot", r"sphere|spher|particle|dot|granul|grain|nut|cluster|irregular|bead"),
    ("Porous / sponge / network", r"porous|sponge|network|foam|mesopor|macropor|framework|3d"),
]


def morph_category(s) -> str | float:
    if not isinstance(s, str) or not s.strip() or s.strip().lower() in {"not stated", "nan", "none"}:
        return np.nan
    low = s.lower()
    for name, pat in _MORPH_RULES:
        if re.search(pat, low):
            return name
    return "Other"


def metal_group(n: int) -> str:
    return "unspecified" if n == 0 else "1 metal" if n == 1 else "2 metals" if n == 2 else "≥3 metals"


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def head_frame(rx: str) -> tuple[pd.DataFrame, dict, str]:
    """One row per catalyst (best value of the headline metric), plus spec and a footnote."""
    spec = dict(SPEC[rx])
    rec = prep(rx)
    n_bad = int(out_of_range(rec).any(axis=1).sum())
    rec = clean_metrics(rec)
    note = f"{n_bad} records with physically implausible values excluded. " if n_bad else ""
    if spec["rhe"]:
        n0 = rec["role"].ne("benchmark").sum()
        rec = rec[is_rhe(rec)]
        dropped = n0 - rec["role"].ne("benchmark").sum()
        if dropped:
            note += f"{dropped} records not referenced to RHE omitted. "
    if spec["metric"] == "nh3_yield_norm":
        counts = rec.dropna(subset=["nh3_yield_norm"]).groupby("nh3_yield_basis").size()
        basis = counts.reindex(["mass", "area"]).fillna(0).idxmax() if len(counts) else "mass"
        rec = rec[rec["nh3_yield_basis"] == basis]
        spec["label"] = BASIS_LABEL[basis]
        note += f"Yield basis shown: {BASIS_SHORT[basis]} (other bases omitted). "

    rec = rec[rec["role"] != "benchmark"].copy()
    rec["morphology"] = rec["morphology"].where(rec["morphology"].astype(str).str.lower() != "not stated")
    rec["morph_cat"] = rec["morphology"].map(morph_category)
    d = rec.dropna(subset=[spec["metric"]])
    if not d.empty:
        grp = d.groupby(["doi", "catalyst_name", "medium"])[spec["metric"]]
        idx = grp.idxmin() if spec["better"] == "low" else grp.idxmax()
        d = d.loc[idx].copy()
        # descriptors may be missing on the best-performing duplicate: borrow them from sibling rows
        for col in ("node_metals", "organic_linkers", "morphology", "morph_cat", "bet_surface_area_m2_g"):
            fill = rec.dropna(subset=[col]).groupby(["doi", "catalyst_name"])[col].first()
            key = list(zip(d["doi"], d["catalyst_name"]))
            d[col] = d[col].where(d[col].notna(), pd.Series([fill.get(k, np.nan) for k in key], index=d.index))
    d["metals"] = d["node_metals"].fillna("").map(lambda s: [m for m in s.split("|") if m])
    d["linkers"] = d["organic_linkers"].fillna("").map(lambda s: [linker_label(x) for x in s.split("|") if x])
    d["n_metals"] = d["metals"].map(len)
    d["metal_group"] = d["n_metals"].map(metal_group)
    return d, spec, note


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _nodata(ax, msg="Insufficient data in this sample"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#9A9A9A", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _tag(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=10.5, fontweight="bold")


def _better_hint(ax, better):
    txt = "← better" if better == "low" else "better →"
    ax.text(0.99 if better == "high" else 0.01, 1.005, txt, transform=ax.transAxes, fontsize=7, color="#777777",
            ha="right" if better == "high" else "left", va="bottom")


def _class_legend(ax, present, ncol=4, y=-0.12):
    h = [Patch(facecolor=CLASS_C[c], edgecolor="black", linewidth=0.4, label=CLASS_L[c]) for c in CLASS_C if c in present]
    if h:
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.1,
                  columnspacing=1.0, fontsize=7)


def _plain_log(ax):
    """Readable log x-axis: ticks at 1, 2, 5 x 10^n written as plain numbers."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _fmt(v, spec):
    if spec["log"]:
        return f"{v:.3g}"
    return f"{v:.0f}" if spec["better"] == "low" else f"{v:.2f}"


def panel_catalysts(ax, d, spec, topn=15):
    _tag(ax, "a", f"Top {topn} catalysts by {spec['short']}")
    metric, better = spec["metric"], spec["better"]
    if d.empty:
        return _nodata(ax)
    t = d.sort_values(metric, ascending=(better == "low")).head(topn).reset_index(drop=True)
    y = np.arange(len(t))
    colors = [CLASS_C.get(c, "#999") for c in t["material_class"]]
    if spec["log"]:
        left = min(0.5 * t[metric].min(), t[metric].min())
        ax.set_xscale("log")
        ax.hlines(y, left, t[metric], color=colors, linewidth=2.2, alpha=0.9)
        ax.scatter(t[metric], y, c=colors, s=62, edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_xlim(left * 0.8, t[metric].max() * 3.5)
    else:
        ax.barh(y, t[metric], color=colors, edgecolor="black", linewidth=0.5, height=0.7)
        lo = t[metric].min()
        if better == "high":  # potentials: do not start the bars at zero
            ax.set_xlim(max(0, lo - 0.25 * (t[metric].max() - lo + 0.1)), t[metric].max() * 1.12)
        else:
            ax.set_xlim(0, t[metric].max() * 1.25)
    xmax = ax.get_xlim()[1]
    for yi, v in zip(y, t[metric]):
        ax.text(v * (1.12 if spec["log"] else 1.0) + (0 if spec["log"] else 0.012 * xmax), yi, _fmt(v, spec),
                va="center", ha="left", fontsize=7, color="#222")
    multi_med = t["medium"].nunique() > 1
    labels = [f"{n}  ({tg})" + (f" [{MED_ABBR.get(m, '')}]" if multi_med and MED_ABBR.get(m, "") else "")
              for n, tg, m in zip(t["name"], t["tag"], t["medium"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(spec["label"])
    ax.grid(axis="y", visible=False)
    _better_hint(ax, better)
    if spec["log"]:
        _plain_log(ax)

    # secondary metric in an aligned side panel (shared rows)
    sec = spec["sec"]
    ax2 = make_axes_locatable(ax).append_axes("right", size="42%", pad=0.14, sharey=ax)
    vals = t[sec]
    if vals.notna().sum() >= 1:
        ax2.hlines(y, 0 if sec != "onset_potential_v" else vals.min() - 0.05, vals, color="#BBBBBB", linewidth=1.0, zorder=1)
        ax2.scatter(vals, y, marker="D", s=30, facecolor="white", edgecolor=SEC_COLOR, linewidth=1.2, zorder=3)
        span = vals.max() - vals.min() if vals.notna().sum() > 1 else abs(vals.max()) * 0.3 + 1e-9
        for yi, v in zip(y, vals):
            if pd.notna(v):
                fmt = f"{v:.0f}" if sec != "onset_potential_v" else f"{v:.2f}"
                ax2.text(v + 0.04 * (vals.max() * 1.3 if sec != "onset_potential_v" else span + 0.1), yi, fmt,
                         va="center", ha="left", fontsize=6.8, color=SEC_COLOR)
        if sec == "onset_potential_v":
            ax2.set_xlim(vals.min() - 0.05, vals.max() + 0.12 + 0.2 * span)
        elif "pct" in sec:
            ax2.set_xlim(0, 118)
        else:
            ax2.set_xlim(0, vals.max() * 1.3)
        if sec == "tafel_slope_mv_dec":
            for x_, c_ in ((40, "#7FB77E"), (120, "#E0A030")):
                ax2.axvline(x_, color=c_, linestyle=":", linewidth=1.0, zorder=0)
    else:
        ax2.text(0.5, 0.5, "not reported", transform=ax2.transAxes, ha="center", color="#9A9A9A", fontsize=8)
    ax2.set_xlabel(spec["sec_label"], fontsize=8.5)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.grid(axis="y", visible=False)
    h = [Patch(facecolor=CLASS_C[c], edgecolor="black", linewidth=0.4, label=CLASS_L[c]) for c in CLASS_C
         if c in set(t["material_class"])]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(h), frameon=False,
              handlelength=1.1, columnspacing=0.9, fontsize=7)


def _long(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explode a list column (metals/linkers) or take a scalar column into a 'grp' column."""
    if col in ("metals", "linkers"):
        x = d.explode(col).rename(columns={col: "grp"})
    else:
        x = d.rename(columns={col: "grp"}).copy()
    return x.dropna(subset=["grp"]).query("grp != ''")


def dot_panel(ax, d, spec, col, letter, title, topn=12, ylabel_prefix=""):
    _tag(ax, letter, title)
    metric, better = spec["metric"], spec["better"]
    x = _long(d, col)
    if x.empty:
        return _nodata(ax, "Not reported in the papers\nof this sample")
    g = x.groupby("grp")[metric]
    stats = pd.DataFrame({"best": g.min() if better == "low" else g.max(), "median": g.median(), "n": g.size(),
                          "papers": x.groupby("grp")["doi"].nunique()})
    # groups backed by >= 2 catalysts first (more trustworthy), then single-catalyst groups; best value within each
    stats["single"] = stats["n"] < 2
    stats = stats.sort_values(["single", "best"], ascending=[True, better == "low"]).head(topn)
    y = {k: i for i, k in enumerate(stats.index)}
    xs = x[x["grp"].isin(y)].reset_index(drop=True)
    # one bar per group = value of its best catalyst; colour = material class of that catalyst
    idx = xs.groupby("grp")[metric].idxmin() if better == "low" else xs.groupby("grp")[metric].idxmax()
    best_rows = xs.loc[idx.values].set_index("grp").reindex(stats.index)
    vals = stats["best"].values
    colors = [CLASS_C.get(c, "#999") for c in best_rows["material_class"]]
    yy = np.arange(len(stats))
    if spec["log"]:
        left = vals.min() / 2.5
        ax.set_xscale("log")
        ax.barh(yy, vals - left, left=left, color=colors, edgecolor="black", linewidth=0.5, height=0.72)
        ax.set_xlim(left, vals.max() * 3.2)
    else:
        ax.barh(yy, vals, color=colors, edgecolor="black", linewidth=0.5, height=0.72)
        if better == "high":
            ax.set_xlim(max(0, vals.min() - 0.3 * (vals.max() - vals.min() + 0.1)), vals.max() * 1.1)
        else:
            ax.set_xlim(0, vals.max() * 1.18)
    for yi, v in zip(yy, vals):
        ax.text(v * (1.08 if spec["log"] else 1.0) + (0 if spec["log"] else 0.012 * ax.get_xlim()[1]), yi,
                _fmt(v, spec), va="center", ha="left", fontsize=7, color="#222")
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([f"{ylabel_prefix}{k}  (n={int(r.n)})" for k, r in stats.iterrows()], fontsize=8)
    n_multi = int((~stats["single"]).sum())
    if 0 < n_multi < len(stats):
        ax.axhline(n_multi - 0.5, color="#999", linestyle="--", linewidth=0.8)
        ax.text(0.99, (n_multi - 0.5), "single-catalyst groups below", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=6.3, color="#888", style="italic")
    ax.invert_yaxis()
    if spec["log"]:
        ax.set_xscale("log")
        _plain_log(ax)
    ax.set_xlabel("Best " + spec["label"][0].lower() + spec["label"][1:] if not spec["label"].startswith("NH") else
                  "Best " + spec["label"])
    ax.grid(axis="y", visible=False)
    _better_hint(ax, better)
    h = [Patch(facecolor=CLASS_C[c], edgecolor="black", linewidth=0.4, label=CLASS_L[c]) for c in CLASS_C
         if c in set(best_rows["material_class"])]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=len(h), frameon=False,
              handlelength=1.1, columnspacing=0.9, fontsize=7)
    n_single = int((stats["n"] == 1).sum())
    if n_single:
        ax.text(0.99, 0.01, f"{n_single} of {len(stats)} groups: single catalyst", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, color="#888", style="italic")


def panel_heatmap(ax, d, spec, letter="d", max_metals=8, max_linkers=8):
    _tag(ax, letter, f"Metal × linker: best {spec['short']}")
    metric, better = spec["metric"], spec["better"]
    rows = []
    for _, r in d.iterrows():
        for m in r["metals"]:
            for l in r["linkers"]:
                rows.append((m, l, r[metric], r["doi"]))
    if not rows:
        return _nodata(ax, "Linkers not named in this sample")
    t = pd.DataFrame(rows, columns=["metal", "linker", "v", "doi"])
    top_m = t.groupby("metal").size().sort_values(ascending=False).head(max_metals).index
    top_l = t.groupby("linker").size().sort_values(ascending=False).head(max_linkers).index
    t = t[t["metal"].isin(top_m) & t["linker"].isin(top_l)]
    agg = t.groupby(["metal", "linker"])["v"].agg("min" if better == "low" else "max").unstack()
    cnt = t.groupby(["metal", "linker"]).size().unstack()
    agg = agg.reindex(index=[m for m in top_m if m in agg.index], columns=[l for l in top_l if l in agg.columns])
    cnt = cnt.reindex_like(agg)
    arr = np.log10(agg.values.astype(float)) if spec["log"] else agg.values.astype(float)
    cmap = (GRID_CMAP_R if better == "low" else GRID_CMAP).copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, aspect="auto")
    ax.set_xticks(range(agg.shape[1]))
    ax.set_xticklabels(agg.columns, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(agg.shape[0]))
    ax.set_yticklabels(agg.index, fontsize=8.5)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            v = agg.values[i, j]
            if np.isnan(v):
                continue
            norm = (arr[i, j] - lo) / (hi - lo + 1e-12)
            dark = (norm < 0.35) if better == "low" else (norm > 0.65)
            ax.text(j, i, f"{_fmt(v, spec)}\nn={int(cnt.values[i, j])}", ha="center", va="center", fontsize=6.3,
                    color="white" if dark else "#222")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, agg.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, agg.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel(f"Organic linker   (cell = best {spec['short']}; grey = no catalyst)", fontsize=8)


def panel_bet(ax, d, spec, letter="f"):
    metric, better = spec["metric"], spec["better"]
    x = d.dropna(subset=["bet_surface_area_m2_g", metric]).copy()
    x = x[x["bet_surface_area_m2_g"] > 0]
    if len(x) < 5:
        # fallback: metal count (single vs multi-metal)
        return dot_panel(ax, d, spec, "metal_group", letter, f"Best {spec['short']} by number of metals", topn=4)
    _tag(ax, letter, f"{spec['short']} vs BET surface area")
    prim = x["metals"].map(lambda ms: ms[0] if ms else "n.s.")
    top = prim.value_counts().head(7).index
    prim = prim.where(prim.isin(top), "other")
    cats = list(prim.value_counts().index)
    cmap = plt.get_cmap("tab10")
    col = {c: ("#BBBBBB" if c in ("other", "n.s.") else cmap(i % 10)) for i, c in enumerate(cats)}
    ax.scatter(x["bet_surface_area_m2_g"], x[metric], c=[col[p] for p in prim], s=60, edgecolor="black", linewidth=0.5,
               zorder=3)
    ax.set_xscale("log")
    if spec["log"]:
        ax.set_yscale("log")
    ax.set_xlabel("BET surface area (m$^2$ g$^{-1}$)")
    ax.set_ylabel(spec["label"], fontsize=8)
    rho = x["bet_surface_area_m2_g"].corr(x[metric], method="spearman")
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}  (n = {len(x)})", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCC"))
    best = x.sort_values(metric, ascending=(better == "low")).head(4)
    texts = [ax.text(r["bet_surface_area_m2_g"], r[metric], r["name"], fontsize=6.5) for _, r in best.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=col[c], markeredgecolor="black", label=c)
                       for c in cats], title="Primary metal", fontsize=6.5, title_fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=min(len(cats), 6), frameon=False, handletextpad=0.2, columnspacing=0.8)
    if better == "low":
        ax.invert_yaxis() if False else None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _panels(d, spec):
    return [
        ("a_catalysts", lambda ax: panel_catalysts(ax, d, spec)),
        ("b_linkers", lambda ax: dot_panel(ax, d, spec, "linkers", "b", f"Top organic linkers by best {spec['short']}")),
        ("c_metals", lambda ax: dot_panel(ax, d, spec, "metals", "c", f"Top metals by best {spec['short']}")),
        ("d_metal_x_linker", lambda ax: panel_heatmap(ax, d, spec)),
        ("e_morphology", lambda ax: dot_panel(ax, d, spec, "morph_cat", "e", f"Morphology by best {spec['short']}")),
        ("f_bet_or_metalcount", lambda ax: panel_bet(ax, d, spec)),
    ]


def _save_png(fig, path: Path, dpi: int = 250, tries: int = 6) -> None:
    """Render to memory, then write with retries (OneDrive can briefly lock files under a synced folder)."""
    import io
    import time

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    data = buf.getvalue()
    for i in range(tries):
        try:
            Path(path).write_bytes(data)
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def make_panel_figures(rx: str, out_root: Path = FIG_ROOT) -> list[Path]:
    d, spec, note = head_frame(rx)
    out: list[Path] = []
    with_linker = int(d["linkers"].map(len).gt(0).sum())
    n_pap = d["doi"].nunique()
    if "linker_check" in d:
        conf = int((d["linker_check"] == "text_confirmed").sum())
        impl = int((d["linker_check"] == "implied_by_code").sum())
        linker_txt = (f"Linker confirmed by a quote in the text for {conf}/{len(d)} catalysts, standard linker implied "
                      f"by MOF code for {impl}, none/unstated for {len(d) - with_linker}. ")
    else:
        linker_txt = f"Linker identified for {with_linker}/{len(d)} catalysts. "
    foot = (f"{rx}: {len(d)} catalysts from {n_pap} full-text-verified MOF papers (LLM extraction). "
            f"{linker_txt}{note}Controls and benchmarks excluded.")

    fig = plt.figure(figsize=(20, 11.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.5, wspace=0.52, left=0.13, right=0.985,
                          top=0.89, bottom=0.12)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax, (_, fn) in zip(axes, _panels(d, spec)):
        fn(ax)
    fig.suptitle(f"{rx} — MOF electrocatalysts: catalysts, organic linkers and metals", fontsize=14, fontweight="bold",
                 x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.012, foot, fontsize=7, color="#666", ha="left")
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{rx}_6panel.png"
    _save_png(fig, path)
    plt.close(fig)
    out.append(path)

    # individual panels
    sub = out_root / "panels" / rx
    sub.mkdir(parents=True, exist_ok=True)
    for name, fn in _panels(d, spec):
        f, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
        fn(ax)
        p = sub / f"{rx}_{name}.png"
        _save_png(f, p)
        plt.close(f)
        out.append(p)

    # tidy table behind the figure
    tab = d[["reaction", "tag", "name", "material_class", "medium", spec["metric"], spec["sec"], "node_metals",
             "organic_linkers", "organic_linkers_raw", "linker_source", "morphology", "morph_cat", "crystallinity",
             "bet_surface_area_m2_g", "synthesis_method"]].copy() if "organic_linkers_raw" in d else d
    tab = tab.sort_values(spec["metric"], ascending=(spec["better"] == "low"))
    try:
        tab.to_csv(sub / f"{rx}_panel_data.csv", index=False)
    except PermissionError:  # file open elsewhere (e.g. Excel); keep a copy under another name
        tab.to_csv(sub / f"{rx}_panel_data_new.csv", index=False)
    return out
