r"""Regenerate ALL five detector-dropout figures of thesis section 4.3 with one
shared, enlarged house style, WITHOUT re-running model inference.

Why this script exists
----------------------
run_robustness.py couples the analysis (model inference on the test subsample)
with the plotting. Re-running it just to restyle the figures costs ~160 model
evaluations. But every number the figures need is already persisted as CSV/JSON
in pipeline_artifacts/diagnostics/. This script reads those artifacts and redraws:

    dropout_single.png       <- robustness_single_detector.csv + robustness_baseline.json
    dropout_progressive.png  <- robustness_progressive_stats.csv + baseline
    dropout_regional.png     <- robustness_regional.csv
    dropout_geometry.png     <- robustness_geometry.csv
    dropout_regions_map.png  <- pipeline_artifacts/detector_catalog.csv   (group map)

Output goes straight to thesis/figures/ under the names the LaTeX expects.

Style
-----
Raquel's review (sep-2026) flagged the figure text as too small; Luis then asked
for ALL of these figures to be enlarged and to read as one set.

What matters is the size the text renders at ON THE PAGE, not in the standalone
PNG. \includegraphics[width=\linewidth] scales the PNG to the text column
(~6.3 in in this book class), so on-page pt = fontpt * 6.3 / figwidth_in. The
old figures were 21 in wide with 13-pt type -> 13 * 6.3/21 ~ 3.9 pt on the page
(what Raquel flagged). Here every figure is 13 in wide with 15-pt type ->
15 * 6.3/13 ~ 7.3 pt on the page -- a normal figure caption size, readable
without being chunky. Panel titles are NOT bold (a bold title larger than its
own small panel looked wrong); only the one-line suptitle is bold.
constrained_layout owns suptitle placement -- passing y= pushed it into the
panels in an earlier try.

Run with the condor-sim python (seaborn not required; its rocket colormap and
"ticks" style are reproduced inline).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ── CONDOR visual identity (verbatim from run_robustness.py) ──────────────────
CONDOR_DARKRED = "#6E1423"
CONDOR_RED     = "#A4243B"
CONDOR_INK     = "#1A1A1A"
CONDOR_GRAY    = "#9E9E9E"
CONDOR_CREAM   = "#F2E8E5"
SAVE_DPI = 200

# seaborn's "rocket" ramp (17 anchors sampled from seaborn 0.13.2); "_r" reversed.
# run_robustness.py uses sns.color_palette("rocket_r", as_cmap=True) for the
# single-detector map -- perceptually uniform, dark-red high end (thesis comment #9).
_ROCKET = ['#03051a', '#180f29', '#30173a', '#481c48', '#611f53', '#7b1f59',
           '#971c5b', '#b21758', '#cb1b4f', '#df2f44', '#ec4c3e', '#f26b49',
           '#f58860', '#f6a37a', '#f6bc99', '#f8d4bc', '#faebdd']
CMAP_IMPORTANCE = LinearSegmentedColormap.from_list("rocket_r", _ROCKET[::-1])

# sns.set_style("ticks") + run_robustness.py rcParams. Type sizes chosen so that,
# after the 13-in canvas is scaled to \linewidth, the body lands ~7 pt on the
# page (a normal figure size). Panel titles are the same size as the body and
# NOT bold.
FIGW = 13.0  # every figure is this wide; on-page pt = fontpt * 6.3 / FIGW
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": SAVE_DPI, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 15,
    "axes.titlesize": 15, "axes.titleweight": "normal",
    "axes.labelsize": 15, "axes.labelcolor": CONDOR_INK,
    "axes.edgecolor": CONDOR_INK, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#DCDCDC", "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "xtick.color": CONDOR_INK, "ytick.color": CONDOR_INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 14, "legend.frameon": False, "lines.linewidth": 1.6,
})
SUPTITLE_KW = dict(fontweight="bold", fontsize=14)  # the only bold text; constrained_layout places it

HERE = Path(__file__).resolve().parent
DIAG = HERE / "pipeline_artifacts" / "diagnostics"
FIGS = HERE.parent / "thesis" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

baseline = json.loads((DIAG / "robustness_baseline.json").read_text())["baseline"]


def save(fig, name):
    fig.savefig(FIGS / name, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  {name}")


# ── 1. dropout_single.png  (Fig. 4.10) ───────────────────────────────────────
def fig_single():
    df = pd.read_csv(DIAG / "robustness_single_detector.csv")
    maps = [("acc_drop",    "Particle - accuracy loss"),
            ("angle_rise",  r"Angle - MAE rise ($^\circ$)"),
            ("energy_rise", "Energy - MAE rise (GeV)")]
    # square panels (set_aspect) don't fill a tall figure, so constrained_layout
    # parks the slack between the suptitle and the panel titles -> a big gap.
    # Height tuned so the panels fill it and the suptitle sits close.
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.4), constrained_layout=True)
    for ax, (col, title) in zip(axes, maps):
        sc = ax.scatter(df["x_center"], df["y_center"], c=df[col].to_numpy(),
                        cmap=CMAP_IMPORTANCE, s=80, edgecolor=CONDOR_INK, linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Degradation when this detector is off", fontsize=11)
        cb.ax.tick_params(labelsize=10)
    fig.suptitle("Single-detector importance map", **SUPTITLE_KW)
    save(fig, "dropout_single.png")


# ── 2. dropout_progressive.png  (Fig. 4.13) ──────────────────────────────────
def fig_progressive():
    st = pd.read_csv(DIAG / "robustness_progressive_stats.csv", header=[0, 1], index_col=0)
    xx = st.index.to_numpy() * 100.0
    panels = [("particle_acc", "Particle accuracy"),
              ("angle_mae", r"Angle MAE ($^\circ$)"),
              ("energy_mae", "Energy MAE (GeV)")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.4), constrained_layout=True)
    for ax, (col, title) in zip(axes, panels):
        mu = st[(col, "mean")].to_numpy()
        sd = np.nan_to_num(st[(col, "std")].to_numpy())
        ax.plot(xx, mu, color=CONDOR_DARKRED, marker="o", ms=5)
        ax.fill_between(xx, mu - sd, mu + sd, color=CONDOR_DARKRED, alpha=0.2)
        ax.axhline(baseline[col], color=CONDOR_INK, ls="--", lw=1.0, label="baseline")
        ax.set_title(title)
        ax.set_xlabel("Detectors switched off (%)")
        ax.set_ylabel(title)
        ax.legend()
    fig.suptitle("Reconstruction under progressive random detector failure", **SUPTITLE_KW)
    save(fig, "dropout_progressive.png")


# ── 3. dropout_regional.png  (Fig. 4.12) ─────────────────────────────────────
def fig_regional():
    df = pd.read_csv(DIAG / "robustness_regional.csv")
    order = df.sort_values("acc_drop")["region"].tolist()
    d = df.set_index("region").loc[order]
    colors = [CONDOR_GRAY if "Half" in r else CONDOR_DARKRED for r in d.index]
    bars = [("acc_drop",    "Accuracy loss"),
            ("angle_rise",  r"Angle MAE rise ($^\circ$)"),
            ("energy_rise", "Energy MAE rise (GeV)")]
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 8.2), constrained_layout=True)
    for j, (col, title) in enumerate(bars):
        axes[0, j].barh(d.index, d[col], color=colors, edgecolor=CONDOR_INK)
        axes[0, j].set_title(title)
        axes[0, j].set_xlabel(title)
        axes[1, j].barh(d.index, d[col + "_per_det"], color=colors, edgecolor=CONDOR_INK)
        axes[1, j].set_title("per detector removed", fontsize=13)
        axes[1, j].set_xlabel(title + " / detector")
    fig.suptitle("Impact of switching off whole detector groups", **SUPTITLE_KW)
    save(fig, "dropout_regional.png")


# ── 4. dropout_geometry.png  (Fig. 4.14) ─────────────────────────────────────
def fig_geometry():
    g = pd.read_csv(DIAG / "robustness_geometry.csv")
    rnd = (g[g["strategy"] == "random_thinning"]
           .groupby("n_kept").mean(numeric_only=True).reset_index())
    crop = g[g["strategy"] == "central_crop"].sort_values("n_kept")
    chk = g[g["strategy"] == "checkerboard"].iloc[0]
    panels = [("particle_acc", "Particle accuracy"),
              ("angle_mae", r"Angle MAE ($^\circ$)"),
              ("energy_mae", "Energy MAE (GeV)")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.4), constrained_layout=True)
    for ax, (col, title) in zip(axes, panels):
        ax.plot(rnd["n_kept"], rnd[col], color=CONDOR_GRAY, marker="o", ms=5,
                label="Random thinning")
        ax.plot(crop["n_kept"], crop[col], color=CONDOR_DARKRED, marker="s", ms=6,
                label="Central crop")
        ax.scatter([chk["n_kept"]], [chk[col]], color=CONDOR_RED, s=130, marker="D",
                   zorder=5, edgecolor=CONDOR_INK, label="Checkerboard")
        ax.set_title(title)
        ax.set_xlabel("Detectors kept")
        ax.set_ylabel(title)
        ax.legend()
    fig.suptitle("Array geometry trade-off at matched detector count", **SUPTITLE_KW)
    save(fig, "dropout_geometry.png")


# ── 5. dropout_regions_map.png  (Fig. 4.11) ──────────────────────────────────
def fig_regions_map():
    cat = pd.read_csv(HERE / "pipeline_artifacts" / "detector_catalog.csv")
    cat["distance"] = np.hypot(cat["x_center"], cat["y_center"])
    grid = cat[cat["detector_id"] < 100]
    core36 = set(grid.nsmallest(36, "distance")["detector_id"].astype(int))
    radial = {
        "Central core (36)": cat["detector_id"].isin(core36),
        "Outer grid (64)":   (cat["detector_id"] < 100) & (~cat["detector_id"].isin(core36)),
        "Peripheral (20)":   cat["detector_id"] >= 100,
    }
    radial_colors = {"Central core (36)": "#5A0C18",
                     "Outer grid (64)":   "#D8703F",
                     "Peripheral (20)":   "#8C8C8C"}
    halves = [("Halves along $x$", "x_center"),
              ("Halves along $y$", "y_center")]

    # constrained_layout + set_aspect("equal") so the three panels stay the same
    # size and the array is not distorted. The legend is ONE row for the three
    # radial groups, placed in reserved space below the panels with
    # loc="outside lower center" (constrained_layout gives it its own band, so it
    # cannot collide with the x-axis labels). The halves panels need no legend:
    # the dashed split line + the panel title say which side is which, and the
    # "60 modules each" count is in the caption.
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.8), layout="constrained")

    MS = 55
    for name, mask in radial.items():
        s = cat[mask]
        axes[0].scatter(s["x_center"], s["y_center"], s=MS, marker="s",
                        color=radial_colors[name], edgecolor=CONDOR_INK,
                        linewidth=0.4, label=name)
    axes[0].set_title("Radial groups")

    for ax, (title, col) in zip(axes[1:], halves):
        neg, pos = cat[col] < 0, cat[col] > 0
        ax.scatter(cat[neg]["x_center"], cat[neg]["y_center"], s=MS, marker="s",
                   color=CONDOR_INK, edgecolor=CONDOR_INK, linewidth=0.4)
        ax.scatter(cat[pos]["x_center"], cat[pos]["y_center"], s=MS, marker="s",
                   color=CONDOR_DARKRED, edgecolor=CONDOR_INK, linewidth=0.4)
        (ax.axvline if col == "x_center" else ax.axhline)(0, color=CONDOR_INK,
                                                          lw=1.0, ls="--", zorder=0)
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False,
               fontsize=17, handletextpad=0.4, columnspacing=2.0)

    fig.suptitle("Detector groups for the regional dropout analysis", **SUPTITLE_KW)
    save(fig, "dropout_regions_map.png")


if __name__ == "__main__":
    print("Regenerating thesis section 4.3 dropout figures -> thesis/figures/")
    fig_single()
    fig_progressive()
    fig_regional()
    fig_geometry()
    fig_regions_map()
    print("done.")
