"""
Regenerate the CONDOR detector-grid figure (thesis Fig 2.5) with the same
visual identity as the rest of the thesis figures.

Originally the figure was copied from `CONDOR_CNN_Transformers_2025/figures/
detector.png`, which uses a plain matplotlib default style and a sans-serif
title. This script produces a replacement with the CONDOR palette + Palatino
serif + bold, consistent with the other figures.

Input : detector positions and IDs from the notebook data pipeline
Output: thesis/figures/condor_modules_grid.png
"""
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only, we just need the catalog

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # notebook cells resolve BASE_DIR from the working directory
NB = os.path.join(HERE, "CONDOR_EAS-Reconstruction.ipynb")
FIGS = os.path.join(HERE, "..", "thesis", "figures")

CONDOR_DARKRED = "#6E1423"
CONDOR_INK     = "#1A1A1A"
CONDOR_GRAY    = "#9E9E9E"
CONDOR_CREAM   = "#F2E8E5"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.labelcolor": CONDOR_INK,
    "axes.edgecolor": CONDOR_INK, "axes.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#DCDCDC", "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "xtick.color": CONDOR_INK, "ytick.color": CONDOR_INK,
})

# --- rebuild detector catalog from the notebook (verbatim) ---
# `detector_catalog["distance"]` is added in cell 22, so we run the pipeline
# up to and including that cell (same as run_robustness.py / run_attention.py).
nb = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "__main__"}
for i in [2, 4, 6, 8, 9, 13, 15, 20, 22]:
    src = "".join(nb["cells"][i]["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)

cat = ns["detector_catalog"]

# Split central grid (IDs 0-99) vs peripheral (IDs 100-119).
central = cat[cat["detector_id"] < 100]
peripheral = cat[cat["detector_id"] >= 100]

fig, ax = plt.subplots(figsize=(9.5, 8.5))

# Central modules: filled squares in cream + darkred border for the innermost.
ax.scatter(central["x_center"], central["y_center"], s=520, marker="s",
           facecolor=CONDOR_CREAM, edgecolor=CONDOR_INK, linewidth=0.9, zorder=2)
# Highlight the 16 innermost (used as central_ids in the pipeline).
inner16 = cat.nsmallest(16, "distance")
ax.scatter(inner16["x_center"], inner16["y_center"], s=520, marker="s",
           facecolor="#F5D7DA", edgecolor=CONDOR_DARKRED, linewidth=1.4, zorder=3,
           label="16 central modules (energy feature)")
# Peripheral: distinct gray fill.
ax.scatter(peripheral["x_center"], peripheral["y_center"], s=520, marker="s",
           facecolor="#E5E5E5", edgecolor=CONDOR_INK, linewidth=0.9, zorder=2,
           label="20 peripheral modules")

# Detector IDs on top of each module.
for _, r in cat.iterrows():
    ax.text(r["x_center"], r["y_center"], str(int(r["detector_id"])),
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=CONDOR_INK, zorder=4)

ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")
ax.set_title("CONDOR detector-module layout (120 modules)")
ax.set_aspect("equal")
# Legend outside the axes to avoid covering peripheral IDs at the corners.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
          frameon=True, framealpha=0.95, fontsize=11)
fig.tight_layout()

out_path = os.path.join(FIGS, "condor_modules_grid.png")
fig.savefig(out_path)
plt.close(fig)
print("Saved:", out_path, flush=True)
