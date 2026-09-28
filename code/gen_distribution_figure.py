"""
gen_distribution_figure.py
Genera la figura de distribuciones before/after balancing (1×3, superpuesta).
Guarda en:
  - pipeline_artifacts/diagnostics/distribution_checks.png
  - ../thesis/figures/data_balancing.png
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT_DIAG   = BASE / "pipeline_artifacts" / "diagnostics" / "distribution_checks.png"
OUT_THESIS = BASE.parent / "thesis" / "figures" / "data_balancing.png"

SEED     = 42
SAVE_DPI = 200

# ── Paleta ─────────────────────────────────────────────────────────────
CONDOR_DARKRED = "#6E1423"
CONDOR_INK     = "#1A1A1A"
CONDOR_GRAY    = "#9E9E9E"


def apply_condor_style():
    sns.set_style("ticks")
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": SAVE_DPI, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "serif", "font.size": 13,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 13, "axes.labelcolor": CONDOR_INK,
        "axes.edgecolor": CONDOR_INK, "axes.linewidth": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#DCDCDC", "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "xtick.color": CONDOR_INK, "ytick.color": CONDOR_INK,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.frameon": False,
    })


def balance_by_group(df, group_cols, random_state=42):
    counts = df.groupby(list(group_cols)).size()
    counts = counts[counts > 0]
    if counts.empty:
        raise RuntimeError("No groups available for balancing.")
    target = int(counts.median())
    parts = []
    for _, group in df.groupby(list(group_cols)):
        replace = len(group) < target
        parts.append(group.sample(n=target, replace=replace, random_state=random_state))
    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced, target


# ── Carga y filtros (idénticos al notebook) ────────────────────────────
ENERGY_FILTER      = ("3E2", "5E2", "8E2")
ANGLE_MAX          = 40.0
MIN_TOTAL_PARTICLES = 30

pkl_path = BASE / "processed_all_data.pkl"
print(f"Loading {pkl_path} ...")
with open(pkl_path, "rb") as f:
    df_raw = pickle.load(f)
print(f"  {len(df_raw):,} events loaded (raw)")

df_full = df_raw[
    (df_raw["energy"].isin(ENERGY_FILTER)) &
    (df_raw["angle"] <= ANGLE_MAX) &
    (df_raw["total_particles"] >= MIN_TOTAL_PARTICLES)
].reset_index(drop=True)
print(f"  {len(df_full):,} events after filtering")

df_balanced, target = balance_by_group(df_full, ("label", "angle", "energy"), SEED)
print(f"  Balanced: {len(df_balanced):,} events  (target={target}/group)")

# ── Figura 1×3 superpuesta ─────────────────────────────────────────────
apply_condor_style()

features = ["label", "angle", "energy"]
titles   = ["Label", "Zenith Angle", "Energy"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for col_idx, (feature, title) in enumerate(zip(features, titles)):
    ax = axes[col_idx]

    counts_before = df_full[feature].value_counts().sort_index()
    counts_after  = df_balanced[feature].value_counts().sort_index()

    # Alinear índices (por seguridad)
    idx = counts_before.index.union(counts_after.index)
    counts_before = counts_before.reindex(idx, fill_value=0)
    counts_after  = counts_after.reindex(idx, fill_value=0)

    x = np.arange(len(idx))

    ax.bar(x, counts_before.values,
           color=CONDOR_GRAY, alpha=0.70,
           edgecolor=CONDOR_INK, linewidth=0.5,
           label="Before balancing", zorder=1)
    ax.bar(x, counts_after.values,
           color=CONDOR_DARKRED, alpha=0.85,
           edgecolor=CONDOR_INK, linewidth=0.5,
           label="After balancing", zorder=2)

    # Etiquetas del eje x
    if feature == "label":
        tick_labels = ["Proton" if v == 0 else r"$\gamma$" for v in idx]
        rot, ha = 0, "center"
    elif feature == "energy":
        tick_labels = [str(v) for v in idx]
        rot, ha = 0, "center"
    else:  # angle — muchos bins, rotar
        tick_labels = [str(v) for v in idx]
        rot, ha = 45, "right"

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=rot, ha=ha)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

# ── Guardado ───────────────────────────────────────────────────────────
for out in [OUT_DIAG, OUT_THESIS]:
    fig.savefig(out, dpi=SAVE_DPI)
    print(f"Saved → {out}")

plt.show()
print("Done.")
