"""
Regenera las 6 figuras de resultados del Cap. 4 desde los artefactos guardados
(`pipeline_artifacts/test_predictions.npz` + `pipeline_artifacts/model/training_history.csv`)
con tamaño de fuente aumentado para legibilidad en la tesis impresa.

Produce:
    thesis/figures/training_curves.png
    thesis/figures/particle_results.png
    thesis/figures/roc_stratified.png
    thesis/figures/angle_results.png
    thesis/figures/energy_analysis.png
    thesis/figures/energy_migration_matrix.png

No requiere GPU ni reentrenar — solo lee los .npz/.csv ya escritos.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

# ----- paleta -----
CONDOR_DARKRED = "#6E1423"
CONDOR_RED     = "#A4243B"
CONDOR_INK     = "#1A1A1A"
CONDOR_GRAY    = "#9E9E9E"
CONDOR_CREAM   = "#F2E8E5"
LIGHTRED       = "#C97B86"
SAVE_DPI = 200

CMAP_SEQ = LinearSegmentedColormap.from_list(
    "condor_seq", [CONDOR_CREAM, LIGHTRED, CONDOR_RED, CONDOR_DARKRED, "#3D0A14"])

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
    "legend.fontsize": 12, "legend.frameon": False, "lines.linewidth": 1.8,
})

HERE = Path(__file__).resolve().parent
ART  = HERE / "pipeline_artifacts"
FIGS = HERE.parent / "thesis" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

pred = np.load(ART / "test_predictions.npz")
hist = pd.read_csv(ART / "model" / "training_history.csv")

# ============================================================ #
# 1. training_curves.png — 4 paneles: loss, F1, Angle MAE, E MAE #
# ============================================================ #
ep = np.arange(1, len(hist) + 1)
# Best epoch: minimum of the combined validation loss (matches the checkpoint
# actually restored by EarlyStopping with restore_best_weights=True).
best_ep = int(np.argmin(hist["val_loss"].to_numpy()) + 1)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

def _mark_best(ax):
    ax.axvline(best_ep, color=CONDOR_GRAY, ls=":", lw=1.2, zorder=1,
               label=f"Best epoch ({best_ep})")

ax = axes[0]
ax.plot(ep, hist["loss"], color=CONDOR_INK, label="Train")
ax.plot(ep, hist["val_loss"], color=CONDOR_DARKRED, label="Validation")
_mark_best(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss"); ax.set_title("Combined loss")
ax.legend()

ax = axes[1]
ax.plot(ep, hist["particle_output_f1_score"], color=CONDOR_INK, label="Train")
ax.plot(ep, hist["val_particle_output_f1_score"], color=CONDOR_DARKRED, label="Val")
_mark_best(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("F1 score")
ax.set_title(r"Gamma-hadron F1")
ax.legend()

ax = axes[2]
ax.plot(ep, hist["angle_output_mae"], color=CONDOR_INK, label="Train")
ax.plot(ep, hist["val_angle_output_mae"], color=CONDOR_DARKRED, label="Val")
_mark_best(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel(r"Angle MAE [$\degree$]")
ax.set_title("Angular MAE")
ax.legend()

ax = axes[3]
ax.plot(ep, hist["energy_output_mae"], color=CONDOR_INK, label="Train")
ax.plot(ep, hist["val_energy_output_mae"], color=CONDOR_DARKRED, label="Val")
_mark_best(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("Energy MAE [GeV]")
ax.set_title("Energy MAE")
ax.legend()

fig.tight_layout()
fig.savefig(FIGS / "training_curves.png")
plt.close(fig)
print(f"  best_epoch = {best_ep}  (min val_loss = {hist['val_loss'].min():.4f})")
print("Saved:", FIGS / "training_curves.png")

# ============================================================ #
# 2. particle_results.png — confusion matrix + ROC               #
# ============================================================ #
y_true = pred["particle_label_true"]
y_pred = pred["particle_label_pred"]
y_prob = pred["particle_probability"]

cm = confusion_matrix(y_true, y_pred, normalize="true")
cm_counts = confusion_matrix(y_true, y_pred)  # raw event counts per cell
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

ax = axes[0]
im = ax.imshow(cm, cmap=CMAP_SEQ, vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Proton", r"$\gamma$"])
ax.set_yticklabels(["Proton", r"$\gamma$"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion matrix (normalized)")
# Each cell shows the row-normalized fraction (large) and the raw event count
# below it (small), so both the rate and the absolute support are legible.
for i in range(2):
    for j in range(2):
        txt_color = "white" if cm[i, j] > 0.5 else CONDOR_INK
        ax.text(j, i - 0.10, f"{cm[i, j]:.3f}", ha="center", va="center",
                color=txt_color, fontsize=16, fontweight="bold")
        ax.text(j, i + 0.16, f"n = {cm_counts[i, j]:,}", ha="center", va="center",
                color=txt_color, fontsize=10.5)

fpr, tpr, _ = roc_curve(y_true, y_prob)
auc_v = auc(fpr, tpr)
ax = axes[1]
ax.plot(fpr, tpr, color=CONDOR_DARKRED, lw=2.2, label=f"AUC = {auc_v:.3f}")
ax.plot([0, 1], [0, 1], "--", color=CONDOR_GRAY, lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "particle_results.png")
plt.close(fig)
print("Saved:", FIGS / "particle_results.png")

# ============================================================ #
# 3. roc_stratified.png — ROC por energia (izq) y zenit (der)    #
# ============================================================ #
energies = list(pred["energy_levels"])
e_true_str = pred["energy_true_str"]
colors = {energies[0]: LIGHTRED, energies[1]: CONDOR_RED, energies[2]: CONDOR_DARKRED}
labels = {"3E2": "300 GeV", "5E2": "500 GeV", "8E2": "800 GeV"}

fig, (axE, axZ) = plt.subplots(1, 2, figsize=(12, 5.2))

# --- left: stratified by primary energy ---
for e in energies:
    m = e_true_str == e
    if m.sum() == 0:
        continue
    fpr_e, tpr_e, _ = roc_curve(y_true[m], y_prob[m])
    auc_e = auc(fpr_e, tpr_e)
    axE.plot(fpr_e, tpr_e, color=colors[e], lw=2.2,
             label=f"{labels.get(e, e)} (AUC={auc_e:.3f})")
axE.plot([0, 1], [0, 1], "--", color=CONDOR_GRAY, lw=1)
axE.set_xlabel("False positive rate"); axE.set_ylabel("True positive rate")
axE.set_title("Stratified by primary energy")
axE.set_xlim(0, 1); axE.set_ylim(0, 1.02)
axE.legend(loc="lower right")

# --- right: stratified by zenith-angle bin ---
ang_true_cls = pred["angle_true"]
zbins = [(0, 10), (10, 20), (20, 30), (30, 40)]
zcolors = [LIGHTRED, "#B85566", CONDOR_RED, CONDOR_DARKRED]
zenith_aucs = {}
for (lo, hi), col in zip(zbins, zcolors):
    m = (ang_true_cls >= lo) & (ang_true_cls < hi)
    if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
        continue
    fpr_z, tpr_z, _ = roc_curve(y_true[m], y_prob[m])
    auc_z = auc(fpr_z, tpr_z)
    zenith_aucs[(lo, hi)] = auc_z
    axZ.plot(fpr_z, tpr_z, color=col, lw=2.2,
             label=fr"${lo}$--${hi}\degree$ (AUC={auc_z:.3f})")
axZ.plot([0, 1], [0, 1], "--", color=CONDOR_GRAY, lw=1)
axZ.set_xlabel("False positive rate"); axZ.set_ylabel("True positive rate")
axZ.set_title("Stratified by zenith angle")
axZ.set_xlim(0, 1); axZ.set_ylim(0, 1.02)
axZ.legend(loc="lower right")

fig.tight_layout()
fig.savefig(FIGS / "roc_stratified.png")
plt.close(fig)
if zenith_aucs:
    zvals = list(zenith_aucs.values())
    print(f"  zenith AUCs: " + ", ".join(f"{lo}-{hi}deg={v:.3f}"
          for (lo, hi), v in zenith_aucs.items()))
    print(f"  zenith AUC range: {min(zvals):.3f}-{max(zvals):.3f}")
print("Saved:", FIGS / "roc_stratified.png")

# ============================================================ #
# 4. angle_results.png — abs error | relative error | linearity  #
# ============================================================ #
ang_t = pred["angle_true"]; ang_p = pred["angle_pred"]
res = ang_p - ang_t
abs_err = np.abs(res)
ptype_a = pred["particle_label_true"]
isg_a = ptype_a == 1; isp_a = ptype_a == 0
mae_a = float(abs_err.mean()); bias_a = float(res.mean())
psf68 = float(np.percentile(abs_err, 68))

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

# Panel 1: absolute error distribution by particle type
ax = axes[0]
bins = np.linspace(0, np.percentile(abs_err, 99), 60)
ax.hist(abs_err[isp_a], bins=bins, color=CONDOR_INK, alpha=0.55, label="Proton", density=True)
ax.hist(abs_err[isg_a], bins=bins, color=CONDOR_DARKRED, alpha=0.6, label=r"$\gamma$", density=True)
ax.axvline(psf68, color=CONDOR_GRAY, ls="--", lw=1.3, label=f"PSF$_{{68}}$ = {psf68:.2f}$\\degree$")
ax.set_xlabel(r"Absolute error $|\hat{\theta}-\theta_\mathrm{true}|$ [$\degree$]")
ax.set_ylabel("Density")
ax.set_title("Absolute error by particle type")
ax.text(0.97, 0.55, f"MAE = {mae_a:.3f}$\\degree$\nBias = {bias_a:+.3f}$\\degree$",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
ax.legend(loc="upper right")

# Panel 2: relative error by particle type (exclude theta_true = 0)
ax = axes[1]
nz = ang_t > 1e-6
rel_a = res[nz] / ang_t[nz]
isg_nz = isg_a[nz]; isp_nz = isp_a[nz]
rb = np.linspace(-1, 1, 60)
ax.hist(rel_a[isp_nz], bins=rb, color=CONDOR_INK, alpha=0.55, label="Proton", density=True)
ax.hist(rel_a[isg_nz], bins=rb, color=CONDOR_DARKRED, alpha=0.6, label=r"$\gamma$", density=True)
ax.axvline(0, color=CONDOR_INK, lw=1)
ax.set_xlabel(r"Relative error $(\hat{\theta}-\theta_\mathrm{true})/\theta_\mathrm{true}$")
ax.set_ylabel("Density")
ax.set_title("Relative error by particle type")
ax.legend(loc="upper right")

# Panel 3: linearity profile E[theta_hat | theta_true] with +/- 1 sigma.
# NOTE: theta_true is DISCRETE in 2-degree steps (0, 2, 4, ..., 40 deg from the
# CORSIKA sampling). Using arbitrary bin centers as x-coords creates a false
# visual offset (a bin [0, 2) contains only theta_true = 0, so E[theta_true|bin]
# is 0, not the bin center 1). We therefore use the empirical mean of theta_true
# per bin as the x-coordinate, so the profile sits exactly where the data live.
ax = axes[2]
edges = np.linspace(0, 40, 21)
xs, means, stds = [], [], []
for i in range(len(edges) - 1):
    m = (ang_t >= edges[i]) & (ang_t < edges[i + 1])
    if m.sum() > 0:
        xs.append(float(ang_t[m].mean()))
        means.append(float(ang_p[m].mean()))
        stds.append(float(ang_p[m].std()))
xs = np.asarray(xs); means = np.asarray(means); stds = np.asarray(stds)
# Ideal reference: thin, light, behind the data.
ax.plot([xs.min(), xs.max()], [xs.min(), xs.max()], "--", color=CONDOR_GRAY,
        lw=1.0, zorder=1, label="Ideal")
ax.fill_between(xs, means - stds, means + stds, color=CONDOR_DARKRED, alpha=0.20,
                zorder=2, label=r"$\pm 1\sigma$")
ax.plot(xs, means, color=CONDOR_DARKRED, lw=2, marker="o", ms=4, zorder=3,
        label=r"$\mathbb{E}[\hat{\theta}\,|\,\theta_\mathrm{true}]$")
ax.set_xlabel(r"True zenith angle $\theta_\mathrm{true}$ [$\degree$]")
ax.set_ylabel(r"Reconstructed $\hat{\theta}$ [$\degree$]")
ax.set_title("Linearity profile")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "angle_results.png")
plt.close(fig)
print(f"  angle: MAE={mae_a:.3f} bias={bias_a:+.3f} PSF68={psf68:.3f}")
print("Saved:", FIGS / "angle_results.png")

# ============================================================ #
# 5. energy_analysis.png — overall pull | pull stratified by class #
# ============================================================ #
E_t = pred["energy_true"]; E_p = pred["energy_pred"]
res_E = E_p - E_t
rel = res_E / E_t
ptype_e = pred["particle_label_true"]
isg_e = ptype_e == 1; isp_e = ptype_e == 0
mu, sigma = float(rel.mean()), float(rel.std())
mu_p, sig_p = float(rel[isp_e].mean()), float(rel[isp_e].std())
mu_g, sig_g = float(rel[isg_e].mean()), float(rel[isg_e].std())

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: overall pull distribution with a Gaussian reference curve
ax = axes[0]
rb = np.linspace(-1, 2, 70)
ax.hist(rel, bins=rb, color=CONDOR_RED, alpha=0.85, edgecolor=CONDOR_INK,
        linewidth=0.3, density=True)
xx = np.linspace(-1, 2, 300)
gauss = np.exp(-0.5 * ((xx - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
ax.plot(xx, gauss, color=CONDOR_INK, lw=1.8,
        label=f"Gaussian fit\n$\\mu$={mu:+.3f}, $\\sigma$={sigma:.3f}")
ax.axvline(0, color=CONDOR_INK, lw=1)
ax.set_xlabel(r"Pull $(\hat{E}-E_\mathrm{true})/E_\mathrm{true}$")
ax.set_ylabel("Density")
ax.set_title("Energy pull distribution")
ax.legend(loc="upper right")

# Panel 2: pull stratified by particle type. Both classes are unbiased at the
# few-percent level, but gamma-induced showers have roughly twice the width of
# proton-induced ones -- proton showers reach many detectors with a more
# stable lateral profile at fixed energy, while gamma showers depend more
# sensitively on the electromagnetic core position.
ax = axes[1]
ax.hist(rel[isp_e], bins=rb, color=CONDOR_INK, alpha=0.55, density=True,
        label=fr"Proton: $\mu$={mu_p:+.3f}, $\sigma$={sig_p:.3f}")
ax.hist(rel[isg_e], bins=rb, color=CONDOR_DARKRED, alpha=0.60, density=True,
        label=fr"$\gamma$: $\mu$={mu_g:+.3f}, $\sigma$={sig_g:.3f}")
ax.axvline(0, color=CONDOR_INK, lw=1)
ax.set_xlabel(r"Pull $(\hat{E}-E_\mathrm{true})/E_\mathrm{true}$")
ax.set_ylabel("Density")
ax.set_title("Energy pull by particle type")
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "energy_analysis.png")
plt.close(fig)
print(f"  energy pull: mu={mu:+.3f} sigma={sigma:.3f}  "
      f"proton: mu={mu_p:+.3f} sigma={sig_p:.3f}  "
      f"gamma: mu={mu_g:+.3f} sigma={sig_g:.3f}")
print("Saved:", FIGS / "energy_analysis.png")

# ============================================================ #
# 6. energy_migration_matrix.png — 3x3 migración                 #
# ============================================================ #
levels = list(pred["energy_levels"])
level_to_idx = {l: i for i, l in enumerate(levels)}

# nearest level for prediction
def closest_level(e_pred):
    arr = np.array([float(l.replace("E", "e")) for l in levels])
    diffs = np.abs(arr[None, :] - e_pred[:, None])
    return np.argmin(diffs, axis=1)

idx_true = np.array([level_to_idx[s] for s in e_true_str])
idx_pred = closest_level(E_p)
mig = confusion_matrix(idx_true, idx_pred, normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mig, cmap=CMAP_SEQ, vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([labels[l] for l in levels])
ax.set_yticklabels([labels[l] for l in levels])
ax.set_xlabel("Reconstructed energy bin")
ax.set_ylabel("True energy")
ax.set_title("Energy migration matrix (row-normalized)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{mig[i, j]:.2f}", ha="center", va="center",
                color="white" if mig[i, j] > 0.5 else CONDOR_INK, fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")
fig.tight_layout()
fig.savefig(FIGS / "energy_migration_matrix.png")
plt.close(fig)
print("Saved:", FIGS / "energy_migration_matrix.png")

print("\nAll 6 results figures regenerated.")
