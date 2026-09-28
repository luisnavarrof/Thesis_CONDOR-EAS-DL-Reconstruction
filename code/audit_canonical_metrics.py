"""Recompute ALL Results-chapter metrics from the canonical test_predictions.npz,
validating against the saved artifacts (evaluation_metrics.json / performance_summary).
Used to reconcile the thesis text to the canonical (0.484-deg) model run.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, r2_score, accuracy_score,
                             precision_recall_fscore_support, mean_absolute_error)

d = np.load(Path(__file__).parent / "pipeline_artifacts" / "test_predictions.npz", allow_pickle=True)
p_prob = d["particle_probability"]; p_pred = d["particle_label_pred"]; p_true = d["particle_label_true"]
a_pred = d["angle_pred"].astype(float); a_true = d["angle_true"].astype(float)
e_pred = d["energy_pred"].astype(float); e_true = d["energy_true"].astype(float)

print("N test =", len(p_true))
# ---- gamma-hadron (label convention: 1 = gamma) ----
acc = accuracy_score(p_true, p_pred)
auc = roc_auc_score(p_true, p_prob)
prec, rec, f1, sup = precision_recall_fscore_support(p_true, p_pred, labels=[0, 1])
# efficiencies at threshold 0.5: eps_gamma = recall(gamma=1); eps_proton = fraction protons kept as gamma (FPR)
eps_g = rec[1]
fpr_p = 1 - rec[0]  # protons misid as gamma
Q = eps_g / np.sqrt(fpr_p) if fpr_p > 0 else float("nan")
print(f"\n[PARTICLE]  acc={acc:.4f}  AUC={auc:.4f}  F1(macro)={f1.mean():.4f}")
print(f"  proton : prec={prec[0]:.4f} rec={rec[0]:.4f} f1={f1[0]:.4f} sup={sup[0]}")
print(f"  gamma  : prec={prec[1]:.4f} rec={rec[1]:.4f} f1={f1[1]:.4f} sup={sup[1]}")
print(f"  eps_gamma={eps_g:.4f}  eps_proton(FPR)={fpr_p:.4f}  Q={Q:.3f}")

# stratified AUC by energy and by zenith bin
print("  stratified AUC by energy:")
for E in (300, 500, 800):
    m = e_true == E
    print(f"    {E} GeV: AUC={roc_auc_score(p_true[m], p_prob[m]):.4f}  (n={m.sum()})")
print("  stratified AUC by zenith bin:")
for lo in range(0, 40, 10):
    m = (a_true >= lo) & (a_true < lo + 10)
    if m.sum() > 0 and len(np.unique(p_true[m])) > 1:
        print(f"    [{lo},{lo+10}): AUC={roc_auc_score(p_true[m], p_prob[m]):.4f}  (n={m.sum()})")

# ---- angle ----
err = a_pred - a_true
mae_a = np.mean(np.abs(err)); rmse_a = np.sqrt(np.mean(err**2))
r2_a = r2_score(a_true, a_pred); bias_a = np.mean(err)
psf68 = np.percentile(np.abs(err), 68)
print(f"\n[ANGLE]  MAE={mae_a:.4f}  RMSE={rmse_a:.4f}  R2={r2_a:.4f}  bias={bias_a:+.4f}  PSF68={psf68:.4f}  RMSE/MAE={rmse_a/mae_a:.3f}")
frac5 = np.mean(np.abs(err) < 5.0)
print(f"  fraction |err|<5deg = {frac5:.4f}")

# ---- energy ----
erre = e_pred - e_true
mae_e = np.mean(np.abs(erre)); rmse_e = np.sqrt(np.mean(erre**2))
r2_e = r2_score(e_true, e_pred)
rel = erre / e_true
print(f"\n[ENERGY]  MAE={mae_e:.4f}  RMSE={rmse_e:.4f}  R2={r2_e:.4f}")
print(f"  pull (rel err): mean={rel.mean():+.4f}  std={rel.std():.4f}  std(ddof=1)={rel.std(ddof=1):.4f}")
# stratified resolution (std of rel err) and bias per particle x energy
print("  stratified (particle x energy): bias%% / resolution%%")
for lbl, name in [(0, "Proton"), (1, "Gamma")]:
    for E in (300, 500, 800):
        m = (p_true == lbl) & (e_true == E)
        r = rel[m]
        print(f"    {name:6s} {E}: bias={100*r.mean():+5.1f}  res={100*r.std():4.1f}  (n={m.sum()})")
