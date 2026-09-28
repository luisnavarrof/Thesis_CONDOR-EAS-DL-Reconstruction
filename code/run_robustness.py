"""
Standalone runner for the detector-robustness study (thesis Section 4.7 / v2 notebook Section 6).

Why this exists
---------------
The robustness study only ever lived inside the v2 notebook builder
(build_improved_notebook.py, Section 6) and its results were never persisted as
data -- only the PNG figures were saved. That made the thesis §4.7 narrative
impossible to verify against numbers and impossible to re-analyse without a full
notebook re-run. This script fixes that: it runs the study in a FRESH process
with a clean GPU, loads the ALREADY-TRAINED model (no retraining), and writes
EVERY intermediate result to CSV/JSON in pipeline_artifacts/diagnostics/ so any
downstream question (which region degrades most per task, percentage impact,
geometry trade-off, ...) can be answered by opening a file.

It reuses the main notebook's own pipeline cells verbatim (so it cannot drift
from the canonical data pipeline), exactly like run_validation.py.

Outputs (pipeline_artifacts/diagnostics/)
-----------------------------------------
  robustness_baseline.json            baseline metrics (all detectors on)
  robustness_single_detector.csv      per-detector degradation + geometry
  robustness_progressive_raw.csv      every (frac, rep) evaluation
  robustness_progressive_stats.csv    mean/std per fraction
  robustness_regional.csv             per-group degradation, absolute AND % of baseline
  robustness_geometry.csv             random-thinning / central-crop / checkerboard
  dropout_single_detector.png         regenerated (perceptually-uniform colormap)
  dropout_progressive.png             regenerated
  dropout_regional.png                regenerated
  dropout_geometry_tradeoff.png       regenerated

Run (env MUST be active so TF sees CUDA):
  conda run -n condor-tf210-gpu --no-capture-output python -u run_robustness.py
Tunables via env: ROB_SUBSAMPLE (default 3000), ROB_BATCH (default 16).
"""
import os, sys, json, time, gc
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # notebook cells resolve BASE_DIR from the working directory
NB = os.path.join(HERE, "CONDOR_EAS-Reconstruction.ipynb")
MODEL_PATH = os.path.join(HERE, "pipeline_artifacts", "model", "condor_multitask_model.keras")
OUT = os.path.join(HERE, "pipeline_artifacts", "diagnostics")
os.makedirs(OUT, exist_ok=True)

ROB_SUBSAMPLE = int(os.environ.get("ROB_SUBSAMPLE", "3000"))
ROB_BATCH = int(os.environ.get("ROB_BATCH", "16"))
# Which blocks to run. Default = everything, as before. Set e.g.
# ROB_STEPS=regional to re-run only the group study after changing the group
# definitions, instead of paying again for the 120 single-detector evaluations
# and the geometry scan, which are unaffected by that change.
ROB_STEPS = set(os.environ.get(
    "ROB_STEPS", "single,progressive,regional,geometry").replace(" ", "").split(","))


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: run the notebook's data pipeline (verbatim) -> data + helper functions
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1/6  Running notebook data pipeline (fresh process, clean GPU)")
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
ns = {"__name__": "__main__"}

# Same pipeline cells as run_validation.py: data load + filters + balancing +
# sequences + globals + split. Excludes plots, model build, training, Section 6.
PIPE = [2, 4, 6, 8, 9, 13, 15, 20, 22, 24]
t0 = time.time()
for i in PIPE:
    src = "".join(cells[i]["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"[pipeline ready in {time.time()-t0:.0f}s]", flush=True)

# Pull the symbols we need out of the notebook namespace.
X_sequences          = ns["X_sequences"]
test_idx             = ns["test_idx"]
y_label_test         = ns["y_label_test"]
y_angle_test         = ns["y_angle_test"]
y_energy_test_cont   = ns["y_energy_test_cont"]
compute_global_features = ns["compute_global_features"]
central_ids          = ns["central_ids"]
max_sequence_length  = ns["max_sequence_length"]
energy_mean          = ns["energy_mean"]
energy_std           = ns["energy_std"]
detector_catalog     = ns["detector_catalog"]
SEED                 = ns.get("SEED", 42)

for col in ("detector_id", "x_center", "y_center", "distance"):
    assert col in detector_catalog.columns, f"detector_catalog missing '{col}'"
print(f"  test events={len(test_idx)}  detectors={len(detector_catalog)}  "
      f"maxlen={max_sequence_length}  SEED={SEED}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: load the already-trained model (no retraining)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2/6  Loading trained model (compile=False, inference only)")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"  loaded {os.path.basename(MODEL_PATH)}  "
      f"inputs={[tuple(i.shape) for i in model.inputs]}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Visual identity (mirrors build_improved_notebook.py) + perceptually-uniform map
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DPI = 200
CONDOR_DARKRED = "#6E1423"
CONDOR_RED     = "#A4243B"
CONDOR_INK     = "#1A1A1A"
CONDOR_GRAY    = "#9E9E9E"
CONDOR_CREAM   = "#F2E8E5"
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "condor_seq", [CONDOR_CREAM, "#C97B86", CONDOR_RED, CONDOR_DARKRED, "#3D0A14"])
# For the single-detector map, CMAP_SEQ compresses the dark end (high and mid
# values look identical). Use a perceptually-uniform, on-brand dark-red ramp so
# the most-critical detectors are clearly distinguishable (thesis comment #9).
CMAP_IMPORTANCE = sns.color_palette("rocket_r", as_cmap=True)

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
    "legend.fontsize": 12, "legend.frameon": False, "lines.linewidth": 1.8,
})

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: evaluation subsample + core switch-off evaluator (verbatim logic)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3/6  Building evaluation subsample and baseline")
test_sequences = [X_sequences[i] for i in test_idx]
rng_drop = np.random.default_rng(SEED)
if ROB_SUBSAMPLE and ROB_SUBSAMPLE < len(test_sequences):
    _sel = np.sort(rng_drop.choice(len(test_sequences), ROB_SUBSAMPLE, replace=False))
else:
    _sel = np.arange(len(test_sequences))

drop_sequences = [test_sequences[i] for i in _sel]
drop_label  = y_label_test[_sel].astype(int)
drop_angle  = y_angle_test[_sel].astype(float)
drop_energy = y_energy_test_cont[_sel].astype(float)
print(f"  detector-dropout evaluation set: {len(drop_sequences)} events "
      f"(batch={ROB_BATCH})", flush=True)


def evaluate_with_detectors_off(off_ids):
    """Drop all hits from off_ids, recompute global features, return task metrics.

    Events that lose ALL hits after removal are excluded (all-zero sequences give
    NaN under LayerNorm/attention). Metrics use the subset with >= 1 hit.
    """
    off = np.asarray(list(off_ids), dtype=np.int32)
    mod, keep = [], []
    for seq in drop_sequences:
        filtered = seq if (off.size == 0 or seq.size == 0) else \
                   seq[~np.isin(seq[:, 0].astype(np.int32), off)]
        mod.append(filtered)
        keep.append(filtered.size > 0)
    keep = np.array(keep, dtype=bool)

    mod_v = [m for m, k in zip(mod, keep) if k]
    lbl_v = drop_label[keep]
    ang_v = drop_angle[keep]
    en_v  = drop_energy[keep]

    _NAN = {k: np.nan for k in ("particle_acc", "particle_auc", "angle_mae", "energy_mae")}
    if len(mod_v) == 0:
        return _NAN

    Xg = compute_global_features(mod_v, central_ids)
    Xs = pad_sequences(mod_v, maxlen=max_sequence_length, padding="post", dtype="float32")
    p_part, p_ang, p_en = model.predict([Xs, Xg], batch_size=ROB_BATCH, verbose=0)
    p_part = p_part.ravel()
    p_ang  = p_ang.ravel()
    en_gev = p_en.ravel() * energy_std + energy_mean
    gc.collect()

    ok = np.isfinite(p_part) & np.isfinite(p_ang) & np.isfinite(en_gev)
    if ok.sum() == 0:
        return _NAN
    p_part, p_ang, en_gev = p_part[ok], p_ang[ok], en_gev[ok]
    lbl_v, ang_v, en_v = lbl_v[ok], ang_v[ok], en_v[ok]

    auc_val = (float(roc_auc_score(lbl_v, p_part))
               if len(np.unique(lbl_v)) == 2 else np.nan)
    return {
        "particle_acc": float(np.mean((p_part > 0.5).astype(int) == lbl_v)),
        "particle_auc": auc_val,
        "angle_mae":    float(np.mean(np.abs(p_ang - ang_v))),
        "energy_mae":   float(np.mean(np.abs(en_gev - en_v))),
    }


baseline = evaluate_with_detectors_off([])
print("  baseline:", {k: round(v, 4) for k, v in baseline.items()}, flush=True)
with open(os.path.join(OUT, "robustness_baseline.json"), "w") as f:
    json.dump({"baseline": baseline,
               "n_events": int(len(drop_sequences)),
               "subsample": ROB_SUBSAMPLE, "seed": int(SEED),
               "batch": ROB_BATCH}, f, indent=2)


def pct_columns(df):
    """Add relative-degradation (%) columns w.r.t. baseline, for every task."""
    df["acc_drop_pct"]    = 100.0 * df["acc_drop"]    / baseline["particle_acc"]
    df["angle_rise_pct"]  = 100.0 * df["angle_rise"]  / baseline["angle_mae"]
    df["energy_rise_pct"] = 100.0 * df["energy_rise"] / baseline["energy_mae"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4a: 6.1 single-detector importance
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 4/6  6.1 Single-detector importance")
all_detectors = detector_catalog["detector_id"].astype(int).tolist()
if "single" in ROB_STEPS:
    single_rows = []
    for k, det in enumerate(all_detectors):
        single_rows.append({"detector_id": det, **evaluate_with_detectors_off([det])})
        if (k + 1) % 25 == 0:
            print(f"  evaluated {k + 1}/{len(all_detectors)} detectors", flush=True)

    single_df = pd.DataFrame(single_rows).merge(
        detector_catalog[["detector_id", "x_center", "y_center", "distance"]], on="detector_id")
    single_df["acc_drop"]    = baseline["particle_acc"] - single_df["particle_acc"]
    single_df["angle_rise"]  = single_df["angle_mae"]  - baseline["angle_mae"]
    single_df["energy_rise"] = single_df["energy_mae"] - baseline["energy_mae"]
    single_df = pct_columns(single_df)
    single_df.round(6).to_csv(os.path.join(OUT, "robustness_single_detector.csv"), index=False)

    maps = [("acc_drop", "Particle - accuracy loss"),
            ("angle_rise", r"Angle - MAE rise ($^\circ$)"),
            ("energy_rise", "Energy - MAE rise (GeV)")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.3), constrained_layout=True)
    for ax, (col, title) in zip(axes, maps):
        sc = ax.scatter(single_df["x_center"], single_df["y_center"],
                        c=single_df[col].to_numpy(), cmap=CMAP_IMPORTANCE,
                        s=180, edgecolor=CONDOR_INK, linewidth=0.6)
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Degradation when this detector is off")
    fig.suptitle("Single-detector importance map", fontweight="bold")
    fig.savefig(os.path.join(OUT, "dropout_single_detector.png"), dpi=SAVE_DPI)
    plt.close(fig)
else:
    print("  [single-detector block skipped: not in ROB_STEPS]", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: 6.2 progressive random failure
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5/6  6.2 Progressive random failure  |  6.3 Regional  |  6.4 Geometry")
if "progressive" in ROB_STEPS:
    frac_grid = np.linspace(0.0, 0.9, 13)
    N_REPS = 4
    det_array = np.array(all_detectors)
    prog_rows = []
    for frac in frac_grid:
        n_off = int(round(frac * len(det_array)))
        for rep in range(N_REPS):
            m = baseline if n_off == 0 else \
                evaluate_with_detectors_off(rng_drop.choice(det_array, n_off, replace=False))
            prog_rows.append({"frac": frac, "n_off": n_off, "rep": rep, **m})
    prog_df = pd.DataFrame(prog_rows)
    prog_df.round(6).to_csv(os.path.join(OUT, "robustness_progressive_raw.csv"), index=False)
    prog_stats = prog_df.groupby("frac").agg(["mean", "std"])
    prog_stats.round(6).to_csv(os.path.join(OUT, "robustness_progressive_stats.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.6), constrained_layout=True)
    panels = [("particle_acc", "Particle accuracy"),
              ("angle_mae", r"Angle MAE ($^\circ$)"),
              ("energy_mae", "Energy MAE (GeV)")]
    xx = prog_stats.index.to_numpy() * 100.0
    for ax, (col, title) in zip(axes, panels):
        mu = prog_stats[(col, "mean")].to_numpy()
        sd = np.nan_to_num(prog_stats[(col, "std")].to_numpy())
        ax.plot(xx, mu, color=CONDOR_DARKRED, marker="o", ms=4)
        ax.fill_between(xx, mu - sd, mu + sd, color=CONDOR_DARKRED, alpha=0.2)
        ax.axhline(baseline[col], color=CONDOR_INK, ls="--", lw=1.0, label="baseline")
        ax.set_title(title); ax.set_xlabel("Detectors switched off (%)"); ax.set_ylabel(title)
        ax.legend()
    fig.suptitle("Reconstruction under progressive random detector failure", fontweight="bold")
    fig.savefig(os.path.join(OUT, "dropout_progressive.png"), dpi=SAVE_DPI)
    plt.close(fig)
else:
    print("  [progressive block skipped: not in ROB_STEPS]", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4c: 6.3 regional dropout (the block the thesis §4.7 narrative rests on)
# ─────────────────────────────────────────────────────────────────────────────
if "regional" in ROB_STEPS:
    # Group definitions, revised 19-ago-2026 (thesis comment). Two changes:
    #
    # (a) THREE radial groups instead of five. The old five (core 16 / inner 32 /
    #     mid 32 / outer 20 / peripheral 20) were finer than this measurement can
    #     resolve and mixed sizes 16..32, so a group could rank high simply for
    #     holding more detectors. The three used now follow how the array is
    #     actually built: the 100-module central grid, split into its 36 innermost
    #     modules and the 64 that surround them, plus the 20 peripheral modules,
    #     which are a structurally separate sub-array. The peripheral set is now
    #     taken by construction (detector_id >= 100) instead of by distance rank --
    #     the old rank cut silently swapped four of them for the four corners of the
    #     central grid, which sit further out (55.7 m) than the nearest peripheral
    #     modules (52.2 m).
    #
    # (b) HALVES instead of the two upper quadrants. Evaluating only (+x,+y) and
    #     (-x,+y) left "why not the other two?" unanswered, and quadrants hold 30
    #     detectors against the 36/64/20 of the radial groups. The four halves are
    #     all exactly 60 detectors, cover the array completely, and are the natural
    #     probe for the one asymmetry this dataset has (fixed azimuth phi = 0).
    grid = detector_catalog[detector_catalog["detector_id"] < 100]
    core36 = grid.nsmallest(36, "distance")["detector_id"].astype(int).tolist()
    _ids = lambda m: detector_catalog[m]["detector_id"].astype(int).tolist()
    regions = {
        "Central core (36)": core36,
        "Outer grid (64)":   [d for d in grid["detector_id"].astype(int).tolist()
                              if d not in set(core36)],
        "Peripheral (20)":   _ids(detector_catalog["detector_id"] >= 100),
        "Half $x<0$":        _ids(detector_catalog["x_center"] < 0),
        "Half $x>0$":        _ids(detector_catalog["x_center"] > 0),
        "Half $y<0$":        _ids(detector_catalog["y_center"] < 0),
        "Half $y>0$":        _ids(detector_catalog["y_center"] > 0),
    }
    print("  group sizes:", {k: len(v) for k, v in regions.items()}, flush=True)
    reg_rows = []
    for name, ids in regions.items():
        reg_rows.append({"region": name, "n_off": len(ids), **evaluate_with_detectors_off(ids)})
    reg_df = pd.DataFrame(reg_rows)
    reg_df["acc_drop"]    = baseline["particle_acc"] - reg_df["particle_acc"]
    reg_df["angle_rise"]  = reg_df["angle_mae"]  - baseline["angle_mae"]
    reg_df["energy_rise"] = reg_df["energy_mae"] - baseline["energy_mae"]
    reg_df = pct_columns(reg_df)
    # Groups differ in size (36 / 64 / 20 / 60) and a bigger group damages more just
    # by being bigger. Report the impact PER DETECTOR REMOVED as well, so the two
    # readings -- "what does losing this block cost?" and "how valuable is a single
    # detector inside it?" -- can be told apart.
    for c in ("acc_drop", "angle_rise", "energy_rise"):
        reg_df[c + "_per_det"] = reg_df[c] / reg_df["n_off"]
    reg_df.round(8).to_csv(os.path.join(OUT, "robustness_regional.csv"), index=False)
    print("\n  Regional degradation (sorted by accuracy loss):", flush=True)
    print(reg_df.sort_values("acc_drop", ascending=False)[
        ["region", "n_off", "acc_drop", "acc_drop_pct", "angle_rise", "angle_rise_pct",
         "energy_rise", "energy_rise_pct"]].round(4).to_string(index=False), flush=True)
    print("\n  Per detector removed (size-normalized):", flush=True)
    print(reg_df.sort_values("acc_drop_per_det", ascending=False)[
        ["region", "n_off", "acc_drop_per_det", "angle_rise_per_det",
         "energy_rise_per_det"]].round(6).to_string(index=False), flush=True)

    order = reg_df.sort_values("acc_drop")["region"].tolist()
    d = reg_df.set_index("region").loc[order]
    _cols = [CONDOR_GRAY if "Half" in r else CONDOR_DARKRED for r in d.index]
    bars = [("acc_drop", "Accuracy loss"),
            ("angle_rise", r"Angle MAE rise ($^\circ$)"),
            ("energy_rise", "Energy MAE rise (GeV)")]
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.0), constrained_layout=True)
    for j, (col, title) in enumerate(bars):
        axes[0, j].barh(d.index, d[col], color=_cols, edgecolor=CONDOR_INK)
        axes[0, j].set_title(title); axes[0, j].set_xlabel(title)
        axes[1, j].barh(d.index, d[col + "_per_det"], color=_cols, edgecolor=CONDOR_INK)
        axes[1, j].set_title(title + "  per detector removed", fontsize=13)
        axes[1, j].set_xlabel(title + " / detector")
    fig.suptitle("Impact of switching off whole detector groups   "
                 "(top: whole block   bottom: normalized by group size   |   "
                 "dark red = radial groups, grey = halves)", fontweight="bold")
    fig.savefig(os.path.join(OUT, "dropout_regional.png"), dpi=SAVE_DPI)
    plt.close(fig)
else:
    print("  [regional block skipped: not in ROB_STEPS]", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4d: 6.4 array geometry trade-off
# ─────────────────────────────────────────────────────────────────────────────
if "geometry" in ROB_STEPS:
    SPACING = 8.7
    det_xy = detector_catalog.set_index("detector_id")[["x_center", "y_center"]]
    n_det = len(det_array)

    random_rows = []
    for kf in np.linspace(0.2, 1.0, 11):
        n_off = int(round((1.0 - kf) * n_det))
        for rep in range(3):
            off = rng_drop.choice(det_array, n_off, replace=False) if n_off > 0 else []
            random_rows.append({"n_kept": n_det - n_off, "strategy": "random_thinning",
                                "rep": rep, **evaluate_with_detectors_off(off)})
    random_raw = pd.DataFrame(random_rows)
    random_df = random_raw.groupby("n_kept").mean(numeric_only=True).reset_index()

    crop_rows = []
    for R in np.linspace(detector_catalog["distance"].quantile(0.25),
                         detector_catalog["distance"].max(), 9):
        off = detector_catalog[detector_catalog["distance"] > R]["detector_id"].tolist()
        crop_rows.append({"n_kept": n_det - len(off), "strategy": "central_crop",
                          "radius": float(R), **evaluate_with_detectors_off(off)})
    crop_df = pd.DataFrame(crop_rows)

    grid_idx = (np.round(det_xy["x_center"] / SPACING) + np.round(det_xy["y_center"] / SPACING))
    checker_off = det_xy.index[(grid_idx % 2 == 1).to_numpy()].tolist()
    m_check = evaluate_with_detectors_off(checker_off)
    check_kept = n_det - len(checker_off)

    # Persist the geometry study as one tidy table.
    geo_out = pd.concat([
        random_raw.assign(radius=np.nan),
        crop_df.assign(rep=np.nan),
        pd.DataFrame([{"n_kept": check_kept, "strategy": "checkerboard", "rep": np.nan,
                       "radius": np.nan, **m_check}]),
    ], ignore_index=True)
    geo_out.round(6).to_csv(os.path.join(OUT, "robustness_geometry.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), constrained_layout=True)
    geo = [("particle_acc", "Particle accuracy"),
           ("angle_mae", r"Angle MAE ($^\circ$)"),
           ("energy_mae", "Energy MAE (GeV)")]
    for ax, (col, title) in zip(axes, geo):
        ax.plot(random_df["n_kept"], random_df[col], color=CONDOR_GRAY, marker="o", ms=4,
                label="Random thinning")
        ax.plot(crop_df["n_kept"], crop_df[col], color=CONDOR_DARKRED, marker="s", ms=5,
                label="Central crop")
        ax.scatter([check_kept], [m_check[col]], color=CONDOR_RED, s=120, marker="D",
                   zorder=5, edgecolor=CONDOR_INK, label="Checkerboard")
        ax.set_title(title); ax.set_xlabel("Detectors kept"); ax.set_ylabel(title)
        ax.legend()
    fig.suptitle("Array geometry trade-off - same detector count, different layout",
                 fontweight="bold")
    fig.savefig(os.path.join(OUT, "dropout_geometry_tradeoff.png"), dpi=SAVE_DPI)
    plt.close(fig)
else:
    print("  [geometry block skipped: not in ROB_STEPS]", flush=True)

# Sync the regenerated figures into the thesis figures/ folder under the names the
# LaTeX expects, so figure == CSV == text stay consistent without a manual copy.
import shutil
FIGS = os.path.join(HERE, "..", "thesis", "figures")
if os.path.isdir(FIGS):
    for src_name, dst_name in [
        ("dropout_single_detector.png",   "dropout_single.png"),
        ("dropout_progressive.png",       "dropout_progressive.png"),
        ("dropout_regional.png",          "dropout_regional.png"),
        ("dropout_geometry_tradeoff.png", "dropout_geometry.png"),
    ]:
        shutil.copyfile(os.path.join(OUT, src_name), os.path.join(FIGS, dst_name))
    print("  synced dropout figures -> thesis/figures/", flush=True)

banner("STEP 6/6  Done. CSV/JSON + regenerated PNGs in pipeline_artifacts/diagnostics/")
for fn in ["robustness_baseline.json", "robustness_single_detector.csv",
           "robustness_progressive_raw.csv", "robustness_progressive_stats.csv",
           "robustness_regional.csv", "robustness_geometry.csv"]:
    print("  wrote", fn, flush=True)
