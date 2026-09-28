"""
Standalone runner for the attention-difference XAI figures (thesis §4.6 / v2 notebook §5.3).

Why this exists
---------------
Like the robustness study, the gamma-vs-proton attention-difference analysis only
lived inside the v2 notebook builder and its numbers were never persisted -- only
the PNGs were saved. This script re-extracts the RAW-stream attention in a FRESH
process, PERSISTS the per-position (temporal) and per-detector (spatial) attention
differences to CSV, and regenerates both figures. The spatial map uses a
perceptually-separated diverging colormap so the two extremes (proton vs gamma)
are clearly distinguishable (thesis comment #9 on Fig. 4.10).

It reuses the notebook's own pipeline cells + attention-extractor cells verbatim
(cannot drift), exactly like run_validation.py / run_robustness.py.

Outputs (pipeline_artifacts/diagnostics/)
-----------------------------------------
  attention_diff_temporal.csv     per task: position, att_gamma, att_proton, delta, null band
  attention_diff_spatial.csv      per task, per detector: x, y, att_gamma, att_proton, delta, null
  attention_null_summary.json     scale of the attention values + label-permutation null test
  attention_diff_temporal.png     regenerated (with permutation-null band)
  attention_diff_spatial.png      regenerated (diverging colormap with distinct extremes)

Why the null test (added 19-ago-2026)
-------------------------------------
A gamma-minus-proton attention difference is never exactly zero even when the
model treats both classes identically: it is estimated from a finite sample of
events, so it fluctuates. Without a reference scale, ANY structure in these maps
invites a physical story. We therefore recompute the same difference under
N_PERM random permutations of the particle labels (group sizes preserved), which
destroys any real class dependence while keeping event-to-event variability
intact. The resulting band is the amount of structure expected from sampling
noise alone; only excursions beyond it can carry information.

Run (env MUST be active so TF sees CUDA), AFTER run_robustness.py has finished
(the 4 GB GPU cannot host two TF processes at once):
  conda run -n condor-tf210-gpu --no-capture-output python -u run_attention.py
"""
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from tensorflow.keras import Model

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # notebook cells resolve BASE_DIR from the working directory
NB = os.path.join(HERE, "CONDOR_EAS-Reconstruction.ipynb")
MODEL_PATH = os.path.join(HERE, "pipeline_artifacts", "model", "condor_multitask_model.keras")
OUT = os.path.join(HERE, "pipeline_artifacts", "diagnostics")
FIGS = os.path.join(HERE, "..", "thesis", "figures")
os.makedirs(OUT, exist_ok=True)

SAMPLE_SIZE = int(os.environ.get("ATTN_SAMPLE", "200"))
ATTN_BATCH = int(os.environ.get("ATTN_BATCH", "16"))


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: notebook data pipeline (verbatim)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1/5  Running notebook data pipeline (fresh process, clean GPU)")
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
ns = {"__name__": "__main__"}
PIPE = [2, 4, 6, 8, 9, 13, 15, 20, 22, 24]
t0 = time.time()
for i in PIPE:
    src = "".join(cells[i]["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"[pipeline ready in {time.time()-t0:.0f}s]", flush=True)

X_test          = ns["X_test"]
Xg_test         = ns["Xg_test"]
y_label_test    = ns["y_label_test"]
detector_catalog = ns["detector_catalog"]
FEATURE_IDX     = ns["FEATURE_IDX"]
SEED            = ns.get("SEED", 42)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: load trained model + build attention extractor (cell 65 verbatim)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2/5  Loading model and building attention extractor")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
ns["model"] = model
ns["Model"] = Model
ns["tf"] = tf
ns["np"] = np
exec(compile("".join(cells[65]["source"]), "<cell 65>", "exec"), ns)
attention_model = ns["attention_model"]
print("  attention extractor built:",
      [o.name.split('/')[0] for o in attention_model.outputs], flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: extract RAW-stream attention on a fixed, reproducible sample
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3/5  Extracting RAW-stream attention scores")
np.random.seed(SEED)  # cell 66 uses global np.random; fix it for reproducibility
sample_size = min(SAMPLE_SIZE, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample  = X_test[sample_indices]
Xg_sample = Xg_test[sample_indices]

(_c_cnn, _a_cnn, _e_cnn,
 attn_class_raw_scores, attn_angle_raw_scores, attn_energy_raw_scores) = \
    attention_model.predict([X_sample, Xg_sample], batch_size=ATTN_BATCH, verbose=0)

valid_masks_raw = ~np.all(np.isclose(X_sample, 0.0, atol=1e-8), axis=-1)
valid_lengths_raw = valid_masks_raw.sum(axis=-1)

particle_types = y_label_test[sample_indices].astype(int)
is_gamma  = particle_types == 1
is_proton = particle_types == 0
print(f"  sample={sample_size}  gamma={is_gamma.sum()}  proton={is_proton.sum()}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Visual identity + helper functions (verbatim from build_improved_notebook.py)
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DPI = 200
CONDOR_DARKRED = "#6E1423"; CONDOR_RED = "#A4243B"; CONDOR_INK = "#1A1A1A"
CONDOR_GRAY = "#9E9E9E"; CONDOR_CREAM = "#F2E8E5"
# Diverging map with clearly distinct extremes (comment #9): proton -> deep blue,
# gamma -> CONDOR dark red, neutral cream center. Replaces the ink<->dark-red ramp
# whose two ends were visually indistinguishable.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "condor_div2", ["#1B4965", "#5FA8C4", CONDOR_CREAM, "#C97B86", CONDOR_DARKRED])
PROTON_BLUE = "#1B4965"
CMAP_ATT = LinearSegmentedColormap.from_list(
    "condor_att", ["#FFFFFF", CONDOR_CREAM, "#C97B86", CONDOR_RED,
                   CONDOR_DARKRED, "#3D0A14"])

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": SAVE_DPI, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelcolor": CONDOR_INK,
    "axes.edgecolor": CONDOR_INK, "axes.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#DCDCDC", "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "legend.fontsize": 12, "legend.frameon": False, "lines.linewidth": 1.8,
})


def _as3d(a):
    a = np.asarray(a)
    return a.mean(axis=1) if a.ndim == 4 else a


def mean_attention_profile(attn, valid_lengths, mask, max_len=60):
    attn = _as3d(attn)
    acc = np.zeros(max_len); cnt = np.zeros(max_len)
    for mat, L, keep in zip(attn, valid_lengths, mask):
        if not keep:
            continue
        Le = int(min(L, mat.shape[0], max_len))
        if Le <= 0:
            continue
        recv = mat[:Le, :Le].mean(axis=0)
        acc[:Le] += recv; cnt[:Le] += 1
    return acc / np.maximum(cnt, 1.0)


def per_detector_attention(attn, sequences, valid_lengths, mask, max_len=120):
    attn = _as3d(attn)
    acc, cnt = {}, {}
    for mat, seq, L, keep in zip(attn, sequences, valid_lengths, mask):
        if not keep:
            continue
        Le = int(min(L, mat.shape[0], seq.shape[0], max_len))
        if Le <= 0:
            continue
        recv = mat[:Le, :Le].mean(axis=0)
        for t in range(Le):
            det = int(seq[t, FEATURE_IDX["detector_id"]])
            acc[det] = acc.get(det, 0.0) + float(recv[t])
            cnt[det] = cnt.get(det, 0) + 1
    imp = {d: acc[d] / cnt[d] for d in acc}
    total = sum(imp.values()) or 1.0
    return {d: v / total for d, v in imp.items()}


tasks_attn = [
    ("Particle Classification", attn_class_raw_scores),
    ("Angle Reconstruction",    attn_angle_raw_scores),
    ("Energy Estimation",       attn_energy_raw_scores),
]

# Crop once to the largest window either estimator uses (temporal 60, spatial
# 120). Bit-identical results, ~15x less memory, and the permutation loop below
# then costs seconds instead of minutes.
CROP = 120
tasks_full = [(t, _as3d(a)) for t, a in tasks_attn]
tasks_attn = [(t, a[:, :CROP, :CROP].astype(np.float32)) for t, a in tasks_full]

# -----------------------------------------------------------------------------
# STEP 3b: what IS an attention value here?  (scale report + permutation null)
# -----------------------------------------------------------------------------
banner("STEP 3b/5  Attention scale report and label-permutation null")

null_summary = {"sample_size": int(sample_size),
                "n_gamma": int(is_gamma.sum()), "n_proton": int(is_proton.sum()),
                "seed": int(SEED), "tasks": {}}

# Scale: the scores come from a softmax over key positions, so each ROW of a
# matrix sums to 1. The profiles plotted below are column means (mean attention
# RECEIVED by a position), i.e. a distribution over positions -> a position of
# average importance sits at 1/L. Measure it instead of asserting it.
_L = valid_lengths_raw
print(f"  valid sequence length L: mean={_L.mean():.1f} median={np.median(_L):.0f} "
      f"min={_L.min()} max={_L.max()}", flush=True)
for title, attn_full in tasks_full:
    rs, inwin = [], []
    for mat, L in zip(attn_full, _L):
        Le = int(min(L, mat.shape[0]))
        if Le <= 0:
            continue
        rows = mat[:Le]                      # valid queries only
        rs.append(float(rows.sum(axis=1).mean()))          # over ALL keys
        inwin.append(float(rows[:, :min(Le, CROP)].sum(axis=1).mean()))
    print(f"  {title:<24} mean row sum over all keys = {np.mean(rs):.4f}  "
          f"(fraction inside the first {CROP} positions: {np.mean(inwin):.3f})",
          flush=True)
    null_summary["tasks"].setdefault(title, {})["mean_row_sum"] = float(np.mean(rs))
    null_summary["tasks"][title]["mass_in_first_%d" % CROP] = float(np.mean(inwin))
del tasks_full
null_summary["mean_valid_length"] = float(_L.mean())
null_summary["uniform_temporal_level"] = float(1.0 / _L.mean())

N_PERM = int(os.environ.get("ATTN_NPERM", "200"))
# 200 permutations put the p-value floor at 1/201 = 0.005; with 40 it sat at
# 0.024 and every real effect piled up on that floor, unresolvable.
_rng_perm = np.random.default_rng(SEED)
PERM_MASKS = []
for _ in range(N_PERM):
    _p = _rng_perm.permutation(particle_types)
    PERM_MASKS.append((_p == 1, _p == 0))
print(f"  label permutations: {N_PERM}", flush=True)

# ---------------------------------------------------------------------------
# A SECOND null, stratified by sequence length. Why it is necessary:
# each attention row is a softmax over the L valid keys, so the attention
# RECEIVED per position is ~1/L. Gamma showers deposit hits in more detectors
# than proton showers of the same energy, so they have larger L and therefore
# SMALLER per-position attention -- for purely arithmetic reasons, with no
# class-dependent behaviour by the model at all. The plain permutation null
# above breaks the class/length association and so cannot separate the two.
# Permuting labels only WITHIN bins of similar L preserves the length
# imbalance, leaving only genuinely class-dependent structure to detect.
# ---------------------------------------------------------------------------
Lg, Lp = _L[is_gamma].mean(), _L[is_proton].mean()
print(f"  mean valid length: gamma={Lg:.1f}  proton={Lp:.1f}  "
      f"(1/L level: {1/Lg:.4f} vs {1/Lp:.4f}, difference {1/Lg - 1/Lp:+.4f})", flush=True)
null_summary["mean_length_gamma"] = float(Lg)
null_summary["mean_length_proton"] = float(Lp)
null_summary["length_arithmetic_offset"] = float(1 / Lg - 1 / Lp)

N_STRATA = int(os.environ.get("ATTN_NSTRATA", "5"))
_edges = np.quantile(_L, np.linspace(0, 1, N_STRATA + 1))
_stratum = np.clip(np.searchsorted(_edges, _L, side="right") - 1, 0, N_STRATA - 1)
PERM_MASKS_STRAT = []
for _ in range(N_PERM):
    _p = particle_types.copy()
    for k in range(N_STRATA):
        sel = np.where(_stratum == k)[0]
        _p[sel] = _rng_perm.permutation(_p[sel])
    PERM_MASKS_STRAT.append((_p == 1, _p == 0))
print(f"  length-stratified permutations: {N_PERM} over {N_STRATA} length strata "
      f"(edges {np.round(_edges, 1).tolist()})", flush=True)


def null_stats(diff, null, band, signed=True):
    """Compare an observed difference vector against its permutation null.

    Three statistics, because they answer different questions and the first one
    alone is misleading:
      max|D|   -- is there a single outstanding position/detector? Dominated by
                  the noisiest element, so it is a weak test of diffuse patterns.
      mean|D|  -- is the map globally more structured than noise?
      mean D   -- is there a COHERENT offset in one direction (e.g. "more on
                  proton nearly everywhere")? A sign-consistent pattern can be
                  highly significant here while max|D| sees nothing. Only
                  meaningful when the two profiles are NOT each normalized to a
                  fixed total: the per-detector maps are (both sum to 1), so
                  their signed mean is exactly 0 and the statistic is skipped
                  there (signed=False) rather than reported as float dust.
    Also reports how many elements exceed the per-element 95% band, calibrated
    against the same count computed for each permutation.
    """
    stats = {}
    which = [("max_abs", lambda a: np.abs(a).max(axis=-1)),
             ("mean_abs", lambda a: np.abs(a).mean(axis=-1))]
    if signed:
        which.append(("mean_signed", lambda a: a.mean(axis=-1)))
    for name, fn in which:
        obs = float(fn(diff))
        nul = fn(null)
        ref = np.abs(nul) if name == "mean_signed" else nul
        cmp = abs(obs) if name == "mean_signed" else obs
        stats[name] = {
            "observed": obs,
            "null_p95": float(np.percentile(ref, 95)),
            "p_value": float((np.sum(ref >= cmp) + 1) / (len(ref) + 1))}
    n_out = int(np.sum(np.abs(diff) > band))
    null_counts = (np.abs(null) > band).sum(axis=1)
    stats["n_outside_band"] = n_out
    stats["n_elements"] = int(len(diff))
    stats["null_median_outside"] = float(np.median(null_counts))
    stats["p_value_n_outside"] = float((np.sum(null_counts >= n_out) + 1)
                                       / (len(null_counts) + 1))
    return stats


def stat_line(tag, st):
    return (f"    {tag:<9} obs={st['observed']:+.2e}  null95={st['null_p95']:.2e}  "
            f"p={st['p_value']:.3f}")

# -----------------------------------------------------------------------------
# STEP 3c: attention distribution per task (main-text figure)
# -----------------------------------------------------------------------------
# The plain view of the learned attention, matching the one reported in the
# companion A&C paper: the mean attention matrix and the mean attention received
# per position. No class split and no derived statistic -- those live in the
# appendix analysis further below.
banner("STEP 3c/5  Attention distribution per task (mean matrix + key profile)")

PATT_W = 120  # window; mean valid length is ~77 hits, max 411
MIN_EVENTS = 10  # cells averaged over fewer events than this are left blank

fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.4),
                         gridspec_kw={"height_ratios": [1.35, 1.0]},
                         constrained_layout=True)
patt_rows = []
for col, (title, attn) in enumerate(tasks_attn):
    acc = np.zeros((PATT_W, PATT_W)); cnt = np.zeros((PATT_W, PATT_W))
    for mat, L in zip(attn, valid_lengths_raw):
        Le = int(min(L, mat.shape[0], PATT_W))
        if Le <= 0:
            continue
        acc[:Le, :Le] += mat[:Le, :Le]; cnt[:Le, :Le] += 1
    mean_mat = np.where(cnt >= MIN_EVENTS, acc / np.maximum(cnt, 1), np.nan)

    ax = axes[0, col]
    im = ax.imshow(mean_mat, cmap=CMAP_ATT, origin="upper", aspect="equal",
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean attention weight")

    # where the mass sits relative to the diagonal (causal-like structure?)
    _m = np.nan_to_num(mean_mat); _tot = _m.sum() or 1.0
    _iu = np.triu_indices(PATT_W, k=1); _il = np.tril_indices(PATT_W, k=-1)
    print(f"  {title:<24} mass later-than-query={_m[_iu].sum()/_tot:.3f}  "
          f"diagonal={np.trace(_m)/_tot:.3f}  "
          f"earlier={_m[_il].sum()/_tot:.3f}", flush=True)

    key_prof = np.nanmean(np.where(cnt >= MIN_EVENTS, acc / np.maximum(cnt, 1), np.nan),
                          axis=0)
    ax2 = axes[1, col]
    ax2.bar(np.arange(PATT_W), key_prof, width=1.0, color=CONDOR_DARKRED,
            edgecolor="none")
    ax2.axhline(1.0 / _L.mean(), color=CONDOR_INK, lw=1.1, ls="--",
                label=r"uniform level $1/\bar{L}$")
    ax2.set_xlim(-0.5, PATT_W - 0.5)
    ax2.set_title(title + ": mean attention received")
    ax2.set_xlabel("Sequence position"); ax2.set_ylabel("Mean attention weight")
    ax2.legend(loc="upper right", fontsize=11)
    for pos in range(PATT_W):
        patt_rows.append({"task": title, "position": pos,
                          "mean_key_attention": float(key_prof[pos])})

fig.suptitle("Attention distribution of the RAW streams", fontweight="bold")
fig.savefig(os.path.join(OUT, "attention_patterns_raw.png"), dpi=SAVE_DPI)
fig.savefig(os.path.join(FIGS, "attention_patterns_raw.png"), dpi=SAVE_DPI)
plt.close(fig)
pd.DataFrame(patt_rows).round(8).to_csv(
    os.path.join(OUT, "attention_patterns_raw.csv"), index=False)
print(f"  wrote attention_patterns_raw.png/.csv  (window {PATT_W}, "
      f"cells with >= {MIN_EVENTS} events)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: temporal attention difference  (persist CSV + regenerate PNG)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 4/5  Temporal attention difference")

def _moving_average(x, w):
    """Centered moving average with edge padding (same length as x)."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    kern = np.ones(w) / w
    return np.convolve(xp, kern, mode="valid")

SMOOTH_W = 5  # small window: keeps trends, suppresses per-position noise
temporal_rows = []
fig, axes = plt.subplots(1, 3, figsize=(20, 5.4), constrained_layout=True)
for ax, (title, attn) in zip(axes, tasks_attn):
    prof_g = mean_attention_profile(attn, valid_lengths_raw, is_gamma)
    prof_p = mean_attention_profile(attn, valid_lengths_raw, is_proton)
    diff_raw = prof_g - prof_p
    diff = _moving_average(diff_raw, SMOOTH_W)

    # permutation nulls: same estimator, labels shuffled (plain and
    # length-stratified). The stratified one is the honest reference.
    def _temporal_null(masks):
        return np.stack([
            _moving_average(mean_attention_profile(attn, valid_lengths_raw, mg)
                            - mean_attention_profile(attn, valid_lengths_raw, mp),
                            SMOOTH_W)
            for mg, mp in masks])

    null = _temporal_null(PERM_MASKS)
    null_s = _temporal_null(PERM_MASKS_STRAT)
    band = np.percentile(np.abs(null), 95, axis=0)
    band_s = np.percentile(np.abs(null_s), 95, axis=0)
    st = null_stats(diff, null, band)
    st["length_stratified"] = null_stats(diff, null_s, band_s)
    null_summary["tasks"][title]["temporal"] = st
    print(f"  {title} -- temporal", flush=True)
    for tag in ("max_abs", "mean_abs", "mean_signed"):
        print(stat_line(tag, st[tag])
              + f"   | length-stratified p={st['length_stratified'][tag]['p_value']:.3f}",
              flush=True)
    print(f"    outside   {st['n_outside_band']}/{st['n_elements']} positions "
          f"(null median {st['null_median_outside']:.0f}, "
          f"p={st['p_value_n_outside']:.3f})"
          f"   | stratified {st['length_stratified']['n_outside_band']}/"
          f"{st['n_elements']}, p={st['length_stratified']['p_value_n_outside']:.3f}",
          flush=True)

    for pos in range(len(diff)):
        temporal_rows.append({"task": title, "position": pos,
                              "att_gamma": prof_g[pos], "att_proton": prof_p[pos],
                              "delta_raw": diff_raw[pos], "delta_smooth": diff[pos],
                              "null_p95_abs": band[pos],
                              "null_p95_abs_strat": band_s[pos]})
    x = np.arange(len(diff))
    ax.fill_between(x, -band_s, band_s, color=CONDOR_GRAY, alpha=0.30, lw=0,
                    zorder=1)
    ax.axhline(0, color=CONDOR_INK, lw=1.0)
    ax.fill_between(x, diff, 0, where=diff >= 0, color=CONDOR_DARKRED, alpha=0.85,
                    interpolate=True, label=r"more on $\gamma$")
    ax.fill_between(x, diff, 0, where=diff < 0, color=PROTON_BLUE, alpha=0.85,
                    interpolate=True, label="more on proton")
    # Expose the y-axis magnitude so the reader sees that Classification's
    # dependence is roughly an order of magnitude smaller than the regression
    # tasks' -- easy to miss with per-panel autoscaling.
    peak = float(max(abs(diff.max()), abs(diff.min())))
    ax.text(0.98, 0.03,
            fr"peak $|\Delta|$ = {peak:.1e}" + "\n" + fr"null 95% = {band_s.max():.1e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.plot(x, band_s, color=CONDOR_INK, lw=1.1, ls="--", zorder=4,
            label="null (95%, length-matched)")
    ax.plot(x, -band_s, color=CONDOR_INK, lw=1.1, ls="--", zorder=4)
    ax.set_title(title); ax.set_xlabel("Temporal position")
    ax.set_ylabel(r"$\Delta$ attention ($\gamma$ - proton)")
    ax.legend(loc="upper left")
fig.suptitle(f"Particle-type dependence of temporal attention "
             f"(moving avg, window = {SMOOTH_W})", fontweight="bold")
fig.savefig(os.path.join(OUT, "attention_diff_temporal.png"), dpi=SAVE_DPI)
fig.savefig(os.path.join(FIGS, "attention_diff_temporal.png"), dpi=SAVE_DPI)
plt.close(fig)
pd.DataFrame(temporal_rows).round(8).to_csv(
    os.path.join(OUT, "attention_diff_temporal.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: spatial (per-detector) attention difference (persist CSV + PNG)
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5/5  Spatial attention difference (distinct-extreme colormap)")
pos = detector_catalog.set_index("detector_id")[["x_center", "y_center"]]
spatial_rows = []
fig, axes2 = plt.subplots(2, 3, figsize=(21, 12.0), constrained_layout=True)
axes = axes2[0]
for col, (ax, (title, attn)) in enumerate(zip(axes, tasks_attn)):
    att_g = per_detector_attention(attn, X_sample, valid_lengths_raw, is_gamma)
    att_p = per_detector_attention(attn, X_sample, valid_lengths_raw, is_proton)
    dets = sorted(set(att_g) | set(att_p))
    diff = np.array([att_g.get(d, 0.0) - att_p.get(d, 0.0) for d in dets])
    xs = np.array([pos.loc[d, "x_center"] for d in dets])
    ys = np.array([pos.loc[d, "y_center"] for d in dets])

    # permutation null on the same per-detector estimator
    def _spatial_null(masks):
        rows = []
        for mg, mp in masks:
            ng = per_detector_attention(attn, X_sample, valid_lengths_raw, mg)
            npr = per_detector_attention(attn, X_sample, valid_lengths_raw, mp)
            rows.append(np.array([ng.get(d, 0.0) - npr.get(d, 0.0) for d in dets]))
        return np.stack(rows)

    null = _spatial_null(PERM_MASKS)
    null_s = _spatial_null(PERM_MASKS_STRAT)
    band = np.percentile(np.abs(null), 95, axis=0)
    band_s = np.percentile(np.abs(null_s), 95, axis=0)
    st = null_stats(diff, null, band, signed=False)
    st["length_stratified"] = null_stats(diff, null_s, band_s, signed=False)
    st["uniform_level"] = float(1.0 / len(dets))
    null_summary["tasks"][title]["spatial"] = st
    print(f"  {title} -- spatial (uniform level 1/N = {1.0/len(dets):.2e})", flush=True)
    for tag in ("max_abs", "mean_abs"):
        print(stat_line(tag, st[tag])
              + f"   | length-stratified p={st['length_stratified'][tag]['p_value']:.3f}",
              flush=True)
    print(f"    outside   {st['n_outside_band']}/{st['n_elements']} detectors "
          f"(null median {st['null_median_outside']:.0f}, "
          f"p={st['p_value_n_outside']:.3f})"
          f"   | stratified {st['length_stratified']['n_outside_band']}/"
          f"{st['n_elements']}, p={st['length_stratified']['p_value_n_outside']:.3f}",
          flush=True)

    for k, d in enumerate(dets):
        spatial_rows.append({"task": title, "detector_id": d,
                            "x_center": float(pos.loc[d, "x_center"]),
                            "y_center": float(pos.loc[d, "y_center"]),
                            "att_gamma": att_g.get(d, 0.0),
                            "att_proton": att_p.get(d, 0.0),
                            "delta": float(diff[k]),
                            "null_p95_abs": float(band[k]),
                            "null_p95_abs_strat": float(band_s[k]),
                            "outside_null": bool(abs(diff[k]) > band_s[k])})
    vmax = float(np.abs(diff).max()) or 1e-9
    sc = ax.scatter(xs, ys, c=diff, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
                    s=170, edgecolor=CONDOR_INK, linewidth=0.6)
    out = np.abs(diff) > band_s
    if out.any():
        ax.scatter(xs[out], ys[out], s=340, facecolor="none",
                   edgecolor=CONDOR_INK, linewidth=1.8, zorder=3,
                   label="beyond null (length-matched)")
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    ax.text(0.02, 0.02, f"{int(out.sum())}/{len(dets)} beyond null (95%)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta$ attention ($\gamma$ - proton)")

    # Second row: the same difference divided by the 95% noise band, i.e. how
    # far beyond sampling noise each detector actually is. Without it the top
    # row misleads -- its most saturated detectors are the low-statistics ones,
    # whose difference is large AND entirely compatible with noise.
    ax2 = axes2[1, col]
    ratio = diff / np.maximum(band_s, 1e-12)
    sc2 = ax2.scatter(xs, ys, c=ratio, cmap=CMAP_DIV, vmin=-2.5, vmax=2.5,
                      s=170, edgecolor=CONDOR_INK, linewidth=0.6)
    if out.any():
        ax2.scatter(xs[out], ys[out], s=340, facecolor="none",
                    edgecolor=CONDOR_INK, linewidth=1.8, zorder=3)
    ax2.set_title(title + r"  --  in units of noise", fontsize=13)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_aspect("equal")
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label(r"$\Delta$ / null 95%   ($|\cdot| > 1$: beyond noise)")
fig.suptitle("Where attention depends on particle type "
             "(top: raw difference   bottom: same difference in units of noise)",
             fontweight="bold")
fig.savefig(os.path.join(OUT, "attention_diff_spatial.png"), dpi=SAVE_DPI)
fig.savefig(os.path.join(FIGS, "attention_diff_spatial.png"), dpi=SAVE_DPI)
plt.close(fig)
pd.DataFrame(spatial_rows).round(8).to_csv(
    os.path.join(OUT, "attention_diff_spatial.csv"), index=False)

with open(os.path.join(OUT, "attention_null_summary.json"), "w") as f:
    json.dump(null_summary, f, indent=2)

banner("Done. CSV + regenerated PNGs in pipeline_artifacts/diagnostics/ and thesis/figures/")
for fn in ["attention_patterns_raw.png", "attention_patterns_raw.csv",
           "attention_diff_temporal.csv", "attention_diff_spatial.csv",
           "attention_null_summary.json",
           "attention_diff_temporal.png", "attention_diff_spatial.png"]:
    print("  wrote", fn, flush=True)
