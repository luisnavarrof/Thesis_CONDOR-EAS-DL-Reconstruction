"""
Standalone runner: regenerate the ablation + PFI diagnostics from the CANONICAL
saved model, in a fresh process.

Motivation: the committed diagnostics CSVs (feature_ablation_results.csv,
pfi_all_heads_sequence_only.csv) were overwritten by a later, non-canonical model
run and no longer match the thesis figures/text (e.g. angle baseline 0.484 deg vs
the canonical 0.498 deg reported everywhere). This script runs the notebook's OWN
ablation/PFI cells verbatim (so it cannot drift) against the saved
condor_multitask_model.keras, writing self-consistent CSVs.

It loads the model from disk (no retraining) and SKIPS training/evaluation/Section 6.
Both methods are deterministic given the saved model: ablation uses median
substitution, PFI is seeded (random_state=42).

Run with the GPU env so TF sees CUDA:
  conda run -n condor-tf210-gpu \
      --no-capture-output python -u rerun_xai_canonical.py
"""
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # notebook cells resolve BASE_DIR from the working directory
NB = os.path.join(HERE, "CONDOR_EAS-Reconstruction.ipynb")
BATCH = int(os.environ.get("XAI_BATCH", "32"))


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

# no-op IPython display + plain tqdm so notebook cells run headless
ns = {"__name__": "__main__", "display": lambda *a, **k: None}

# Data pipeline cells (same set run_validation.py uses): load + filter + balance
# + sequences + global features + split/scaling. Defines X_train/X_test, Xg_test,
# y_*_test, energy_mean/std, FEATURE_NAMES, F1Score, BATCH_SIZE.
PIPE = [2, 4, 6, 8, 9, 13, 15, 20, 22, 24]

banner("STEP 1/4  Running notebook data pipeline (fresh process)")
t0 = time.time()
for i in PIPE:
    src = "".join(cells[i]["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)
ns["BATCH_SIZE"] = BATCH
# guarantee the metrics used by the ablation/PFI helpers are present
exec("from sklearn.metrics import accuracy_score, mean_absolute_error", ns)
print(f"[pipeline ready in {time.time()-t0:.0f}s]  "
      f"test={ns['X_test'].shape}  BATCH={BATCH}", flush=True)

banner("STEP 2/4  Loading CANONICAL saved model from disk (no retraining)")
# cell 28 defines model_path; cell 34 loads from disk because `model` is absent
# from the namespace (use_in_memory_model short-circuits to False).
for i in (28, 34):
    exec(compile("".join(cells[i]["source"]), f"<cell {i}>", "exec"), ns)

# quick sanity: report the model's own test angle MAE so we know which run this is
import numpy as np  # noqa
_p, _a, _e = ns["model"].predict([ns["X_test"], ns["Xg_test"]], batch_size=BATCH, verbose=0)
_mae = float(np.mean(np.abs(_a.reshape(-1) - ns["y_angle_test"].astype(float))))
print(f"[canonical model] angle MAE on test = {_mae:.4f} deg", flush=True)

banner("STEP 3/4  Feature ablation (median, deterministic)")
for i in (56, 57, 59):
    src = "".join(cells[i]["source"]).replace("from tqdm.notebook import tqdm",
                                              "from tqdm import tqdm")
    exec(compile(src, f"<cell {i}>", "exec"), ns)

banner("STEP 4/4  Permutation feature importance (seeded, random_state=42)")
src62 = "".join(cells[62]["source"]).replace("tqdm.notebook", "tqdm")
exec(compile(src62, "<cell 62>", "exec"), ns)

banner("DONE.  Fresh CSVs written to pipeline_artifacts/diagnostics/")
print("  - feature_ablation_results.csv")
print("  - pfi_all_heads_sequence_only.csv")
