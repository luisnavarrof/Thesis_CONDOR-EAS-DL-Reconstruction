"""
Standalone runner for the thesis validation experiments (notebook Section 8).

Why this exists: on the 4 GB laptop GPU, running the full notebook first
(main training + Section 6 dropout/attention loops) grows TensorFlow's VRAM pool
to near capacity and does NOT release it (memory growth never shrinks), so building
another model inside the same kernel OOMs ("Dst tensor is not initialized").

This script runs in a FRESH process with a clean GPU. It executes the notebook's
own pipeline cells (data + function defs) verbatim — so it cannot drift from the
notebook — but SKIPS the heavy cells (main training, evaluation, Section 6). Then it
runs the three validation experiments with their flags forced to True.

Results are written incrementally to pipeline_artifacts/diagnostics/validation_*.
"""
import os, sys, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
# Async CUDA allocator reduces fragmentation OOMs on the small 4 GB GPU.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import matplotlib
matplotlib.use("Agg")  # no GUI backend in a headless run

# Smaller batch than the notebook default (32): the laptop GPU has only ~2.7 GB
# free once the desktop/other kernels are accounted for. Tunable via env VAL_BATCH.
VAL_BATCH = int(os.environ.get("VAL_BATCH", "16"))

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # notebook cells resolve BASE_DIR from the working directory
NB = os.path.join(HERE, "CONDOR_EAS-Reconstruction.ipynb")

def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
ns = {"__name__": "__main__"}

# Pipeline cells: data load + filters + balancing + sequences + globals + split.
# Deliberately EXCLUDES: 10 (df.head display), 17/18/25/26 (plots),
# 28/31/34/36+ (model build, training, evaluation, Section 6).
PIPE = [2, 4, 6, 8, 9, 13, 15, 20, 22, 24]

banner("STEP 1/3  Running notebook data pipeline (fresh process, clean GPU)")
t0 = time.time()
for i in PIPE:
    src = "".join(cells[i]["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)
ns["BATCH_SIZE"] = VAL_BATCH  # override notebook's 32 to fit the 4 GB GPU
print(f"[pipeline ready in {time.time()-t0:.0f}s]  "
      f"train={ns['X_train'].shape}  test={ns['X_test'].shape}  BATCH_SIZE={VAL_BATCH}", flush=True)

# build_multitask_model definition only (cut the trailing global `model` build).
src29 = "".join(cells[29]["source"]).split("with strategy.scope():")[0]
exec(compile(src29, "<cell 29 def>", "exec"), ns)
assert "build_multitask_model" in ns, "build_multitask_model not defined"

banner("STEP 2/3  Loading validation utilities and forcing experiment flags True")
val_cells = [c for c in cells
             if "validation-section" in c.get("metadata", {}).get("tags", [])
             and c["cell_type"] == "code"]
for c in val_cells:
    src = "".join(c["source"])
    src = (src.replace("RUN_EXP1_LEAKAGE = False", "RUN_EXP1_LEAKAGE = True")
              .replace("RUN_EXP2_ABLATION = False", "RUN_EXP2_ABLATION = True")
              .replace("RUN_EXP3_MULTISEED = False", "RUN_EXP3_MULTISEED = True"))
    exec(compile(src, "<validation cell>", "exec"), ns)

banner("STEP 3/3  Done. Results in pipeline_artifacts/diagnostics/validation_*")
