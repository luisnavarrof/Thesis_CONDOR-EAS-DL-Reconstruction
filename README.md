# Deep learning for extensive air showers reconstruction: an interpretable multi-task CNN-Transformer model for the CONDOR observatory

[![Thesis PDF](https://github.com/luisnavarrof/Thesis_CONDOR-EAS-DL-Reconstruction/actions/workflows/thesis.yml/badge.svg)](https://github.com/luisnavarrof/Thesis_CONDOR-EAS-DL-Reconstruction/actions/workflows/thesis.yml)
[![Paper DOI](https://img.shields.io/badge/paper-10.1016%2Fj.ascom.2026.101173-blue)](https://doi.org/10.1016/j.ascom.2026.101173)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17717722.svg)](https://doi.org/10.5281/zenodo.17717722)

Master's thesis of **Luis Felipe Navarro Farías**, Magíster en Ciencias de la Ingeniería Informática, Departamento de Informática, Universidad Técnica Federico Santa María (Valparaíso, Chile).
Advisor: Raquel Pezoa, Ph.D. Co-advisor: Nicolás Viaux, Ph.D.

This repository holds the LaTeX source of the thesis, the code that produces every result in it, and the trained model and result files those results come from.

## Summary

CONDOR is a proposed ground-based array of plastic scintillator detectors at 5,300 m a.s.l. in the Atacama Desert, aimed at gamma rays and cosmic rays in the sub-TeV range. The thesis presents a single multi-task model that reconstructs three properties of an extensive air shower from the detector hits:

1. gamma/hadron discrimination (photon or proton primary);
2. zenith angle of arrival;
3. primary energy.

Each event is fed to the model as two inputs: a time-ordered sequence of detector hits (`detector_id`, `particle_count`, `t_bin`, `total_energy`, `x_center`, `y_center`; zero-padded to 472 hits with a mask) and a vector of five event-level features. A shared 1D-CNN backbone is followed by six single-head Transformer streams, two per task: one attends over the CNN features and one over the raw hit sequence. The raw streams make the attention maps readable in terms of detector hits. The model has 531,375 parameters.

Interpretability is part of the analysis, not an add-on: feature ablation, permutation feature importance, the distribution of attention in the raw streams, and a detector switch-off study that measures how much each detector, region and layout contributes to each task.

## Results (test set, 11,209 events)

| Task | Metric | Value |
|---|---|---|
| Gamma/hadron | AUC · F1 · accuracy | 0.991 · 0.970 · 96.99% |
| Gamma/hadron | Q-factor · ε<sub>γ</sub> | 5.48 · 0.971 |
| Zenith angle | MAE · PSF<sub>68</sub> · RMSE | 0.484° · 0.50° · 0.736° |
| Energy | MAE · R² · σ<sub>E</sub>/E | 103.7 GeV · 0.453 · 33.1% |

Values from [`code/pipeline_artifacts/model/evaluation_metrics.json`](code/pipeline_artifacts/model/evaluation_metrics.json) and [`performance_summary.csv`](code/pipeline_artifacts/performance_summary.csv); [`code/audit_canonical_metrics.py`](code/audit_canonical_metrics.py) recomputes all of them from the saved test predictions.

## Data

CORSIKA simulations with EPOS and GHEISHA as hadronic interaction models, observation level 5,300 m, photon and proton primaries at 300, 500 and 800 GeV, zenith angle 0° to 40° and azimuth fixed at φ = 0°. Events with fewer than 30 particles on the detector plane are removed. After stratified balancing by particle type, energy and zenith bin the dataset has 74,718 events, split 70/15/15 (seed 42).

The processed dataset (`processed_all_data.pkl`, 397 MB) is archived on Zenodo, [doi:10.5281/zenodo.17717722](https://doi.org/10.5281/zenodo.17717722). The main notebook downloads it on first run.

## Repository layout

```
thesis/                         LaTeX source (book class, apacite)
  main.tex  build.bat  references.bib
  chapters/  appendix/  frontmatter/  figures/
code/
  CONDOR_EAS-Reconstruction.ipynb       data pipeline, model, training, evaluation, XAI
  CONDOR_EAS-Data-Visualization.ipynb   dataset and shower visualizations
  CONDORPROCESSING_MULTI.py             CORSIKA DAT files -> detector-binned hits
  run_robustness.py                     detector switch-off study (Sec. 4.3)
  run_attention.py                      attention analysis (Sec. 4.2.2)
  rerun_xai_canonical.py                feature ablation + permutation importance
  run_validation.py                     methodological validation checks
  audit_canonical_metrics.py            recompute all Results-chapter metrics
  gen_*.py                              thesis figures, drawn from the saved results
  condor_style.py                       shared matplotlib style
  environment-*.yml  requirements.txt
  pipeline_artifacts/
    model/              trained model (.keras), weights, training history, metrics
    diagnostics/        CSV/JSON results of every analysis + figures
    eas_visualizations/ shower visualizations
    test_predictions.npz  detector_catalog.csv  preprocessing_metadata.json
```

The CSV and JSON files in `pipeline_artifacts/` are the source of every number in the thesis. The `run_*.py` scripts execute the notebook's own cells, so the analyses cannot drift from the data pipeline.

## Reproducing the results

```bash
conda env create -f code/environment-cuda.yml     # TensorFlow 2.10 + CUDA 11.2
conda activate condor-tf210-gpu
cd code
jupyter notebook CONDOR_EAS-Reconstruction.ipynb  # downloads the dataset on first run
```

`environment-cpu.yml` gives a CPU-only environment and `environment-gpu.yml` uses the DirectML plugin for non-NVIDIA GPUs on Windows. Training takes about 45 minutes on an NVIDIA RTX 3050 Ti Laptop GPU (4 GB). The analysis scripts load the saved model and do not retrain:

```bash
python run_robustness.py
python run_attention.py
python gen_results_figures.py
```

To regenerate the dataset from raw CORSIKA output, `CONDORPROCESSING_MULTI.py` needs the `panama` package:

```bash
python CONDORPROCESSING_MULTI.py --photon-dir <photon DAT dir> --proton-dir <proton DAT dir> --out-dir <output dir> --energy 300
```

## Building the thesis

With a TeX distribution on the path, run `thesis/build.bat` (Windows) or

```bash
cd thesis
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every push that touches `thesis/` builds the PDF on GitHub Actions; download it from the latest run of the *Thesis PDF* workflow.

## Relation to the published paper

The model was first published in *Astronomy and Computing* (2026), article 101173. The paper describes the 3-stream version (CNN streams only, 521,379 parameters); its code is at [luisnavarrof/CONDOR_EAS-Reconstruction](https://github.com/luisnavarrof/CONDOR_EAS-Reconstruction). The thesis extends it with the raw-sequence attention streams, feature ablation, Q-factor and PSF<sub>68</sub>, stratified ROC analysis, and the detector robustness study.

## Citation

```bibtex
@mastersthesis{navarro2026thesis,
  author = {Navarro Far{\'i}as, Luis Felipe},
  title  = {Deep learning for extensive air showers reconstruction: an interpretable multi-task {CNN}-Transformer model for the {CONDOR} observatory},
  school = {Universidad T{\'e}cnica Federico Santa Mar{\'i}a},
  address = {Valpara{\'i}so, Chile},
  type   = {Master's thesis},
  year   = {2026}
}

@article{navarro2026ascom,
  author  = {Navarro, Luis and Pezoa, Raquel and Viaux, Nicol{\'a}s and Tapia, Sebasti{\'a}n and Valdivieso, Constanza and Mendizabal, Sebasti{\'a}n and Arratia, Miguel and Huang, Jiajun and Brooks, W. K.},
  title   = {Deep learning for extensive air showers reconstruction: An interpretable multi-task {CNN}-Transformer model for the {CONDOR} observatory},
  journal = {Astronomy and Computing},
  volume  = {57},
  pages   = {101173},
  year    = {2026},
  doi     = {10.1016/j.ascom.2026.101173}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The thesis text and figures are © 2026 Luis Felipe Navarro Farías.
