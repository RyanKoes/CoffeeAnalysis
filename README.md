# CoffeeAnalysis

Predicting the chemical composition and sensory profile of brewed coffee from
cyclic-voltammetry (CV) measurements. The repository contains the data pipeline,
neural-network training/evaluation code, and the plot-generation scripts used
for the accompanying thesis.

A simple, low-cost three-electrode cell is used to record a CV trace of a
coffee sample. From that single 0.0–2.0 V sweep we train models that estimate:

- **HPLC-Caff** — caffeine concentration (ppm) measured by HPLC
- **HPLC-CGA**  — total chlorogenic acid (CGA) concentration (ppm) by HPLC
- **TDS**       — total dissolved solids (% by refractometer)
- **Roast level** and coffee identity / origin (classification)
- **Sensory attributes & flavors** from cupping scores (Brightness, Clean Cup,
  Finish, Uniformity, Flavor: Caramel, Citrus, Rustic, …)

## Repository layout

```
CoffeeAnalysis/
├── voltammetry-files/        # Raw 3-column CVs: time, potential (V), current
├── extra-voltammetry-files/  # Additional held-out / new-sample CVs
├── data/                     # CSV search results, model checkpoints, caches
├── src/
│   ├── nn/                   # Neural-net training, window search, evaluation
│   ├── regression/           # Classical regression baselines
│   ├── regression_modeling/  # Higher-level regression experiments
│   ├── network_analysis/     # Diagnostics on trained networks
│   ├── cbc_analysis/         # CBC (coffee-brewing-control) analysis
│   ├── ocp_analysis/         # Open-circuit-potential analysis
│   ├── metrohm/              # Metrohm-instrument-specific helpers
│   ├── variability_tests/    # Repeatability / day-to-day variability checks
│   ├── web_scraper/          # Sensory-score / metadata scraping
│   └── thesis_plots/         # ⭐ Plot generation for the thesis (most current)
├── plots/                    # Earlier exploratory plots (PNG)
├── normalized_plots/         # Normalized-input experiments (PNG)
├── scripts/                  # Misc one-off scripts (debug, etc.)
├── util/                     # Shared utilities (paths, mpl setup)
├── archive/                  # Older method snapshots, kept for reference
└── logs/                     # Training logs (gitignored)
```

The most up-to-date experimental results, figures and write-ups live in
[src/thesis_plots/](src/thesis_plots/).

## Quick start

The project is developed against the `venv2` virtual environment.

```bash
# Activate the project venv
source venv2/bin/activate

# Sanity-check the environment
python -c "import torch, numpy, pandas, matplotlib; print('ok')"
```

Common entry points:

```bash
# Train / evaluate a single neural network
python src/nn/nn_1_train_model.py
python src/nn/nn_2_evaluate_model.py

# Architecture × voltage-window search for a target
python src/nn/nn_model_window_search.py
python src/nn/nn_attributes_model_window_search.py
python src/nn/nn_flavors_model_window_search.py

# Regenerate thesis figures (writes PDFs into src/thesis_plots/plots/)
python src/thesis_plots/plot_best_hplc_caff.py
python src/thesis_plots/plot_best_models.py            # HPLC-Caff, HPLC-CGA, TDS
python src/thesis_plots/plot_best_sensory_models.py    # Sensory attributes/flavors
python src/thesis_plots/plot_best_roast.py
python src/thesis_plots/plot_caffeine_standards.py
python src/thesis_plots/plot_cga_standards.py
python src/thesis_plots/sample_repeats.py
python src/thesis_plots/model_accuracies.py            # Classifier accuracy bar chart
```

## Method in one paragraph

Each coffee sample is measured with cyclic voltammetry from 0.0 V to 2.0 V.
For every regression target (HPLC-Caff, HPLC-CGA, TDS, sensory scores) we
perform a joint **architecture × voltage-window** search: candidate MLP
architectures (Tiny / Small / Large / DeepBN / Wide, with optional BatchNorm
and Dropout) are trained on every contiguous voltage sub-window of the CV
trace, and the best (architecture, window) pair is selected by leave-one-coffee-out
(LOCO) cross-validation. The winning configuration is retrained and reported
with $R^2$ and MAE on held-out coffees.

## Selected results

The PDFs below are produced by the scripts in `src/thesis_plots/` and are the
canonical figures for the thesis. PNG previews from earlier runs in
`normalized_plots/` are embedded inline because GitHub does not render PDFs.

### Chemistry targets (HPLC + TDS)

| Target   | Best architecture     | Voltage window | $R^2$ | MAE    |
| -------- | --------------------- | -------------- | ----- | ------ |
| HPLC-Caff | SmallBN-128-64-1     | 0.4–1.4 V      | 0.928 | 52.98 ppm |
| HPLC-CGA  | Large-1024-512-256-1 | 0.0–0.8 V      | (see [scatter_HPLC_CGA.pdf](src/thesis_plots/plots/scatter_HPLC_CGA.pdf)) | |
| TDS       | Wide-768-256-1       | 0.2–1.2 V      | 0.774 | 0.080 % |

Predicted-vs-actual scatter plots:

| | |
|---|---|
| ![HPLC-Caffeine predictions](normalized_plots/HPLC_Caff_predictions_vs_actual.png) | ![TDS predictions](normalized_plots/TDS_predictions_vs_actual.png) |
| **HPLC-Caffeine** &nbsp; ([scatter_HPLC_Caff.pdf](src/thesis_plots/plots/scatter_HPLC_Caff.pdf)) | **TDS** &nbsp; ([scatter_TDS.pdf](src/thesis_plots/plots/scatter_TDS.pdf)) |
| ![HPLC-CGA predictions](plots/general/HPLC_CGA_predictions_vs_actual.png) | ![Caffeine model detail](normalized_plots/HPLC_Caff_best_model_detailed.png) |
| **HPLC-CGA** &nbsp; ([scatter_HPLC_CGA.pdf](src/thesis_plots/plots/scatter_HPLC_CGA.pdf)) | **Best caffeine model — detail** |

Calibration / standard curves used for HPLC quantification:

- Caffeine standards: [caffeine_standards.pdf](src/thesis_plots/plots/caffeine_standards.pdf)
- CGA standards: [cga_standards.pdf](src/thesis_plots/plots/cga_standards.pdf)

### Roast level

- Best regression model fit: [scatter_Roast.pdf](src/thesis_plots/plots/scatter_Roast.pdf)
- Per-model statistics: [roast_model_stats.txt](src/thesis_plots/plots/roast_model_stats.txt)

### Sensory attributes & flavors

LOCO predictions for the sensory targets are summarized in
[sensory_model_stats_loco_epochs5000.txt](src/thesis_plots/plots/sensory_model_stats_loco_epochs5000.txt).
Per-target scatter plots:

- [scatter_Attribute_Brightness.pdf](src/thesis_plots/plots/scatter_Attribute_Brightness.pdf)
- [scatter_Attribute_Clean_Cup.pdf](src/thesis_plots/plots/scatter_Attribute_Clean_Cup.pdf)
- [scatter_Attribute_Finish.pdf](src/thesis_plots/plots/scatter_Attribute_Finish.pdf)
- [scatter_Attribute_Uniformity.pdf](src/thesis_plots/plots/scatter_Attribute_Uniformity.pdf)
- [scatter_Flavor_Caramel.pdf](src/thesis_plots/plots/scatter_Flavor_Caramel.pdf)
- [scatter_Flavor_Citrus.pdf](src/thesis_plots/plots/scatter_Flavor_Citrus.pdf)
- [scatter_Flavor_Rustic.pdf](src/thesis_plots/plots/scatter_Flavor_Rustic.pdf)

### Classification (coffee identity / roast / class)

Bar chart of test accuracies for Logistic Regression, Decision Tree and SVM
classifiers on Name / Roast / Class targets:

[model_accuracies.pdf](src/thesis_plots/model_accuracies.pdf)

### Repeatability

Sample-repeat overlay showing CV reproducibility across replicate measurements:
[sample_repeats.pdf](src/thesis_plots/plots/sample_repeats.pdf)

## Data files

- `voltammetry-files/*.txt` — 3-column CSV (time s, potential V, current A)
  for each measurement. File names encode coffee identity / replicate.
- `extra-voltammetry-files/` — additional measurements used for held-out
  prediction demos via `src/thesis_plots/predict_folder_targets.py`.
- `data/architecture_search_summary_*.csv`, `data/full_window_search_results_*.csv`,
  `data/model_search_ranking_*.csv` — outputs of the architecture × window search.
- `data/*_loco_results.csv` — LOCO-CV predictions for the selected best models.
- `data/*.pth`, `data/*.pkl` — trained checkpoints and cached search state
  (gitignored; regenerate with the training scripts).

## Plot style

All thesis figures share a publication style configured in
[`src/thesis_plots/plot_generator.py`](src/thesis_plots/plot_generator.py)
(serif fonts, embedded TrueType, single/double column presets). Import it at
the top of any new plotting script:

```python
from plot_generator import apply_publication_style, FIGSIZE_SINGLE_COLUMN
apply_publication_style(figsize=FIGSIZE_SINGLE_COLUMN)
```

## Notes

- Large artifacts (`*.pth`, `*.pkl`, `*.npz`, `logs/`, `loco_cache/`) are
  gitignored — they are produced by re-running the search/training scripts.
- Earlier methods that have been superseded live under `archive/` for
  reference.
