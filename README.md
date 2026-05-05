# Deep Learning Approaches for Voltammetric Analysis of Coffee

This repository contains the code and data pipelines developed for my honors thesis **“Deep Learning Approaches for Voltammetric Analysis of Coffee”**, completed at Bucknell University in May 2026. This project investigates how low‑cost electrochemical measurements and deep learning can be combined to quantify key coffee compounds and explore links to sensory attributes.

## Project Overview

Traditional analytical methods such as **high‑performance liquid chromatography (HPLC)** and **gas chromatography–mass spectrometry (GC‑MS)** provide precise quantification of caffeine, chlorogenic acids (CGAs), and related compounds in coffee, but require expensive instrumentation and specialized expertise. This limits accessibility for small roasters, producers, and cafés.

This project explores an alternative approach:

- Use a **low‑cost open‑source potentiostat (Rodeostat)** and **disposable screen‑printed electrodes (SPEs)** to perform cyclic voltammetry on brewed coffee.
- Record voltammograms for brewed coffee samples, standards, and mixtures under controlled conditions.
- Use **HPLC** and **refractometry** to obtain ground‑truth **caffeine**, **CGA**, and **total dissolved solids (TDS)** labels.
- Train **deep learning models** on full voltammograms to predict caffeine, CGA, TDS, roast level, and exploratory sensory attributes, and compare performance against traditional feature‑extraction plus linear models.

In a dataset of 132 brewed coffee samples, the best neural network models achieved mean absolute errors of approximately **52.98 ppm for caffeine**, **70.48 ppm for CGAs**, and **0.08% for TDS**, demonstrating the promise of coupling low‑cost electrochemistry with deep learning for quantitative coffee analysis.

## Repository Structure

```text
CoffeeAnalysis/
├── README.md
├── .gitignore
├── voltammetry-files/        # Raw Rodeostat CSVs (time s, potential V, current A)
├── extra-voltammetry-files/  # Additional / held-out CV measurements
├── data/                     # Search-result CSVs, trained checkpoints, caches
│   ├── architecture_search_summary_HPLC_Caff.csv
│   ├── architecture_search_summary_HPLC_CGA.csv
│   ├── architecture_search_summary_TDS.csv
│   ├── full_window_search_results_*.csv
│   ├── model_search_ranking_*.csv
│   ├── *_loco_results.csv     # Leave-one-coffee-out predictions for best models
│   ├── *.pth, *.pkl           # Trained model checkpoints (gitignored)
│   └── raw_data_cache.pkl.bak
├── src/
│   ├── nn/                   # Neural-network training, evaluation, window search
│   │   ├── nn_0_synthetic_data_gen.py
│   │   ├── nn_1_train_model.py
│   │   ├── nn_2_evaluate_model.py
│   │   ├── nn_model_window_search.py            # HPLC-Caff / HPLC-CGA / TDS
│   │   ├── nn_attributes_model_window_search.py # Sensory attributes
│   │   ├── nn_flavors_model_window_search.py    # Flavor descriptors
│   │   ├── nn_roast_window_search.py
│   │   └── nn_top_model_window_search.py
│   ├── regression/           # Classical regression baselines (single-voltage features)
│   ├── regression_modeling/  # Higher-level regression experiments
│   ├── network_analysis/     # Diagnostics on trained networks
│   ├── cbc_analysis/         # Coffee-brewing-control feature analysis
│   ├── ocp_analysis/         # Open-circuit-potential analysis
│   ├── metrohm/              # Metrohm-instrument-specific helpers
│   ├── variability_tests/    # Repeatability / day-to-day variability checks
│   ├── web_scraper/          # Sensory-score / metadata scraping
│   └── thesis_plots/         # ⭐ Publication-figure generation (most current)
│       ├── plot_generator.py            # Shared mpl publication style
│       ├── plot_best_models.py          # HPLC-Caff, HPLC-CGA, TDS scatter plots
│       ├── plot_best_hplc_caff.py
│       ├── plot_best_sensory_models.py  # Attributes & flavor descriptors
│       ├── plot_best_roast.py
│       ├── plot_caffeine_standards.py
│       ├── plot_cga_standards.py
│       ├── model_accuracies.py          # Classifier accuracy bar chart
│       ├── sample_repeats.py
│       ├── predict_folder_targets.py    # Predict on a folder of new CVs
│       ├── plotting_notebook.ipynb
│       └── plots/                       # Output PDFs + per-target *_stats.txt
├── plots/                    # Earlier exploratory PNG plots
├── normalized_plots/         # Normalized-input experiment PNG plots
├── scripts/                  # Misc one-off scripts (debug helpers, etc.)
├── util/                     # Shared utilities (paths, mpl setup)
├── archive/                  # Older, superseded methods (kept for reference)
├── logs/                     # Training logs (gitignored)
├── venv2/                    # Project virtualenv (gitignored)
├── .python-version
└── .idea/                    # PyCharm project files
```

The most up-to-date results, figures, and per-target statistics live in
[src/thesis_plots/](src/thesis_plots/) and
[src/thesis_plots/plots/](src/thesis_plots/plots/).

## Experimental Pipeline

### 1. Preliminary Laboratory Exploration

Before voltammetry, the project explored how **lab‑accessible physicochemical parameters** relate to coffee identity and quality classes.

For 144 K‑Cup brews, the following were measured:

- pH  
- Potassium ion concentration  
- Total dissolved solids (TDS) by refractometry  
- Turbidity by nephelometry  
- Color (L*, a*, b*) by reflectance spectrophotometry  
- Conductivity  
- Temperature

These features were used to train standard classifiers (logistic regression, decision tree, SVM) to predict:

- Coffee name (product line)
- Roast level (e.g., medium, dark)
- Broad quality class (commodity, premium, specialty)

Support vector machines achieved up to **100% accuracy for coffee name** and **98% for quality class** under cross‑validation, showing that relatively simple measurements encode strong information about brand and roast.

### 2. Voltammetry and Ground‑Truth Labels

For the main electrochemical study, specialty‑grade coffees were sourced from:

- **Fresh Roasted Coffee (Sunbury, PA)**  
- **Alabaster Coffee Roaster and Tea Company (Williamsport, PA)**  
- **Sweet Maria’s (green beans roasted in‑house on an Aillio Bullet R1 V2)**

Brewing protocol:

- Hario Switch immersion brewer with Hario V60 paper filter  
- Brew ratio 16.7:1 (water:coffee)  
- Initial water temperature 96 °C  
- Immersion time 5 minutes  
- Brewed coffee cooled and stored at 3 °C for at least 12 hours before analysis

Voltammetry protocol:

- **Instrument:** IO Rodeo **Rodeostat** open‑source potentiostat with SPE adapter  
- **Electrodes:** Single‑use carbon working and Ag/AgCl reference SPEs from Zimmer & Peacock  
- Coffee mixed 1:1 with **0.1 M H₂SO₄** supporting electrolyte  
- Quiet step at −0.8 V for 45 s  
- Cyclic voltammetry scan from 0 to 2.0 V at 100 mV/s  
- Outputs: time (s), potential (V), current (µA)

Ground‑truth labels:

- **Caffeine** and **total CGAs** quantified by HPLC with caffeine and 5‑CQA standards using a Thermo Fisher Ultimate 3000 with C18 column.
- **TDS** measured by VST Lab Coffee III refractometer.
- **Roast level** quantified as percent mass loss during roasting.

All data analysis was performed in Python using `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `PyTorch`.

### 3. Traditional Feature Extraction and Regression

Voltammograms were preprocessed by:

- Removing quiet‑time and reduction segments to retain only the **oxidation** sweep.  
- Applying a moving‑window average (window size 25) to reduce sensor noise.

Hand‑crafted features included:

- **Brew‑strength point:** current at 0.8 V in the CGA region, which scales with brew strength and correlated weakly with CGA concentration.
- **Caffeine region features:** single fixed‑voltage current, dynamically extracted **plateau point** (lowest slope between 1.1–1.4 V), and **bump point** (steepest slope decrease) to approximate the caffeine “bump.”

Linear regression on these features yielded:

- CGA prediction from 0.8 V current with \(R^2 \approx 0.343\) and MAE ≈ 186 ppm.
- Best caffeine feature (plateau) with \(R^2 \approx 0.397\) and MAE ≈ 151 ppm, improving to \(R^2 \approx 0.524\) and MAE ≈ 139 ppm after brew‑strength normalization.

These results highlight the limitations of single‑point electrochemical features in noisy, complex matrices such as brewed coffee and motivate learning directly from the full voltammogram.

### 4. Deep Learning Models

Neural networks were trained on **voltage windows of the full voltammogram** to predict:

- Caffeine concentration  
- CGA concentration  
- TDS  
- Roast level

Architectures include:

- **SmallBN‑128‑64‑1:** two hidden fully connected layers (128 and 64 units) with batch normalization, trained with Huber loss and Adam.
- **LargeBN‑1024‑512‑1** and **Wide‑768‑256‑1:** larger multilayer perceptrons with batch normalization tailored for higher‑dimensional voltage windows.

Using appropriate voltage windows for each analyte (e.g., 0.4–1.4 V for caffeine, 0.0–1.2 V for CGAs), the best networks substantially outperformed linear baselines:

- Caffeine prediction: \(R^2 \approx 0.928\), MAE ≈ 52.98 ppm (0.4–1.4 V window).
- CGA prediction: \(R^2 \approx 0.859\), MAE ≈ 70.48 ppm (0.0–1.2 V window).
- TDS prediction: \(R^2 \approx 0.774\), MAE ≈ 0.080% (0.2–1.2 V window).
- Roast prediction: \(R^2 \approx 0.186\), MAE ≈ 0.972 roast‑loss units (1.0–1.8 V window), indicating roast is harder to recover directly from these signals.

These findings show that deep learning can extract richer information from voltammograms than traditional single‑point features when sensor noise and overlapping redox processes complicate manual feature design.

### 5. Future Work: Learning to Taste

The project concludes by exploring **prediction of sensory cupping attributes and flavor descriptors** from voltammetry and related data.

Using small sensory datasets from roasters (e.g., cupping forms from Sweet Maria’s for specific coffees), models were trained to predict:

- SCA‑style cupping attributes such as acidity, sweetness, and body (0–10 scale).  
- Flavor descriptors such as fruit, chocolate, or floral notes (0–5 scale).

Due to limited sample size and narrow score ranges, performance was modest and highlighted the need for:

- Larger, standardized sensory datasets.  
- Consistent panel calibration across roasters.  
- Joint modeling of chemical and electrochemical features for flavor prediction.

A long‑term goal is to approximate aspects of **human tasting** with a low‑cost, objective electrochemical system augmented by deep learning.

## Installation

The project is developed against Python (see [.python-version](.python-version))
in a local virtualenv named `venv2/`.

```bash
# Clone
git clone <this-repo-url> CoffeeAnalysis
cd CoffeeAnalysis

# Create the venv (the rest of the README assumes it is named venv2)
python -m venv venv2
source venv2/bin/activate          # on macOS/Linux
# venv2\Scripts\activate           # on Windows

# Install dependencies
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn scikit-learn \
            torch tqdm tabulate jupyter
```

Key Python dependencies:

- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `torch` (CPU is sufficient for the models in this repo)
- `tqdm`, `tabulate`
- `jupyter` / `jupyterlab` (for [src/thesis_plots/plotting_notebook.ipynb](src/thesis_plots/plotting_notebook.ipynb))

Sanity-check the environment:

```bash
python -c "import torch, numpy, pandas, matplotlib, sklearn; print('ok')"
```

> **Note:** `venv2/`, `logs/`, and large training artifacts (`*.pth`, `*.pkl`,
> `*.npz`, `loco_cache/`) are gitignored. They are regenerated by the training
> and search scripts described below.

## Usage

All commands below assume the `venv2` virtualenv is active and the working
directory is the repository root.

### 1. Data layout

The pipeline reads voltammograms directly from the in-tree data folders — no
separate import step is required.

- `voltammetry-files/*.txt` — main 3-column CVs (time s, potential V,
  current A). File names encode the coffee identity and replicate.
- `extra-voltammetry-files/*.txt` — additional / held-out measurements used
  for prediction demos.
- `data/` — search-result CSVs and cached / trained model artifacts produced
  by the scripts below.

The first call to [src/nn/nn_0_synthetic_data_gen.py](src/nn/nn_0_synthetic_data_gen.py)
(via `build_model_data(...)`) builds a labeled dataset from the raw CVs and
caches it; subsequent runs reuse the cache.

### 2. Classical regression baselines

```bash
python src/regression/regression_voltage.py        # Caffeine, single-voltage features
python src/regression/regression_cga_at_0p8V.py    # CGA at 0.8 V brew-strength point
python src/regression/model_search_evaluation.py   # Compare baseline regressors
```

### 3. Neural-network training & architecture × voltage-window search

```bash
# Train / evaluate a single network
python src/nn/nn_1_train_model.py
python src/nn/nn_2_evaluate_model.py

# Joint architecture × voltage-window search per target
python src/nn/nn_model_window_search.py            # HPLC-Caff, HPLC-CGA, TDS
python src/nn/nn_attributes_model_window_search.py # Sensory attributes
python src/nn/nn_flavors_model_window_search.py    # Flavor descriptors
python src/nn/nn_roast_window_search.py            # Roast level
python src/nn/nn_top_model_window_search.py        # Top-N model search summary
```

Each search writes a ranking CSV to `data/` (e.g.
`architecture_search_summary_HPLC_Caff.csv`,
`model_search_ranking_HPLC_CGA.csv`) and a `BEST_<target>_fixed.pkl`
checkpoint that the thesis-plot scripts then consume.

### 4. Regenerate thesis figures

All PDFs land in [src/thesis_plots/plots/](src/thesis_plots/plots/) and reuse
the shared publication style from
[src/thesis_plots/plot_generator.py](src/thesis_plots/plot_generator.py).

```bash
# Predicted-vs-actual scatter plots for the chemistry targets
python src/thesis_plots/plot_best_hplc_caff.py        # HPLC-Caffeine
python src/thesis_plots/plot_best_models.py           # HPLC-Caff, HPLC-CGA, TDS
python src/thesis_plots/plot_best_roast.py            # Roast level

# Sensory attributes & flavor descriptors (LOCO-CV)
python src/thesis_plots/plot_best_sensory_models.py

# Standard curves used for HPLC quantification
python src/thesis_plots/plot_caffeine_standards.py
python src/thesis_plots/plot_cga_standards.py

# Repeatability and classifier-accuracy summaries
python src/thesis_plots/sample_repeats.py
python src/thesis_plots/model_accuracies.py

# Predict targets for a folder of new voltammograms
python src/thesis_plots/predict_folder_targets.py
```

For interactive exploration of voltammograms and figures, open
[src/thesis_plots/plotting_notebook.ipynb](src/thesis_plots/plotting_notebook.ipynb)
in Jupyter.

## Data and Reproducibility

Because some raw datasets may include proprietary or identifying information (e.g., roaster‑specific cupping forms), this repository is designed to separate:

- **Code and configuration** (public)  
- **Raw data** (may be private or partially shared)  
- **Processed, anonymized summary data** (where feasible)

If you plan to release data, consider:

- Including **anonymized metadata** (e.g., origin region instead of specific farm).  
- Sharing **standard curves and example voltammograms** for CGA and caffeine standards, which are useful for benchmarking without exposing proprietary coffee sources.

## Acknowledgments

This project was advised by **Dr. Katsuyuki Wakabayashi**, with additional guidance from **Dr. Alan Marchiori** and **Dr. Brian King** in the Bucknell University Department of Computer Science. Laboratory support, HPLC access, and electrochemical instrumentation were provided by the Bucknell **Chemical Engineering** and **Environmental Engineering** departments, and coffee samples were generously supplied by **Fresh Roasted Coffee**, **Sweet Maria’s**, and **Alabaster Coffee**. [file](file:///Users/ryankoes/Downloads/Honors_Thesis__Coffee_2025_%20(14).pdf)