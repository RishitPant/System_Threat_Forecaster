---
title: System Threat Forecaster
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🛡️ System Threat Forecaster

A production-grade end-to-end ML system that predicts Windows malware infection from hardware and OS telemetry. Upload a CSV of system records and receive a binary threat classification per row — downloadable as `submission.csv`.

> **Live Demo** → [rishitpant/system-threat-forecaster on Hugging Face Spaces](https://huggingface.co/spaces/rishitpant/system-threat-forecaster)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Web Application](#web-application)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Overview

System Threat Forecaster is a binary classification system built on a modified version of some malware prediction dataset with windows device telemetry covering engine versions, OS metadata, hardware specs, antivirus signature data, and more. The goal is to predict `target` (0 = clean, 1 = malware-infected) for each system record.

This was a graded academic ML project with an associated viva. The dataset was intentionally modified from the original source, making it a hard problem: **the top leaderboard score in the class was 64.95% accuracy.** The judging metric was accuracy.

**What this project covers end-to-end:**

- Raw data cleaning (duplicate removal, column pruning, missing-target handling)
- Rich feature engineering — version splitting, date decomposition, categorical grouping, derived hardware features
- Sklearn `ColumnTransformer` preprocessing with type-aware pipelines
- 7-model baseline evaluation (accuracy, F1, recall, ROC-AUC)
- Bayesian hyperparameter tuning via **Optuna** (TPE sampler, 20 trials, AUC-optimized) on the top 3 models
- Recursive Feature Elimination with Cross-Validation (**RFECV**) on the best tuned model
- Per-stage visual eval reports: confusion matrix, ROC curve, PR curve for every model at every stage
- Serialized artifacts (`model.pkl`, `preprocessor.pkl`) for reproducible inference
- Flask web app with drag-and-drop CSV upload, prediction summary, and CSV download
- Dockerized deployment on Hugging Face Spaces (port 7860)
- GitHub Actions CI/CD: pytest on every push, auto-deploy to HF Space on green

---

## Project Structure

```
STF-F3/
│
├── app.py                            # Flask entry point (routes: /, /predict, /download)
├── Dockerfile                        # python:3.11-slim, exposes port 7860
├── requirements.txt
├── setup.py                          # Installable package: threatforecaster v0.0.1
│
├── src/
│   ├── exception.py                  # CustomException with file + line traceback
│   ├── logger.py                     # Timestamped file logger → logs/<timestamp>.log/
│   ├── utils.py                      # save_object/load_object (dill), evaluate_models,
│   │                                 #   evaluate_final_model (3-panel plots), param I/O
│   │
│   ├── components/
│   │   ├── data_cleaning.py          # Drops duplicates, 16 bad columns, null targets
│   │   ├── data_ingestion.py         # Stratified 80/20 train/val split
│   │   ├── data_transformation.py    # Feature engineering + ColumnTransformer fit/save
│   │   ├── model_trainer.py          # Baseline → Optuna tuning → RFECV → best model saved
│   │   └── model_pusher.py           # (reserved)
│   │
│   └── pipeline/
│       ├── train_pipeline.py         # Orchestrates full training run
│       └── predict_pipeline.py       # Inference: FE → preprocess → RFECV mask → predict
│
├── artifacts/
│   ├── model.pkl                     # {model, selected (RFECV mask), name}
│   ├── preprocessor.pkl              # Fitted ColumnTransformer
│   ├── eval/
│   │   ├── baseline/                 # 7 × 3-panel eval PNGs (confusion, ROC, PR)
│   │   ├── tuned/                    # 3 × 3-panel eval PNGs (post-Optuna)
│   │   └── final/                    # 1 × 3-panel eval PNG (post-RFECV best model)
│   └── params/
│       ├── lightgbm_best_params.json
│       ├── xgboost_best_params.json
│       └── random_forest_best_params.json
│
├── templates/
│   ├── index.html                    # Drag-and-drop upload UI
│   ├── results.html                  # Prediction summary + 30-row preview + download
│   └── home.html
│
├── notebook/
│   └── EDA.ipynb                     # Exploratory data analysis
│
├── tests/
│   └── test_app.py                   # pytest: homepage 200 OK
│
├── logs/                             # Timestamped runtime logs
│
└── .github/
    └── workflows/
        └── cicd.yaml                 # CI: pytest | CD: push to HF Space
```

---

## ML Pipeline

### Step 0 — Data Cleaning (`data_cleaning.py`)

Processes raw CSVs before ingestion:

- Drops exact duplicate rows
- Removes 16 irrelevant/redundant columns (`MachineID`, `IsBetaUser`, `SMode`, `IsVirtualDevice`, `OSBuildLab`, `Processor`, `OSVersion`, and others)
- Drops rows with null `target` labels

Outputs `train_eda_clean.csv` and `test_eda_clean.csv` to `notebook/data/`.

---

### Step 1 — Data Ingestion (`data_ingestion.py`)

Reads the cleaned CSVs and performs a **80/20 train/val split** (random state 42). Writes `artifacts/train.csv`, `artifacts/val.csv`, and `artifacts/test.csv`.

---

### Step 2 — Feature Engineering (`data_transformation.py`)

Applied identically at training and inference time via `_apply_feature_engineering()`.

**Version Splitting** — `EngineVersion`, `AppVersion`, `SignatureVersion`, `NumericOSVersion` → `_Major`, `_Minor`, `_Build`, `_Revision` numeric columns.

**Date Decomposition** — `DateAS` → `Malware_year/month/day/hour/minute`; `DateOS` → `OS_year/month/day`.

**Categorical Grouping** — high-cardinality string columns bucketed into clean groups:

| Column | Groups |
|---|---|
| `MDC2FormFactor` | Desktop, Notebook, Tablet, Server |
| `PrimaryDiskType` | HDD, SSD, Others |
| `ChassisType` | Desktop, Notebook, Tablet, Others |
| `PowerPlatformRole` | Desktop, Portable, Server, Others |
| `OSEdition` | Core, Professional, Enterprise, Others |
| `OSInstallType` | Upgrade, Clean, Others |
| `AutoUpdateOptionsName` | Auto, Manual, Off, Unknown |
| `LicenseActivationChannel` | Retail, Volume, OEM |
| `FlightRing` | Retail, Insider, Disabled, Unknown |

**Derived Features:**

| Feature | Formula |
|---|---|
| `Days_since_OS_Installation` | `Malware_day − OS_day` |
| `Ram_per_core` | `TotalPhysicalRAMMB / ProcessorCoreCount` |
| `Aspect_Ratio` | `ResolutionH / ResolutionV` |
| `Pixel_Density` | `(ResH × ResV) / DiagonalInches` |
| `Primary_Disk_Allocated` | `PrimaryDiskCapacityMB / SystemVolumeCapacityMB` |
| `Free_Disk_Space` | `(SysVolCapacity − PrimaryDiskCapacity) / PrimaryDiskCapacity` |

---

### Step 3 — Preprocessing (`data_transformation.py`)

A `ColumnTransformer` is **fit on training data only**, serialized to `artifacts/preprocessor.pkl`:

| Column Type | Detection | Pipeline |
|---|---|---|
| **Numerical** | float64/int64, non-binary, non-ID | `SimpleImputer(mean)` → `MinMaxScaler` |
| **Binary** | exactly 2 unique values | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |
| **ID columns** | int/float with "ID"/"Identifier" in name | `SimpleImputer(most_frequent)` |
| **Categorical** | object dtype | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |

---

### Step 4 — Model Training (`model_trainer.py`)

**Phase 1 — Baseline evaluation** across 7 classifiers (default hyperparameters), scored on val accuracy, F1, recall, and ROC-AUC. Eval reports (3-panel PNGs) saved for every model.

> **Context:** This dataset was modified from its original source for an academic setting. The class leaderboard ceiling was **64.95% accuracy**.

| Model | Val Accuracy |
|---|---|
| LightGBM | 0.6204 |
| XGBoost | 0.6146 |
| Random Forest | 0.6090 |
| AdaBoost | 0.6035 |
| Bagging | 0.5724 |
| Logistic Regression | 0.5539 |
| Decision Tree | 0.5370 |

**Phase 2 — Bayesian hyperparameter tuning (Optuna)** on the top 3 models by AUC, using TPE sampler (20 trials, seed 42). Tuning maximizes ROC-AUC on the validation set. Best params are cached as JSON so re-runs skip re-tuning.

| Model | Key tuned params |
|---|---|
| LightGBM | `n_estimators`, `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `reg_lambda`, `colsample_bytree` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `min_child_weight` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |

**Phase 3 — RFECV feature selection** on the best-tuned model (step=2, cv=5, scoring=`roc_auc`, n_jobs=-1). The boolean support mask is stored alongside the model for exact replay at inference time.

**Phase 4 — Final model save.** The winning model dict `{model, selected, name}` is written to `artifacts/model.pkl`. A floor of AUC ≥ 0.50 is enforced; training raises a `CustomException` if unmet.

---

### Step 5 — Prediction Pipeline (`predict_pipeline.py`)

`PredictPipeline.predict(df)` mirrors the training path exactly:

1. Load `artifacts/preprocessor.pkl` and `artifacts/model.pkl`
2. Apply `_apply_feature_engineering()`
3. Transform via the fitted `ColumnTransformer`
4. Apply the RFECV boolean mask
5. Return binary predictions + probability scores (`0` = clean, `1` = infected)

---

## Web Application

Flask app (`app.py`) with three routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Drag-and-drop CSV upload form |
| `/predict` | POST | Accepts `.csv`, runs `PredictPipeline`, renders results with summary stats, 30-row preview, and confidence scores |
| `/download` | POST | Streams full predictions as `submission.csv` |

**Input:** CSV matching the Kaggle test data column layout. If no `id` column is present, a sequential index is used.

**Output:** `id, target, confidence` — one row per input record.

The UI (`index.html`) is a dark cyberpunk-themed single-page app with drag-and-drop file upload, a pulsing live badge, and inline flash error messages. The results page (`results.html`) shows infected/clean counts and a scrollable preview table.

---

## Getting Started

### Prerequisites

- Python 3.11+
- EDA-cleaned CSVs at `notebook/data/train_eda_clean.csv` and `notebook/data/test_eda_clean.csv`

  *(If starting from raw data, run `data_cleaning.py` first — see below.)*

### Install

```bash
pip install -r requirements.txt

# Or as an editable package
pip install -e .
```

### (Optional) Run Data Cleaning

```bash
python src/components/data_cleaning.py
```

### Train

```bash
python src/components/data_ingestion.py
```

Writes `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Run the App

```bash
python app.py
# → http://0.0.0.0:7860
```

---

## Docker

```bash
docker build -t system-threat-forecaster .
docker run -p 7860:7860 system-threat-forecaster
```

The Dockerfile uses `python:3.11-slim`, installs `libgomp1` (required by LightGBM), and serves on port `7860` to match Hugging Face Spaces.

---

## CI/CD

GitHub Actions (`.github/workflows/cicd.yaml`) runs on every push to `main`:

**`test` job:**
1. Checkout repo
2. Setup Python 3.11
3. Install requirements + pytest
4. Set `PYTHONPATH` to workspace root
5. Run `pytest -v`

**`deploy` job** (runs only after tests pass):
1. Checkout with full history and LFS
2. Authenticate to Hugging Face via `HF_TOKEN` secret
3. Force-push to `rishitpant/system-threat-forecaster` on HF Spaces

---

## Testing

```bash
pytest -v
```

| Test | What it checks |
|---|---|
| `test_homepage` | `GET /` returns HTTP 200 |

---

## Dependencies

```
pandas, numpy
scikit-learn==1.8.0
xgboost, lightgbm, catboost
optuna
flask
dill
seaborn, matplotlib
pytest
```

---

## Author

**Rishit Pant** — [rishitpant100@gmail.com](mailto:rishitpant100@gmail.com)