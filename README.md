---
title: System Threat Forecaster
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🛡️ System Threat Forecaster

A production-ready machine learning web application that predicts whether a Windows system is infected with malware, using hardware and OS telemetry features. Upload a CSV of system records and get back a binary threat classification for each row — downloadable as `submission.csv`.

> **Live on Hugging Face Spaces** → [rishitpant/system-threat-forecaster](https://huggingface.co/spaces/rishitpant/system-threat-forecaster)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Full ML Pipeline](#full-ml-pipeline)
- [Web Application](#web-application)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Overview

System Threat Forecaster is an end-to-end binary classification system built around a large dataset of Windows device telemetry — engine versions, OS metadata, hardware specs, antivirus signature data, and more. The goal is to predict `target` (0 = clean, 1 = malware-infected) for each system record.

The pipeline covers:

- **Data Cleaning** — drops duplicates, irrelevant columns, and rows with missing targets
- **Feature Engineering** — version splitting, date decomposition, categorical grouping, derived features
- **Preprocessing** — a sklearn `ColumnTransformer` with separate pipelines per column type
- **Model Selection** — 7-model baseline evaluation → RFECV tuning on the top 3 → best model saved
- **Inference** — a `PredictPipeline` that mirrors training transformations exactly
- **Web UI** — Flask app with drag-and-drop CSV upload, a results summary, preview table, and CSV download

The app is containerized with Docker and deployed on [Hugging Face Spaces](https://huggingface.co/spaces) (port `7860`). GitHub Actions handles CI (pytest) and CD (push to HF Space) on every `main` push.

---

## Project Structure

```
SystemThreatForecaster/
│
├── app.py                            # Flask entry point (routes: /, /predict, /download)
├── Dockerfile                        # Python 3.11-slim image, exposes port 7860
├── requirements.txt                  # All Python dependencies
├── setup.py                          # Installable package: threatforecaster v0.0.1
│
├── src/
│   ├── exception.py                  # CustomException with file + line traceback
│   ├── logger.py                     # Timestamped file logger → logs/<timestamp>.log
│   ├── utils.py                      # save_object, load_object (dill), evaluate_models
│   │
│   ├── components/
│   │   ├── data_cleaning.py          # Drops duplicates, bad columns, missing targets
│   │   ├── data_ingestion.py         # Reads EDA-cleaned CSVs, 80/20 train/val split
│   │   ├── data_transformation.py    # Feature engineering + ColumnTransformer fit/save
│   │   ├── model_trainer.py          # Baseline eval → RFECV top-3 → best model saved
│   │   └── model_pusher.py           # (reserved for future deployment automation)
│   │
│   └── pipeline/
│       ├── train_pipeline.py         # Orchestrates full training run end-to-end
│       └── predict_pipeline.py       # Inference: FE → preprocess → RFECV mask → predict
│
├── artifacts/
│   ├── model.pkl                     # Best model dict: {model, selected (RFECV mask), name}
│   └── preprocessor.pkl              # Fitted ColumnTransformer
│
├── templates/
│   ├── index.html                    # Drag-and-drop upload UI (cyberpunk dark theme)
│   ├── results.html                  # Prediction summary + 50-row preview table
│   └── home.html                     # (reserved)
│
├── notebook/
│   └── EDA.ipynb                     # Exploratory data analysis
│
├── tests/
│   └── test_app.py                   # pytest: homepage 200 OK
│
├── logs/                             # Auto-generated timestamped log files
│
├── .github/
│   └── workflows/
│       └── cicd.yaml                 # GitHub Actions: test → deploy to HF Spaces
│
├── .gitignore
├── .dockerignore
└── .gitattributes
```

---

## Full ML Pipeline

### Step 0 — Data Cleaning (`data_cleaning.py`)

Before ingestion, raw CSV data is cleaned:

- **Duplicate removal** — drops exact duplicate rows from the training set
- **Column pruning** — removes 16 irrelevant/redundant columns including `MachineID`, `IsBetaUser`, `SMode`, `IsVirtualDevice`, `OSBuildLab`, `Processor`, `OSVersion`, and several others
- **Missing target handling** — drops rows where the `target` label is null

Outputs `train_eda_clean.csv` and `test_eda_clean.csv` to `notebook/data/`.

---

### Step 1 — Data Ingestion (`data_ingestion.py`)

Reads the EDA-cleaned CSVs from `notebook/data/` and performs an **80/20 stratified train/val split** (random state 42). Writes three files to `artifacts/`:

| File | Description |
|---|---|
| `train.csv` | 80% of training data (features + target) |
| `val.csv` | 20% of training data (features + target) |
| `test.csv` | Hold-out Kaggle test set (no target) |

---

### Step 2 — Feature Engineering (`data_transformation.py`)

Applied identically at training and inference time via `_apply_feature_engineering()`:

**Version Splitting** — `EngineVersion`, `AppVersion`, `SignatureVersion`, `NumericOSVersion` are split on `.` into `_Major`, `_Minor`, `_Build`, `_Revision` numeric columns.

**Date Decomposition** — `DateAS` (antivirus signature date) → `Malware_year/month/day/hour/minute`; `DateOS` (OS install date) → `OS_year/month/day`.

**Categorical Grouping** — High-cardinality string columns are bucketed into clean groups:

| Column | Groups |
|---|---|
| `MDC2FormFactor` | Desktop, Notebook, Tablet, Server |
| `PrimaryDiskType` | HDD, SSD, Others |
| `ChassisType` | Desktop, Notebook, Tablet, Others |
| `PowerPlatformRole` | Desktop, Portable, Server, Others |
| `OSBranch` | rs_release, th_release |
| `OSEdition` | Core, Professional, Enterprise, Others |
| `OSInstallType` | Upgrade, Clean, Others |
| `AutoUpdateOptionsName` | Auto, Manual, Off, Unknown |
| `LicenseActivationChannel` | Retail, Volume, OEM |
| `FlightRing` | Retail, Insider, Disabled, Unknown |

**Derived Features** — computed from existing columns:

| Feature | Formula |
|---|---|
| `Days_since_OS_Installation` | `Malware_day − OS_day` |
| `Ram_per_core` | `TotalPhysicalRAMMB / ProcessorCoreCount` |
| `Aspect_Ratio` | `ResolutionH / ResolutionV` |
| `Pixel_Density` | `(ResH × ResV) / DiagonalInches` |
| `Primary_Disk_Allocated` | `PrimaryDiskCapacityMB / SystemVolumeCapacityMB` |
| `Free_Disk_Space` | `(SysVolCapacity − PrimaryDiskCapacity) / PrimaryDiskCapacity` |

**Column Cleanup** — drops original version/date columns and a set of redundant post-FE columns (`SignatureVersion_Minor/Major/Revision`, `AppVersion_Major`, `NumericOS_*`, `ProductName`, `OsPlatformSubRelease`, `OSBuildRevisionOnly`).

---

### Step 3 — Preprocessing (`data_transformation.py`)

A `ColumnTransformer` is **fit on training data only** and applied to train, val, and test:

| Column Type | Detection | Pipeline |
|---|---|---|
| **Numerical** | float64/int64, not binary, not ID | `SimpleImputer(mean)` → `MinMaxScaler` |
| **Binary** | exactly 2 unique values | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |
| **ID columns** | int/float with "ID" or "Identifier" in name | `SimpleImputer(most_frequent)` |
| **Categorical** | object dtype | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |

The fitted preprocessor is saved to `artifacts/preprocessor.pkl` using `dill`.

---

### Step 4 — Model Training (`model_trainer.py`)

**Phase 1 — Baseline evaluation** across 7 classifiers (default hyperparameters):

| Model | Val Accuracy |
|---|---|
| LightGBM | 0.6204 |
| XGBoost | 0.6146 |
| Random Forest | 0.6090 |
| AdaBoost | 0.6035 |
| Bagging | 0.5724 |
| Logistic Regression | 0.5539 |
| Decision Tree | 0.5370 |

**Phase 2 — RFECV tuning** on the top 3 models with tuned hyperparameters:

| Model | Hyperparameters | RFECV step | CV folds |
|---|---|---|---|
| XGBoost | `max_depth=5, lr=0.1, n_estimators=200, reg_lambda=10` | 2 | 5 |
| LightGBM | `max_depth=7, lr=0.1, num_leaves=31, min_child_samples=30, reg_lambda=10` | 2 | 5 |
| Random Forest | `n_estimators=50, max_depth=8, min_samples_leaf=10` | 10 | 2 |

Each model undergoes Recursive Feature Elimination with Cross-Validation (`RFECV`) to select the optimal feature subset, then is retrained on the selected features. The best-performing model, along with its boolean RFECV feature mask and name, is saved to `artifacts/model.pkl`.

A minimum val accuracy of 0.60 is enforced — training raises a `CustomException` if no model clears this threshold.

---

### Step 5 — Prediction Pipeline (`predict_pipeline.py`)

At inference time, `PredictPipeline.predict(df)`:

1. Loads `artifacts/preprocessor.pkl` and `artifacts/model.pkl`
2. Applies `_apply_feature_engineering()` (same as training)
3. Transforms through the fitted `ColumnTransformer`
4. Applies the RFECV boolean feature mask
5. Returns binary predictions (`0` = clean, `1` = infected)

---

## Web Application

Flask app (`app.py`) with three routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the drag-and-drop CSV upload form |
| `/predict` | POST | Accepts a `.csv` file, runs `PredictPipeline`, renders results page with summary stats and a 50-row preview |
| `/download` | POST | Streams the full prediction results as `submission.csv` |

**Input format:** CSV matching the Kaggle test data column layout (75 feature columns + optional `id`). Only `.csv` files are accepted. If no `id` column is present, a sequential index is used.

**Output format:** `id, target` — one row per input record.

The UI (`index.html`) features a dark cyberpunk theme with a drag-and-drop dropzone, a pulsing "live" badge, and inline flash error messages. The results page shows total records, infected count, clean count, and a scrollable preview table.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Raw data CSVs placed at:
  - `notebook/data/train_eda_clean.csv`
  - `notebook/data/test_eda_clean.csv`

  *(If starting from truly raw data, run `data_cleaning.py` first — see below.)*

### Install

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

### (Optional) Run Data Cleaning

If you have the original raw `train.csv` / `test.csv` in `artifacts/`:

```bash
python src/components/data_cleaning.py
```

### Train the Model

Runs the full ingestion → transformation → training pipeline:

```bash
python src/components/data_ingestion.py
```

This writes `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Run the App

```bash
python app.py
```

The app starts at `http://0.0.0.0:7860`.

---

## Docker

```bash
# Build
docker build -t system-threat-forecaster .

# Run
docker run -p 7860:7860 system-threat-forecaster
```

The Dockerfile uses `python:3.11-slim`, installs `libgomp1` (required by LightGBM), and copies the full project. The app is served on port `7860` to match Hugging Face Spaces expectations.

---

## CI/CD

GitHub Actions (`.github/workflows/cicd.yaml`) runs automatically on every push to `main`:

**`test` job:**
1. Checks out the repo
2. Sets up Python 3.11
3. Installs requirements + pytest
4. Sets `PYTHONPATH` to workspace root
5. Runs `pytest -v`

**`deploy` job** (runs only after tests pass):
1. Checks out repo with full history and LFS
2. Authenticates to Hugging Face using the `HF_TOKEN` secret
3. Force-pushes to the HF Space remote: `rishitpant/system-threat-forecaster`

---

## Testing

```bash
pytest -v
```

Current tests (`tests/test_app.py`):

| Test | What it checks |
|---|---|
| `test_homepage` | `GET /` returns HTTP 200 |

---

## Dependencies

```
pandas, numpy
scikit-learn==1.8.0
xgboost, lightgbm, catboost
flask
dill
seaborn, matplotlib
pytest
```

---

## Author

**Rishit Pant** — [rishitpant100@gmail.com](mailto:rishitpant100@gmail.com)