# System Threat Forecaster

A machine learning web application that predicts whether a Windows system is infected with malware, based on hardware and OS telemetry features. Upload a CSV of system records and get back a binary threat classification for each row.

---

## Overview

System Threat Forecaster is a binary classification pipeline built around a large dataset of Windows device telemetry (engine versions, OS metadata, hardware specs, antivirus signature data, etc.). It trains an ensemble model — selected from XGBoost, LightGBM, and Random Forest via RFECV feature selection — and serves predictions through a Flask web UI.

The app is containerized and configured to run on [Hugging Face Spaces](https://huggingface.co/spaces), listening on port `7860`.

---

## Project Structure

```
SystemThreatForecaster/
├── app.py                        # Flask application entry point
├── Dockerfile                    # Container definition (Python 3.9-slim)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup (name: threatforecaster)
│
├── src/
│   ├── exception.py              # Custom exception with traceback info
│   ├── logger.py                 # Timestamped file logger
│   ├── utils.py                  # save_object, load_object, evaluate_models
│   ├── components/
│   │   ├── data_ingestion.py     # Reads cleaned CSVs, performs train/val split
│   │   ├── data_transformation.py# Feature engineering + sklearn preprocessor
│   │   └── model_trainer.py      # Baseline eval → RFECV → best model saved
│   └── pipeline/
│       └── predict_pipeline.py   # Inference: feature eng → preprocess → predict
│
├── artifacts/
│   ├── model.pkl                 # Serialized best model + RFECV feature mask
│   ├── preprocessor.pkl          # Fitted ColumnTransformer
│   ├── train.csv                 # 80% split of training data
│   ├── val.csv                   # 20% split of training data
│   └── test.csv                  # Hold-out test set
│
├── notebook/
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── Model_Training.ipynb      # Interactive model development
│   └── data/
│       ├── train.csv / test.csv              # Raw data
│       ├── train_eda_clean.csv               # EDA-cleaned train set
│       ├── test_eda_clean.csv                # EDA-cleaned test set
│       └── submission.csv                    # Kaggle-format predictions
│
├── templates/
│   ├── index.html                # Upload form UI
│   └── results.html              # Predictions table + download
│
└── logs/                         # Timestamped run logs
```

---

## ML Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
Reads `train_eda_clean.csv` and `test_eda_clean.csv` from `notebook/data/`, performs an 80/20 train/val split, and writes the splits to `artifacts/`.

### 2. Feature Engineering (`data_transformation.py`)
Applied identically at training and inference time:

- **Version splitting** — `EngineVersion`, `AppVersion`, `SignatureVersion`, and `NumericOSVersion` are split on `.` into Major / Minor / Build / Revision numeric columns.
- **Date decomposition** — `DateAS` (malware signature date) and `DateOS` (OS install date) are expanded into year, month, day, hour, and minute components.
- **Categorical grouping** — High-cardinality fields like `MDC2FormFactor`, `PrimaryDiskType`, `ChassisType`, `PowerPlatformRole`, `OSBranch`, and `OSEdition` are grouped into consolidated buckets.
- **Preprocessing** — A `ColumnTransformer` applies `MinMaxScaler` to numerical columns, `OrdinalEncoder` to binary columns, and `OneHotEncoder` to remaining categoricals. Missing values are imputed before scaling/encoding. The fitted preprocessor is saved to `artifacts/preprocessor.pkl`.

### 3. Model Training (`model_trainer.py`)

**Baseline evaluation** across 7 classifiers:

| Model | Val Accuracy |
|---|---|
| LightGBM | 0.6204 |
| XGBoost | 0.6146 |
| Random Forest | 0.6090 |
| AdaBoost | 0.6035 |
| Bagging | 0.5724 |
| Logistic Regression | 0.5539 |
| Decision Tree | 0.5370 |

**RFECV tuning** is then run on the top 3 models (XGBoost, LightGBM, Random Forest). Each model undergoes recursive feature elimination with cross-validation to select an optimal feature subset, followed by training with tuned hyperparameters. The best-performing model and its RFECV feature mask are saved together to `artifacts/model.pkl`.

### 4. Prediction Pipeline (`predict_pipeline.py`)
At inference time: loads `preprocessor.pkl` and `model.pkl`, applies the same feature engineering, transforms the input through the preprocessor, applies the RFECV feature mask, and returns binary predictions (0 = clean, 1 = infected).

---

## Web Application

The Flask app (`app.py`) exposes three routes:

- `GET /` — renders the upload form
- `POST /predict` — accepts a CSV file, runs the prediction pipeline, and renders a results page showing a summary (total records, infected count, clean count) and a preview table of the first 50 predictions
- `POST /download` — streams the full results as a downloadable `submission.csv`

Only `.csv` files are accepted. If the uploaded CSV contains an `id` column it is preserved in the output; otherwise a sequential index is used.

---

## Getting Started

### Prerequisites

- Python 3.9+
- The EDA-cleaned datasets at `notebook/data/train_eda_clean.csv` and `notebook/data/test_eda_clean.csv`

### Install dependencies

```bash
pip install -r requirements.txt
# or install as a package
pip install -e .
```

### Train the model

Run the full pipeline (ingestion → transformation → training) from the project root:

```bash
python src/components/data_ingestion.py
```

This writes `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Run the app

```bash
python app.py
```

The app starts on `http://0.0.0.0:7860`.

### Run with Docker

```bash
docker build -t system-threat-forecaster .
docker run -p 7860:7860 system-threat-forecaster
```

---

## Dependencies

```
pandas, numpy, scikit-learn
xgboost, lightgbm, catboost
flask
dill
seaborn, matplotlib
```

---

## Author

Rishit Pant — `rishitpant100@gmail.com`