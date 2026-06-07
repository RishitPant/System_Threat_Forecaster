# model_trainer.py
import os
import sys
import json
import numpy as np
from dataclasses import dataclass

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    recall_score, roc_auc_score, classification_report
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_final_model, save_best_params, load_best_params


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    eval_report_path: str        = os.path.join("artifacts", "eval_report.png")

def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=20, params_path="artifacts/params/lightgbm_best_params.json"):
    if os.path.exists(params_path):
        logging.info("LightGBM: found saved params, skipping Tuning.")
        return LGBMClassifier(**load_best_params(params_path), random_state=42, verbosity=-1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logging.info(f"LightGBM best params: {study.best_params} | AUC: {study.best_value: .4f}")
    save_best_params(study.best_params, params_path)
    return LGBMClassifier(**study.best_params, random_state=42, verbosity=-1)


def tune_xgboost(X_train, y_train, X_val, y_val,
                n_trials=20,
                params_path="artifacts/params/xgboost_best_params.json"):
    if os.path.exists(params_path):
        logging.info("XGBoost: found saved params, skipping Optuna.")
        return XGBClassifier(**load_best_params(params_path), random_state=42, eval_metric='logloss', verbosity=0)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        model = XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)
        model.fit(X_train, y_train)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logging.info(f"XGBoost best params: {study.best_params} | AUC: {study.best_value:.4f}")
    save_best_params(study.best_params, params_path)
    return XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss', verbosity=0)


def tune_random_forest(X_train, y_train, X_val, y_val,
                    n_trials=20,
                    params_path="artifacts/params/rf_best_params.json"):
    if os.path.exists(params_path):
        logging.info("Random Forest: found saved params, skipping Optuna.")
        return RandomForestClassifier(**load_best_params(params_path), random_state=42)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 5, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        }
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logging.info(f"Random Forest best params: {study.best_params} | AUC: {study.best_value:.4f}")
    save_best_params(study.best_params, params_path)
    return RandomForestClassifier(**study.best_params, random_state=42)

TUNING_FUNCTIONS = {
    'LightGBM':     tune_lightgbm,
    'XGBoost':      tune_xgboost,
    'Random Forest': tune_random_forest,
}

mlflow.set_experiment("Threat_Forecaster")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, val_array):
        try:
            logging.info("Splitting train and val arrays into features and target.")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_val   = val_array[:, :-1]
            y_val   = val_array[:, -1]

            models = {
                'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000),
                'Random Forest':       RandomForestClassifier(random_state=1),
                'LightGBM':            LGBMClassifier(random_state=1, verbosity=-1),
                'XGBoost':             XGBClassifier(random_state=1, verbosity=0, eval_metric='logloss'),
                'Decision Tree':       DecisionTreeClassifier(random_state=1),
                'AdaBoost':            AdaBoostClassifier(random_state=1),
                'Bagging':             BaggingClassifier(random_state=1),
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                models=models
            )
            logging.info(f"Baseline model report: {model_report}")

            for name, model in models.items():
                baseline_metrics = evaluate_final_model(
                    model      = model,
                    X_val      = X_val,
                    y_val      = y_val,
                    selected   = None,
                    name       = f"Baseline_{name}",
                    report_dir = "artifacts/eval/baseline"
                )
                with mlflow.start_run(run_name=f"Baseline_{name}"):
                    mlflow.set_tag("stage", "baseline")
                    mlflow.log_metric("val_roc_auc",   baseline_metrics['auc'])
                    mlflow.log_metric("val_recall",    baseline_metrics['recall'])
                    mlflow.log_metric("val_precision", baseline_metrics['precision'])
                    mlflow.log_metric("val_f1",        baseline_metrics['f1'])
                    mlflow.log_metric("val_ap",        baseline_metrics['ap'])

            top_3_names = sorted(
                model_report.keys(),
                key=lambda k: model_report[k]['val_roc_auc'],
                reverse=True
            )[:3]

            logging.info(f"Top 3 models by AUC: {top_3_names}")
            print(f"\nTop 3 selected for tuning: {top_3_names}")

            tuned_results = {}

            for name in top_3_names:
                if name not in TUNING_FUNCTIONS:
                    logging.warning(f"No tuning function for {name} — using baseline model as-is.")
                    tuned_model = models[name]
                else:
                    tune_fn     = TUNING_FUNCTIONS[name]
                    params_path = f"artifacts/params/{name.replace(' ', '_').lower()}_best_params.json"
                    tuned_model = tune_fn(X_train, y_train, X_val, y_val, params_path=params_path)

                tuned_model.fit(X_train, y_train)

                tuned_metrics = evaluate_final_model(
                    model      = tuned_model,
                    X_val      = X_val,
                    y_val      = y_val,
                    selected   = None,
                    name       = f"Tuned_{name}",
                    report_dir = "artifacts/eval/tuned"
                )
                logging.info(f"Tuned {name} → AUC: {tuned_metrics['auc']:.4f} | Recall: {tuned_metrics['recall']:.4f}")

                tuned_results[name] = {
                    'model':       tuned_model,
                    'val_roc_auc': tuned_metrics['auc'],
                    'recall':      tuned_metrics['recall'],
                }

                with mlflow.start_run(run_name=f"Tuned_{name}"):
                    mlflow.set_tag("stage", "tuning")
                    mlflow.log_params(tuned_model.get_params())
                    mlflow.log_metric("val_roc_auc",   tuned_metrics['auc'])
                    mlflow.log_metric("val_recall",    tuned_metrics['recall'])
                    mlflow.log_metric("val_precision", tuned_metrics['precision'])
                    mlflow.log_metric("val_f1",        tuned_metrics['f1'])
                    mlflow.log_metric("val_ap",        tuned_metrics['ap'])

            best_name = max(tuned_results, key=lambda k: tuned_results[k]['val_roc_auc'])
            best      = tuned_results[best_name]

            if best['val_roc_auc'] < 0.5:
                raise CustomException("No best model found — AUC below 0.5.", sys)

            logging.info(f"Best model: {best_name} | AUC: {best['val_roc_auc']:.4f}")
            print(f"\nBest model: {best_name} — running RFECV for feature selection.")

            rfecv = RFECV(
                estimator = best['model'],
                step      = 2,
                cv        = 5,
                scoring   = 'roc_auc',
                n_jobs    = -1
            )
            rfecv.fit(X_train, y_train)
            selected = rfecv.support_
            logging.info(f"RFECV selected {selected.sum()} / {len(selected)} features.")

            X_train_sel = X_train[:, selected]
            X_val_sel   = X_val[:,   selected]

            best['model'].fit(X_train_sel, y_train)

            final_metrics = evaluate_final_model(
                model      = best['model'],
                X_val      = X_val,
                y_val      = y_val,
                selected   = selected,
                name       = f"Final_{best_name}",
                report_dir = "artifacts/eval/final"
            )

            with mlflow.start_run(run_name=f"Final_{best_name}"):
                mlflow.set_tag("stage", "final")
                mlflow.log_params(best['model'].get_params())
                mlflow.log_metric("val_roc_auc",      final_metrics['auc'])
                mlflow.log_metric("val_recall",       final_metrics['recall'])
                mlflow.log_metric("val_precision",    final_metrics['precision'])
                mlflow.log_metric("val_f1",           final_metrics['f1'])
                mlflow.log_metric("val_ap",           final_metrics['ap'])
                mlflow.log_metric("features_selected", int(selected.sum()))
                mlflow.sklearn.log_model(
                    best['model'],
                    artifact_path="model",
                    registered_model_name="Threat_Forecaster"
                )

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = {
                    'model':    best['model'],
                    'selected': selected,
                    'name':     best_name
                }
            )
            logging.info("Best model saved to artifacts/.")

            return best['val_roc_auc']

        except Exception as e:
            raise CustomException(e, sys)
