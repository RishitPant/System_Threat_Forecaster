import os
import sys
import json
import numpy as np
import pandas as pd
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
    precision_recall_curve, average_precision_score
)
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_val, y_val, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred   = model.predict(X_val)

            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc   = accuracy_score(y_val,   y_val_pred)

            logging.info(f"{name} — train: {train_acc:.4f}, val: {val_acc:.4f}")
            report[name] = {
                'val_acc':     accuracy_score(y_val, y_val_pred),
                'val_recall':  recall_score(y_val, y_val_pred),
                'val_f1':      f1_score(y_val, y_val_pred),
                'val_roc_auc': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


def save_best_params(params: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    logging.info(f"Best params saved to {path}")


def load_best_params(path: str):
    with open(path, "r") as f:
        params = json.load(f)
    logging.info(f"Loaded best params from {path}")
    return params

    

def evaluate_final_model(model, X_val, y_val, selected, name, report_dir="artifacts/eval"):
    """
    Prints full classification report, logs all metrics, and saves
    a three-panel eval figure (confusion matrix, ROC, PR curve)
    to report_dir/<name>.png
    """
    try:
        os.makedirs(report_dir, exist_ok=True)

        # Apply feature selection mask if provided
        X_sel  = X_val[:, selected] if selected is not None else X_val
        y_pred = model.predict(X_sel)
        y_prob = model.predict_proba(X_sel)[:, 1]

        auc = roc_auc_score(y_val, y_prob)
        f1  = f1_score(y_val, y_pred)
        pre = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        ap  = average_precision_score(y_val, y_prob)

        print(f"\n{'='*52}")
        print(f"  Evaluation — {name}")
        print(f"{'='*52}")
        print(classification_report(y_val, y_pred, target_names=["Clean (0)", "Infected (1)"]))
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  Precision: {pre:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Avg Prec : {ap:.4f}")
        print(f"{'='*52}\n")

        logging.info(
            f"{name} — AUC: {auc:.4f} | F1: {f1:.4f} | "
            f"Precision: {pre:.4f} | Recall: {rec:.4f} | AP: {ap:.4f}"
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle(f"Evaluation — {name}", fontsize=13, fontweight='bold')

        # Panel 1 — Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred,
            display_labels=["Clean", "Infected"],
            cmap="Blues", ax=axes[0]
        )
        axes[0].set_title("Confusion Matrix")

        # Panel 2 — ROC Curve
        RocCurveDisplay.from_predictions(
            y_val, y_prob, name=name, ax=axes[1]
        )
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label="Random")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        # Panel 3 — Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob)
        axes[2].plot(recall_vals, precision_vals, label=f"{name} (AP = {ap:.2f})")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")
        axes[2].legend()

        plt.tight_layout()

        # Save as artifacts/eval/<model_name>.png
        safe_name = name.replace(" ", "_")
        save_path  = os.path.join(report_dir, f"{safe_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info(f"Eval report saved → {save_path}")

        return {"auc": auc, "f1": f1, "precision": pre, "recall": rec, "ap": ap}

    except Exception as e:
        raise CustomException(e, sys)
