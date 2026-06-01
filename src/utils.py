import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import accuracy_score
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
            report[name] = val_acc

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)