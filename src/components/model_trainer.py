# model_trainer.py

import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


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
                'Logistic Regression': LogisticRegression(random_state=1),
                'Random Forest':       RandomForestClassifier(random_state=1),
                'LightGBM':            LGBMClassifier(random_state=1),
                'XGBoost':             XGBClassifier(random_state=1),
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

            top_3_configs = {
                'XGBoost': {
                    'model': XGBClassifier(
                        max_depth=5, learning_rate=0.1,
                        n_estimators=200, reg_lambda=10,
                        random_state=42
                    ),
                    'step': 2, 'cv': 5
                },
                'LightGBM': {
                    'model': LGBMClassifier(
                        max_depth=7, learning_rate=0.1, num_leaves=31,
                        min_child_samples=30, reg_lambda=10,
                        random_state=42
                    ),
                    'step': 2, 'cv': 5
                },
                'Random Forest': {
                    'model': RandomForestClassifier(
                        n_estimators=50, max_depth=8,
                        min_samples_leaf=10,
                        random_state=42
                    ),
                    'step': 10, 'cv': 2
                }
            }

            tuned_results = {}

            for name, config in top_3_configs.items():
                logging.info(f"Running RFECV for {name}.")

                rfecv = RFECV(
                    estimator=config['model'],
                    step=config['step'],
                    cv=config['cv'],
                    scoring='accuracy'
                )
                rfecv.fit(X_train, y_train)
                selected = rfecv.support_

                X_train_sel = X_train[:, selected]
                X_val_sel   = X_val[:,   selected]
                logging.info(f"{name} RFECV selected {selected.sum()} features.")

                config['model'].fit(X_train_sel, y_train)
                val_acc = accuracy_score(y_val, config['model'].predict(X_val_sel))
                logging.info(f"{name} val accuracy: {val_acc:.4f}")

                tuned_results[name] = {
                    'model':    config['model'],
                    'selected': selected,
                    'val_acc':  val_acc
                }

            best_name = max(tuned_results, key=lambda k: tuned_results[k]['val_acc'])
            best      = tuned_results[best_name]

            if best['val_acc'] < 0.6:
                raise CustomException("No best model found — val accuracy below 0.6.", sys)

            logging.info(f"Best model: {best_name} with val accuracy {best['val_acc']:.4f}")
            print(classification_report(y_val, best['model'].predict(X_val[:, best['selected']])))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    'model':    best['model'],
                    'selected': best['selected'],
                    'name':     best_name
                }
            )
            logging.info(f"Best model saved to artifacts/.")

            return best['val_acc']

        except Exception as e:
            raise CustomException(e, sys)