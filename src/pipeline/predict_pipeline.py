import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes a raw dataframe (same format as test_eda_clean.csv),
        applies feature engineering + preprocessing, then predicts.
        """
        try:
            # Load preprocessor and model pickels
            preprocessor = load_object("artifacts/preprocessor.pkl")
            model_artifact = load_object("artifacts/model.pkl")

            model    = model_artifact['model']
            selected = model_artifact['selected']  # boolean mask from RFECV
            name     = model_artifact['name']
            logging.info(f"Loaded model: {name}")

            # Apply feature engineering
            transformation = DataTransformation()
            df = transformation._apply_feature_engineering(df)
            logging.info(f"Feature engineering applied. Shape: {df.shape}")

            # Apply preprocessor 
            arr = preprocessor.transform(df)
            logging.info(f"Preprocessing applied. Shape: {arr.shape}")

            # Apply RFECV feature selection mask
            arr_selected = arr[:, selected]
            logging.info(f"Feature selection applied. Shape: {arr_selected.shape}")

            preds = model.predict(arr_selected)
            logging.info(f"Predictions generated: {len(preds)} rows.")

            return preds

        except Exception as e:
            raise CustomException(e, sys)