# data_ingestion.py

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str   = os.path.join('artifacts', 'val.csv')
    test_data_path: str  = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component.")
        try:
            train_df = pd.read_csv('notebook/data/train_eda_clean.csv')
            test_df  = pd.read_csv('notebook/data/test_eda_clean.csv')
            logging.info(f'Read train {train_df.shape} and test {test_df.shape} datasets.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            X = train_df.drop(columns=["target"])
            y = train_df["target"]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            train_split = pd.concat([X_train, y_train], axis=1)
            val_split   = pd.concat([X_val,   y_val],   axis=1)

            train_split.to_csv(self.ingestion_config.train_data_path, index=False)
            val_split.to_csv(self.ingestion_config.val_data_path,     index=False)
            test_df.to_csv(self.ingestion_config.test_data_path,      index=False)

            logging.info("Train/val split and Kaggle test saved to artifacts/.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Ingest
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train: {train_path} | Val: {val_path} | Test: {test_path}")

    # Transform
    transformation = DataTransformation()
    train_arr, val_arr, test_arr, preprocessor_path = \
        transformation.initiate_data_transformation(train_path, val_path, test_path)
    print(f"Train: {train_arr.shape} | Val: {val_arr.shape} | Test: {test_arr.shape}")

    # Train
    trainer = ModelTrainer()
    val_accuracy = trainer.initiate_model_trainer(train_arr, val_arr)
    print(f"Best model val accuracy: {val_accuracy:.4f}")