import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component.")
        try:
            train_df = pd.read_csv('notebook/data/train.csv')
            test_df = pd.read_csv('notebook/data/test.csv')
            logging.info(f'Read train {train_df.shape} and test {test_df.shape} datasets.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Created artifacts directory.")

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved to artifacts/.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")