from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys

class TrainPipeline:
    def run(self):
        try:
            logging.info("Starting data cleaning.")
            cleaner = DataCleaning()
            cleaner.initiate_data_cleaning()

            logging.info("Starting data ingestion.")
            ingestion = DataIngestion()
            train_path, val_path, test_path = ingestion.initiate_data_ingestion()

            logging.info("Starting data transformation.")
            transformation = DataTransformation()
            train_arr, val_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_path, val_path, test_path
            )

            logging.info("Starting model training.")
            trainer = ModelTrainer()
            val_accuracy = trainer.initiate_model_trainer(train_arr, val_arr)
            logging.info(f"Training complete. Val accuracy: {val_accuracy:.4f}")

            return val_accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run()