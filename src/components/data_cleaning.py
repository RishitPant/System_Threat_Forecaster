import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DataCleaningConfig:
    raw_train_data_path: str = os.path.join('notebook', 'data', 'train.csv')
    raw_test_data_path: str = os.path.join('notebook', 'data', 'test.csv')

    cleaned_train_data_path: str = os.path.join('notebook', 'data', 'train_eda_clean.csv')
    cleaned_test_data_path: str = os.path.join('notebook', 'data', 'test_eda_clean.csv')


class DataCleaning:
    def __init__(self):
        self.cleaning_config = DataCleaningConfig()

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            initial_shape = df.shape
            df = df.drop_duplicates()
            final_shape = df.shape
            logging.info(f"Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            cols_to_drop = ['MachineID', 'IsBetaUser', 'AutoSampleSubmissionEnabled', 'IsFlightsDisabled', 'SMode', 'HasTpm', 'IsVirtualDevice', 'IsPortableOS', "DeviceFamily", 'EnableLUA',
                            "OSBuildLab", "OSBuildNumberOnly", "SKUEditionName", "OSSkuFriendlyName", "OSInstallLanguageID", "Processor", "OSVersion"
                            ]
            
            existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(columns=existing_cols_to_drop)
                logging.info(f"Dropped columns: {existing_cols_to_drop}")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'target' in df.columns:
                initial_shape = df.shape
                df = df.dropna(subset=['target'])
                logging.info(f"Dropped {initial_shape[0] - df.shape[0]} rows due to missing target.")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_cleaning(self):
        logging.info(f"Entered the data cleaning component...")
        try:
            train_df = pd.read_csv(self.cleaning_config.raw_train_data_path)
            test_df = pd.read_csv(self.cleaning_config.raw_test_data_path)
            logging.info(f"Read raw train [{train_df.shape}] and test [{test_df.shape}] dataset")

            os.makedirs(os.path.dirname(self.cleaning_config.cleaned_train_data_path), exist_ok=True)

            logging.info("Cleaning train data...")
            train_df = self._drop_duplicates(train_df)
            train_df = self._drop_unnecessary_columns(train_df)
            train_df = self._handle_missing_values(train_df)

            logging.info("Cleaning test data...")
            # test_df = self._drop_duplicates(test_df)
            test_df = self._drop_unnecessary_columns(test_df)
            test_df = self._handle_missing_values(test_df)

            train_df.to_csv(self.cleaning_config.cleaned_train_data_path, index=False)
            test_df.to_csv(self.cleaning_config.cleaned_test_data_path, index=False)
            logging.info("Cleaned train and test data saved successfully.")

            return (
                self.cleaning_config.cleaned_train_data_path,
                self.cleaning_config.cleaned_test_data_path,
                train_df.shape,
                test_df.shape
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Test the Data Cleaning Component
    cleaner = DataCleaning()
    cleaned_train_path, cleaned_test_path, train_shape, test_shape = cleaner.initiate_data_cleaning()
    print(f"Data Cleaning Completed!\nCleaned Train shape: {train_shape}\nCleaned Test Path: {test_shape}")