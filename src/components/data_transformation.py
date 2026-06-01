# data_transformation.py

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _apply_feature_engineering(self, df):
        """Mirrors cells 05-24 of Model_Training.ipynb exactly."""
        df = df.copy()

        # ── Feature Splitting (cell 05) ──────────────────────────────────────
        for col, prefix in [
            ('EngineVersion',    'EngineVersion'),
            ('AppVersion',       'AppVersion'),
            ('SignatureVersion', 'SignatureVersion'),
            ('NumericOSVersion', 'NumericOS'),
        ]:
            if col in df.columns:
                parts = df[col].str.split('.', expand=True).astype(float)
                parts.columns = [f"{prefix}_Major", f"{prefix}_Minor",
                                 f"{prefix}_Build", f"{prefix}_Revision"]
                df = pd.concat([df, parts], axis=1)

        if 'DateAS' in df.columns:
            df['Malware_year']   = pd.to_datetime(df['DateAS']).dt.year
            df['Malware_month']  = pd.to_datetime(df['DateAS']).dt.month
            df['Malware_day']    = pd.to_datetime(df['DateAS']).dt.day
            df['Malware_hour']   = pd.to_datetime(df['DateAS']).dt.hour
            df['Malware_minute'] = pd.to_datetime(df['DateAS']).dt.minute

        if 'DateOS' in df.columns:
            df['OS_year']  = pd.to_datetime(df['DateOS']).dt.year
            df['OS_month'] = pd.to_datetime(df['DateOS']).dt.month
            df['OS_day']   = pd.to_datetime(df['DateOS']).dt.weekday

        # Drop original version/date columns (cell 07)
        df.drop(columns=[
            'EngineVersion', 'AppVersion', 'SignatureVersion',
            'NumericOSVersion', 'DateAS', 'DateOS'
        ], inplace=True, errors='ignore')

        # ── Feature Grouping (cell 09) ────────────────────────────────────────
        if 'MDC2FormFactor' in df.columns:
            df.loc[df['MDC2FormFactor'].isin(['Desktop', 'PCOther', 'AllInOne']),       'MDC2FormFactor_Grouped'] = 'Desktop'
            df.loc[df['MDC2FormFactor'].isin(['Notebook', 'Convertible', 'Detachable']), 'MDC2FormFactor_Grouped'] = 'Notebook'
            df.loc[df['MDC2FormFactor'].isin(['LargeTablet', 'SmallTablet']),            'MDC2FormFactor_Grouped'] = 'Tablet'
            df.loc[df['MDC2FormFactor'].isin(['SmallServer', 'MediumServer', 'LargeServer']), 'MDC2FormFactor_Grouped'] = 'Server'

        if 'PrimaryDiskType' in df.columns:
            df.loc[df['PrimaryDiskType'] == 'HDD',                          'PrimaryDiskType_Grouped'] = 'HDD'
            df.loc[df['PrimaryDiskType'] == 'SSD',                          'PrimaryDiskType_Grouped'] = 'SSD'
            df.loc[~df['PrimaryDiskType'].isin(['HDD', 'SSD']),             'PrimaryDiskType_Grouped'] = 'Others'

        if 'ChassisType' in df.columns:
            df.loc[df['ChassisType'].isin(['Desktop', 'Tower', 'MiniTower', 'LowProfileDesktop', 'MiniPC', 'AllinOne']), 'ChassisType_Grouped'] = 'Desktop'
            df.loc[df['ChassisType'].isin(['Notebook', 'Portable', 'Laptop']),                                           'ChassisType_Grouped'] = 'Notebook'
            df.loc[df['ChassisType'].isin(['Tablet', 'Convertible', 'Detachable', 'HandHeld']),                          'ChassisType_Grouped'] = 'Tablet'
            df.loc[~df['ChassisType'].isin(['Desktop', 'Tower', 'MiniTower', 'LowProfileDesktop', 'MiniPC', 'AllinOne',
                                            'Notebook', 'Portable', 'Laptop', 'Tablet', 'Convertible',
                                            'Detachable', 'HandHeld']),                                                  'ChassisType_Grouped'] = 'Others'

        if 'PowerPlatformRole' in df.columns:
            df.loc[df['PowerPlatformRole'] == 'Desktop',                                                          'PowerPlatformRole_Grouped'] = 'Desktop'
            df.loc[df['PowerPlatformRole'].isin(['Mobile', 'Slate']),                                             'PowerPlatformRole_Grouped'] = 'Portable'
            df.loc[df['PowerPlatformRole'].isin(['SOHOServer', 'EnterpriseServer', 'PerformanceServer']),         'PowerPlatformRole_Grouped'] = 'Server'
            df.loc[~df['PowerPlatformRole'].isin(['Desktop', 'Mobile', 'Slate', 'SOHOServer',
                                                   'EnterpriseServer', 'PerformanceServer']),                     'PowerPlatformRole_Grouped'] = 'Others'

        if 'OSBranch' in df.columns:
            df.loc[df['OSBranch'].isin(['rs1_release', 'rs2_release', 'rs3_release', 'rs3_release_svc_escrow',
                                        'rs3_release_svc_escrow_im', 'rs4_release', 'rs5_release',
                                        'rs_prerelease_flt', 'rs_prerelease']),      'OSBranch_Grouped'] = 'rs_release'
            df.loc[df['OSBranch'].isin(['th1_st1', 'th1', 'th2_release', 'th2_release_sec']), 'OSBranch_Grouped'] = 'th_release'

        if 'OSEdition' in df.columns:
            core  = ['Core', 'CoreSingleLanguage', 'CoreCountrySpecific', 'CoreN']
            pro   = ['Professional', 'ProfessionalN', 'ProfessionalEducation', 'Education',
                     'EducationN', 'ProfessionalEducationN', 'ProfessionalWorkstation',
                     'ProfessionalSingleLanguage', 'ProfessionalCountrySpecific']
            ent   = ['Enterprise', 'EnterpriseN', 'EnterpriseS', 'EnterpriseSN']
            df.loc[df['OSEdition'].isin(core), 'OSEdition_Grouped']  = 'Core'
            df.loc[df['OSEdition'].isin(pro),  'OSEdition_Grouped']  = 'Professional'
            df.loc[df['OSEdition'].isin(ent),  'OSEdition_Grouped']  = 'Enterprise'
            df.loc[~df['OSEdition'].isin(core + pro + ent), 'OSEdition_Grouped'] = 'Others'

        if 'OSInstallType' in df.columns:
            df.loc[df['OSInstallType'].isin(['UUPUGrade', 'Update', 'Upgrade']),                          'OSInstallType_Grouped'] = 'Upgrade'
            df.loc[df['OSInstallType'].isin(['Reset', 'Refresh', 'CleanPCRefresh', 'Clean', 'IBSClean']), 'OSInstallType_Grouped'] = 'Clean'
            df.loc[~df['OSInstallType'].isin(['UUPUGrade', 'Update', 'Upgrade',
                                              'Reset', 'Refresh', 'CleanPCRefresh', 'Clean', 'IBSClean']), 'OSInstallType_Grouped'] = 'Others'

        if 'AutoUpdateOptionsName' in df.columns:
            df.loc[df['AutoUpdateOptionsName'].isin(['FullAuto', 'AutoInstallAndRebootAtMaintenanceTime']), 'AutoUpdateOptionsName_Grouped'] = 'Auto'
            df.loc[df['AutoUpdateOptionsName'].isin(['Notify', 'DownloadNotify']),                         'AutoUpdateOptionsName_Grouped'] = 'Manual'
            df.loc[df['AutoUpdateOptionsName'] == 'Off',                                                   'AutoUpdateOptionsName_Grouped'] = 'Off'
            df.loc[~df['AutoUpdateOptionsName'].isin(['FullAuto', 'AutoInstallAndRebootAtMaintenanceTime',
                                                      'Notify', 'DownloadNotify', 'Off']),                 'AutoUpdateOptionsName_Grouped'] = 'Unknown'

        if 'LicenseActivationChannel' in df.columns:
            df.loc[df['LicenseActivationChannel'].isin(['Retail', 'Retail:TB:Eval']),   'LicenseActivationChannel_Grouped'] = 'Retail'
            df.loc[df['LicenseActivationChannel'].isin(['Volume:GVLK', 'Volume:MAK']), 'LicenseActivationChannel_Grouped'] = 'Volume'
            df.loc[~df['LicenseActivationChannel'].isin(['Retail', 'Retail:TB:Eval',
                                                          'Volume:GVLK', 'Volume:MAK']), 'LicenseActivationChannel_Grouped'] = 'OEM'

        if 'FlightRing' in df.columns:
            df.loc[df['FlightRing'] == 'Retail',                          'FlightRing_Grouped'] = 'Retail'
            df.loc[df['FlightRing'].isin(['WIS', 'RP', 'WIF']),           'FlightRing_Grouped'] = 'Insider'
            df.loc[df['FlightRing'] == 'Disabled',                        'FlightRing_Grouped'] = 'Disabled'
            df.loc[~df['FlightRing'].isin(['Retail', 'WIS', 'RP', 'WIF', 'Disabled']), 'FlightRing_Grouped'] = 'Unknown'

        # Drop original grouping columns (cell 11)
        df.drop(columns=[
            'MDC2FormFactor', 'PrimaryDiskType', 'ChassisType', 'PowerPlatformRole',
            'OSBranch', 'OSEdition', 'OSInstallType', 'AutoUpdateOptionsName',
            'LicenseActivationChannel', 'FlightRing'
        ], inplace=True, errors='ignore')

        # ── Feature Creation (cell 13) ────────────────────────────────────────
        if 'Malware_day' in df.columns and 'OS_day' in df.columns:
            df['Days_since_OS_Installation'] = df['Malware_day'] - df['OS_day']

        if 'TotalPhysicalRAMMB' in df.columns:
            df['Ram_per_core'] = df['TotalPhysicalRAMMB'] / df['ProcessorCoreCount']

        if 'PrimaryDisplayResolutionHorizontal' in df.columns:
            df['Aspect_Ratio']   = df['PrimaryDisplayResolutionHorizontal'] / df['PrimaryDisplayResolutionVertical']
            df['Pixel_Density']  = (df['PrimaryDisplayResolutionHorizontal'] * df['PrimaryDisplayResolutionVertical']) / df['PrimaryDisplayDiagonalInches']

        if 'SystemVolumeCapacityMB' in df.columns:
            df['Primary_Disk_Allocated'] = df['PrimaryDiskCapacityMB'] / df['SystemVolumeCapacityMB']
            df['Free_Disk_Space']        = (df['SystemVolumeCapacityMB'] - df['PrimaryDiskCapacityMB']) / df['PrimaryDiskCapacityMB']

        # ── Drop redundant post-FE columns (cell 24) ─────────────────────────
        df.drop(columns=[
            'SignatureVersion_Minor', 'SignatureVersion_Major', 'SignatureVersion_Revision',
            'AppVersion_Major', 'EngineVersion_Major', 'EngineVersion_Minor',
            'NumericOS_Major', 'NumericOS_Minor', 'NumericOS_Revision', 'NumericOS_Build',
            'ProductName', 'OsPlatformSubRelease', 'OSBuildRevisionOnly'
        ], inplace=True, errors='ignore')

        return df

    def get_data_transformer_object(self, df):
        try:
            binary_columns = [
                col for col in df.columns
                if df[col].nunique() == 2 and col != 'target'
            ]
            ID_columns = [
                col for col in df.columns
                if df[col].dtype in ['int64', 'float64']
                and any(k in col for k in ["ID", "Identifier"])
            ]
            numerical_columns = [
                col for col in df.columns
                if df[col].dtype in ['int64', 'float64']
                and col not in binary_columns
                and col not in ID_columns
                and col != 'target'
            ]
            categorical_columns = [
                col for col in df.columns
                if df[col].dtype == 'object'
            ]

            logging.info(f"Numerical: {len(numerical_columns)}, Binary: {len(binary_columns)}, "
                         f"ID: {len(ID_columns)}, Categorical: {len(categorical_columns)}")

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",  MinMaxScaler())
            ])
            binary_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder())
            ])
            id_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ])
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numerical_pipeline,    numerical_columns),
                ("bin", binary_pipeline,       binary_columns),
                ("id",  id_pipeline,           ID_columns),
                ("cat", categorical_pipeline,  categorical_columns)
            ])

            return preprocessor, numerical_columns, binary_columns, ID_columns, categorical_columns

        except Exception as e:
            raise CustomException(e, sys)

    # data_transformation.py  — only initiate_data_transformation changes

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            val_df   = pd.read_csv(val_path)
            test_df  = pd.read_csv(test_path)
            logging.info(f"Read train {train_df.shape}, val {val_df.shape}, test {test_df.shape}.")

            # Apply feature engineering to all three
            train_df = self._apply_feature_engineering(train_df)
            val_df   = self._apply_feature_engineering(val_df)
            test_df  = self._apply_feature_engineering(test_df)
            logging.info("Feature engineering applied to train, val, and test.")

            # Separate features and target
            X_train = train_df.drop(columns=["target"])
            y_train = train_df["target"]

            X_val   = val_df.drop(columns=["target"])
            y_val   = val_df["target"]

            X_test  = test_df  # no target column

            # Build preprocessor from X_train shape
            preprocessor, _, _, _, _ = self.get_data_transformer_object(X_train)

            # fit on X_train only, transform all three (mirrors cell 27)
            X_train_arr = preprocessor.fit_transform(X_train)
            X_val_arr   = preprocessor.transform(X_val)
            X_test_arr  = preprocessor.transform(X_test)
            logging.info("Preprocessing applied to train, val, and test.")

            # Attach targets back to train and val arrays
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            val_arr   = np.c_[X_val_arr,   np.array(y_val)]
            test_arr  = X_test_arr  # no target — for final submission only

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor saved to artifacts/.")

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)