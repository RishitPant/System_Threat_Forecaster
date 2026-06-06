import pandas as pd
import numpy as np
import pytest
from src.components.data_transformation import DataTransformation

@pytest.fixture
def transformer():
    return DataTransformation()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'EngineVersion':    ['1.2.3.4'],
        'AppVersion':       ['5.6.7.8'],
        'SignatureVersion': ['9.10.11.12'],
        'NumericOSVersion': ['10.0.19041.1'],
        'DateAS': ['2021-06-01 10:30:00'],
        'DateOS': ['2020-01-15 00:00:00'],
        'TotalPhysicalRAMMB': [8192],
        'ProcessorCoreCount': [4],
    })

def test_version_splitting_creates_columns(transformer, sample_df):
    result = transformer._apply_feature_engineering(sample_df)
    assert 'EngineVersion_Build' in result.columns
    assert 'AppVersion_Minor' in result.columns

def test_original_version_columns_dropped(transformer, sample_df):
    result = transformer._apply_feature_engineering(sample_df)
    assert 'EngineVersion' not in result.columns
    assert 'AppVersion' not in result.columns

def test_date_decomposition_creates_columns(transformer, sample_df):
    result = transformer._apply_feature_engineering(sample_df)
    assert 'Malware_year' in result.columns
    assert 'Malware_month' in result.columns
    assert 'OS_year' in result.columns

def test_ram_per_core_calculated(transformer, sample_df):
    result = transformer._apply_feature_engineering(sample_df)
    assert 'Ram_per_core' in result.columns
    assert result['Ram_per_core'].iloc[0] == 8192/4

def test_feature_engineering_handles_missing_columns_gracefully(transformer):
    # should not crash on a minimal df missing most columns
    df = pd.DataFrame({'some_col': [1, 2, 3]})
    result = transformer._apply_feature_engineering(df)
    assert len(result) == 3