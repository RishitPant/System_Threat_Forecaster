import pandas as pd
import pytest
from src.components.data_cleaning import DataCleaning


@pytest.fixture
def cleaner():
    return DataCleaning()


def test_drop_duplicates_removes_duplicate_rows(cleaner):
    df = pd.DataFrame({'a': [1,1,2], 'b': [3,3,4]})
    result = cleaner._drop_duplicates(df)
    assert len(result) == 2

def test_drop_duplicates_keeps_unique_rows(cleaner):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    result = cleaner._drop_duplicates(df)
    assert len(result) == 3

def test_drop_unnecessary_columns_removes_known_cols(cleaner):
    df = pd.DataFrame({'MachineID': [1], 'SMode': [0], 'keep_me': [99]})
    result = cleaner._drop_unnecessary_columns(df)
    assert 'MachineID' not in result.columns
    assert 'SMode' not in result.columns
    assert 'keep_me' in result.columns

def test_drop_unnecessary_columns_ignores_missing_cols(cleaner):
    df = pd.DataFrame({'keep_me': [1, 2]})
    result = cleaner._drop_unnecessary_columns(df)
    assert 'keep_me' in result.columns

def test_handle_missing_values_drops_null_target_rows(cleaner):
    df = pd.DataFrame({'a': [1, 2, 3], 'target': [0, None, 1]})
    result = cleaner._handle_missing_values(df)
    assert len(result) == 2
    assert result['target'].isna().sum() == 0

def test_handle_missing_values_ignores_df_without_target(cleaner):
    df = pd.DataFrame({'a': [1, 2], 'b': [None, 4]})
    result = cleaner._handle_missing_values(df)
    assert len(result) == 2