import pandas as pd
import numpy as np
import pytest
import os
from src.pipeline.predict_pipeline import PredictPipeline

pytestmark = pytest.mark.skipif(
    not os.path.exists("artifacts/model.pkl") or 
    not os.path.exists("notebook/data/test_eda_clean.csv"),
    reason="Trained artifacts or data not found — run train_pipeline.py first"
)

@pytest.fixture
def pipeline():
    return PredictPipeline()

@pytest.fixture
def sample_input():
    # Minimal row matching your test CSV schema
    return pd.read_csv("notebook/data/test_eda_clean.csv").head(5)

def test_predict_returns_two_outputs(pipeline, sample_input):
    preds, proba = pipeline.predict(sample_input)
    assert preds is not None
    assert proba is not None

def test_predict_output_length_matches_input(pipeline, sample_input):
    preds, proba = pipeline.predict(sample_input)
    assert len(preds) == len(sample_input)
    assert len(proba) == len(sample_input)

def test_predictions_are_binary(pipeline, sample_input):
    preds, _ = pipeline.predict(sample_input)
    assert set(preds).issubset({0, 1})

def test_probabilities_are_between_0_and_1(pipeline, sample_input):
    _, proba = pipeline.predict(sample_input)
    assert (proba >= 0).all() and (proba <= 1).all()