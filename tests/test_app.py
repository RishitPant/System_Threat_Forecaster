import io
import pandas as pd
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200

def test_predict_with_no_file_returns_error(client):
    response = client.post("/predict", data={})
    # should redirect back or return an error, not crash with 500
    assert response.status_code in (200, 302, 400)

def test_predict_with_wrong_file_type_returns_error(client):
    data = {'file': (io.BytesIO(b"not a csv"), 'test.txt')}
    response = client.post("/predict", data=data, content_type='multipart/form-data')
    assert response.status_code in (200, 302, 400)