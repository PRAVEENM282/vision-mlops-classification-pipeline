import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.config import settings
import torch
import os

client = TestClient(app)

def test_config():
    assert settings.API_PORT is not None
    assert settings.MODEL_PATH is not None

def test_health_check_no_model():
    # If model is not trained, health check might be 503 or 200 depending on implementation details.
    # We mocked it to be 503 if not ready.
    # However, running in CI/Test without a trained model file means 503.
    # We'll check if file exists to assertion can be dynamic.
    response = client.get("/health")
    if os.path.exists(settings.MODEL_PATH):
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    else:
        assert response.status_code == 503

def test_config_defaults():
    assert settings.DATASET_NAME == "Caltech101"

# We skip real training tests here to avoid long runtime during 'pytest', 
# but we can verify imports and simple function calls.

def test_imports():
    from src import preprocess, train, evaluate
    assert preprocess is not None
    assert train is not None
    assert evaluate is not None

