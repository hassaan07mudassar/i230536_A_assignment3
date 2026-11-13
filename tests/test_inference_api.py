"""Tests for FastAPI Inference Microservice (/health, /ready, /predict, /version, /metrics).

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, load_model


def test_inference_health(inference_client):
    resp = inference_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "customer-churn-inference"


def test_inference_ready_loaded(inference_client):
    load_model()
    resp = inference_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"


def test_inference_version(inference_client):
    resp = inference_client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_version"] == "1.0.0"
    assert data["service"] == "customer-churn-inference"


def test_inference_predict(inference_client):
    load_model()
    payload = {
        "tenure_months": 24,
        "contract_type": 1,
        "monthly_charges": 65.5,
        "total_charges": 1500.0,
        "tech_support": 1,
        "online_security": 1,
        "payment_method": 2,
    }
    resp = inference_client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_inference_metrics(inference_client):
    resp = inference_client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "uptime_seconds" in data
    assert "total_predictions" in data

