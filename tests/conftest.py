"""Pytest Test Configuration and Mock Fixtures.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from agent.llm_client import MockLLMClient
from agent.orchestrator import Orchestrator
from agent.main import app as agent_app
from api.main import app as inference_app


@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch):
    """Enforces offline test configuration with zero secrets and local mocks."""
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://mock-airflow:8080")
    monkeypatch.setenv("ALLOWED_NAMESPACES", "mlops-agent,default")
    monkeypatch.setenv("ALLOWED_DAG_IDS", "ml_training_pipeline")
    monkeypatch.setenv("ALLOWED_K8S_ACTIONS", "restart,scale")


@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def agent_client():
    return TestClient(agent_app)


@pytest.fixture
def inference_client():
    return TestClient(inference_app)


@pytest.fixture
def mock_k8s_apis():
    """Provides mocked Kubernetes CoreV1Api and AppsV1Api."""
    with patch("agent.tools.kubernetes_tools._init_k8s_clients") as mock_init:
        mock_core = MagicMock()
        mock_apps = MagicMock()
        mock_init.return_value = (mock_core, mock_apps, None)
        yield mock_core, mock_apps
