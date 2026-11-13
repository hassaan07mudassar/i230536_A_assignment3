"""Tests for FastAPI Agent Endpoints (/health, /tools, /chat).

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import pytest


def test_agent_health(agent_client):
    resp = agent_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mlops-ai-agent"
    assert data["registered_tools_count"] >= 20


def test_agent_list_tools(agent_client):
    resp = agent_client.get("/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert data["total"] >= 20
    # Verify tool metadata
    tool_names = [t["name"] for t in data["tools"]]
    assert "dvc_get_dataset_version" in tool_names
    assert "mlflow_best_run" in tool_names
    assert "airflow_trigger_dag" in tool_names
    assert "k8s_restart_deployment" in tool_names


def test_agent_chat_endpoint(agent_client):
    payload = {
        "message": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
        "confirm": False,
    }
    resp = agent_client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "KNOWN FACTS" in data["answer"]
    assert "UNCERTAINTIES" in data["answer"]
    assert "RECOMMENDED ACTIONS" in data["answer"]
    assert len(data["tool_trace"]) >= 4

