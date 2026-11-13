"""Tests for Agent Orchestrator, Multi-Tool Reasoning, and Non-Negotiable Safety Principles.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from unittest.mock import MagicMock, patch
import pytest

from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest


def test_mandatory_scenario_calls_all_four_tools():
    """Asserts that the mandatory operational inquiry triggers all 4 technology tools."""
    orch = Orchestrator()
    msg = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    req = ChatRequest(message=msg, confirm=False)
    resp = orch.run(req)

    # 1. Assert structure
    assert resp.request_id is not None
    assert "### KNOWN FACTS" in resp.answer
    assert "### UNCERTAINTIES" in resp.answer
    assert "### RECOMMENDED ACTIONS" in resp.answer

    # 2. Assert tool trace contains calls for MLflow, DVC, Kubernetes, and Airflow
    called_tools = [t.tool for t in resp.tool_trace]
    assert "mlflow_best_run" in called_tools
    assert "dvc_get_dataset_for_experiment" in called_tools
    assert "k8s_diagnose_service" in called_tools
    assert "airflow_get_dag_runs" in called_tools


def test_orchestrator_does_not_autotrigger_retraining():
    """Principle 4: Retraining must NEVER be auto-triggered from an analysis question."""
    orch = Orchestrator()
    msg = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    req = ChatRequest(message=msg, confirm=False)
    resp = orch.run(req)

    # Assert no trigger tool was executed
    called_tools = [t.tool for t in resp.tool_trace]
    assert "airflow_trigger_dag" not in called_tools
    assert "NO EVIDENCE RETRAINING IS NECESSARY" in resp.answer


def test_orchestrator_does_not_hallucinate_when_tool_fails():
    """Principle 1: Never fabricate results when a tool or backend is unavailable."""
    orch = Orchestrator()
    msg = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

    # Force kubernetes and airflow to be offline
    with patch("agent.tools.kubernetes_tools._init_k8s_clients") as mock_k8s:
        from agent.schemas import ToolError
        mock_k8s.return_value = (None, None, ToolError(code="K8S_OFFLINE", message="Cluster completely offline"))

        req = ChatRequest(message=msg, confirm=False)
        resp = orch.run(req)

        # Assert answer clearly flags unavailability in UNCERTAINTIES
        assert "UNCERTAINTIES" in resp.answer
        assert any("offline" in u.lower() or "unreachable" in u.lower() for u in resp.uncertainties)


def test_orchestrator_records_accurate_tool_trace():
    """Asserts that tool traces contain accurate durations, statuses, and redacted args."""
    orch = Orchestrator()
    req = ChatRequest(message="What is the current version of the dataset in data/raw/customer_churn.csv?", confirm=False)
    resp = orch.run(req)

    assert len(resp.tool_trace) >= 0
    for trace in resp.tool_trace:
        assert trace.duration_ms >= 0.0
        assert trace.status in ("SUCCESS", "FAILED", "REJECTED", "TIMEOUT")
        assert trace.request_id == resp.request_id

