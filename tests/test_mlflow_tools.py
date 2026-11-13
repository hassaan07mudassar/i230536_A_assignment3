"""Tests for MLflow Experiment Intelligence Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from unittest.mock import patch

from agent.schemas import (
    MlflowBestRunInput,
    MlflowCompareExperimentsInput,
    MlflowCompareRunsInput,
    MlflowGetRegistryInfoInput,
    MlflowGetRunDetailsInput,
    MlflowListExperimentsInput,
    MlflowListRunsInput,
)
from agent.tools.mlflow_tools import (
    mlflow_best_run,
    mlflow_compare_experiments,
    mlflow_compare_runs,
    mlflow_get_registry_info,
    mlflow_get_run_details,
    mlflow_list_experiments,
    mlflow_list_runs,
)


def test_mlflow_list_experiments():
    inp = MlflowListExperimentsInput()
    result = mlflow_list_experiments(inp)
    assert result.ok is True
    assert result.data["count"] >= 1
    names = [e["name"] for e in result.data["experiments"]]
    assert "customer_churn_prediction" in names


def test_mlflow_list_runs():
    inp = MlflowListRunsInput(experiment_name="customer_churn_prediction", max_results=20)
    result = mlflow_list_runs(inp)
    assert result.ok is True
    assert result.data["total"] >= 10
    first_run = result.data["runs"][0]
    assert "metrics" in first_run
    assert "params" in first_run
    assert "dataset_version" in first_run


def test_mlflow_best_run():
    inp = MlflowBestRunInput(experiment_name="customer_churn_prediction", metric="f1_score", mode="max")
    result = mlflow_best_run(inp)
    assert result.ok is True
    assert "best_run_id" in result.data
    assert "defensive_audit" in result.data
    assert "overfitting_gap" in result.data["defensive_audit"]
    assert "reasoning" in result.data["defensive_audit"]


def test_mlflow_get_run_details():
    list_res = mlflow_list_runs(MlflowListRunsInput(experiment_name="customer_churn_prediction", max_results=1))
    run_id = list_res.data["runs"][0]["run_id"]

    inp = MlflowGetRunDetailsInput(run_id=run_id)
    result = mlflow_get_run_details(inp)
    assert result.ok is True
    assert result.data["run_id"] == run_id
    assert "artifacts" in result.data
    assert "provenance" in result.data


def test_mlflow_compare_runs():
    inp = MlflowCompareRunsInput(experiment_name="customer_churn_prediction", last_n=3)
    result = mlflow_compare_runs(inp)
    assert result.ok is True
    assert len(result.data["comparison_table"]) == 3
    for entry in result.data["comparison_table"]:
        assert "f1_score" in entry
        assert "accuracy" in entry
        assert "dataset_version" in entry


def test_mlflow_get_registry_info():
    inp = MlflowGetRegistryInfoInput(model_name="customer_churn_classifier")
    result = mlflow_get_registry_info(inp)
    assert result.ok is True
    assert result.data["model_name"] == "customer_churn_classifier"
    assert "versions" in result.data
    assert result.data["total_versions"] >= 1


def test_mlflow_connection_failure_handled():
    with patch("agent.tools.mlflow_tools._get_client") as mock_client:
        from agent.schemas import ToolError
        mock_client.return_value = (None, ToolError(code="MLFLOW_UNAVAILABLE", message="Connection refused"))
        res = mlflow_list_experiments(MlflowListExperimentsInput())
        assert res.ok is False
        assert res.error.code == "MLFLOW_UNAVAILABLE"

def test_mlflow_compare_experiments():
    res = mlflow_compare_experiments(MlflowCompareExperimentsInput(experiment_names=["customer_churn_prediction"]))
    assert res.ok is True
    assert len(res.data["experiments"]) >= 1
    assert res.data["experiments"][0]["experiment_name"] == "customer_churn_prediction"
