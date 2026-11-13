"""Tests for Apache Airflow 3 Tools and REST Client.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from agent.schemas import (
    AirflowGetDagRunsInput,
    AirflowGetFailedTasksInput,
    AirflowGetTaskInstancesInput,
    AirflowGetTaskLogsInput,
    AirflowListDagsInput,
    AirflowTriggerDagInput,
)
from agent.tools.airflow_tools import (
    airflow_get_dag_runs,
    airflow_get_failed_tasks,
    airflow_get_task_instances,
    airflow_get_task_logs,
    airflow_list_dags,
    airflow_trigger_dag,
)


def test_airflow_list_dags_success():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dags": [
                {"dag_id": "ml_training_pipeline", "is_paused": False, "is_active": True, "description": "Training DAG"}
            ]
        }
        mock_req.return_value = mock_resp

        res = airflow_list_dags(AirflowListDagsInput(limit=10))
        assert res.ok is True
        assert res.data["count"] == 1
        assert res.data["dags"][0]["dag_id"] == "ml_training_pipeline"


def test_airflow_get_dag_runs():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dag_runs": [
                {"dag_run_id": "run-1", "state": "success", "logical_date": "2026-11-13T10:00:00Z"}
            ]
        }
        mock_req.return_value = mock_resp

        res = airflow_get_dag_runs(AirflowGetDagRunsInput(dag_id="ml_training_pipeline"))
        assert res.ok is True
        assert res.data["total"] == 1
        assert res.data["runs"][0]["state"] == "success"


def test_airflow_trigger_unconfirmed_rejected():
    inp = AirflowTriggerDagInput(dag_id="ml_training_pipeline", confirm=False)
    res = airflow_trigger_dag(inp)
    assert res.ok is False
    assert res.error.code == "CONFIRMATION_REQUIRED"


def test_airflow_trigger_unauthorized_dag_rejected():
    inp = AirflowTriggerDagInput(dag_id="unauthorized_malicious_dag", confirm=True)
    res = airflow_trigger_dag(inp)
    assert res.ok is False
    assert res.error.code == "DAG_NOT_ALLOWED"


def test_airflow_trigger_confirmed_success():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "dag_run_id": "manual__2026-11-13T12:00:00Z",
            "state": "queued",
            "logical_date": "2026-11-13T12:00:00Z",
        }
        mock_req.return_value = mock_resp

        inp = AirflowTriggerDagInput(dag_id="ml_training_pipeline", confirm=True)
        res = airflow_trigger_dag(inp)
        assert res.ok is True
        assert res.data["dag_run_id"] == "manual__2026-11-13T12:00:00Z"
        assert res.data["state"] == "queued"


def test_airflow_get_failed_tasks():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "task_instances": [
                {"task_id": "validate_dataset", "state": "success", "try_number": 1},
                {"task_id": "apply_quality_gate", "state": "failed", "try_number": 2},
            ]
        }
        mock_req.return_value = mock_resp

        inp = AirflowGetFailedTasksInput(dag_id="ml_training_pipeline", dag_run_id="run-101")
        res = airflow_get_failed_tasks(inp)
        assert res.ok is True
        assert res.data["count"] == 1
        assert res.data["failed_tasks"][0]["task_id"] == "apply_quality_gate"


def test_airflow_get_task_instances():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "task_instances": [
                {"task_id": "validate_dataset", "state": "success", "try_number": 1, "duration": 2.1},
                {"task_id": "train_model", "state": "success", "try_number": 1, "duration": 5.4},
            ]
        }
        mock_req.return_value = mock_resp

        inp = AirflowGetTaskInstancesInput(dag_id="ml_training_pipeline", dag_run_id="run-101")
        res = airflow_get_task_instances(inp)
        assert res.ok is True
        assert len(res.data["tasks"]) == 2


def test_airflow_get_task_logs():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Executing task validate_dataset with API token secret_token_value_abc"
        mock_req.return_value = mock_resp

        inp = AirflowGetTaskLogsInput(dag_id="ml_training_pipeline", dag_run_id="run-101", task_id="validate_dataset")
        res = airflow_get_task_logs(inp)
        assert res.ok is True
        assert "validate_dataset" in res.data["logs"]
        assert "secret_token_value_abc" not in res.data["logs"]
        assert "[REDACTED_SECRET]" in res.data["logs"]


def test_airflow_service_failure_handled():
    with patch("agent.tools.airflow_tools._airflow_client.request") as mock_req:
        mock_req.side_effect = requests.exceptions.ConnectionError("Connection refused")
        res = airflow_list_dags(AirflowListDagsInput(limit=5))
        assert res.ok is False
        assert res.error.code == "AIRFLOW_CONNECTION_REFUSED"
