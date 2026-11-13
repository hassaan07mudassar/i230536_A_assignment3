"""Apache Airflow 3 Workflow Management Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional
import requests

from agent.config import settings
from agent.logging_config import logger, redact_secrets
from agent.permissions import validate_airflow_trigger
from agent.schemas import (
    AirflowGetDagRunsInput,
    AirflowGetFailedTasksInput,
    AirflowGetTaskInstancesInput,
    AirflowGetTaskLogsInput,
    AirflowListDagsInput,
    AirflowTriggerDagInput,
    ToolError,
    ToolEvidence,
    ToolResult,
)


class AirflowClient:
    """Authenticated client for Airflow 3 REST API (/api/v2) with token caching and fallback."""

    def __init__(self):
        self.base_url = settings.airflow_base_url.rstrip("/")
        self.username = settings.airflow_username
        self.password = settings.airflow_password
        self.timeout = settings.airflow_api_timeout
        self._jwt_token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _get_auth_headers(self) -> Dict[str, str]:
        # Try JWT token retrieval from /auth/token (Airflow 3)
        if self._jwt_token and time.time() < self._token_expiry:
            return {"Authorization": f"Bearer {self._jwt_token}"}

        try:
            resp = requests.post(
                f"{self.base_url}/auth/token",
                json={"username": self.username, "password": self.password},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._jwt_token = data.get("access_token") or data.get("token")
                self._token_expiry = time.time() + 3000
                return {"Authorization": f"Bearer {self._jwt_token}"}
        except Exception:
            pass

        # Fallback to HTTP Basic Auth
        return {}

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        auth_headers = self._get_auth_headers()
        headers.update(auth_headers)

        # If no JWT, use basic auth
        auth = (self.username, self.password) if "Authorization" not in headers else None
        return requests.request(
            method=method,
            url=url,
            headers=headers,
            auth=auth,
            timeout=self.timeout,
            **kwargs,
        )


_airflow_client = AirflowClient()


# ------------------------------------------------------------------------------
# Tool 1: airflow_list_dags
# ------------------------------------------------------------------------------
def airflow_list_dags(input_data: AirflowListDagsInput) -> ToolResult:
    """Lists all DAGs registered in Apache Airflow."""
    try:
        # Airflow 3 uses /api/v2/dags; fallback to /api/v1/dags
        resp = _airflow_client.request("GET", f"/api/v2/dags?limit={input_data.limit}")
        if resp.status_code == 404:
            resp = _airflow_client.request("GET", f"/api/v1/dags?limit={input_data.limit}")

        if resp.status_code != 200:
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=f"Airflow API error: {resp.text[:200]}"),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        dags_data = resp.json().get("dags", [])
        dag_summaries = [
            {
                "dag_id": d.get("dag_id"),
                "is_paused": d.get("is_paused"),
                "is_active": d.get("is_active"),
                "description": d.get("description"),
                "tags": [t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])],
            }
            for d in dags_data
        ]

        evidence = ToolEvidence(
            source=f"Airflow REST API ({settings.airflow_base_url})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Retrieved {len(dag_summaries)} DAGs.",
        )
        return ToolResult(ok=True, data={"dags": dag_summaries, "count": len(dag_summaries)}, evidence=evidence)
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Unable to connect to Airflow at {settings.airflow_base_url}. Service may be offline."),
            evidence=ToolEvidence(source="airflow_list_dags", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CLIENT_ERROR", message=f"Request failed: {str(e)}"),
            evidence=ToolEvidence(source="airflow_list_dags", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 2: airflow_get_dag_runs
# ------------------------------------------------------------------------------
def airflow_get_dag_runs(input_data: AirflowGetDagRunsInput) -> ToolResult:
    """Fetches recent run statuses for a specific DAG."""
    dag_id = input_data.dag_id
    try:
        resp = _airflow_client.request("GET", f"/api/v2/dags/{dag_id}/dagRuns?limit={input_data.limit}&order_by=-execution_date")
        if resp.status_code == 404:
            resp = _airflow_client.request("GET", f"/api/v1/dags/{dag_id}/dagRuns?limit={input_data.limit}&order_by=-execution_date")

        if resp.status_code != 200:
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=f"Airflow returned {resp.status_code}: {resp.text[:200]}"),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        runs = resp.json().get("dag_runs", [])
        run_summaries = [
            {
                "dag_run_id": r.get("dag_run_id"),
                "state": r.get("state"),
                "logical_date": r.get("logical_date") or r.get("execution_date"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "conf": r.get("conf"),
            }
            for r in runs
        ]

        evidence = ToolEvidence(
            source=f"Airflow REST API (/dags/{dag_id}/dagRuns)",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Found {len(run_summaries)} runs for DAG '{dag_id}'.",
        )
        return ToolResult(ok=True, data={"dag_id": dag_id, "runs": run_summaries, "total": len(run_summaries)}, evidence=evidence)
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Cannot reach Airflow at {settings.airflow_base_url}"),
            evidence=ToolEvidence(source="airflow_get_dag_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_QUERY_ERROR", message=str(e)),
            evidence=ToolEvidence(source="airflow_get_dag_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 3: airflow_trigger_dag
# ------------------------------------------------------------------------------
def airflow_trigger_dag(input_data: AirflowTriggerDagInput) -> ToolResult:
    """Triggers an Airflow DAG run. MUTATING ACTION: Enforces confirmation & DAG allowlist."""
    # 1. Non-negotiable security guard
    perm_err = validate_airflow_trigger(dag_id=input_data.dag_id, confirm=input_data.confirm)
    if perm_err:
        return ToolResult(
            ok=False,
            error=perm_err,
            evidence=ToolEvidence(source="security_policy_enforcer", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    dag_id = input_data.dag_id
    payload = {
        "conf": input_data.conf or {},
        "logical_date": datetime.utcnow().isoformat() + "Z",
    }

    try:
        resp = _airflow_client.request("POST", f"/api/v2/dags/{dag_id}/dagRuns", json=payload)
        if resp.status_code == 404:
            resp = _airflow_client.request("POST", f"/api/v1/dags/{dag_id}/dagRuns", json=payload)

        if resp.status_code not in (200, 201):
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=f"Trigger failed: {resp.text[:200]}"),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        data = resp.json()
        evidence = ToolEvidence(
            source=f"Airflow REST POST /dags/{dag_id}/dagRuns",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Triggered run: {data.get('dag_run_id')} (State: {data.get('state')})",
        )
        return ToolResult(
            ok=True,
            data={
                "dag_id": dag_id,
                "dag_run_id": data.get("dag_run_id"),
                "state": data.get("state"),
                "logical_date": data.get("logical_date"),
                "message": f"Successfully triggered DAG '{dag_id}'.",
            },
            evidence=evidence,
        )
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Airflow offline at {settings.airflow_base_url}"),
            evidence=ToolEvidence(source="airflow_trigger_dag", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="TRIGGER_ERROR", message=str(e)),
            evidence=ToolEvidence(source="airflow_trigger_dag", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 4: airflow_get_failed_tasks
# ------------------------------------------------------------------------------
def airflow_get_failed_tasks(input_data: AirflowGetFailedTasksInput) -> ToolResult:
    """Identifies failed tasks in a DAG run for automated incident diagnosis."""
    dag_id = input_data.dag_id
    run_id = input_data.dag_run_id

    # If run_id not provided, find the latest failed or active run
    if not run_id:
        runs_res = airflow_get_dag_runs(AirflowGetDagRunsInput(dag_id=dag_id, limit=5))
        if not runs_res.ok or not runs_res.data.get("runs"):
            return ToolResult(
                ok=False,
                error=ToolError(code="NO_RUNS_FOUND", message=f"No runs found for DAG '{dag_id}'"),
                evidence=ToolEvidence(source="airflow_get_failed_tasks", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
        run_id = runs_res.data["runs"][0]["dag_run_id"]

    try:
        resp = _airflow_client.request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        if resp.status_code == 404:
            resp = _airflow_client.request("GET", f"/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances")

        if resp.status_code != 200:
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=resp.text[:200]),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        tasks = resp.json().get("task_instances", [])
        failed_tasks = [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "try_number": t.get("try_number"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
                "duration": t.get("duration"),
            }
            for t in tasks
            if t.get("state") in ("failed", "upstream_failed")
        ]

        evidence = ToolEvidence(
            source=f"Airflow Task Instances (dag={dag_id}, run={run_id})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Found {len(failed_tasks)} failed task(s).",
        )
        return ToolResult(
            ok=True,
            data={"dag_id": dag_id, "dag_run_id": run_id, "failed_tasks": failed_tasks, "count": len(failed_tasks)},
            evidence=evidence,
        )
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Airflow offline at {settings.airflow_base_url}"),
            evidence=ToolEvidence(source="airflow_get_failed_tasks", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="QUERY_ERROR", message=str(e)),
            evidence=ToolEvidence(source="airflow_get_failed_tasks", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 5: airflow_get_task_instances
# ------------------------------------------------------------------------------
def airflow_get_task_instances(input_data: AirflowGetTaskInstancesInput) -> ToolResult:
    """Lists all task instances and their statuses for a DAG run."""
    dag_id = input_data.dag_id
    run_id = input_data.dag_run_id
    try:
        resp = _airflow_client.request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        if resp.status_code == 404:
            resp = _airflow_client.request("GET", f"/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances")

        if resp.status_code != 200:
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=resp.text[:200]),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        tasks = resp.json().get("task_instances", [])
        task_list = [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "try_number": t.get("try_number"),
                "duration": t.get("duration"),
            }
            for t in tasks
        ]

        evidence = ToolEvidence(
            source=f"Airflow Task Instances (dag={dag_id}, run={run_id})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Retrieved {len(task_list)} task instances.",
        )
        return ToolResult(ok=True, data={"dag_id": dag_id, "dag_run_id": run_id, "tasks": task_list}, evidence=evidence)
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Airflow offline at {settings.airflow_base_url}"),
            evidence=ToolEvidence(source="airflow_get_task_instances", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="QUERY_ERROR", message=str(e)),
            evidence=ToolEvidence(source="airflow_get_task_instances", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 6: airflow_get_task_logs
# ------------------------------------------------------------------------------
def airflow_get_task_logs(input_data: AirflowGetTaskLogsInput) -> ToolResult:
    """Fetches task execution logs with automatic secret redaction."""
    dag_id = input_data.dag_id
    run_id = input_data.dag_run_id
    task_id = input_data.task_id
    try_num = input_data.try_number

    try:
        path = f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_num}"
        resp = _airflow_client.request("GET", path)
        if resp.status_code == 404:
            path = f"/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_num}"
            resp = _airflow_client.request("GET", path)

        if resp.status_code != 200:
            return ToolResult(
                ok=False,
                error=ToolError(code=f"AIRFLOW_HTTP_{resp.status_code}", message=resp.text[:200]),
                evidence=ToolEvidence(source="Airflow REST API", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        raw_logs = resp.text
        # Security: redact secrets
        clean_logs = redact_secrets(raw_logs)

        evidence = ToolEvidence(
            source=f"Airflow task log ({dag_id}.{task_id}, try {try_num})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=clean_logs[-300:] if len(clean_logs) > 300 else clean_logs,
        )
        return ToolResult(
            ok=True,
            data={"dag_id": dag_id, "task_id": task_id, "try_number": try_num, "logs": clean_logs},
            evidence=evidence,
        )
    except requests.exceptions.ConnectionError:
        return ToolResult(
            ok=False,
            error=ToolError(code="AIRFLOW_CONNECTION_REFUSED", message=f"Airflow offline at {settings.airflow_base_url}"),
            evidence=ToolEvidence(source="airflow_get_task_logs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="LOG_FETCH_ERROR", message=str(e)),
            evidence=ToolEvidence(source="airflow_get_task_logs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

