"""Structured Pydantic Schemas for Tools, Orchestrator, and API.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Common Tool Result & Evidence Model
# ==============================================================================

class ToolEvidence(BaseModel):
    source: str = Field(..., description="Tool or backend providing the evidence (e.g. 'dvc CLI', 'MLflow Client')")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    raw_output: Optional[str] = Field(None, description="Verbatim raw command/API response snippet")


class ToolError(BaseModel):
    code: str = Field(..., description="Standardized error code (e.g. PERMISSION_DENIED, TIMEOUT, NOT_FOUND)")
    message: str = Field(..., description="Human-readable explanation of error")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool = Field(..., description="True if operation succeeded, False otherwise")
    data: Optional[Any] = Field(None, description="Structured payload returned by the tool")
    error: Optional[ToolError] = Field(None, description="Error structure if operation failed")
    evidence: ToolEvidence = Field(..., description="Provenance and evidence metadata")


# ==============================================================================
# DVC Tool Schemas (Task 1)
# ==============================================================================

class DvcListTrackedDatasetsInput(BaseModel):
    repo_path: str = Field(".", description="Path to DVC repository root")


class DvcGetDatasetVersionInput(BaseModel):
    dataset_path: str = Field("data/raw/customer_churn.csv", description="Relative path of dataset")
    repo_path: str = Field(".", description="Path to DVC repository root")


class DvcCompareVersionsInput(BaseModel):
    rev1: str = Field("v1.0", description="Base git revision or tag")
    rev2: str = Field("v2.0", description="Target git revision or tag")
    repo_path: str = Field(".", description="Path to DVC repository root")


class DvcDetectChangesInput(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Specific dataset or None for all")
    repo_path: str = Field(".", description="Path to DVC repository root")


class DvcExplainReproduceInput(BaseModel):
    revision: str = Field("v1.0", description="Git revision/tag to reproduce")
    dataset_path: str = Field("data/raw/customer_churn.csv", description="Dataset path")


class DvcGetDatasetForExperimentInput(BaseModel):
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    experiment_id: Optional[str] = Field(None, description="MLflow experiment ID")


# ==============================================================================
# MLflow Tool Schemas (Task 2)
# ==============================================================================

class MlflowListExperimentsInput(BaseModel):
    view_type: Optional[str] = Field("ACTIVE_ONLY", description="ACTIVE_ONLY, DELETED_ONLY, ALL")


class MlflowListRunsInput(BaseModel):
    experiment_name: str = Field("customer_churn_prediction", description="Experiment name")
    max_results: int = Field(20, ge=1, le=100)
    order_by: Optional[str] = Field("metrics.f1_score DESC", description="Ordering metric or timestamp")
    filter_string: Optional[str] = Field(None, description="SQL-like filter string (e.g. params.model_type = 'RandomForestClassifier')")


class MlflowGetRunDetailsInput(BaseModel):
    run_id: str = Field(..., description="Target MLflow run ID")


class MlflowBestRunInput(BaseModel):
    experiment_name: str = Field("customer_churn_prediction", description="Experiment name")
    metric: str = Field("f1_score", description="Metric name to optimize (f1_score, accuracy, recall)")
    mode: Literal["max", "min"] = Field("max", description="Maximize or minimize metric")


class MlflowCompareRunsInput(BaseModel):
    run_ids: Optional[List[str]] = Field(None, description="List of 2-5 run IDs to compare")
    last_n: Optional[int] = Field(3, description="If run_ids not given, compare latest N runs")
    experiment_name: str = Field("customer_churn_prediction", description="Experiment name")


class MlflowCompareExperimentsInput(BaseModel):
    experiment_names: List[str] = Field(..., description="List of experiment names to compare")


class MlflowGetRegistryInfoInput(BaseModel):
    model_name: str = Field("customer_churn_classifier", description="Registered model name")


# ==============================================================================
# Airflow 3 Tool Schemas (Task 3)
# ==============================================================================

class AirflowListDagsInput(BaseModel):
    limit: int = Field(25, ge=1, le=100)


class AirflowGetDagRunsInput(BaseModel):
    dag_id: str = Field("ml_training_pipeline", description="Target DAG ID")
    limit: int = Field(5, ge=1, le=25)


class AirflowTriggerDagInput(BaseModel):
    dag_id: str = Field("ml_training_pipeline", description="Target DAG ID")
    conf: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration parameters for the DAG run")
    confirm: bool = Field(False, description="Explicit confirmation flag required for mutating action")


class AirflowGetFailedTasksInput(BaseModel):
    dag_id: str = Field("ml_training_pipeline", description="Target DAG ID")
    dag_run_id: Optional[str] = Field(None, description="Specific DAG run ID or None for latest failed run")


class AirflowGetTaskInstancesInput(BaseModel):
    dag_id: str = Field("ml_training_pipeline", description="Target DAG ID")
    dag_run_id: str = Field(..., description="DAG run ID")


class AirflowGetTaskLogsInput(BaseModel):
    dag_id: str = Field("ml_training_pipeline", description="Target DAG ID")
    dag_run_id: str = Field(..., description="DAG run ID")
    task_id: str = Field(..., description="Task ID (e.g. apply_quality_gate)")
    try_number: int = Field(1, ge=1)


# ==============================================================================
# Kubernetes Tool Schemas (Task 4)
# ==============================================================================

class K8sListPodsInput(BaseModel):
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sListDeploymentsInput(BaseModel):
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sGetPodStatusInput(BaseModel):
    pod_name: str = Field(..., description="Pod name")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sGetPodLogsInput(BaseModel):
    pod_name: str = Field(..., description="Pod name")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")
    tail_lines: int = Field(50, ge=1, le=500)
    previous: bool = Field(False, description="Fetch previous terminated container logs if crashed")


class K8sListServicesInput(BaseModel):
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sGetEventsInput(BaseModel):
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")
    limit: int = Field(20, ge=1, le=100)


class K8sFindUnhealthyPodsInput(BaseModel):
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sGetDeployedVersionInput(BaseModel):
    deployment_name: str = Field("churn-inference-deployment", description="Deployment name")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sDiagnoseServiceInput(BaseModel):
    deployment_name: str = Field("churn-inference-deployment", description="Deployment name")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")


class K8sRestartDeploymentInput(BaseModel):
    deployment_name: str = Field("churn-inference-deployment", description="Deployment name")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")
    confirm: bool = Field(False, description="Explicit confirmation flag required for mutating action")


class K8sScaleDeploymentInput(BaseModel):
    deployment_name: str = Field("churn-inference-deployment", description="Deployment name")
    replicas: int = Field(..., ge=0, le=10, description="Desired replica count")
    namespace: str = Field("mlops-agent", description="Kubernetes namespace")
    confirm: bool = Field(False, description="Explicit confirmation flag required for mutating action")


# ==============================================================================
# Agent API & Multi-Tool Orchestrator Schemas (Task 5)
# ==============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or command for the MLOps agent")
    request_id: Optional[str] = Field(None, description="Correlation / request ID for distributed tracing")
    confirm: bool = Field(False, description="Explicit confirmation flag for mutating actions")


class ToolTraceItem(BaseModel):
    tool: str
    args: Dict[str, Any]
    status: Literal["SUCCESS", "FAILED", "REJECTED", "TIMEOUT"]
    duration_ms: float
    request_id: str
    error_code: Optional[str] = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    known_facts: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    tool_trace: List[ToolTraceItem] = Field(default_factory=list)

