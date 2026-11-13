"""Central Tool Catalog and Dispatcher.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from typing import Any, Dict

# DVC Tools
from agent.tools.dvc_tools import (
    dvc_compare_versions,
    dvc_detect_changes,
    dvc_explain_reproduce,
    dvc_get_dataset_for_experiment,
    dvc_get_dataset_version,
    dvc_list_tracked_datasets,
)
from agent.schemas import (
    DvcCompareVersionsInput,
    DvcDetectChangesInput,
    DvcExplainReproduceInput,
    DvcGetDatasetForExperimentInput,
    DvcGetDatasetVersionInput,
    DvcListTrackedDatasetsInput,
)

# MLflow Tools
from agent.tools.mlflow_tools import (
    mlflow_best_run,
    mlflow_compare_experiments,
    mlflow_compare_runs,
    mlflow_get_registry_info,
    mlflow_get_run_details,
    mlflow_list_experiments,
    mlflow_list_runs,
)
from agent.schemas import (
    MlflowBestRunInput,
    MlflowCompareExperimentsInput,
    MlflowCompareRunsInput,
    MlflowGetRegistryInfoInput,
    MlflowGetRunDetailsInput,
    MlflowListExperimentsInput,
    MlflowListRunsInput,
)

# Airflow Tools
from agent.tools.airflow_tools import (
    airflow_get_dag_runs,
    airflow_get_failed_tasks,
    airflow_get_task_instances,
    airflow_get_task_logs,
    airflow_list_dags,
    airflow_trigger_dag,
)
from agent.schemas import (
    AirflowGetDagRunsInput,
    AirflowGetFailedTasksInput,
    AirflowGetTaskInstancesInput,
    AirflowGetTaskLogsInput,
    AirflowListDagsInput,
    AirflowTriggerDagInput,
)

# Kubernetes Tools
from agent.tools.kubernetes_tools import (
    k8s_diagnose_service,
    k8s_find_unhealthy_pods,
    k8s_get_deployed_version,
    k8s_get_events,
    k8s_get_pod_logs,
    k8s_get_pod_status,
    k8s_list_deployments,
    k8s_list_pods,
    k8s_list_services,
    k8s_restart_deployment,
    k8s_scale_deployment,
)
from agent.schemas import (
    K8sDiagnoseServiceInput,
    K8sFindUnhealthyPodsInput,
    K8sGetDeployedVersionInput,
    K8sGetEventsInput,
    K8sGetPodLogsInput,
    K8sGetPodStatusInput,
    K8sListDeploymentsInput,
    K8sListPodsInput,
    K8sListServicesInput,
    K8sRestartDeploymentInput,
    K8sScaleDeploymentInput,
)

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # DVC
    "dvc_list_tracked_datasets": {
        "func": dvc_list_tracked_datasets,
        "schema": DvcListTrackedDatasetsInput,
        "domain": "DVC",
        "description": "Lists all datasets currently tracked by DVC (.dvc files).",
        "mutating": False,
    },
    "dvc_get_dataset_version": {
        "func": dvc_get_dataset_version,
        "schema": DvcGetDatasetVersionInput,
        "domain": "DVC",
        "description": "Gets current DVC hash, git commit, and active tag for a dataset.",
        "mutating": False,
    },
    "dvc_compare_versions": {
        "func": dvc_compare_versions,
        "schema": DvcCompareVersionsInput,
        "domain": "DVC",
        "description": "Compares dataset pointer files and hashes between two git revisions/tags.",
        "mutating": False,
    },
    "dvc_detect_changes": {
        "func": dvc_detect_changes,
        "schema": DvcDetectChangesInput,
        "domain": "DVC",
        "description": "Runs dvc status to check if working tree data has changed from cache.",
        "mutating": False,
    },
    "dvc_explain_reproduce": {
        "func": dvc_explain_reproduce,
        "schema": DvcExplainReproduceInput,
        "domain": "DVC",
        "description": "Explains exact git checkout and dvc checkout steps to reproduce a version.",
        "mutating": False,
    },
    "dvc_get_dataset_for_experiment": {
        "func": dvc_get_dataset_for_experiment,
        "schema": DvcGetDatasetForExperimentInput,
        "domain": "DVC",
        "description": "Retrieves authentic DVC dataset provenance recorded in MLflow run tags.",
        "mutating": False,
    },
    # MLflow
    "mlflow_list_experiments": {
        "func": mlflow_list_experiments,
        "schema": MlflowListExperimentsInput,
        "domain": "MLflow",
        "description": "Lists registered experiments from the MLflow tracking store.",
        "mutating": False,
    },
    "mlflow_list_runs": {
        "func": mlflow_list_runs,
        "schema": MlflowListRunsInput,
        "domain": "MLflow",
        "description": "Queries MLflow runs for an experiment with filters, ordering, and metrics.",
        "mutating": False,
    },
    "mlflow_get_run_details": {
        "func": mlflow_get_run_details,
        "schema": MlflowGetRunDetailsInput,
        "domain": "MLflow",
        "description": "Retrieves metrics, params, tags, artifacts, and provenance for a run ID.",
        "mutating": False,
    },
    "mlflow_best_run": {
        "func": mlflow_best_run,
        "schema": MlflowBestRunInput,
        "domain": "MLflow",
        "description": "Finds the best run by metric with defensive ML audit (overfitting & quality checks).",
        "mutating": False,
    },
    "mlflow_compare_runs": {
        "func": mlflow_compare_runs,
        "schema": MlflowCompareRunsInput,
        "domain": "MLflow",
        "description": "Compares multiple runs side-by-side across metrics, params, and dataset version.",
        "mutating": False,
    },
    "mlflow_compare_experiments": {
        "func": mlflow_compare_experiments,
        "schema": MlflowCompareExperimentsInput,
        "domain": "MLflow",
        "description": "Compares summary performance metrics across experiments.",
        "mutating": False,
    },
    "mlflow_get_registry_info": {
        "func": mlflow_get_registry_info,
        "schema": MlflowGetRegistryInfoInput,
        "domain": "MLflow",
        "description": "Queries the MLflow Model Registry for registered versions and aliases (production/candidate).",
        "mutating": False,
    },
    # Airflow
    "airflow_list_dags": {
        "func": airflow_list_dags,
        "schema": AirflowListDagsInput,
        "domain": "Airflow",
        "description": "Lists all registered DAGs in Apache Airflow.",
        "mutating": False,
    },
    "airflow_get_dag_runs": {
        "func": airflow_get_dag_runs,
        "schema": AirflowGetDagRunsInput,
        "domain": "Airflow",
        "description": "Fetches recent run statuses and execution dates for a DAG.",
        "mutating": False,
    },
    "airflow_trigger_dag": {
        "func": airflow_trigger_dag,
        "schema": AirflowTriggerDagInput,
        "domain": "Airflow",
        "description": "Triggers a DAG run in Airflow. MUTATING: Requires confirm=True and allowlisted DAG.",
        "mutating": True,
    },
    "airflow_get_failed_tasks": {
        "func": airflow_get_failed_tasks,
        "schema": AirflowGetFailedTasksInput,
        "domain": "Airflow",
        "description": "Finds all failed tasks in a DAG run for incident root-cause diagnosis.",
        "mutating": False,
    },
    "airflow_get_task_instances": {
        "func": airflow_get_task_instances,
        "schema": AirflowGetTaskInstancesInput,
        "domain": "Airflow",
        "description": "Lists all task instances in a DAG run and their current execution states.",
        "mutating": False,
    },
    "airflow_get_task_logs": {
        "func": airflow_get_task_logs,
        "schema": AirflowGetTaskLogsInput,
        "domain": "Airflow",
        "description": "Retrieves task execution logs from Airflow (with secrets redacted).",
        "mutating": False,
    },
    # Kubernetes
    "k8s_list_pods": {
        "func": k8s_list_pods,
        "schema": K8sListPodsInput,
        "domain": "Kubernetes",
        "description": "Lists pods in a namespace with phase, readiness, and restart counts.",
        "mutating": False,
    },
    "k8s_list_deployments": {
        "func": k8s_list_deployments,
        "schema": K8sListDeploymentsInput,
        "domain": "Kubernetes",
        "description": "Lists deployments in a namespace with replica counts and image tags.",
        "mutating": False,
    },
    "k8s_get_pod_status": {
        "func": k8s_get_pod_status,
        "schema": K8sGetPodStatusInput,
        "domain": "Kubernetes",
        "description": "Inspects pod container statuses, termination reasons, and conditions.",
        "mutating": False,
    },
    "k8s_get_pod_logs": {
        "func": k8s_get_pod_logs,
        "schema": K8sGetPodLogsInput,
        "domain": "Kubernetes",
        "description": "Tails pod container logs with secret redaction.",
        "mutating": False,
    },
    "k8s_list_services": {
        "func": k8s_list_services,
        "schema": K8sListServicesInput,
        "domain": "Kubernetes",
        "description": "Lists services, ports, and cluster IPs in a namespace.",
        "mutating": False,
    },
    "k8s_get_events": {
        "func": k8s_get_events,
        "schema": K8sGetEventsInput,
        "domain": "Kubernetes",
        "description": "Fetches warning and normal events in the namespace.",
        "mutating": False,
    },
    "k8s_find_unhealthy_pods": {
        "func": k8s_find_unhealthy_pods,
        "schema": K8sFindUnhealthyPodsInput,
        "domain": "Kubernetes",
        "description": "Finds pods in CrashLoopBackOff, ImagePullBackOff, or with high restarts.",
        "mutating": False,
    },
    "k8s_get_deployed_version": {
        "func": k8s_get_deployed_version,
        "schema": K8sGetDeployedVersionInput,
        "domain": "Kubernetes",
        "description": "Extracts image tag, app version, and replicas for a deployment.",
        "mutating": False,
    },
    "k8s_diagnose_service": {
        "func": k8s_diagnose_service,
        "schema": K8sDiagnoseServiceInput,
        "domain": "Kubernetes",
        "description": "Correlates deployment, pods, probes, and events into an automated diagnosis.",
        "mutating": False,
    },
    "k8s_restart_deployment": {
        "func": k8s_restart_deployment,
        "schema": K8sRestartDeploymentInput,
        "domain": "Kubernetes",
        "description": "Rollout restarts a deployment. MUTATING: Requires confirm=True and allowed namespace.",
        "mutating": True,
    },
    "k8s_scale_deployment": {
        "func": k8s_scale_deployment,
        "schema": K8sScaleDeploymentInput,
        "domain": "Kubernetes",
        "description": "Scales deployment replicas. MUTATING: Requires confirm=True and allowed namespace.",
        "mutating": True,
    },
}

