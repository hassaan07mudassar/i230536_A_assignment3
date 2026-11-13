"""Security, Authorization, and Policy Enforcement Engine.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from typing import Optional, Tuple
from agent.config import settings
from agent.logging_config import logger
from agent.schemas import ToolError


class PermissionDeniedError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_k8s_mutation(
    action: str,
    namespace: str,
    confirm: bool,
) -> Optional[ToolError]:
    """Validates cluster-state-changing operations against safety policies."""
    action_clean = action.strip().lower()

    # 1. Action Allowlist
    if action_clean not in settings.allowed_k8s_actions:
        err = ToolError(
            code="ACTION_NOT_ALLOWED",
            message=f"Kubernetes action '{action}' is not in the allowed actions list: {sorted(settings.allowed_k8s_actions)}",
            details={"action": action, "allowed": list(settings.allowed_k8s_actions)},
        )
        logger.warning(f"Rejected unauthorized K8s action: {err.message}")
        return err

    # 2. Namespace Allowlist
    if namespace not in settings.allowed_namespaces:
        err = ToolError(
            code="NAMESPACE_NOT_ALLOWED",
            message=f"Kubernetes namespace '{namespace}' is not in the approved allowlist: {sorted(settings.allowed_namespaces)}",
            details={"namespace": namespace, "allowed": list(settings.allowed_namespaces)},
        )
        logger.warning(f"Rejected K8s mutation on unauthorized namespace: {err.message}")
        return err

    # 3. Explicit User Confirmation Requirement
    if not confirm:
        err = ToolError(
            code="CONFIRMATION_REQUIRED",
            message=(
                f"Cluster mutation '{action}' in namespace '{namespace}' requires explicit user confirmation. "
                "Please resubmit the request with confirm=true."
            ),
            details={"action": action, "namespace": namespace, "confirmed": False},
        )
        logger.warning(f"Rejected unconfirmed K8s mutation: {err.message}")
        return err

    logger.info(f"Authorized mutating K8s action: {action} on namespace {namespace}")
    return None


def validate_airflow_trigger(
    dag_id: str,
    confirm: bool,
) -> Optional[ToolError]:
    """Validates DAG trigger requests against DAG allowlist and confirmation flag."""
    dag_clean = dag_id.strip()

    # 1. DAG Allowlist
    if dag_clean not in settings.allowed_dag_ids:
        err = ToolError(
            code="DAG_NOT_ALLOWED",
            message=f"DAG '{dag_id}' is not in the approved trigger allowlist: {sorted(settings.allowed_dag_ids)}",
            details={"dag_id": dag_id, "allowed": list(settings.allowed_dag_ids)},
        )
        logger.warning(f"Rejected trigger on unauthorized DAG: {err.message}")
        return err

    # 2. Confirmation Requirement
    if not confirm:
        err = ToolError(
            code="CONFIRMATION_REQUIRED",
            message=(
                f"Triggering DAG '{dag_id}' modifies pipeline execution state and requires explicit user confirmation. "
                "Please resubmit with confirm=true."
            ),
            details={"dag_id": dag_id, "confirmed": False},
        )
        logger.warning(f"Rejected unconfirmed Airflow DAG trigger: {err.message}")
        return err

    logger.info(f"Authorized Airflow trigger for DAG: {dag_id}")
    return None


def validate_dvc_subcommand(subcommand: str) -> Optional[ToolError]:
    """Enforces strict allowlisting for DVC subprocess execution."""
    subcmd_clean = subcommand.strip().lower()
    if subcmd_clean not in settings.allowed_dvc_subcommands:
        err = ToolError(
            code="DVC_COMMAND_DISALLOWED",
            message=f"DVC subcommand '{subcommand}' is rejected. Permitted subcommands: {sorted(settings.allowed_dvc_subcommands)}",
            details={"subcommand": subcommand, "allowed": list(settings.allowed_dvc_subcommands)},
        )
        logger.warning(f"Rejected disallowed DVC subcommand: {err.message}")
        return err
    return None

