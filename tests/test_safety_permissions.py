"""Tests for Permissions, Safety Guards, and Allowlist Policies.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import pytest

from agent.permissions import (
    validate_airflow_trigger,
    validate_dvc_subcommand,
    validate_k8s_mutation,
)


def test_validate_k8s_mutation_allowed():
    err = validate_k8s_mutation(action="restart", namespace="mlops-agent", confirm=True)
    assert err is None


def test_validate_k8s_mutation_unconfirmed():
    err = validate_k8s_mutation(action="restart", namespace="mlops-agent", confirm=False)
    assert err is not None
    assert err.code == "CONFIRMATION_REQUIRED"


def test_validate_k8s_mutation_disallowed_namespace():
    err = validate_k8s_mutation(action="restart", namespace="kube-system", confirm=True)
    assert err is not None
    assert err.code == "NAMESPACE_NOT_ALLOWED"


def test_validate_k8s_mutation_disallowed_action():
    err = validate_k8s_mutation(action="delete_all_pods", namespace="mlops-agent", confirm=True)
    assert err is not None
    assert err.code == "ACTION_NOT_ALLOWED"


def test_validate_airflow_trigger_allowed():
    err = validate_airflow_trigger(dag_id="ml_training_pipeline", confirm=True)
    assert err is None


def test_validate_airflow_trigger_unconfirmed():
    err = validate_airflow_trigger(dag_id="ml_training_pipeline", confirm=False)
    assert err is not None
    assert err.code == "CONFIRMATION_REQUIRED"


def test_validate_airflow_trigger_disallowed_dag():
    err = validate_airflow_trigger(dag_id="unregistered_arbitrary_dag", confirm=True)
    assert err is not None
    assert err.code == "DAG_NOT_ALLOWED"


def test_validate_dvc_subcommand_allowlist():
    assert validate_dvc_subcommand("status") is None
    assert validate_dvc_subcommand("diff") is None
    assert validate_dvc_subcommand("checkout") is None
    assert validate_dvc_subcommand("rm -rf /") is not None

