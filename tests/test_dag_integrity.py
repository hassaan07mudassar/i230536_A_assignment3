"""Tests for Apache Airflow DAG Structure, Task Dependencies, and Integrity.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import pytest


def test_dag_imports_and_structure():
    """Asserts that training_pipeline.py imports cleanly and configures valid parameters."""
    try:
        from dags.training_pipeline import dag_instance
    except ImportError as e:
        pytest.skip(f"Airflow not in current python environment: {e}")

    assert dag_instance is not None
    assert dag_instance.dag_id == "ml_training_pipeline"
    assert dag_instance.default_args["retries"] == 2

    # Check task IDs
    task_ids = [t.task_id for t in dag_instance.tasks]
    assert "validate_dataset" in task_ids
    assert "preprocess_data" in task_ids
    assert "train_model" in task_ids
    assert "evaluate_and_log_mlflow" in task_ids
    assert "apply_quality_gate" in task_ids


def test_dag_task_dependency_order():
    """Verifies strict execution ordering:
    validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow -> apply_quality_gate
    """
    try:
        from dags.training_pipeline import dag_instance
    except ImportError:
        pytest.skip("Airflow not in current python environment")

    tasks = {t.task_id: t for t in dag_instance.tasks}

    assert "preprocess_data" in [down.task_id for down in tasks["validate_dataset"].downstream_list]
    assert "train_model" in [down.task_id for down in tasks["preprocess_data"].downstream_list]
    assert "evaluate_and_log_mlflow" in [down.task_id for down in tasks["train_model"].downstream_list]
    assert "apply_quality_gate" in [down.task_id for down in tasks["evaluate_and_log_mlflow"].downstream_list]

