"""Tests for DVC Data Management Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import os
from unittest.mock import patch
import pytest

from agent.schemas import (
    DvcCompareVersionsInput,
    DvcDetectChangesInput,
    DvcExplainReproduceInput,
    DvcGetDatasetForExperimentInput,
    DvcGetDatasetVersionInput,
    DvcListTrackedDatasetsInput,
)
from agent.tools.dvc_tools import (
    dvc_compare_versions,
    dvc_detect_changes,
    dvc_explain_reproduce,
    dvc_get_dataset_for_experiment,
    dvc_get_dataset_version,
    dvc_list_tracked_datasets,
)


def test_dvc_list_tracked_datasets():
    inp = DvcListTrackedDatasetsInput(repo_path=".")
    result = dvc_list_tracked_datasets(inp)
    assert result.ok is True
    assert "tracked_datasets" in result.data
    assert result.data["count"] >= 1
    # Check customer churn dataset is tracked
    paths = [d["data_path"] for d in result.data["tracked_datasets"]]
    assert any("customer_churn.csv" in p for p in paths)


def test_dvc_get_dataset_version():
    inp = DvcGetDatasetVersionInput(dataset_path="data/raw/customer_churn.csv", repo_path=".")
    result = dvc_get_dataset_version(inp)
    assert result.ok is True
    assert result.data["dvc_md5_hash"] != "unknown"
    assert len(result.data["dvc_md5_hash"]) == 32
    assert result.data["git_commit"] != "unknown"


def test_dvc_compare_versions():
    inp = DvcCompareVersionsInput(rev1="v1.0", rev2="v2.0", repo_path=".")
    result = dvc_compare_versions(inp)
    assert result.ok is True
    assert result.data["revision_base"] == "v1.0"
    assert result.data["revision_target"] == "v2.0"
    assert "data_changed" in result.data


def test_dvc_detect_changes():
    inp = DvcDetectChangesInput(dataset_path="data/raw/customer_churn.csv", repo_path=".")
    result = dvc_detect_changes(inp)
    assert result.ok is True
    assert "is_up_to_date" in result.data


def test_dvc_explain_reproduce():
    inp = DvcExplainReproduceInput(revision="v1.0", dataset_path="data/raw/customer_churn.csv")
    result = dvc_explain_reproduce(inp)
    assert result.ok is True
    assert any("git checkout v1.0" in cmd for cmd in result.data["commands"])
    assert any("dvc checkout" in cmd for cmd in result.data["commands"])


def test_dvc_get_dataset_for_experiment():
    inp = DvcGetDatasetForExperimentInput(run_id=None, experiment_id=None)
    result = dvc_get_dataset_for_experiment(inp)
    assert result.ok is True
    assert "dvc_dataset_version" in result.data
    assert result.data["dvc_dataset_version"] in ("v1.0", "v2.0")


def test_dvc_disallowed_subcommand_rejected():
    from agent.permissions import validate_dvc_subcommand
    err = validate_dvc_subcommand("destroy_repo")
    assert err is not None
    assert err.code == "DVC_COMMAND_DISALLOWED"

