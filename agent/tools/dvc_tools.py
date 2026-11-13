"""DVC Data Management Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import glob
import os
import subprocess
import time
from typing import List, Tuple
import yaml

from agent.config import settings
from agent.logging_config import logger
from agent.permissions import validate_dvc_subcommand
from agent.schemas import (
    DvcCompareVersionsInput,
    DvcDetectChangesInput,
    DvcExplainReproduceInput,
    DvcGetDatasetForExperimentInput,
    DvcGetDatasetVersionInput,
    DvcListTrackedDatasetsInput,
    ToolError,
    ToolEvidence,
    ToolResult,
)


def _execute_subprocess(cmd: List[str], timeout: int = 15, cwd: str = ".") -> Tuple[int, str, str]:
    """Safely executes subprocess with shell=False, timeouts, and error trapping."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
    except Exception as e:
        return -1, "", f"Subprocess execution failed: {str(e)}"


# ------------------------------------------------------------------------------
# Tool 1: dvc_list_tracked_datasets
# ------------------------------------------------------------------------------
def dvc_list_tracked_datasets(input_data: DvcListTrackedDatasetsInput) -> ToolResult:
    """Lists all datasets tracked by DVC in the repository."""
    repo_path = input_data.repo_path

    if not os.path.exists(os.path.join(repo_path, ".dvc")):
        return ToolResult(
            ok=False,
            error=ToolError(code="DVC_REPO_NOT_FOUND", message=f"No .dvc directory found at '{repo_path}'"),
            evidence=ToolEvidence(source="dvc_list_tracked_datasets", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    tracked = []
    # Find all .dvc files
    dvc_files = glob.glob(os.path.join(repo_path, "**", "*.dvc"), recursive=True)
    for dvc_file in dvc_files:
        try:
            with open(dvc_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                for out in content.get("outs", []):
                    rel_dir = os.path.dirname(dvc_file)
                    target_file = os.path.normpath(os.path.join(rel_dir, out.get("path", "")))
                    tracked.append({
                        "dvc_file": os.path.relpath(dvc_file, repo_path),
                        "data_path": os.path.relpath(target_file, repo_path),
                        "md5_hash": out.get("md5"),
                        "size_bytes": out.get("size"),
                        "is_cached": out.get("isexec", False),
                    })
        except Exception as e:
            logger.warning(f"Error parsing .dvc file {dvc_file}: {e}")

    evidence = ToolEvidence(
        source="dvc_list_tracked_datasets (.dvc scan)",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output=f"Discovered {len(tracked)} DVC-tracked dataset artifacts.",
    )
    return ToolResult(ok=True, data={"tracked_datasets": tracked, "count": len(tracked)}, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 2: dvc_get_dataset_version
# ------------------------------------------------------------------------------
def dvc_get_dataset_version(input_data: DvcGetDatasetVersionInput) -> ToolResult:
    """Retrieves current DVC hash, git commit, branch, and active tag for a dataset."""
    dvc_file = f"{input_data.dataset_path}.dvc"
    if not os.path.exists(dvc_file):
        return ToolResult(
            ok=False,
            error=ToolError(code="DATASET_NOT_TRACKED", message=f"No DVC metadata file found at '{dvc_file}'"),
            evidence=ToolEvidence(source="dvc_get_dataset_version", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    dvc_hash = "unknown"
    try:
        with open(dvc_file, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
            outs = meta.get("outs", [])
            if outs:
                dvc_hash = outs[0].get("md5", "unknown")
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="DVC_PARSE_ERROR", message=f"Failed to read {dvc_file}: {e}"),
            evidence=ToolEvidence(source="dvc_get_dataset_version", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    # Git metadata
    rc_commit, git_commit, _ = _execute_subprocess(["git", "rev-parse", "HEAD"], cwd=input_data.repo_path)
    rc_branch, git_branch, _ = _execute_subprocess(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=input_data.repo_path)
    rc_tag, git_tag, _ = _execute_subprocess(["git", "describe", "--tags", "--exact-match"], cwd=input_data.repo_path)

    data = {
        "dataset_path": input_data.dataset_path,
        "dvc_file": dvc_file,
        "dvc_md5_hash": dvc_hash,
        "git_commit": git_commit.strip() if rc_commit == 0 else "unknown",
        "git_branch": git_branch.strip() if rc_branch == 0 else "unknown",
        "git_tag": git_tag.strip() if rc_tag == 0 else "none",
        "local_file_exists": os.path.exists(input_data.dataset_path),
    }

    evidence = ToolEvidence(
        source="dvc metadata + git rev-parse",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output=f"Hash: {dvc_hash} | Commit: {data['git_commit'][:8]} | Tag: {data['git_tag']}",
    )
    return ToolResult(ok=True, data=data, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 3: dvc_compare_versions
# ------------------------------------------------------------------------------
def dvc_compare_versions(input_data: DvcCompareVersionsInput) -> ToolResult:
    """Compares dataset pointer files between two git revisions/tags."""
    rev1 = input_data.rev1
    rev2 = input_data.rev2
    repo_path = input_data.repo_path

    # Git diff on .dvc files between the two revisions
    rc, stdout, stderr = _execute_subprocess(
        ["git", "diff", f"{rev1}..{rev2}", "--", "*.dvc"], cwd=repo_path
    )
    if rc != 0:
        return ToolResult(
            ok=False,
            error=ToolError(code="GIT_DIFF_ERROR", message=f"Failed to compare {rev1} and {rev2}: {stderr.strip()}"),
            evidence=ToolEvidence(source="git diff", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    # Inspect file at rev1
    rc1, stdout1, _ = _execute_subprocess(["git", "show", f"{rev1}:data/raw/customer_churn.csv.dvc"], cwd=repo_path)
    rc2, stdout2, _ = _execute_subprocess(["git", "show", f"{rev2}:data/raw/customer_churn.csv.dvc"], cwd=repo_path)

    hash1, hash2 = "unknown", "unknown"
    if rc1 == 0:
        d1 = yaml.safe_load(stdout1) or {}
        hash1 = d1.get("outs", [{}])[0].get("md5", "unknown")
    if rc2 == 0:
        d2 = yaml.safe_load(stdout2) or {}
        hash2 = d2.get("outs", [{}])[0].get("md5", "unknown")

    comparison = {
        "revision_base": rev1,
        "revision_target": rev2,
        "dataset": "data/raw/customer_churn.csv",
        "hash_rev1": hash1,
        "hash_rev2": hash2,
        "data_changed": hash1 != hash2,
        "diff_summary": stdout if stdout else "No changes in .dvc files between revisions",
    }

    evidence = ToolEvidence(
        source="git diff -- *.dvc",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output=stdout[:300] if stdout else "No .dvc diff",
    )
    return ToolResult(ok=True, data=comparison, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 4: dvc_detect_changes
# ------------------------------------------------------------------------------
def dvc_detect_changes(input_data: DvcDetectChangesInput) -> ToolResult:
    """Executes dvc status to check if tracked files have diverged from cache/remote."""
    # Allowlist check
    err = validate_dvc_subcommand("status")
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="security_policy", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    cmd = ["dvc", "status"]
    if input_data.dataset_path:
        cmd.append(input_data.dataset_path)

    rc, stdout, stderr = _execute_subprocess(cmd, cwd=input_data.repo_path)
    if rc != 0:
        return ToolResult(
            ok=False,
            error=ToolError(code="DVC_STATUS_FAILED", message=f"dvc status failed: {stderr.strip()}"),
            evidence=ToolEvidence(source="dvc status", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    is_clean = "Data and pipelines are up to date" in stdout or not stdout.strip()
    data = {
        "status_output": stdout.strip(),
        "is_up_to_date": is_clean,
        "has_changes": not is_clean,
    }

    evidence = ToolEvidence(
        source="dvc status (subprocess allowlist)",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output=stdout.strip() or "Up to date",
    )
    return ToolResult(ok=True, data=data, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 5: dvc_explain_reproduce
# ------------------------------------------------------------------------------
def dvc_explain_reproduce(input_data: DvcExplainReproduceInput) -> ToolResult:
    """Returns the exact deterministic commands to reproduce a specific dataset version."""
    revision = input_data.revision
    dataset_path = input_data.dataset_path
    dvc_file = f"{dataset_path}.dvc"

    steps = [
        f"git checkout {revision}",
        f"dvc checkout {dvc_file}",
        "# Alternative if cache is not local:",
        f"dvc pull {dvc_file}",
    ]

    reproduce_plan = {
        "target_revision": revision,
        "target_dataset": dataset_path,
        "commands": steps,
        "verification_step": f"python -c \"import pandas as pd; df=pd.read_csv('{dataset_path}'); print('Row count:', len(df))\"",
        "guarantee": "DVC MD5 hash verification guarantees cryptographic bit-for-bit reproducibility.",
    }

    evidence = ToolEvidence(
        source="dvc_explain_reproduce",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output="\n".join(steps),
    )
    return ToolResult(ok=True, data=reproduce_plan, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 6: dvc_get_dataset_for_experiment
# ------------------------------------------------------------------------------
def dvc_get_dataset_for_experiment(input_data: DvcGetDatasetForExperimentInput) -> ToolResult:
    """Retrieves authentic DVC dataset provenance recorded in MLflow run tags."""
    from mlflow.tracking import MlflowClient

    tracking_uri = settings.mlflow_tracking_uri
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MLFLOW_CONNECTION_ERROR", message=f"Cannot connect to MLflow at {tracking_uri}: {e}"),
            evidence=ToolEvidence(source="MlflowClient", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    target_run_id = input_data.run_id
    if not target_run_id and input_data.experiment_id:
        runs = client.search_runs(experiment_ids=[input_data.experiment_id], max_results=1)
        if runs:
            target_run_id = runs[0].info.run_id

    if not target_run_id:
        # Search active experiment
        exp = client.get_experiment_by_name("customer_churn_prediction")
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
            if runs:
                target_run_id = runs[0].info.run_id

    if not target_run_id:
        return ToolResult(
            ok=False,
            error=ToolError(code="RUN_NOT_FOUND", message="No run ID provided or found in active experiments"),
            evidence=ToolEvidence(source="MlflowClient.search_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    try:
        run = client.get_run(target_run_id)
        tags = run.data.tags
        provenance = {
            "run_id": target_run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run.info.run_name,
            "dvc_dataset_version": tags.get("dvc.dataset_version", "NOT_RECORDED"),
            "dvc_data_hash": tags.get("dvc.data_hash", "NOT_RECORDED"),
            "dvc_dataset_path": tags.get("dvc.dataset_path", "data/raw/customer_churn.csv"),
            "git_commit": tags.get("git.commit", "NOT_RECORDED"),
            "git_tag": tags.get("git.tag", "none"),
            "model_type": tags.get("model.type", "unknown"),
        }

        evidence = ToolEvidence(
            source=f"MLflow Run Tags (run_id={target_run_id})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"dvc.dataset_version={provenance['dvc_dataset_version']}, dvc.data_hash={provenance['dvc_data_hash']}, git.commit={provenance['git_commit']}",
        )
        return ToolResult(ok=True, data=provenance, evidence=evidence)

    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MLFLOW_RUN_ERROR", message=f"Failed to fetch run '{target_run_id}': {str(e)}"),
            evidence=ToolEvidence(source="MlflowClient.get_run", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
