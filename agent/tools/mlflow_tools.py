"""MLflow Experiment Intelligence Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import time
from typing import Optional, Tuple
from mlflow.tracking import MlflowClient

from agent.config import settings
from agent.schemas import (
    MlflowBestRunInput,
    MlflowCompareExperimentsInput,
    MlflowCompareRunsInput,
    MlflowGetRegistryInfoInput,
    MlflowGetRunDetailsInput,
    MlflowListExperimentsInput,
    MlflowListRunsInput,
    ToolError,
    ToolEvidence,
    ToolResult,
)


def _get_client() -> Tuple[MlflowClient, Optional[ToolError]]:
    tracking_uri = settings.mlflow_tracking_uri
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        return client, None
    except Exception as e:
        return None, ToolError(
            code="MLFLOW_UNAVAILABLE",
            message=f"Failed to connect to MLflow tracking server at '{tracking_uri}': {str(e)}",
            details={"tracking_uri": tracking_uri},
        )


# ------------------------------------------------------------------------------
# Tool 1: mlflow_list_experiments
# ------------------------------------------------------------------------------
def mlflow_list_experiments(input_data: MlflowListExperimentsInput) -> ToolResult:
    """Lists experiments recorded in the MLflow tracking store."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_list_experiments", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    try:
        experiments = client.search_experiments()
        result = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags,
            }
            for exp in experiments
        ]
        evidence = ToolEvidence(
            source=f"MLflow tracking API ({settings.mlflow_tracking_uri})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Retrieved {len(result)} experiments.",
        )
        return ToolResult(ok=True, data={"experiments": result, "count": len(result)}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MLFLOW_QUERY_ERROR", message=f"Failed to list experiments: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_list_experiments", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 2: mlflow_list_runs
# ------------------------------------------------------------------------------
def mlflow_list_runs(input_data: MlflowListRunsInput) -> ToolResult:
    """Queries runs for an experiment with optional filtering, ordering, and metrics."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_list_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    try:
        exp = client.get_experiment_by_name(input_data.experiment_name)
        if not exp:
            return ToolResult(
                ok=False,
                error=ToolError(code="EXPERIMENT_NOT_FOUND", message=f"Experiment '{input_data.experiment_name}' not found"),
                evidence=ToolEvidence(source="mlflow_list_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=input_data.filter_string or "",
            order_by=[input_data.order_by] if input_data.order_by else ["attributes.start_time DESC"],
            max_results=input_data.max_results,
        )

        run_summaries = []
        for r in runs:
            run_summaries.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                "params": r.data.params,
                "dataset_version": r.data.tags.get("dvc.dataset_version", "unknown"),
                "git_commit": r.data.tags.get("git.commit", "unknown")[:8],
                "quality_passed": r.data.tags.get("quality_gate.passed", "unknown"),
            })

        evidence = ToolEvidence(
            source=f"MLflow search_runs (exp={input_data.experiment_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Matched {len(run_summaries)} runs for experiment '{input_data.experiment_name}'.",
        )
        return ToolResult(ok=True, data={"runs": run_summaries, "total": len(run_summaries)}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MLFLOW_SEARCH_ERROR", message=f"Search failed: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_list_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 3: mlflow_get_run_details
# ------------------------------------------------------------------------------
def mlflow_get_run_details(input_data: MlflowGetRunDetailsInput) -> ToolResult:
    """Retrieves full details for a run: metrics, params, tags, artifacts."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_get_run_details", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    try:
        run = client.get_run(input_data.run_id)
        artifacts = [a.path for a in client.list_artifacts(input_data.run_id)]

        details = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "lifecycle_stage": run.info.lifecycle_stage,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifacts": artifacts,
            "provenance": {
                "dvc_version": run.data.tags.get("dvc.dataset_version"),
                "dvc_hash": run.data.tags.get("dvc.data_hash"),
                "git_commit": run.data.tags.get("git.commit"),
            },
        }

        evidence = ToolEvidence(
            source=f"MlflowClient.get_run({input_data.run_id})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Run details for {run.info.run_name} (Status: {run.info.status})",
        )
        return ToolResult(ok=True, data=details, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="RUN_NOT_FOUND", message=f"Run '{input_data.run_id}' not found or inaccessible: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_get_run_details", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 4: mlflow_best_run
# ------------------------------------------------------------------------------
def mlflow_best_run(input_data: MlflowBestRunInput) -> ToolResult:
    """Finds best run by metric with defensive ML analysis (checking overfitting & gates)."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_best_run", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    metric = input_data.metric
    order_direction = "DESC" if input_data.mode == "max" else "ASC"
    order_by = f"metrics.{metric} {order_direction}"

    try:
        exp = client.get_experiment_by_name(input_data.experiment_name)
        if not exp:
            return ToolResult(
                ok=False,
                error=ToolError(code="EXPERIMENT_NOT_FOUND", message=f"Experiment '{input_data.experiment_name}' not found"),
                evidence=ToolEvidence(source="mlflow_best_run", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[order_by],
            max_results=10,
        )

        if not runs:
            return ToolResult(
                ok=False,
                error=ToolError(code="NO_RUNS_FOUND", message=f"No runs found in experiment '{input_data.experiment_name}'"),
                evidence=ToolEvidence(source="mlflow_best_run", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

        best_candidate = runs[0]
        # Multi-factor defensive validation:
        val_f1 = best_candidate.data.metrics.get("f1_score", 0.0)
        train_f1 = best_candidate.data.metrics.get("train_f1", val_f1)
        overfitting_gap = round(train_f1 - val_f1, 4)
        run_status = best_candidate.info.status

        overfitting_warning = overfitting_gap > 0.15
        quality_gate_passed = best_candidate.data.tags.get("quality_gate.passed") == "True"

        recommendation_reasoning = (
            f"Selected run '{best_candidate.info.run_name}' with {metric}={best_candidate.data.metrics.get(metric, 0):.4f}. "
            f"Dataset version: {best_candidate.data.tags.get('dvc.dataset_version', 'unknown')}. "
            f"Overfitting gap (train - val): {overfitting_gap:.4f}. "
            f"Quality gate status: {'PASSED' if quality_gate_passed else 'FAILED/UNCONFIRMED'}."
        )

        data = {
            "best_run_id": best_candidate.info.run_id,
            "run_name": best_candidate.info.run_name,
            "status": run_status,
            "metric_name": metric,
            "metric_value": best_candidate.data.metrics.get(metric),
            "all_metrics": {k: round(v, 4) for k, v in best_candidate.data.metrics.items()},
            "params": best_candidate.data.params,
            "provenance": {
                "dvc_dataset_version": best_candidate.data.tags.get("dvc.dataset_version"),
                "dvc_data_hash": best_candidate.data.tags.get("dvc.data_hash"),
                "git_commit": best_candidate.data.tags.get("git.commit"),
            },
            "defensive_audit": {
                "overfitting_gap": overfitting_gap,
                "is_overfitted": overfitting_warning,
                "run_status": run_status,
                "is_production_ready": quality_gate_passed and not overfitting_warning and run_status == "FINISHED",
                "reasoning": recommendation_reasoning,
            },
        }

        evidence = ToolEvidence(
            source=f"MLflow search_runs (ordered by {order_by})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Best run: {data['run_name']} | {metric}: {data['metric_value']} | DVC: {data['provenance']['dvc_dataset_version']}",
        )
        return ToolResult(ok=True, data=data, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MLFLOW_BEST_RUN_ERROR", message=f"Failed finding best run: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_best_run", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 5: mlflow_compare_runs
# ------------------------------------------------------------------------------
def mlflow_compare_runs(input_data: MlflowCompareRunsInput) -> ToolResult:
    """Compares metrics, params, dataset version, and defensive signals across multiple runs."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_compare_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    try:
        run_ids = input_data.run_ids
        if not run_ids:
            exp = client.get_experiment_by_name(input_data.experiment_name)
            if not exp:
                return ToolResult(
                    ok=False,
                    error=ToolError(code="EXPERIMENT_NOT_FOUND", message=f"Experiment '{input_data.experiment_name}' not found"),
                    evidence=ToolEvidence(source="mlflow_compare_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                )
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=input_data.last_n or 3,
                order_by=["attributes.start_time DESC"],
            )
            run_ids = [r.info.run_id for r in runs]

        comparison = []
        for rid in run_ids:
            run = client.get_run(rid)
            metrics = run.data.metrics
            tags = run.data.tags
            comparison.append({
                "run_id": rid,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "accuracy": round(metrics.get("accuracy", 0.0), 4),
                "precision": round(metrics.get("precision", 0.0), 4),
                "recall": round(metrics.get("recall", 0.0), 4),
                "f1_score": round(metrics.get("f1_score", 0.0), 4),
                "train_f1": round(metrics.get("train_f1", 0.0), 4),
                "train_val_gap": round(metrics.get("train_val_f1_gap", 0.0), 4),
                "training_time": metrics.get("training_time_seconds", 0.0),
                "model_type": run.data.params.get("model_type"),
                "n_estimators": run.data.params.get("n_estimators"),
                "max_depth": run.data.params.get("max_depth"),
                "dataset_version": tags.get("dvc.dataset_version", "unknown"),
                "quality_passed": tags.get("quality_gate.passed", "unknown"),
            })

        evidence = ToolEvidence(
            source=f"MlflowClient.compare({len(comparison)} runs)",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Compared {len(comparison)} runs across metrics, params, and dataset versions.",
        )
        return ToolResult(ok=True, data={"comparison_table": comparison, "count": len(comparison)}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="COMPARE_RUNS_ERROR", message=f"Failed to compare runs: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_compare_runs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 6: mlflow_compare_experiments
# ------------------------------------------------------------------------------
def mlflow_compare_experiments(input_data: MlflowCompareExperimentsInput) -> ToolResult:
    """Compares high-level aggregate statistics across multiple experiments."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_compare_experiments", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    summaries = []
    for exp_name in input_data.experiment_names:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            continue
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
        f1_vals = [r.data.metrics.get("f1_score", 0.0) for r in runs if "f1_score" in r.data.metrics]
        summaries.append({
            "experiment_name": exp_name,
            "experiment_id": exp.experiment_id,
            "total_runs": len(runs),
            "max_f1": max(f1_vals) if f1_vals else None,
            "mean_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
        })

    evidence = ToolEvidence(
        source="mlflow_compare_experiments",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output=f"Compared {len(summaries)} experiments.",
    )
    return ToolResult(ok=True, data={"experiments": summaries}, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 7: mlflow_get_registry_info
# ------------------------------------------------------------------------------
def mlflow_get_registry_info(input_data: MlflowGetRegistryInfoInput) -> ToolResult:
    """Retrieves registered models, versions, and stage/alias mappings (production/candidate)."""
    client, err = _get_client()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="mlflow_get_registry_info", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    model_name = input_data.model_name
    try:
        reg_model = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")

        version_list = []
        for v in versions:
            version_list.append({
                "version": v.version,
                "current_stage": v.current_stage,
                "aliases": v.aliases if hasattr(v, "aliases") else [],
                "run_id": v.run_id,
                "description": v.description,
                "tags": v.tags,
                "status": v.status,
            })

        aliases = getattr(reg_model, "aliases", {})
        data = {
            "model_name": model_name,
            "description": reg_model.description,
            "aliases": aliases,
            "versions": version_list,
            "total_versions": len(version_list),
        }

        evidence = ToolEvidence(
            source=f"MlflowClient (Model Registry: {model_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Registry '{model_name}': {len(version_list)} versions. Aliases: {aliases}",
        )
        return ToolResult(ok=True, data=data, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="MODEL_NOT_IN_REGISTRY", message=f"Model '{model_name}' not found in registry: {str(e)}"),
            evidence=ToolEvidence(source="mlflow_get_registry_info", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
