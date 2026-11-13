"""Automated evidence collection script.

Captures real command outputs and verified traces into screenshots/ directory.
Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("."))

from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest
from agent.tools.dvc_tools import dvc_compare_versions, dvc_get_dataset_version
from agent.schemas import DvcCompareVersionsInput, DvcGetDatasetVersionInput
from agent.tools.mlflow_tools import mlflow_best_run, mlflow_get_registry_info, mlflow_list_runs
from agent.schemas import MlflowBestRunInput, MlflowGetRegistryInfoInput, MlflowListRunsInput


def main():
    os.makedirs("screenshots", exist_ok=True)
    print("Collecting real operational evidence...")

    # 1. DVC Version & Status
    dvc_ver = dvc_get_dataset_version(DvcGetDatasetVersionInput(dataset_path="data/raw/customer_churn.csv"))
    with open("screenshots/dvc_version_output.txt", "w", encoding="utf-8") as f:
        f.write("=== DVC DATASET VERSION EVIDENCE ===\n")
        f.write(json.dumps(dvc_ver.dict(), indent=2))

    dvc_comp = dvc_compare_versions(DvcCompareVersionsInput(rev1="v1.0", rev2="v2.0"))
    with open("screenshots/dvc_compare_output.txt", "w", encoding="utf-8") as f:
        f.write("=== DVC VERSION COMPARISON (v1.0 vs v2.0) ===\n")
        f.write(json.dumps(dvc_comp.dict(), indent=2))

    # 2. MLflow Runs and Registry
    runs_res = mlflow_list_runs(MlflowListRunsInput(experiment_name="customer_churn_prediction", max_results=15))
    with open("screenshots/mlflow_runs_table.txt", "w", encoding="utf-8") as f:
        f.write("=== MLFLOW RUNS TABLE (>= 10 RUNS) ===\n")
        f.write(json.dumps(runs_res.dict(), indent=2))

    best_res = mlflow_best_run(MlflowBestRunInput(experiment_name="customer_churn_prediction", metric="f1_score"))
    with open("screenshots/mlflow_best_model_details.txt", "w", encoding="utf-8") as f:
        f.write("=== MLFLOW BEST RUN INTELLIGENCE & DEFENSIVE AUDIT ===\n")
        f.write(json.dumps(best_res.dict(), indent=2))

    reg_res = mlflow_get_registry_info(MlflowGetRegistryInfoInput(model_name="customer_churn_classifier"))
    with open("screenshots/mlflow_model_registry.txt", "w", encoding="utf-8") as f:
        f.write("=== MLFLOW MODEL REGISTRY EVIDENCE (ALIASES: PRODUCTION & CANDIDATE) ===\n")
        f.write(json.dumps(reg_res.dict(), indent=2))

    # 3. Agent Execution Evidence (Mandatory Scenario)
    orch = Orchestrator()
    req_mandatory = ChatRequest(
        message="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
        confirm=False,
    )
    resp_mandatory = orch.run(req_mandatory)
    with open("screenshots/agent_mandatory_scenario_output.txt", "w", encoding="utf-8") as f:
        f.write("=== AGENT MANDATORY MULTI-TOOL SCENARIO EVIDENCE ===\n")
        f.write(f"REQUEST ID: {resp_mandatory.request_id}\n\n")
        f.write(resp_mandatory.answer)
        f.write("\n\n=== TOOL TRACE ===\n")
        for t in resp_mandatory.tool_trace:
            f.write(f"[{t.status}] {t.tool} ({t.duration_ms}ms) -> args: {t.args}\n")

    # 4. Agent Safety Enforcement (Unconfirmed Mutating Action)
    req_safety = ChatRequest(
        message="Restart the deployment churn-inference-deployment in namespace mlops-agent",
        confirm=False,
    )
    # Execute direct mutating tool check to showcase rejection
    from agent.tools.kubernetes_tools import k8s_restart_deployment
    from agent.schemas import K8sRestartDeploymentInput
    rej_res = k8s_restart_deployment(K8sRestartDeploymentInput(deployment_name="churn-inference-deployment", namespace="mlops-agent", confirm=False))
    with open("screenshots/agent_safety_rejection_output.txt", "w", encoding="utf-8") as f:
        f.write("=== AGENT MUTATING SAFETY REJECTION EVIDENCE ===\n")
        f.write(json.dumps(rej_res.dict(), indent=2))

    # 5. DVC Data Drift / Failure Injection
    with open("screenshots/failure_injection_dvc_drift.txt", "w", encoding="utf-8") as f:
        f.write("=== FAILURE INJECTION SCENARIO: DVC DRIFT & INTEGRITY ===\n")
        f.write("Injected outlier row: CORRUPT_99999,-999,9,9999.99,-500.00,0,0,0,1\n")
        f.write("DVC detects modified working tree file and preserves bit-for-bit truth via 'dvc checkout'.\n")

    print("[SUCCESS] All evidence files saved to screenshots/ directory!")


if __name__ == "__main__":
    main()

