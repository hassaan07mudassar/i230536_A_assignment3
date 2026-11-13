# Live Demonstration Runbook: 5 Core MLOps Scenarios

**Course:** MLOps & Production AI Engineering (Fall 2026)  
**Student:** Hassaan Mudassar (Roll No: i230536, Section: CS-A)  

This runbook provides copy-paste ready commands and verified execution traces to demonstrate each requirement of the MLOps AI Platform Agent.

---

## Prerequisites
Ensure the virtual environment is activated and dependencies are installed:
```bash
# In directory: assignment-03-mlops-agent
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

## Scenario 1: Dataset Provenance & Reproducibility (DVC)

### Command:
```bash
python -m agent.main --message "What is the current version and hash of the customer churn dataset, and how do I reproduce version v1.0?"
```

### Verified Agent Response:
```markdown
### KNOWN FACTS (with tool + evidence source)
- Current DVC dataset version in repository: v2.0 (Commit 2fae37a3). (Source: `dvc metadata + git rev-parse`)
- Cryptographic MD5 hash: `c59d992161f4c7594589d970e5b721e0`. (Source: `data/raw/customer_churn.csv.dvc`)
- Reproducibility sequence for v1.0:
  1. `git checkout v1.0`
  2. `dvc checkout data/raw/customer_churn.csv.dvc` (or `dvc pull`)
  3. Row count validation: 1000 records.

### UNCERTAINTIES / MISSING DATA
- None. Dataset pointer files and Git tags verified directly on disk.

### RECOMMENDED ACTIONS
- Run `dvc status` before training to verify local cache alignment.
```

---

## Scenario 2: Model Selection & Defensive ML Intelligence (MLflow)

### Command:
```bash
python -m agent.main --message "Which is the best performing model in experiment customer_churn_prediction and is it production-ready?"
```

### Verified Agent Response:
```markdown
### KNOWN FACTS (with tool + evidence source)
- Best performing model: `GradientBoostingClassifier_v2.0_trees150_depth6` (Run ID: `6a8834dad1784649a5a544782003b7f3`).
  - Validation Accuracy: 0.8960
  - Validation F1-score: 0.4179 (minority churn class)
  - Training Time: 0.38 seconds
  - Dataset version: `v2.0` (Git `2fae37a3`)
  (Source: `MLflow search_runs (ordered by metrics.accuracy DESC)`)
- Defensive Overfitting Audit:
  - Overfitting gap (Train F1 - Val F1): 0.082 (within allowed threshold <= 0.15).
  - Model registered in MLflow Model Registry under name `customer_churn_classifier` with alias `production`.

### UNCERTAINTIES / MISSING DATA
- None. Run parameters and serialized artifact signatures verified via MlflowClient.

### RECOMMENDED ACTIONS
- The model satisfies production quality gates (Accuracy >= 0.85, Overfitting Gap <= 0.15). Proceed with canary deployment.
```

---

## Scenario 3: Pipeline Execution & Airflow 3 Task Diagnosis

### Command:
```bash
python -m agent.main --message "What is the status of the ml_training_pipeline DAG and are there any failed tasks?"
```

### Verified Agent Response:
```markdown
### KNOWN FACTS (with tool + evidence source)
- DAG ID: `ml_training_pipeline`
- Task execution graph:
  `validate_dataset >> preprocess_data >> train_model >> evaluate_and_log_mlflow >> apply_quality_gate`
- Retries configured: 2 retries per task with 10s backoff. (Source: `dags/training_pipeline.py`)

### UNCERTAINTIES / MISSING DATA
- When Airflow standalone server is offline, agent plainly reports:
  `Airflow 3 orchestration status unavailable: Cannot reach Airflow at http://localhost:8080`
  (Agent does NOT fabricate task run IDs or execution dates).

### RECOMMENDED ACTIONS
- Launch local Airflow stack via `docker-compose up airflow` or `airflow standalone` to inspect live execution.
```

---

## Scenario 4: Deployment Diagnosis & Safety Enforcement (Kubernetes)

### Safe Query (Read-Only Diagnosis):
```bash
python -m agent.main --message "Diagnose the customer churn deployment in namespace mlops-agent"
```

### Mutating Action Without Confirmation (REJECTED):
```bash
python -m agent.main --message "Restart the deployment churn-inference-deployment in namespace mlops-agent"
```
**Result:** Rejected with `CONFIRMATION_REQUIRED`:
> *"Cluster mutation 'restart' in namespace 'mlops-agent' requires explicit user confirmation. Please resubmit the request with confirm=true."*

### Mutating Action With Explicit Confirmation (AUTHORIZED):
```bash
python -m agent.main --message "Restart the deployment churn-inference-deployment in namespace mlops-agent" --confirm
```
**Result:** Rollout restart triggered via `patch_namespaced_deployment` annotation update.

---

## Scenario 5: Mandatory Integrated Multi-Tool Reasoning

### Command:
```bash
python -m agent.main --message "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```

### Real Agent Output:
```markdown
### KNOWN FACTS (with tool + evidence source)
- Best MLflow model is 'GradientBoostingClassifier_v2.0_trees150_depth6' with Accuracy 0.8960 / F1-score 0.4179. (Source: `MLflow search_runs (ordered by metrics.accuracy DESC)`)
- Dataset version used for best model: v2.0 (Git 2fae37a3). (Source: `MLflow Run Tags (run_id=6a8834dad1784649a5a544782003b7f3)`)
- Current DVC dataset version in repository: v2.0 (Commit 2fae37a3). (Source: `dvc_get_dataset_version`)

### UNCERTAINTIES / MISSING DATA
- Kubernetes cluster status unreachable: No valid kubeconfig found and in-cluster config is disabled.
- Airflow 3 orchestration status unavailable: Cannot reach Airflow at http://localhost:8080

### RECOMMENDED ACTIONS
- NO EVIDENCE RETRAINING IS NECESSARY: The candidate model achieves Accuracy >= 0.85 with no significant overfitting gap (gap = 0.082).
- Conditions that would trigger retraining:
  (a) production model age > 30 days,
  (b) verified data drift / concept drift in incoming customer churn features, or
  (c) failed validation quality gates.
```
