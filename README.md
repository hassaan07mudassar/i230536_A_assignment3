# Production-Ready MLOps AI Platform Agent

[![CI Status](https://github.com/hassaan07mudassar/assignment-03-mlops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hassaan07mudassar/assignment-03-mlops-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Course:** MLOps & Production AI Engineering (CS-4061, Assignment 03, Fall 2026)  
**Student Name:** Hassaan Mudassar  
**Roll Number:** i230536  
**Section:** CS-A  

---

## 1. Overview & Non-Negotiable Engineering Principles

This repository contains an autonomous, evidence-driven AI platform agent that manages, diagnoses, and audits end-to-end Machine Learning Operations across four core platforms:
- **DVC:** Dataset versioning, pointer tracking, cryptographic hash validation, and bit-for-bit reproducible checkouts.
- **MLflow:** Multi-run experiment tracking, defensive overfitting analysis, artifact management, and Model Registry aliases (`production`, `candidate`).
- **Apache Airflow 3:** Orchestrated machine learning pipeline DAGs with quality gates and authenticated REST API (`/api/v2`) interactions.
- **Kubernetes:** Production container deployment, health probes (liveness/readiness), automated pod diagnostics, and safe, authorized rollout operations.

### Non-Negotiable Operational Principles
1. **Zero Hallucination:** The agent NEVER answers operational questions from pre-trained LLM memory. Every operational claim originates from real tool executions. If a backend is unavailable, the agent states so plainly without fabricating values.
2. **Secret-Free Repository:** No API keys, passwords, or kubeconfig secrets in the repository. Logs and tool traces feature automated regex masking (`[REDACTED_SECRET]`).
3. **Safety Authorization:** Any cluster-state-changing action (`restart`, `scale`, DAG trigger) requires explicit confirmation (`confirm=True`) and is restricted to approved namespace allowlists (`mlops-agent`, `default`).
4. **Defensive Retraining:** Retraining is NEVER automatically triggered from an analysis question. Recommendations require concrete evidence (model age expiry, concept drift, or failed quality gates).

---

## 2. Repository Structure

```
assignment-03-mlops-agent/
├── agent/                       # Agent core reasoning & execution engine
│   ├── config.py                # Environment configuration & allowlists
│   ├── llm_client.py            # LLM abstraction (Gemini 2.5 Flash + MockLLM)
│   ├── logging_config.py        # Structured JSON logger & secret redaction
│   ├── main.py                  # FastAPI agent gateway & interactive CLI
│   ├── orchestrator.py          # Bounded tool-calling loop (max 8 steps)
│   ├── permissions.py           # Safety authorization engine
│   ├── schemas.py               # Strict Pydantic I/O models
│   └── tools/                   # Multi-domain tool implementations (30 tools)
│       ├── airflow_tools.py     # Airflow 3 REST client tools (6)
│       ├── dvc_tools.py         # DVC subprocess allowlist tools (6)
│       ├── kubernetes_tools.py  # Kubernetes Core & Apps API tools (11)
│       └── mlflow_tools.py      # MLflow tracking & registry tools (7)
├── api/                         # Production inference service deployed to Kubernetes
│   └── main.py                  # FastAPI app (/health, /ready, /predict, /version)
├── dags/                        # Apache Airflow 3 pipelines
│   └── training_pipeline.py     # 5-stage training DAG with quality gate
├── data/                        # DVC-tracked dataset storage
│   └── raw/customer_churn.csv.dvc # Pointer metadata committed to Git (raw data ignored)
├── kubernetes/                  # Kubernetes production manifests
│   ├── configmap.yaml           # Service configuration
│   ├── deployment.yaml          # RollingUpdate deployment with probes & limits
│   ├── namespace.yaml           # Dedicated mlops-agent namespace
│   ├── rbac.yaml                # Least-privilege ServiceAccount, Role, RoleBinding
│   └── service.yaml             # ClusterIP service
├── screenshots/                 # Captured authentic command evidence & exports
├── scripts/                     # Cluster setup, run generation, and failure injections
│   ├── demo.sh                  # 5-scenario live demonstration runner
│   ├── export_evidence.py       # Programmatic evidence collection
│   ├── generate_runs.py         # Generates v1.0/v2.0 data & 11 MLflow runs
│   ├── register_models.py       # Model Registry promotion
│   ├── setup_cluster.sh         # kind cluster creation and deployment
│   ├── inject_failure_k8s.sh    # Injects ImagePullBackOff
│   ├── restore_k8s.sh           # Restores healthy pod deployment
│   ├── inject_failure_airflow.sh# Injects validation failure
│   ├── restore_airflow.sh       # Restores dataset
│   ├── inject_failure_dvc.sh    # Injects untracked data drift
│   └── restore_dvc.sh           # Restores canonical DVC data
├── tests/                       # Complete offline test suite (67 tests, 82% coverage)
├── .github/workflows/ci.yml     # GitHub Actions automated lint & test pipeline
├── docker-compose.yml           # Local multi-service stack (MLflow + Airflow 3 + Agent API)
├── Dockerfile                   # Multi-stage container for inference microservice
├── dvc.yaml                     # DVC pipeline specification
├── params.yaml                  # Model hyperparameters and quality gate thresholds
├── pyproject.toml               # Pinned package dependencies
├── README.md                    # System documentation
├── architecture.md              # Architectural diagrams and defense Q&A
└── REPORT.md                    # Official technical evaluation report
```

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git & DVC
- Docker Desktop (with kind or minikube for local cluster execution)

### 3.2 Installation
```bash
# Clone the repository
git clone https://github.com/hassaan07mudassar/assignment-03-mlops-agent.git
cd assignment-03-mlops-agent

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies in editable mode with development tools
pip install -e ".[dev]"
```

### 3.3 Environment Configuration (`.env`)
Copy the provided template:
```bash
cp .env.example .env
```
Key configuration settings:
- `GEMINI_API_KEY`: Set your Google Gemini API key for live LLM mode.
- `MLFLOW_TRACKING_URI`: `sqlite:///mlflow.db` (local SQLite tracking store) or `http://localhost:5000`.
- `AIRFLOW_BASE_URL`: `http://localhost:8080`.
- `ALLOWED_NAMESPACES`: `mlops-agent,default`.
- `ALLOWED_DAG_IDS`: `ml_training_pipeline`.
- `ALLOWED_K8S_ACTIONS`: `restart,scale`.

---

## 4. How to Launch and Verify Each Component

### 4.1 Step 1: DVC Dataset & MLflow Experiment Runs
Generate dataset versions `v1.0` and `v2.0`, commit Git tags, and execute 11 training runs with diverse hyperparameters:
```bash
python scripts/generate_runs.py
python scripts/register_models.py
```
Verify DVC and Git tags:
```bash
git tag
# Output: v1.0, v2.0
dvc status
```

### 4.2 Step 2: Launch Local Stack via Docker Compose
To launch MLflow tracking server and Airflow 3:
```bash
docker compose up -d
```
- MLflow UI: `http://localhost:5000`
- Airflow UI: `http://localhost:8080` (admin / admin)

### 4.3 Step 3: Kubernetes Cluster Deployment (kind)
Deploy the inference microservice into kind:
```bash
bash scripts/setup_cluster.sh
```
Or apply manifests manually:
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/rbac.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```
Verify pod health:
```bash
kubectl get pods -n mlops-agent -o wide
```

### 4.4 Step 4: Run the MLOps AI Agent

#### Interactive CLI Console:
```bash
python -m agent.main
```
#### Single Command Execution:
```bash
python -m agent.main --message "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```

#### Launch FastAPI Agent REST Gateway:
```bash
python -m agent.main --serve
# API available at http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

---

## 5. Running Tests & Verifying Safety Guardrails

The repository includes a 67-test pytest suite runnable completely offline using mock backends:
```bash
# Run tests with coverage report
pytest --cov=agent --cov=api --cov-report=term-missing tests/

# Run Ruff linter
ruff check .
```

---

## 6. Deliberate Failure Injection Scripts

Three automated failure scenarios test the diagnostic capabilities of the agent:
```bash
# Scenario A: Kubernetes broken image tag
bash scripts/inject_failure_k8s.sh
python -m agent.main --message "Diagnose the deployment churn-inference-deployment in namespace mlops-agent"
bash scripts/restore_k8s.sh

# Scenario B: Airflow dataset missing
bash scripts/inject_failure_airflow.sh
bash scripts/restore_airflow.sh

# Scenario C: DVC untracked data drift
bash scripts/inject_failure_dvc.sh
python -m agent.main --message "Check if working tree data has changed with dvc status"
bash scripts/restore_dvc.sh
```

---

## 7. Submission Artifacts Checklist
- `architecture.md`: Architecture diagrams, tool catalog, and Defense Q&A.
- `REPORT.md`: Comprehensive evaluation report with evidence tables and screenshot checklist.
- `DEMO.md`: Runbook covering the 5 live demonstration scenarios.
- `screenshots/`: 8 real captured command outputs and evidence logs.

