#!/usr/bin/env bash
# ==============================================================================
# Comprehensive Live Demonstration Script: 5 Core Scenarios
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo " MLOps AI Platform Agent - Live Automated Demonstration"
echo " Student: Hassaan Mudassar | Roll No: i230536 | Section: CS-A"
echo "======================================================================"

echo ""
echo ">>> [SCENARIO 1: Dataset Provenance & Reproducibility (DVC)]"
python -m agent.main --message "What is the current version and hash of the customer churn dataset, and how do I reproduce version v1.0?"

echo ""
echo ">>> [SCENARIO 2: Model Selection & Defensive ML Intelligence (MLflow)]"
python -m agent.main --message "Which is the best performing model in experiment customer_churn_prediction and is it production-ready?"

echo ""
echo ">>> [SCENARIO 3: Pipeline Execution & Airflow 3 Task Diagnosis]"
python -m agent.main --message "What is the status of the ml_training_pipeline DAG and are there any failed tasks?"

echo ""
echo ">>> [SCENARIO 4: Kubernetes Deployment Diagnosis & Safety Enforcement]"
python -m agent.main --message "Diagnose the customer churn deployment in namespace mlops-agent"

echo ""
echo ">>> [SCENARIO 4B: Attempting State-Changing Action Without Confirmation (Safety Check)]"
python -m agent.main --message "Restart the deployment churn-inference-deployment in namespace mlops-agent"

echo ""
echo ">>> [SCENARIO 5: Mandatory Integrated Multi-Tool Reasoning Across All 4 Domains]"
python -m agent.main --message "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

echo ""
echo "======================================================================"
echo " Demonstration finished successfully."
echo "======================================================================"

