#!/usr/bin/env bash
# ==============================================================================
# Failure Injection Scenario 2: Missing Dataset Validation Failure in Airflow
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [FAILURE INJECTION] Renaming dataset to trigger validation failure in Airflow DAG..."
if [ -f "data/raw/customer_churn.csv" ]; then
    mv data/raw/customer_churn.csv data/raw/customer_churn.csv.bak
    echo "==> Dataset moved to data/raw/customer_churn.csv.bak"
    echo "==> Next Airflow run will fail at validate_dataset task."
else
    echo "Dataset already moved or missing."
fi

