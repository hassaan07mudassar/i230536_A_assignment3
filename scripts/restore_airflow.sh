#!/usr/bin/env bash
# ==============================================================================
# Restoration Script: Restore Dataset for Airflow Pipeline
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [RESTORATION] Restoring dataset from backup or via DVC checkout..."
if [ -f "data/raw/customer_churn.csv.bak" ]; then
    mv data/raw/customer_churn.csv.bak data/raw/customer_churn.csv
    echo "==> Restored customer_churn.csv from backup."
else
    dvc checkout data/raw/customer_churn.csv.dvc
    echo "==> Restored customer_churn.csv via DVC checkout."
fi

