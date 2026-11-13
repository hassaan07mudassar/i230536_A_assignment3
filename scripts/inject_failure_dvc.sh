#!/usr/bin/env bash
# ==============================================================================
# Failure Injection Scenario 3: Untracked Data Corruption / Drift
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [FAILURE INJECTION] Appending corrupted outlier rows to customer_churn.csv..."
echo "CORRUPT_99999,-999,9,9999.99,-500.00,0,0,0,1" >> data/raw/customer_churn.csv
echo "==> Corrupted rows added. DVC hash divergence introduced."
echo "==> Verify with agent: 'Check if working tree data has changed with dvc status'"

