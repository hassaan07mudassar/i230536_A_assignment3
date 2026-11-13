#!/usr/bin/env bash
# ==============================================================================
# Restoration Script: Restore Authentic DVC Dataset via Checkout
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [RESTORATION] Discarding uncommitted changes and checking out canonical DVC data..."
dvc checkout data/raw/customer_churn.csv.dvc --force
echo "==> Restored data/raw/customer_churn.csv to authentic cached state."

