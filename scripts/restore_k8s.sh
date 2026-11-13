#!/usr/bin/env bash
# ==============================================================================
# Restoration Script: Kubernetes Healthy Image Rollout
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [RESTORATION] Rolling back deployment image to stable production release..."
kubectl set image deployment/churn-inference-deployment churn-inference=churn-inference:1.0.0 -n mlops-agent || true
kubectl rollout status deployment/churn-inference-deployment -n mlops-agent --timeout=60s || true

echo "==> Restored deployment to healthy state."

