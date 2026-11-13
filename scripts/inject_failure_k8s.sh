#!/usr/bin/env bash
# ==============================================================================
# Failure Injection Scenario 1: Kubernetes ImagePullBackOff / Broken Tag
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

echo "==> [FAILURE INJECTION] Updating Kubernetes Deployment with non-existent image tag..."
kubectl set image deployment/churn-inference-deployment churn-inference=churn-inference:broken-v999 -n mlops-agent || true

echo "==> Injected failure: Pods will enter ImagePullBackOff / ErrImagePull."
echo "==> Verify with agent query: 'Diagnose the deployment churn-inference-deployment in namespace mlops-agent'"

