#!/usr/bin/env bash
# ==============================================================================
# Kubernetes Cluster Setup & Deployment Script (kind / minikube)
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================
set -euo pipefail

CLUSTER_NAME="mlops-cluster"

echo "==> Step 1: Checking kind installation..."
if ! command -v kind &> /dev/null; then
    echo "kind is not installed. Please install kind from https://kind.sigs.k8s.io/"
    exit 1
fi

echo "==> Step 2: Creating kind Kubernetes cluster: ${CLUSTER_NAME}..."
if ! kind get clusters | grep -q "${CLUSTER_NAME}"; then
    kind create cluster --name "${CLUSTER_NAME}"
else
    echo "Cluster '${CLUSTER_NAME}' already exists."
fi

echo "==> Step 3: Building inference service Docker image..."
docker build -t churn-inference:1.0.0 .

echo "==> Step 4: Loading image into kind cluster..."
kind load docker-image churn-inference:1.0.0 --name "${CLUSTER_NAME}"

echo "==> Step 5: Applying namespace and RBAC..."
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/rbac.yaml

echo "==> Step 6: Applying ConfigMap, Deployment, and Service..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

echo "==> Step 7: Waiting for deployment rollout..."
kubectl rollout status deployment/churn-inference-deployment -n mlops-agent --timeout=120s

echo "==> Step 8: Cluster verification:"
kubectl get pods -n mlops-agent -o wide
kubectl get svc -n mlops-agent

echo "==> Kubernetes deployment completed successfully!"

