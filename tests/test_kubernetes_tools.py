"""Tests for Kubernetes Operations, Diagnostics, and State Mutations.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from unittest.mock import MagicMock

from agent.schemas import (
    K8sDiagnoseServiceInput,
    K8sFindUnhealthyPodsInput,
    K8sGetDeployedVersionInput,
    K8sGetEventsInput,
    K8sGetPodLogsInput,
    K8sGetPodStatusInput,
    K8sListDeploymentsInput,
    K8sListPodsInput,
    K8sListServicesInput,
    K8sRestartDeploymentInput,
    K8sScaleDeploymentInput,
)
from agent.tools.kubernetes_tools import (
    k8s_diagnose_service,
    k8s_find_unhealthy_pods,
    k8s_get_deployed_version,
    k8s_get_events,
    k8s_get_pod_logs,
    k8s_get_pod_status,
    k8s_list_deployments,
    k8s_list_pods,
    k8s_list_services,
    k8s_restart_deployment,
    k8s_scale_deployment,
)


def test_k8s_list_pods(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    mock_pod = MagicMock()
    mock_pod.metadata.name = "churn-inference-776f8b8-x1"
    mock_pod.status.phase = "Running"
    mock_pod.status.pod_ip = "10.244.0.5"
    mock_pod.spec.node_name = "kind-control-plane"
    mock_pod.metadata.labels = {"app": "churn-inference"}
    cs = MagicMock()
    cs.ready = True
    cs.restart_count = 0
    mock_pod.status.container_statuses = [cs]
    mock_pod.spec.containers = [MagicMock()]
    mock_core.list_namespaced_pod.return_value.items = [mock_pod]

    res = k8s_list_pods(K8sListPodsInput(namespace="mlops-agent"))
    assert res.ok is True
    assert res.data["total"] == 1
    assert res.data["pods"][0]["name"] == "churn-inference-776f8b8-x1"


def test_k8s_list_deployments(mock_k8s_apis):
    _, mock_apps = mock_k8s_apis
    mock_dep = MagicMock()
    mock_dep.metadata.name = "churn-inference-deployment"
    mock_dep.spec.replicas = 2
    mock_dep.status.ready_replicas = 2
    mock_dep.status.available_replicas = 2
    mock_container = MagicMock()
    mock_container.image = "churn-inference:1.0.0"
    mock_dep.spec.template.spec.containers = [mock_container]
    mock_dep.metadata.labels = {"app": "churn-inference"}
    mock_apps.list_namespaced_deployment.return_value.items = [mock_dep]

    res = k8s_list_deployments(K8sListDeploymentsInput(namespace="mlops-agent"))
    assert res.ok is True
    assert len(res.data["deployments"]) == 1
    assert res.data["deployments"][0]["replicas_ready"] == 2


def test_k8s_get_pod_status(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    mock_pod = MagicMock()
    mock_pod.status.phase = "Running"
    cs = MagicMock()
    cs.name = "inference"
    cs.ready = True
    cs.restart_count = 0
    cs.state.running = MagicMock()
    cs.state.running.started_at = "2026-11-13T10:00:00Z"
    cs.state.waiting = None
    cs.state.terminated = None
    cs.image = "churn-inference:1.0.0"
    mock_pod.status.container_statuses = [cs]
    cond = MagicMock()
    cond.type = "Ready"
    cond.status = "True"
    cond.reason = None
    cond.message = None
    mock_pod.status.conditions = [cond]
    mock_core.read_namespaced_pod.return_value = mock_pod

    res = k8s_get_pod_status(K8sGetPodStatusInput(pod_name="pod-1", namespace="mlops-agent"))
    assert res.ok is True
    assert res.data["phase"] == "Running"
    assert res.data["containers"][0]["ready"] is True


def test_k8s_get_pod_logs(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    mock_core.read_namespaced_pod_log.return_value = "Starting Uvicorn server on port 8000 with token secret_token_xyz"

    res = k8s_get_pod_logs(K8sGetPodLogsInput(pod_name="pod-1", namespace="mlops-agent", tail_lines=20))
    assert res.ok is True
    assert "Uvicorn" in res.data["logs"]
    assert "secret_token_xyz" not in res.data["logs"]
    assert "[REDACTED_SECRET]" in res.data["logs"]


def test_k8s_list_services(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    svc = MagicMock()
    svc.metadata.name = "churn-inference-service"
    svc.spec.type = "ClusterIP"
    svc.spec.cluster_ip = "10.96.100.25"
    p = MagicMock()
    p.port = 80
    p.target_port = 8000
    svc.spec.ports = [p]
    svc.spec.selector = {"app": "churn-inference"}
    mock_core.list_namespaced_service.return_value.items = [svc]

    res = k8s_list_services(K8sListServicesInput(namespace="mlops-agent"))
    assert res.ok is True
    assert len(res.data["services"]) == 1
    assert res.data["services"][0]["cluster_ip"] == "10.96.100.25"


def test_k8s_get_events(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    ev = MagicMock()
    ev.type = "Normal"
    ev.reason = "Scheduled"
    ev.message = "Successfully assigned mlops-agent/pod-1 to node-1"
    ev.involved_object.kind = "Pod"
    ev.involved_object.name = "pod-1"
    ev.count = 1
    ev.last_timestamp = "2026-11-13T10:00:00Z"
    mock_core.list_namespaced_event.return_value.items = [ev]

    res = k8s_get_events(K8sGetEventsInput(namespace="mlops-agent", limit=5))
    assert res.ok is True
    assert len(res.data["events"]) == 1


def test_k8s_find_unhealthy_pods_detects_crashloop(mock_k8s_apis):
    mock_core, _ = mock_k8s_apis
    mock_pod = MagicMock()
    mock_pod.metadata.name = "failing-pod-1"
    mock_pod.status.phase = "Running"
    cs = MagicMock()
    cs.name = "inference"
    cs.ready = False
    cs.restart_count = 5
    cs.state.waiting.reason = "CrashLoopBackOff"
    cs.state.waiting.message = "Back-off 5m restarting failed container"
    cs.last_state.terminated = None
    mock_pod.status.container_statuses = [cs]
    mock_pod.spec.containers = [MagicMock()]
    mock_core.list_namespaced_pod.return_value.items = [mock_pod]

    res = k8s_find_unhealthy_pods(K8sFindUnhealthyPodsInput(namespace="mlops-agent"))
    assert res.ok is True
    assert res.data["unhealthy_count"] == 1
    assert "CrashLoopBackOff" in res.data["unhealthy_pods"][0]["issues"][0]


def test_k8s_get_deployed_version(mock_k8s_apis):
    _, mock_apps = mock_k8s_apis
    dep = MagicMock()
    c = MagicMock()
    c.image = "churn-inference:1.0.0"
    dep.spec.template.spec.containers = [c]
    dep.metadata.labels = {"version": "1.0.0"}
    dep.spec.replicas = 2
    dep.status.ready_replicas = 2
    mock_apps.read_namespaced_deployment.return_value = dep

    res = k8s_get_deployed_version(K8sGetDeployedVersionInput(deployment_name="churn-inference-deployment", namespace="mlops-agent"))
    assert res.ok is True
    assert res.data["image_tag"] == "1.0.0"
    assert res.data["replicas_ready"] == 2


def test_k8s_diagnose_service_healthy(mock_k8s_apis):
    _, mock_apps = mock_k8s_apis
    mock_core, _ = mock_k8s_apis
    dep = MagicMock()
    c = MagicMock()
    c.image = "churn-inference:1.0.0"
    dep.spec.template.spec.containers = [c]
    dep.metadata.labels = {"version": "1.0.0"}
    dep.spec.replicas = 2
    dep.status.ready_replicas = 2
    mock_apps.read_namespaced_deployment.return_value = dep
    mock_core.list_namespaced_pod.return_value.items = []
    mock_core.list_namespaced_event.return_value.items = []

    res = k8s_diagnose_service(K8sDiagnoseServiceInput(deployment_name="churn-inference-deployment", namespace="mlops-agent"))
    assert res.ok is True
    assert res.data["is_healthy"] is True


def test_k8s_mutation_unconfirmed_rejected(mock_k8s_apis):
    inp = K8sScaleDeploymentInput(deployment_name="churn-inference-deployment", replicas=4, namespace="mlops-agent", confirm=False)
    res = k8s_scale_deployment(inp)
    assert res.ok is False
    assert res.error.code == "CONFIRMATION_REQUIRED"

    inp2 = K8sRestartDeploymentInput(deployment_name="churn-inference-deployment", namespace="mlops-agent", confirm=False)
    res2 = k8s_restart_deployment(inp2)
    assert res2.ok is False
    assert res2.error.code == "CONFIRMATION_REQUIRED"


def test_k8s_mutation_unauthorized_namespace_rejected(mock_k8s_apis):
    inp = K8sScaleDeploymentInput(deployment_name="kube-dns", replicas=0, namespace="kube-system", confirm=True)
    res = k8s_scale_deployment(inp)
    assert res.ok is False
    assert res.error.code == "NAMESPACE_NOT_ALLOWED"


def test_k8s_mutation_confirmed_success(mock_k8s_apis):
    _, mock_apps = mock_k8s_apis
    inp = K8sScaleDeploymentInput(deployment_name="churn-inference-deployment", replicas=3, namespace="mlops-agent", confirm=True)
    res = k8s_scale_deployment(inp)
    assert res.ok is True
    assert res.data["desired_replicas"] == 3
    mock_apps.patch_namespaced_deployment_scale.assert_called_once()


def test_k8s_restart_confirmed_success(mock_k8s_apis):
    _, mock_apps = mock_k8s_apis
    inp = K8sRestartDeploymentInput(deployment_name="churn-inference-deployment", namespace="mlops-agent", confirm=True)
    res = k8s_restart_deployment(inp)
    assert res.ok is True
    assert res.data["status"] == "RESTART_TRIGGERED"
    mock_apps.patch_namespaced_deployment.assert_called_once()
