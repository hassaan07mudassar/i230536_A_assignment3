"""Kubernetes Operations and Automated Cluster Diagnostics Tools.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from agent.config import settings
from agent.logging_config import logger, redact_secrets
from agent.permissions import validate_k8s_mutation
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
    ToolError,
    ToolEvidence,
    ToolResult,
)


def _init_k8s_clients() -> Tuple[Optional[client.CoreV1Api], Optional[client.AppsV1Api], Optional[ToolError]]:
    """Initializes official Kubernetes API clients with in-cluster / kubeconfig fallback."""
    try:
        if settings.kubernetes_in_cluster:
            config.load_incluster_config()
        elif settings.kubeconfig_path and os.path.exists(settings.kubeconfig_path):
            config.load_kube_config(config_file=settings.kubeconfig_path)
        else:
            try:
                config.load_kube_config()
            except Exception:
                return None, None, ToolError(
                    code="K8S_CONFIG_NOT_FOUND",
                    message="No valid kubeconfig found and in-cluster config is disabled.",
                )

        core_api = client.CoreV1Api()
        apps_api = client.AppsV1Api()
        return core_api, apps_api, None
    except Exception as e:
        return None, None, ToolError(
            code="K8S_INIT_FAILED",
            message=f"Failed to initialize Kubernetes client: {str(e)}",
        )


# ------------------------------------------------------------------------------
# Tool 1: k8s_list_pods
# ------------------------------------------------------------------------------
def k8s_list_pods(input_data: K8sListPodsInput) -> ToolResult:
    """Lists pods in the specified namespace with status, phase, and restart counts."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_list_pods", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    ns = input_data.namespace
    try:
        pods = core_api.list_namespaced_pod(namespace=ns)
        pod_list = []
        for pod in pods.items:
            restart_count = sum(cs.restart_count for cs in (pod.status.container_statuses or []))
            ready_count = sum(1 for cs in (pod.status.container_statuses or []) if cs.ready)
            total_containers = len(pod.spec.containers)

            pod_list.append({
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "ready": f"{ready_count}/{total_containers}",
                "restarts": restart_count,
                "pod_ip": pod.status.pod_ip,
                "node_name": pod.spec.node_name,
                "labels": pod.metadata.labels,
            })

        evidence = ToolEvidence(
            source=f"Kubernetes CoreV1Api (Namespace: {ns})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Listed {len(pod_list)} pods in namespace '{ns}'.",
        )
        return ToolResult(ok=True, data={"namespace": ns, "pods": pod_list, "total": len(pod_list)}, evidence=evidence)
    except ApiException as e:
        return ToolResult(
            ok=False,
            error=ToolError(code=f"K8S_API_ERROR_{e.status}", message=f"Kubernetes API error: {e.reason}"),
            evidence=ToolEvidence(source="k8s_list_pods", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="K8S_CONNECTION_FAILED", message=f"Cluster connection error: {str(e)}"),
            evidence=ToolEvidence(source="k8s_list_pods", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 2: k8s_list_deployments
# ------------------------------------------------------------------------------
def k8s_list_deployments(input_data: K8sListDeploymentsInput) -> ToolResult:
    """Lists deployments in namespace with desired vs available replica counts."""
    _, apps_api, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_list_deployments", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    ns = input_data.namespace
    try:
        deps = apps_api.list_namespaced_deployment(namespace=ns)
        deployment_list = []
        for d in deps.items:
            images = [c.image for c in d.spec.template.spec.containers]
            deployment_list.append({
                "name": d.metadata.name,
                "replicas_desired": d.spec.replicas,
                "replicas_ready": d.status.ready_replicas or 0,
                "replicas_available": d.status.available_replicas or 0,
                "images": images,
                "labels": d.metadata.labels,
            })

        evidence = ToolEvidence(
            source=f"Kubernetes AppsV1Api (Namespace: {ns})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Listed {len(deployment_list)} deployments in namespace '{ns}'.",
        )
        return ToolResult(ok=True, data={"namespace": ns, "deployments": deployment_list}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="K8S_DEPLOYMENTS_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_list_deployments", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 3: k8s_get_pod_status
# ------------------------------------------------------------------------------
def k8s_get_pod_status(input_data: K8sGetPodStatusInput) -> ToolResult:
    """Inspects detailed lifecycle status, readiness conditions, and container states."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_get_pod_status", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    pod_name = input_data.pod_name
    ns = input_data.namespace
    try:
        pod = core_api.read_namespaced_pod(name=pod_name, namespace=ns)
        statuses = []
        for cs in (pod.status.container_statuses or []):
            state_dict = {}
            if cs.state.running:
                state_dict = {"state": "running", "started_at": str(cs.state.running.started_at)}
            elif cs.state.waiting:
                state_dict = {"state": "waiting", "reason": cs.state.waiting.reason, "message": cs.state.waiting.message}
            elif cs.state.terminated:
                state_dict = {"state": "terminated", "reason": cs.state.terminated.reason, "exit_code": cs.state.terminated.exit_code}

            statuses.append({
                "container": cs.name,
                "ready": cs.ready,
                "restart_count": cs.restart_count,
                "state": state_dict,
                "image": cs.image,
            })

        data = {
            "name": pod_name,
            "namespace": ns,
            "phase": pod.status.phase,
            "containers": statuses,
            "conditions": [
                {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
                for c in (pod.status.conditions or [])
            ],
        }

        evidence = ToolEvidence(
            source=f"Kubernetes read_namespaced_pod({pod_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Pod {pod_name} is in phase {pod.status.phase}.",
        )
        return ToolResult(ok=True, data=data, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="POD_NOT_FOUND", message=str(e)),
            evidence=ToolEvidence(source="k8s_get_pod_status", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 4: k8s_get_pod_logs
# ------------------------------------------------------------------------------
def k8s_get_pod_logs(input_data: K8sGetPodLogsInput) -> ToolResult:
    """Tails container logs with automatic secret redaction."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_get_pod_logs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    pod_name = input_data.pod_name
    ns = input_data.namespace
    try:
        raw_logs = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=ns,
            tail_lines=input_data.tail_lines,
            previous=input_data.previous,
        )
        cleaned_logs = redact_secrets(raw_logs)

        evidence = ToolEvidence(
            source=f"Kubernetes read_namespaced_pod_log({pod_name}, tail={input_data.tail_lines})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=cleaned_logs[-250:] if len(cleaned_logs) > 250 else cleaned_logs,
        )
        return ToolResult(
            ok=True,
            data={"pod_name": pod_name, "namespace": ns, "tail_lines": input_data.tail_lines, "logs": cleaned_logs},
            evidence=evidence,
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="POD_LOG_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_get_pod_logs", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 5: k8s_list_services
# ------------------------------------------------------------------------------
def k8s_list_services(input_data: K8sListServicesInput) -> ToolResult:
    """Lists services, cluster IPs, and port configurations."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_list_services", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    ns = input_data.namespace
    try:
        svcs = core_api.list_namespaced_service(namespace=ns)
        svc_list = [
            {
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "ports": [{"port": p.port, "target_port": str(p.target_port)} for p in s.spec.ports],
                "selector": s.spec.selector,
            }
            for s in svcs.items
        ]

        evidence = ToolEvidence(
            source=f"Kubernetes CoreV1Api list_namespaced_service({ns})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Found {len(svc_list)} services in {ns}.",
        )
        return ToolResult(ok=True, data={"namespace": ns, "services": svc_list}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="SERVICE_LIST_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_list_services", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 6: k8s_get_events
# ------------------------------------------------------------------------------
def k8s_get_events(input_data: K8sGetEventsInput) -> ToolResult:
    """Fetches warning and normal events in the namespace."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_get_events", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    ns = input_data.namespace
    try:
        events = core_api.list_namespaced_event(namespace=ns)
        event_list = [
            {
                "type": e.type,
                "reason": e.reason,
                "message": e.message,
                "object": f"{e.involved_object.kind}/{e.involved_object.name}",
                "count": e.count,
                "last_timestamp": str(e.last_timestamp),
            }
            for e in events.items[: input_data.limit]
        ]

        evidence = ToolEvidence(
            source=f"Kubernetes Events ({ns})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Retrieved {len(event_list)} events in {ns}.",
        )
        return ToolResult(ok=True, data={"namespace": ns, "events": event_list}, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="EVENTS_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_get_events", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 7: k8s_find_unhealthy_pods
# ------------------------------------------------------------------------------
def k8s_find_unhealthy_pods(input_data: K8sFindUnhealthyPodsInput) -> ToolResult:
    """Detects pods in CrashLoopBackOff, ImagePullBackOff, NotReady, or with high restarts."""
    core_api, _, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_find_unhealthy_pods", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    ns = input_data.namespace
    try:
        pods = core_api.list_namespaced_pod(namespace=ns)
        unhealthy = []
        for pod in pods.items:
            issues = []
            restart_count = sum(cs.restart_count for cs in (pod.status.container_statuses or []))

            # Container waiting state checks
            for cs in (pod.status.container_statuses or []):
                if cs.state.waiting:
                    reason = cs.state.waiting.reason
                    if reason in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "ContainerCreating"):
                        issues.append(f"Container '{cs.name}' in {reason}: {cs.state.waiting.message}")
                if cs.last_state.terminated:
                    term = cs.last_state.terminated
                    if term.reason in ("OOMKilled", "Error") or term.exit_code != 0:
                        issues.append(f"Container '{cs.name}' terminated with {term.reason} (exit code: {term.exit_code})")

            # Readiness check
            ready_containers = sum(1 for cs in (pod.status.container_statuses or []) if cs.ready)
            total_containers = len(pod.spec.containers)
            if ready_containers < total_containers and pod.status.phase == "Running":
                issues.append(f"Readiness probe not passing ({ready_containers}/{total_containers} ready)")

            if restart_count > 3:
                issues.append(f"High restart count: {restart_count} restarts")

            if issues:
                unhealthy.append({
                    "pod_name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "restarts": restart_count,
                    "issues": issues,
                })

        evidence = ToolEvidence(
            source=f"Kubernetes Pod Health Audit ({ns})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Found {len(unhealthy)} unhealthy pod(s) in {ns}.",
        )
        return ToolResult(
            ok=True,
            data={"namespace": ns, "unhealthy_pods": unhealthy, "unhealthy_count": len(unhealthy)},
            evidence=evidence,
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="POD_HEALTH_AUDIT_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_find_unhealthy_pods", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 8: k8s_get_deployed_version
# ------------------------------------------------------------------------------
def k8s_get_deployed_version(input_data: K8sGetDeployedVersionInput) -> ToolResult:
    """Extracts image tag, app version label, and verifies model version from Deployment."""
    _, apps_api, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_get_deployed_version", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    dep_name = input_data.deployment_name
    ns = input_data.namespace
    try:
        dep = apps_api.read_namespaced_deployment(name=dep_name, namespace=ns)
        containers = dep.spec.template.spec.containers
        image = containers[0].image if containers else "none"
        labels = dep.metadata.labels or {}
        annotations = dep.metadata.annotations or {}

        # Parse image tag
        image_tag = image.split(":")[-1] if ":" in image else "latest"

        data = {
            "deployment_name": dep_name,
            "namespace": ns,
            "image": image,
            "image_tag": image_tag,
            "version_label": labels.get("version", "unknown"),
            "replicas_desired": dep.spec.replicas,
            "replicas_ready": dep.status.ready_replicas or 0,
        }

        evidence = ToolEvidence(
            source=f"Kubernetes Deployment ({dep_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Image: {image} | Version label: {data['version_label']} | Ready: {data['replicas_ready']}/{data['replicas_desired']}",
        )
        return ToolResult(ok=True, data=data, evidence=evidence)
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="DEPLOYMENT_VERSION_ERROR", message=str(e)),
            evidence=ToolEvidence(source="k8s_get_deployed_version", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 9: k8s_diagnose_service
# ------------------------------------------------------------------------------
def k8s_diagnose_service(input_data: K8sDiagnoseServiceInput) -> ToolResult:
    """Correlates deployment, pods, probes, events, and logs into an automated diagnosis."""
    dep_name = input_data.deployment_name
    ns = input_data.namespace

    # Gather evidence across all sub-resources
    dep_ver = k8s_get_deployed_version(K8sGetDeployedVersionInput(deployment_name=dep_name, namespace=ns))
    unhealthy = k8s_find_unhealthy_pods(K8sFindUnhealthyPodsInput(namespace=ns))
    events = k8s_get_events(K8sGetEventsInput(namespace=ns, limit=10))

    if not dep_ver.ok:
        return ToolResult(
            ok=False,
            error=dep_ver.error,
            evidence=ToolEvidence(source="k8s_diagnose_service", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

    is_healthy = dep_ver.data["replicas_ready"] == dep_ver.data["replicas_desired"] and (
        not unhealthy.ok or unhealthy.data.get("unhealthy_count", 0) == 0
    )

    diagnosis_findings = []
    if is_healthy:
        diagnosis_findings.append(f"Deployment '{dep_name}' is healthy: all {dep_ver.data['replicas_desired']} pods ready.")
    else:
        diagnosis_findings.append(f"Deployment '{dep_name}' is DEGRADED: {dep_ver.data['replicas_ready']}/{dep_ver.data['replicas_desired']} pods ready.")
        if unhealthy.ok and unhealthy.data.get("unhealthy_pods"):
            for up in unhealthy.data["unhealthy_pods"]:
                for iss in up["issues"]:
                    diagnosis_findings.append(f"Issue on pod {up['pod_name']}: {iss}")

    data = {
        "deployment": dep_name,
        "namespace": ns,
        "is_healthy": is_healthy,
        "replicas_status": f"{dep_ver.data['replicas_ready']}/{dep_ver.data['replicas_desired']}",
        "image": dep_ver.data["image"],
        "findings": diagnosis_findings,
        "recent_events": events.data.get("events", []) if events.ok else [],
    }

    evidence = ToolEvidence(
        source="k8s_diagnose_service (multi-resource correlation)",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_output="; ".join(diagnosis_findings),
    )
    return ToolResult(ok=True, data=data, evidence=evidence)


# ------------------------------------------------------------------------------
# Tool 10: k8s_restart_deployment
# ------------------------------------------------------------------------------
def k8s_restart_deployment(input_data: K8sRestartDeploymentInput) -> ToolResult:
    """Executes a rolling restart on deployment. MUTATING: Requires confirm=True & allowed namespace."""
    perm_err = validate_k8s_mutation(action="restart", namespace=input_data.namespace, confirm=input_data.confirm)
    if perm_err:
        return ToolResult(ok=False, error=perm_err, evidence=ToolEvidence(source="security_policy_enforcer", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    _, apps_api, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_restart_deployment", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    dep_name = input_data.deployment_name
    ns = input_data.namespace
    try:
        # Standard Kubernetes rollout restart: update restart annotation
        now_str = datetime.utcnow().isoformat() + "Z"
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now_str
                        }
                    }
                }
            }
        }
        apps_api.patch_namespaced_deployment(name=dep_name, namespace=ns, body=patch_body)
        evidence = ToolEvidence(
            source=f"Kubernetes AppsV1Api.patch_namespaced_deployment({dep_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Deployment '{dep_name}' rollout restart triggered at {now_str}.",
        )
        return ToolResult(
            ok=True,
            data={"deployment": dep_name, "namespace": ns, "status": "RESTART_TRIGGERED", "timestamp": now_str},
            evidence=evidence,
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="RESTART_FAILED", message=str(e)),
            evidence=ToolEvidence(source="k8s_restart_deployment", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


# ------------------------------------------------------------------------------
# Tool 11: k8s_scale_deployment
# ------------------------------------------------------------------------------
def k8s_scale_deployment(input_data: K8sScaleDeploymentInput) -> ToolResult:
    """Scales deployment replicas. MUTATING: Requires confirm=True & allowed namespace."""
    perm_err = validate_k8s_mutation(action="scale", namespace=input_data.namespace, confirm=input_data.confirm)
    if perm_err:
        return ToolResult(ok=False, error=perm_err, evidence=ToolEvidence(source="security_policy_enforcer", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    _, apps_api, err = _init_k8s_clients()
    if err:
        return ToolResult(ok=False, error=err, evidence=ToolEvidence(source="k8s_scale_deployment", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")))

    dep_name = input_data.deployment_name
    ns = input_data.namespace
    replicas = input_data.replicas
    try:
        scale_patch = {"spec": {"replicas": replicas}}
        apps_api.patch_namespaced_deployment_scale(name=dep_name, namespace=ns, body=scale_patch)

        evidence = ToolEvidence(
            source=f"Kubernetes AppsV1Api.patch_namespaced_deployment_scale({dep_name})",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            raw_output=f"Deployment '{dep_name}' scaled to {replicas} replicas.",
        )
        return ToolResult(
            ok=True,
            data={"deployment": dep_name, "namespace": ns, "desired_replicas": replicas, "status": "SCALED"},
            evidence=evidence,
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(code="SCALE_FAILED", message=str(e)),
            evidence=ToolEvidence(source="k8s_scale_deployment", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )

