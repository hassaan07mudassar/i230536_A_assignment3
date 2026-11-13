"""Agent Configuration and Security Policies.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

from typing import Set
from pydantic import Field
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    # LLM Settings
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # Tool Backends
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_TRACKING_URI")
    airflow_base_url: str = Field(default="http://localhost:8080", env="AIRFLOW_BASE_URL")
    airflow_username: str = Field(default="admin", env="AIRFLOW_USERNAME")
    airflow_password: str = Field(default="admin", env="AIRFLOW_PASSWORD")
    airflow_api_timeout: int = Field(default=10, env="AIRFLOW_API_TIMEOUT_SECONDS")

    # Kubernetes Settings
    kubernetes_namespace: str = Field(default="mlops-agent", env="KUBERNETES_NAMESPACE")
    kubeconfig_path: str = Field(default="", env="KUBECONFIG_PATH")
    kubernetes_in_cluster: bool = Field(default=False, env="KUBERNETES_IN_CLUSTER")

    # Raw string inputs from environment
    allowed_namespaces_raw: str = Field(default="mlops-agent,default", env="ALLOWED_NAMESPACES")
    allowed_dag_ids_raw: str = Field(default="ml_training_pipeline", env="ALLOWED_DAG_IDS")
    allowed_k8s_actions_raw: str = Field(default="restart,scale", env="ALLOWED_K8S_ACTIONS")

    # Retraining & Quality Governance
    max_model_age_days: int = Field(default=30, env="MAX_MODEL_AGE_DAYS")
    quality_gate_min_f1: float = Field(default=0.80, env="QUALITY_GATE_MIN_F1")

    # Service & Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    service_name: str = Field(default="mlops-agent", env="SERVICE_NAME")

    # Strict DVC Subprocess Command Allowlist
    allowed_dvc_subcommands: Set[str] = {
        "status",
        "diff",
        "list",
        "checkout",
        "pull",
        "params diff",
        "metrics diff",
    }

    @property
    def allowed_namespaces(self) -> Set[str]:
        return {ns.strip() for ns in self.allowed_namespaces_raw.split(",") if ns.strip()}

    @property
    def allowed_dag_ids(self) -> Set[str]:
        return {dag.strip() for dag in self.allowed_dag_ids_raw.split(",") if dag.strip()}

    @property
    def allowed_k8s_actions(self) -> Set[str]:
        return {act.strip() for act in self.allowed_k8s_actions_raw.split(",") if act.strip()}

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = AgentConfig()
