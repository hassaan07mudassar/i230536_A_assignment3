"""Apache Airflow 3 Machine Learning Pipeline DAG.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)

Pipeline Flow:
validate_dataset >> preprocess_data >> train_model >> evaluate_and_log_mlflow >> apply_quality_gate
"""

from datetime import datetime, timedelta
import logging
import os
from typing import Any, Dict

# Standard Airflow task decorators
try:
    from airflow.sdk import dag, task  # Airflow 3 modern SDK
except ImportError:
    from airflow.decorators import dag, task  # Airflow 2.x fallback

from airflow.models.param import Param

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "Hassaan Mudassar",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


@dag(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end reproducible MLOps training pipeline with DVC and quality gates",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "production"],
    params={
        "dataset_path": Param("data/raw/customer_churn.csv", type="string"),
        "min_f1_score": Param(0.80, type="number"),
        "max_train_val_gap": Param(0.15, type="number"),
        "dataset_version": Param("v2.0", type="string"),
        "n_estimators": Param(100, type="integer"),
        "max_depth": Param(8, type="integer"),
    },
)
def ml_training_pipeline():

    @task(task_id="validate_dataset")
    def validate_dataset(**context) -> Dict[str, Any]:
        """Validates dataset presence, schema completeness, and integrity."""
        params = context.get("params", {})
        dataset_path = params.get("dataset_path", "data/raw/customer_churn.csv")
        logger.info(f"[Task 1/5] Validating dataset at: {dataset_path}")

        if not os.path.exists(dataset_path):
            error_msg = f"Dataset validation failed: File not found at '{dataset_path}'"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        import pandas as pd

        df = pd.read_csv(dataset_path)
        required_cols = {
            "customer_id",
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "total_charges",
            "tech_support",
            "online_security",
            "payment_method",
            "churn",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset missing required schema columns: {missing_cols}")

        if len(df) < 100:
            raise ValueError(f"Dataset sample count ({len(df)}) below minimum threshold (100)")

        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Dataset contains {null_count} null values.")

        logger.info(f"Dataset validation passed: {len(df)} rows, schema intact.")
        return {
            "dataset_path": dataset_path,
            "row_count": len(df),
            "columns": list(df.columns),
            "status": "VALIDATED",
        }

    @task(task_id="preprocess_data")
    def preprocess_data(validation_info: Dict[str, Any], **context) -> Dict[str, Any]:
        """Prepares feature matrices, train/test split, and scaling."""
        logger.info(f"[Task 2/5] Preprocessing dataset from: {validation_info['dataset_path']}")
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(validation_info["dataset_path"])
        feature_cols = [
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "total_charges",
            "tech_support",
            "online_security",
            "payment_method",
        ]
        X = df[feature_cols]
        y = df["churn"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        os.makedirs("data/processed", exist_ok=True)
        train_path = "data/processed/train.parquet"
        val_path = "data/processed/val.parquet"

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        logger.info(f"Preprocessing completed. Train shape: {train_df.shape}, Val shape: {val_df.shape}")
        return {
            "train_path": train_path,
            "val_path": val_path,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
        }

    @task(task_id="train_model")
    def train_model(preprocessed_info: Dict[str, Any], **context) -> Dict[str, Any]:
        """Trains candidate model on preprocessed data."""
        params = context.get("params", {})
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params.get("max_depth", 8))
        logger.info(f"[Task 3/5] Training RandomForest (trees={n_estimators}, max_depth={max_depth})")

        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        train_df = pd.read_parquet(preprocessed_info["train_path"])
        val_df = pd.read_parquet(preprocessed_info["val_path"])

        X_train = train_df.drop(columns=["churn"])
        y_train = train_df["churn"]
        X_val = val_df.drop(columns=["churn"])
        y_val = val_df["churn"]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        metrics = {
            "train_f1": float(f1_score(y_train, train_preds)),
            "train_accuracy": float(accuracy_score(y_train, train_preds)),
            "val_f1": float(f1_score(y_val, val_preds)),
            "val_accuracy": float(accuracy_score(y_val, val_preds)),
            "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
            "train_val_gap": float(f1_score(y_train, train_preds) - f1_score(y_val, val_preds)),
        }

        os.makedirs("models", exist_ok=True)
        model_artifact_path = "models/pipeline_candidate.joblib"
        joblib.dump(model, model_artifact_path)

        logger.info(f"Training completed. Metrics: {metrics}")
        return {
            "metrics": metrics,
            "model_path": model_artifact_path,
        }

    @task(task_id="evaluate_and_log_mlflow")
    def evaluate_and_log_mlflow(training_info: Dict[str, Any], **context) -> Dict[str, Any]:
        """Logs metrics, artifacts, and parameters to MLflow tracking."""
        params = context.get("params", {})
        logger.info("[Task 4/5] Logging pipeline run to MLflow")
        import mlflow
        from training.train import get_git_info, get_dvc_provenance

        dataset_path = params.get("dataset_path", "data/raw/customer_churn.csv")
        dataset_version = params.get("dataset_version", "v2.0")

        git_info = get_git_info()
        dvc_info = get_dvc_provenance(dataset_path)

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("customer_churn_prediction")

        with mlflow.start_run(run_name=f"airflow_pipeline_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            mlflow.set_tag("pipeline.orchestrator", "Airflow_3")
            mlflow.set_tag("dvc.dataset_version", dataset_version)
            mlflow.set_tag("dvc.data_hash", dvc_info["dvc_hash"])
            mlflow.set_tag("git.commit", git_info["git_commit"])
            mlflow.log_params(params)
            mlflow.log_metrics(training_info["metrics"])
            mlflow.log_artifact(training_info["model_path"])

        logger.info(f"MLflow run logged successfully: run_id={run_id}")
        return {
            "run_id": run_id,
            "metrics": training_info["metrics"],
            "model_path": training_info["model_path"],
        }

    @task(task_id="apply_quality_gate")
    def apply_quality_gate(eval_info: Dict[str, Any], **context) -> Dict[str, Any]:
        """Applies configurable quality gate threshold. Halts promotion if gate fails."""
        params = context.get("params", {})
        min_f1 = float(params.get("min_f1_score", 0.80))
        max_gap = float(params.get("max_train_val_gap", 0.15))

        val_f1 = eval_info["metrics"]["val_f1"]
        gap = eval_info["metrics"]["train_val_gap"]

        logger.info(f"[Task 5/5] Quality Gate evaluation: val_f1={val_f1:.4f} (min={min_f1}), gap={gap:.4f} (max={max_gap})")

        if val_f1 < min_f1:
            error_msg = f"QUALITY GATE FAILED: Validation F1 ({val_f1:.4f}) is below threshold ({min_f1})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if gap > max_gap:
            error_msg = f"QUALITY GATE FAILED: Overfitting gap ({gap:.4f}) exceeds threshold ({max_gap})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Gate passed: Promote to candidate/production in MLflow
        logger.info("Quality gate passed. Registering model in MLflow Model Registry.")
        import mlflow
        client = mlflow.tracking.MlflowClient()
        reg_name = "customer_churn_classifier"
        try:
            reg_version = client.create_model_version(
                name=reg_name,
                source=eval_info["model_path"],
                run_id=eval_info["run_id"],
                tags={"promoted_by": "Airflow_3", "status": "PASSED_QUALITY_GATE"},
            )
            client.set_registered_model_alias(reg_name, "candidate", reg_version.version)
            logger.info(f"Model version {reg_version.version} assigned alias 'candidate'")
        except Exception as e:
            logger.warning(f"Could not update registry alias: {e}")

        return {
            "status": "PROMOTED",
            "val_f1": val_f1,
            "train_val_gap": gap,
            "quality_gate_passed": True,
        }

    # Task dependency graph
    v_data = validate_dataset()
    p_data = preprocess_data(v_data)
    t_model = train_model(p_data)
    e_model = evaluate_and_log_mlflow(t_model)
    apply_quality_gate(e_model)


# Instantiate the DAG
dag_instance = ml_training_pipeline()

