"""Model training pipeline with strict DVC provenance logging and MLflow Model Registry integration.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_git_info() -> Dict[str, str]:
    info = {"git_commit": "unknown", "git_branch": "unknown", "git_tag": "none"}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        info["git_commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        info["git_branch"] = branch
    except Exception:
        pass

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        info["git_tag"] = tag
    except Exception:
        pass

    return info


def get_dvc_provenance(dataset_path: str) -> Dict[str, str]:
    """Extracts DVC tracking hash from the associated .dvc file."""
    provenance = {
        "dataset_path": dataset_path,
        "dvc_hash": "untracked",
        "dvc_file": "none",
    }
    dvc_file = f"{dataset_path}.dvc"
    if os.path.exists(dvc_file):
        provenance["dvc_file"] = dvc_file
        try:
            with open(dvc_file, "r", encoding="utf-8") as f:
                dvc_data = yaml.safe_load(f)
                outs = dvc_data.get("outs", [])
                if outs and isinstance(outs, list):
                    provenance["dvc_hash"] = outs[0].get("md5", "unknown")
        except Exception:
            pass
    return provenance


def train_and_evaluate(
    data_path: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    learning_rate: float = 0.1,
    model_type: str = "RandomForestClassifier",
    dataset_version_tag: str = "v1.0",
    experiment_name: str = "customer_churn_prediction",
    tracking_uri: Optional[str] = None,
    register_model: bool = True,
    fail_run: bool = False,
) -> Dict[str, Any]:
    """Trains a model, logs full DVC provenance + metrics to MLflow, and registers models passing quality gates."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    start_time = time.time()

    git_info = get_git_info()
    dvc_info = get_dvc_provenance(data_path)

    run_name = f"{model_type}_{dataset_version_tag}_trees{n_estimators}_depth{max_depth or 'None'}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Non-negotiable Provenance Tags
        mlflow.set_tag("dvc.dataset_version", dataset_version_tag)
        mlflow.set_tag("dvc.data_hash", dvc_info["dvc_hash"])
        mlflow.set_tag("dvc.dataset_path", dvc_info["dataset_path"])
        mlflow.set_tag("dvc.file", dvc_info["dvc_file"])
        mlflow.set_tag("git.commit", git_info["git_commit"])
        mlflow.set_tag("git.branch", git_info["git_branch"])
        mlflow.set_tag("git.tag", git_info["git_tag"])
        mlflow.set_tag("model.type", model_type)
        mlflow.set_tag("author", "Hassaan Mudassar")
        mlflow.set_tag("student_id", "i230536")

        if fail_run:
            mlflow.set_tag("run.status", "FAILED")
            raise RuntimeError("Deliberate training failure injected for testing/intelligence")

        # 2. Data Loading & Feature Preparation
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Run DVC checkout first.")

        df = pd.read_csv(data_path)
        feature_cols = [
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "total_charges",
            "tech_support",
            "online_security",
            "payment_method",
        ]
        target_col = "churn"

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # 3. Model Training
        if model_type == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else 3,
                learning_rate=learning_rate,
                random_state=42,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 4)

        # 4. Predictions & Metrics
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        train_f1 = f1_score(y_train, train_preds)
        val_f1 = f1_score(y_val, val_preds)
        val_prec = precision_score(y_val, val_preds, zero_division=0)
        val_rec = recall_score(y_val, val_preds, zero_division=0)
        train_val_gap = float(train_f1 - val_f1)

        # Log Parameters
        params = {
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "learning_rate": learning_rate,
            "dataset_rows": len(df),
            "features_count": len(feature_cols),
        }
        mlflow.log_params(params)

        # Log Metrics
        metrics = {
            "accuracy": round(val_acc, 4),
            "precision": round(val_prec, 4),
            "recall": round(val_rec, 4),
            "f1_score": round(val_f1, 4),
            "train_accuracy": round(train_acc, 4),
            "train_f1": round(train_f1, 4),
            "train_val_f1_gap": round(train_val_gap, 4),
            "training_time_seconds": training_time,
        }
        mlflow.log_metrics(metrics)

        # 5. Artifacts: Confusion Matrix & Saved Model
        cm = confusion_matrix(y_val, val_preds)
        os.makedirs("outputs", exist_ok=True)
        cm_path = "outputs/confusion_matrix.png"
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix ({dataset_version_tag})")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Churn", "Churn"])
        plt.yticks(tick_marks, ["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), horizontalalignment="center")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

        # Save local copy for inference service default
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        mlflow.log_artifact("models/model.joblib", artifact_path="serialized_model")

        # Log Model with MLflow Model Signature
        signature = infer_signature(X_val, val_preds)
        input_example = X_val.iloc[:2]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="pickle",
        )

        # 6. Quality Gate Evaluation & Model Registry
        quality_passed = val_f1 >= 0.80 and train_val_gap <= 0.15
        mlflow.set_tag("quality_gate.passed", str(quality_passed))
        mlflow.set_tag("quality_gate.threshold_f1", "0.80")
        mlflow.set_tag("quality_gate.max_gap", "0.15")

        registry_name = "customer_churn_classifier"
        registered_version = None

        if register_model and quality_passed:
            model_uri = f"runs:/{run_id}/model"
            reg_desc = (
                f"Customer Churn model trained on DVC dataset {dataset_version_tag} "
                f"(Git {git_info['git_commit'][:8]}). Val F1: {val_f1:.4f}"
            )
            reg_model = mlflow.register_model(
                model_uri=model_uri,
                name=registry_name,
                tags={
                    "dvc_version": dataset_version_tag,
                    "val_f1": str(val_f1),
                    "commit": git_info["git_commit"],
                },
            )
            registered_version = reg_model.version
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registry_name,
                version=registered_version,
                description=reg_desc,
            )
            # Set alias (candidate vs production)
            alias = "production" if val_f1 >= 0.88 else "candidate"
            try:
                client.set_registered_model_alias(registry_name, alias, registered_version)
            except Exception:
                pass

        return {
            "run_id": run_id,
            "metrics": metrics,
            "provenance": {
                "dvc_version": dataset_version_tag,
                "dvc_hash": dvc_info["dvc_hash"],
                "git_commit": git_info["git_commit"],
            },
            "quality_passed": quality_passed,
            "registered_version": registered_version,
        }


def main():
    parser = argparse.ArgumentParser(description="Train customer churn prediction model")
    parser.add_argument("--data-path", default="data/raw/customer_churn.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--model-type", default="RandomForestClassifier")
    parser.add_argument("--dataset-version", default="v1.0")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    result = train_and_evaluate(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        dataset_version_tag=args.dataset_version,
        tracking_uri=args.tracking_uri,
        register_model=not args.no_register,
    )
    print(f"Training completed successfully:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
