"""Registers top performing runs into the MLflow Model Registry."""

import mlflow
from mlflow.tracking import MlflowClient

def register_models():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    exp = client.get_experiment_by_name("customer_churn_prediction")
    if not exp:
        print("Experiment not found!")
        return

    runs = client.search_runs([exp.experiment_id], order_by=["metrics.accuracy DESC"], max_results=3)
    reg_name = "customer_churn_classifier"

    for i, run in enumerate(runs, 1):
        model_uri = f"runs:/{run.info.run_id}/model"
        acc = run.data.metrics.get("accuracy", 0.0)
        f1 = run.data.metrics.get("f1_score", 0.0)
        desc = f"Customer churn model. Val Acc: {acc:.4f}, F1: {f1:.4f}"
        print(f"Registering run {run.info.run_id} ({run.info.run_name})...")
        reg_ver = mlflow.register_model(
            model_uri=model_uri,
            name=reg_name,
            tags={"dvc_version": run.data.tags.get("dvc.dataset_version", "v2.0")},
        )
        alias = "production" if i == 1 else "candidate"
        try:
            client.set_registered_model_alias(reg_name, alias, reg_ver.version)
        except Exception:
            pass
        client.update_model_version(reg_name, reg_ver.version, description=desc)

    print("Successfully registered models into MLflow Model Registry!")

if __name__ == "__main__":
    register_models()

