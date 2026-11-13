"""Prints clean terminal summary of MLflow runs and model registry for screenshot capture."""

import os
from mlflow.tracking import MlflowClient

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    client = MlflowClient(tracking_uri=tracking_uri)

    print("=" * 80)
    print(" MLflow Tracking Server Summary: sqlite:///mlflow.db")
    print(" Course: MLOps & Production AI Engineering (Hassaan Mudassar | i230536)")
    print("=" * 80)

    exp = client.get_experiment_by_name("customer_churn_prediction")
    if not exp:
        print("Experiment 'customer_churn_prediction' not found.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=20,
    )

    print(f"\n[EXPERIMENT] {exp.name} (ID: {exp.experiment_id}) - {len(runs)} Recorded Runs:\n")
    header = f"{'Run Name':<46} {'Accuracy':<10} {'F1-Score':<10} {'DVC Ver':<8} {'Status':<8}"
    print(header)
    print("-" * 84)

    for r in runs:
        name = r.info.run_name[:44]
        acc = f"{r.data.metrics.get('accuracy', 0.0):.4f}"
        f1 = f"{r.data.metrics.get('f1_score', 0.0):.4f}"
        dvc_v = r.data.tags.get("dvc.dataset_version", "N/A")
        status = r.info.status
        print(f"{name:<46} {acc:<10} {f1:<10} {dvc_v:<8} {status:<8}")

    print("\n" + "=" * 80)
    print(" MLflow Model Registry (Registered Models & Aliases):")
    print("=" * 80)

    try:
        reg_model = client.get_registered_model("customer_churn_classifier")
        aliases = getattr(reg_model, "aliases", {})
        versions = client.search_model_versions("name='customer_churn_classifier'")
        print(f"Model Name: {reg_model.name}")
        print(f"Production Aliases: {aliases}\n")
        print(f"{'Version':<10} {'Stage/Alias':<20} {'Run ID':<36} {'Status':<10}")
        print("-" * 78)
        for v in versions:
            alias_list = ", ".join(v.aliases) if hasattr(v, "aliases") and v.aliases else v.current_stage
            print(f"v{v.version:<9} {alias_list:<20} {v.run_id:<36} {v.status:<10}")
    except Exception as e:
        print(f"Registry lookup note: {e}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
