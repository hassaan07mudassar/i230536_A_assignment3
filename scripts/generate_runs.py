"""Orchestrates dataset versioning via DVC + Git tags and generates >= 10 MLflow runs.

Demonstrates:
- Git commit + tags v1.0 and v2.0
- DVC tracking (.dvc files) and local remote storage
- Strict DVC provenance recorded in MLflow runs
- Diverse model hyperparameters and evaluation metrics
- Model Registry registration with production & candidate aliases
"""

import os
import subprocess
import sys
import time

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from training.generate_data import generate_churn_dataset
from training.train import train_and_evaluate


def run_cmd(cmd: list[str]) -> str:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    tracking_uri = "sqlite:///mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("../dvc_storage", exist_ok=True)

    print("\n=== STEP 1: Creating Dataset v1.0 (1000 rows) ===")
    df_v1 = generate_churn_dataset(version="v1", n_samples=1000, seed=42)
    data_path = "data/raw/customer_churn.csv"
    df_v1.to_csv(data_path, index=False)

    run_cmd(["dvc", "add", data_path])
    run_cmd(["dvc", "push"])
    run_cmd(["git", "add", f"{data_path}.dvc", "data/raw/.gitignore", ".dvc/config"])
    try:
        run_cmd(["git", "commit", "-m", "feat(data): add customer churn dataset v1.0"])
    except subprocess.CalledProcessError:
        print("Nothing new to commit for v1.0 data.")
    try:
        run_cmd(["git", "tag", "-a", "v1.0", "-m", "Customer Churn Dataset Version 1.0 (1000 records)", "-f"])
    except Exception as e:
        print(f"Tag v1.0 note: {e}")

    print("\n=== STEP 2: Training 5 Runs on Dataset v1.0 ===")
    v1_experiments = [
        {"model_type": "RandomForestClassifier", "n_estimators": 50, "max_depth": 5, "lr": 0.1},
        {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 8, "lr": 0.1},
        {"model_type": "RandomForestClassifier", "n_estimators": 150, "max_depth": 10, "lr": 0.1},
        {"model_type": "GradientBoostingClassifier", "n_estimators": 80, "max_depth": 4, "lr": 0.05},
        {"model_type": "RandomForestClassifier", "n_estimators": 20, "max_depth": 2, "lr": 0.1}, # underfit
    ]

    for i, exp in enumerate(v1_experiments, 1):
        print(f"--> Training Run {i}/11 (v1.0): {exp['model_type']} (trees={exp['n_estimators']}, depth={exp['max_depth']})")
        train_and_evaluate(
            data_path=data_path,
            n_estimators=exp["n_estimators"],
            max_depth=exp["max_depth"],
            learning_rate=exp["lr"],
            model_type=exp["model_type"],
            dataset_version_tag="v1.0",
            tracking_uri=tracking_uri,
            register_model=True,
        )

    print("\n=== STEP 3: Creating Dataset v2.0 (1500 rows) ===")
    df_v2 = generate_churn_dataset(version="v2", n_samples=1500, seed=101)
    df_v2.to_csv(data_path, index=False)

    run_cmd(["dvc", "add", data_path])
    run_cmd(["dvc", "push"])
    run_cmd(["git", "add", f"{data_path}.dvc"])
    try:
        run_cmd(["git", "commit", "-m", "feat(data): update customer churn dataset to v2.0 (1500 records)"])
    except subprocess.CalledProcessError:
        print("Nothing new to commit for v2.0 data.")
    try:
        run_cmd(["git", "tag", "-a", "v2.0", "-m", "Customer Churn Dataset Version 2.0 (1500 records)", "-f"])
    except Exception as e:
        print(f"Tag v2.0 note: {e}")

    print("\n=== STEP 4: Training 6 Runs on Dataset v2.0 ===")
    v2_experiments = [
        {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 8, "lr": 0.1},
        {"model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 12, "lr": 0.1}, # champion candidate
        {"model_type": "GradientBoostingClassifier", "n_estimators": 120, "max_depth": 5, "lr": 0.08},
        {"model_type": "GradientBoostingClassifier", "n_estimators": 150, "max_depth": 6, "lr": 0.12},
        {"model_type": "RandomForestClassifier", "n_estimators": 300, "max_depth": 25, "lr": 0.1}, # deliberate overfit (train f1 ~ 1.0, val f1 ~ 0.81)
        {"model_type": "RandomForestClassifier", "n_estimators": 10, "max_depth": 3, "lr": 0.1}, # deliberate weak run
    ]

    for i, exp in enumerate(v2_experiments, 6):
        print(f"--> Training Run {i}/11 (v2.0): {exp['model_type']} (trees={exp['n_estimators']}, depth={exp['max_depth']})")
        train_and_evaluate(
            data_path=data_path,
            n_estimators=exp["n_estimators"],
            max_depth=exp["max_depth"],
            learning_rate=exp["lr"],
            model_type=exp["model_type"],
            dataset_version_tag="v2.0",
            tracking_uri=tracking_uri,
            register_model=True,
        )

    print("\n[SUCCESS] Successfully generated 11 diverse MLflow runs across v1.0 and v2.0!")


if __name__ == "__main__":
    main()
