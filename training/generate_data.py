"""Data generation utility for Customer Churn dataset.

Generates realistic tabular data for DVC versioning demonstration.
Version 1: 1,000 records
Version 2: 1,500 records with expanded cohort
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate_churn_dataset(version: str, n_samples: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 72, size=n_samples)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    total_charges = np.round(tenure_months * monthly_charges + np.random.normal(0, 5, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    online_security = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    payment_method = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.25, 0.20, 0.20])

    # Logit model for realistic target generation
    # Higher monthly charges + month-to-month + no tech support -> higher churn
    logits = (
        -1.5
        + 0.02 * monthly_charges
        - 0.04 * tenure_months
        - 1.2 * contract_type
        - 0.8 * tech_support
        - 0.5 * online_security
    )
    probs = 1 / (1 + np.exp(-logits))
    churn = (np.random.uniform(0, 1, size=n_samples) < probs).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "tech_support": tech_support,
            "online_security": online_security,
            "payment_method": payment_method,
            "churn": churn,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate dataset for DVC tracking")
    parser.add_argument("--version", choices=["v1", "v2"], required=True, help="Dataset version")
    parser.add_argument("--output", default="data/raw/customer_churn.csv", help="Output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.version == "v1":
        df = generate_churn_dataset(version="v1", n_samples=1000, seed=42)
    else:
        df = generate_churn_dataset(version="v2", n_samples=1500, seed=101)

    df.to_csv(args.output, index=False)
    print(f"Generated {args.version} dataset with {len(df)} rows at {args.output}")


if __name__ == "__main__":
    main()

