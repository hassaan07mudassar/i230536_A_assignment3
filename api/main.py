"""Production FastAPI inference service deployed to Kubernetes.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import os
import time
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Lifespan and app setup
app = FastAPI(
    title="Customer Churn Prediction Inference Service",
    description="Production inference microservice with liveness, readiness, and model versioning.",
    version="1.0.0",
)

START_TIME = time.time()
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
GIT_COMMIT = os.getenv("GIT_COMMIT", "HEAD")

# Global model state
_model = None
_model_metadata: Dict[str, Any] = {
    "loaded": False,
    "model_path": MODEL_PATH,
    "model_version": os.getenv("MODEL_VERSION", "candidate"),
    "dataset_version": os.getenv("DATASET_VERSION", "v2.0"),
    "predictions_count": 0,
    "last_prediction_time": None,
}


def load_model(path: str = MODEL_PATH):
    global _model, _model_metadata
    if os.path.exists(path):
        try:
            _model = joblib.load(path)
            _model_metadata["loaded"] = True
            _model_metadata["model_path"] = path
            return True
        except Exception as e:
            _model_metadata["loaded"] = False
            _model_metadata["load_error"] = str(e)
            return False
    _model_metadata["loaded"] = False
    return False


@app.on_event("startup")
def startup_event():
    load_model()


class PredictionInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    contract_type: int = Field(..., ge=0, le=2, description="0: Month-to-month, 1: One year, 2: Two year")
    monthly_charges: float = Field(..., ge=0.0, le=500.0, description="Monthly charges in USD")
    total_charges: float = Field(..., ge=0.0, le=50000.0, description="Total accumulated charges")
    tech_support: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    online_security: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    payment_method: int = Field(..., ge=0, le=3, description="0: Electronic, 1: Mailed, 2: Bank, 3: Credit")


class PredictionOutput(BaseModel):
    churn_prediction: int = Field(..., description="0: Retained, 1: Churned")
    churn_probability: float = Field(..., description="Probability of churn [0, 1]")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH")
    model_version: str
    dataset_version: str


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Probes"])
def health():
    """Kubernetes liveness probe: indicates the process is alive."""
    return {
        "status": "healthy",
        "service": "customer-churn-inference",
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.get("/ready", tags=["Probes"])
def ready():
    """Kubernetes readiness probe: fails with 503 if the ML model is not loaded."""
    if not _model_metadata["loaded"] or _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": "Model artifact is not loaded into memory",
                "path": _model_metadata["model_path"],
            },
        )
    return {
        "status": "ready",
        "model_version": _model_metadata["model_version"],
        "dataset_version": _model_metadata["dataset_version"],
    }


@app.get("/version", tags=["Metadata"])
def version():
    """Reports deployed app version, underlying model metadata, and provenance."""
    return {
        "service": "customer-churn-inference",
        "app_version": APP_VERSION,
        "git_commit": GIT_COMMIT,
        "model_version": _model_metadata["model_version"],
        "dataset_version": _model_metadata["dataset_version"],
        "model_loaded": _model_metadata["loaded"],
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: PredictionInput):
    """Executes churn inference on input customer features."""
    if not _model_metadata["loaded"] or _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model not ready",
        )

    features = pd.DataFrame(
        [
            {
                "tenure_months": payload.tenure_months,
                "contract_type": payload.contract_type,
                "monthly_charges": payload.monthly_charges,
                "total_charges": payload.total_charges,
                "tech_support": payload.tech_support,
                "online_security": payload.online_security,
                "payment_method": payload.payment_method,
            }
        ]
    )

    try:
        prob = float(_model.predict_proba(features)[0][1])
        pred = int(prob >= 0.5)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}",
        )

    risk = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")

    _model_metadata["predictions_count"] += 1
    _model_metadata["last_prediction_time"] = time.time()

    return PredictionOutput(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
        model_version=_model_metadata["model_version"],
        dataset_version=_model_metadata["dataset_version"],
    )


@app.get("/metrics", tags=["Observability"])
def metrics():
    """Observability metrics for inference requests and health."""
    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_predictions": _model_metadata["predictions_count"],
        "model_loaded": _model_metadata["loaded"],
        "last_prediction_timestamp": _model_metadata["last_prediction_time"],
    }

