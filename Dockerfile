# ==============================================================================
# Multi-Stage Production Dockerfile for Customer Churn Inference Service
# Course: MLOps & Production AI Engineering (Fall 2026)
# Student: Hassaan Mudassar (i230536, CS-A)
# ==============================================================================

# Build Stage: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --user \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    pydantic==2.6.4 \
    scikit-learn==1.4.1.post1 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    joblib==1.3.2

# Runtime Stage: Minimal attack surface & non-root user
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user and group
RUN addgroup --system --gid 10001 appgroup && \
    adduser --system --uid 10001 --ingroup appgroup appuser

# Copy installed wheels from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and model artifacts
COPY --chown=appuser:appgroup api /app/api
COPY --chown=appuser:appgroup models /app/models

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    APP_VERSION=1.0.0

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

