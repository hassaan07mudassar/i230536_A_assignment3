"""Structured JSON Logging with Request ID propagation and strict Secret Redaction.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import contextvars
import json
import logging
import re
import sys
from typing import Any, Dict, Union

from pythonjsonlogger import jsonlogger

# Context variable for distributed request ID tracing
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="req-system")

# Sensitive key names and regex patterns to redact
SENSITIVE_KEY_NAMES = {
    "password",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "kubeconfig",
    "private_key",
}

SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(api[_\-]?key\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(password\s*[:=]\s*['\"]?)[^\s'\",]{3,}", re.IGNORECASE),
    re.compile(r"(token\s*[:=]?\s*['\"]?)[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE),
    re.compile(r"(AQ\.)[A-Za-z0-9_\-]{15,}", re.IGNORECASE),  # Gemini / Google keys
    re.compile(r"(sk-)[A-Za-z0-9_\-]{20,}", re.IGNORECASE),   # OpenAI keys
]


def redact_string(value: str) -> str:
    """Redacts known token patterns and secret substrings from plain text."""
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED_SECRET]", redacted)
    return redacted


def redact_secrets(data: Any) -> Any:
    """Recursively redacts dictionary keys and strings containing secrets."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if any(s in str(k).lower() for s in SENSITIVE_KEY_NAMES):
                cleaned[k] = "[REDACTED_SECRET]"
            else:
                cleaned[k] = redact_secrets(v)
        return cleaned
    elif isinstance(data, list):
        return [redact_secrets(item) for item in data]
    elif isinstance(data, str):
        return redact_string(data)
    return data


class RedactingJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JsonFormatter that injects request_id and redacts secrets."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["request_id"] = request_id_var.get()
        log_record["service"] = "mlops-agent"
        log_record["level"] = record.levelname

        # Redact secrets in message or extra fields
        for k, v in list(log_record.items()):
            log_record[k] = redact_secrets(v)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configures root and agent loggers with structured JSON formatting."""
    logger = logging.getLogger("mlops_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-configuration
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = RedactingJsonFormatter(
            "%(timestamp)s %(level)s %(service)s %(request_id)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = configure_logging()
