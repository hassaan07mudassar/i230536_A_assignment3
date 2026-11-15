"""Tests for Secret Redaction and Structured JSON Logging.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import json
import logging
from agent.logging_config import RedactingJsonFormatter, redact_secrets, redact_string


def test_redact_string_api_keys():
    raw = "Connecting with key AQ.FAKE_TEST_KEY_STRING_123456789 to service"
    cleaned = redact_string(raw)
    assert "AQ." in cleaned
    assert "[REDACTED_SECRET]" in cleaned
    assert "FAKE_TEST_KEY_STRING_123456789" not in cleaned


def test_redact_string_bearer_tokens():
    raw = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload"
    cleaned = redact_string(raw)
    assert "[REDACTED_SECRET]" in cleaned
    assert "secretpayload" not in cleaned


def test_redact_dict_sensitive_keys():
    payload = {
        "user": "admin",
        "password": "SuperSecretPassword123!",
        "token": "ghp_1234567890abcdefghijklmnopqrstuv",
        "nested": {
            "api_key": "sk-proj-1234567890abcdefghijklmnopqrst",
            "normal_field": "public_data",
        },
    }
    cleaned = redact_secrets(payload)
    assert cleaned["password"] == "[REDACTED_SECRET]"
    assert cleaned["token"] == "[REDACTED_SECRET]"
    assert cleaned["nested"]["api_key"] == "[REDACTED_SECRET]"
    assert cleaned["nested"]["normal_field"] == "public_data"


def test_redacting_json_formatter():
    formatter = RedactingJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Authenticated user with token secret_token_value_xyz",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["service"] == "mlops-agent"
    assert "request_id" in parsed

