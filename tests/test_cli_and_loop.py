"""Tests for Interactive CLI and General LLM Orchestration Loop.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import sys
from unittest.mock import patch
import pytest

from agent.llm_client import MockLLMClient
from agent.main import run_cli
from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest


def test_cli_single_message_execution(capsys):
    test_args = ["agent.main", "--message", "What is the dataset version?"]
    mock_orch = Orchestrator(llm_client=MockLLMClient())
    with patch.object(sys, "argv", test_args):
        run_cli(orch_instance=mock_orch)
    captured = capsys.readouterr()
    assert "REQUEST ID:" in captured.out
    assert "TOOL TRACE" in captured.out


def test_orchestrator_general_llm_loop():
    # Test orchestrator when LLM calls a tool via function call JSON
    mock_llm = MockLLMClient(
        responses=[
            {
                "content": "Checking dataset version.",
                "tool_calls": [
                    {
                        "name": "dvc_get_dataset_version",
                        "args": {"dataset_path": "data/raw/customer_churn.csv"},
                    }
                ],
            },
            {
                "content": "### KNOWN FACTS\n- Dataset is verified on Git v2.0.\n\n### UNCERTAINTIES\n- None.\n\n### RECOMMENDED ACTIONS\n- Monitor model accuracy.",
                "tool_calls": [],
            },
        ]
    )
    orch = Orchestrator(llm_client=mock_llm)
    req = ChatRequest(message="Check dataset version", confirm=False)
    resp = orch.run(req)

    assert "KNOWN FACTS" in resp.answer
    assert len(resp.tool_trace) >= 1
    assert resp.tool_trace[0].tool == "dvc_get_dataset_version"
    assert resp.tool_trace[0].status == "SUCCESS"


def test_orchestrator_tool_not_found():
    orch = Orchestrator()
    res, trace = orch.execute_tool("non_existent_tool", {}, "req-test-99")
    assert res.ok is False
    assert res.error.code == "TOOL_NOT_FOUND"
    assert trace.status == "REJECTED"


def test_orchestrator_tool_exception_handling():
    orch = Orchestrator()
    # Invalid args causing validation exception
    res, trace = orch.execute_tool("k8s_scale_deployment", {"invalid_key": 123}, "req-test-99")
    assert res.ok is False
    assert trace.status == "FAILED"

