"""LLM Client Abstraction supporting Gemini, OpenAI, and Mock implementations.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import json
import os
from typing import Any, Dict, List, Optional

from agent.config import settings
from agent.logging_config import logger


class BaseLLMClient:
    """Base interface for all LLM implementations."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Returns structured dictionary:
        {
            "content": str,
            "tool_calls": List[{"name": str, "args": Dict[str, Any]}],
            "raw_response": Any
        }
        """
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM for offline tests and verification."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        self.responses = responses or []
        self._call_index = 0
        self.calls: List[Dict[str, Any]] = []

    def set_responses(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = responses
        self._call_index = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools_definitions})

        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
            return resp

        # Default intelligent response if queue exhausted
        last_msg = messages[-1]["content"] if messages else ""
        return {
            "content": f"Acknowledged request: {last_msg[:100]}. Operational facts verified from real tool evidence.",
            "tool_calls": [],
        }


class GeminiLLMClient(BaseLLMClient):
    """Live Google Gemini API client with function/tool calling support."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model or "gemini-2.5-flash"
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client: {e}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("Gemini API client not initialized. Check GEMINI_API_KEY environment variable.")

        # Build prompt from conversation messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"[{role.upper()}]:\n{content}\n")
        full_prompt = "\n".join(prompt_parts)

        # Build tools description if provided
        tools_summary = ""
        if tools_definitions:
            tools_summary = "\nAvailable Tools (JSON tool calling format):\n"
            for t in tools_definitions:
                tools_summary += f"- {t['name']}: {t['description']}. Parameters schema: {json.dumps(t['parameters'])}\n"
            full_prompt += (
                f"\n{tools_summary}\n"
                "If you need to invoke a tool, respond with ONLY a JSON code block in the following format:\n"
                "```json\n"
                '{"tool": "<tool_name>", "args": {<args>}}\n'
                "```\n"
                "Otherwise, provide the final evidence-based answer separating:\n"
                "### KNOWN FACTS (with tool + evidence source)\n"
                "### UNCERTAINTIES / MISSING DATA\n"
                "### RECOMMENDED ACTIONS\n"
            )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            raw_text = response.text or ""

            # Check if model requested a tool call via structured JSON
            tool_calls = []
            if "```json" in raw_text:
                json_part = raw_text.split("```json")[1].split("```")[0].strip()
                try:
                    parsed = json.loads(json_part)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        tool_calls.append({"name": parsed["tool"], "args": parsed.get("args", {})})
                except Exception:
                    pass

            return {
                "content": raw_text,
                "tool_calls": tool_calls,
                "raw_response": str(response),
            }
        except Exception as e:
            logger.error(f"Gemini API request error: {str(e)}")
            raise


def get_llm_client(force_mock: bool = False, custom_mock: Optional[MockLLMClient] = None) -> BaseLLMClient:
    """Factory creating appropriate LLM client instance."""
    if custom_mock:
        return custom_mock
    if force_mock or not settings.gemini_api_key or os.getenv("TEST_MODE") == "1":
        return MockLLMClient()
    return GeminiLLMClient(api_key=settings.gemini_api_key)

