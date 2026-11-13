"""FastAPI Agent Interface and Interactive CLI.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import argparse
import json
import sys
from typing import Optional
import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel

from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest, ChatResponse
from agent.tools import TOOL_REGISTRY

app = FastAPI(
    title="MLOps Autonomous AI Agent Service",
    description="Production-Ready AI Agent orchestrating DVC, MLflow, Airflow 3, and Kubernetes.",
    version="1.0.0",
)

orchestrator = Orchestrator()


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health():
    """Agent service liveness and diagnostic probe."""
    return {
        "status": "healthy",
        "service": "mlops-ai-agent",
        "registered_tools_count": len(TOOL_REGISTRY),
    }


@app.get("/tools", tags=["Capabilities"])
def list_tools():
    """Catalog of all registered operational tools, domains, and security permissions."""
    catalog = []
    for name, entry in TOOL_REGISTRY.items():
        catalog.append({
            "name": name,
            "domain": entry["domain"],
            "description": entry["description"],
            "mutating": entry["mutating"],
            "requires_confirmation": entry["mutating"],
        })
    return {"tools": catalog, "total": len(catalog)}


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """Processes natural language operational requests, runs tools, and returns evidence."""
    return orchestrator.run(request)


def run_cli(orch_instance: Optional[Orchestrator] = None):
    active_orch = orch_instance or orchestrator
    parser = argparse.ArgumentParser(description="MLOps Agent Interactive CLI")
    parser.add_argument("--message", type=str, help="Single query to send to agent")
    parser.add_argument("--confirm", action="store_true", help="Set confirmation flag for mutating actions")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI agent server on port 8000")
    args = parser.parse_args()

    if args.serve:
        print("Starting MLOps Agent FastAPI server on port 8000...")
        uvicorn.run("agent.main:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.message:
        req = ChatRequest(message=args.message, confirm=args.confirm)
        resp = active_orch.run(req)
        print("\n" + "=" * 70)
        print(f"REQUEST ID: {resp.request_id}")
        print("=" * 70)
        print(resp.answer)
        print("\n" + "-" * 70)
        print(f"TOOL TRACE ({len(resp.tool_trace)} calls):")
        for t in resp.tool_trace:
            print(f"  [{t.status}] {t.tool} ({t.duration_ms}ms) -> args: {t.args}")
        print("=" * 70 + "\n")
        return

    # Interactive REPL
    print("\n" + "=" * 70)
    print(" MLOps AI Agent - Interactive Console (Type 'exit' to quit)")
    print(" Course: MLOps & Production AI Engineering (Hassaan Mudassar | i230536)")
    print("=" * 70)

    while True:
        try:
            query = input("\nagent> ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break

            confirm = False
            if query.endswith("--confirm"):
                query = query.replace("--confirm", "").strip()
                confirm = True

            req = ChatRequest(message=query, confirm=confirm)
            resp = active_orch.run(req)
            print("\n" + resp.answer)
            print("\n--- Tool Trace ---")
            for t in resp.tool_trace:
                print(f"  • {t.tool}: {t.status} ({t.duration_ms}ms)")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    run_cli()

