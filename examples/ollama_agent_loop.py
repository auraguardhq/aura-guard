"""ollama_agent_loop.py

Minimal local-model (Ollama-style) agent loop demo with Aura Guard.

This script uses a tiny mocked Ollama tool-call protocol so it runs offline
with no API keys. It demonstrates Aura Guard blocking/re-writing a search
jitter loop and then catching a repeated stall output.

Run:
    python examples/ollama_agent_loop.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List

# Allow running from a repo clone without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aura_guard import AgentGuard, AuraGuardConfig, CostModel, PolicyAction


def mock_ollama_turns() -> Iterable[Dict[str, Any]]:
    """Yield mocked Ollama-style responses with tool calls and text."""
    # Shape matches Ollama chat responses at a high level:
    # {"message": {"tool_calls": [{"function": {"name": ..., "arguments": ...}}]}}
    yield {
        "message": {
            "tool_calls": [
                {"function": {"name": "search_kb", "arguments": {"query": "refund policy"}}}
            ]
        }
    }
    yield {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_kb",
                        "arguments": {"query": "refund policy in the EU"},
                    }
                }
            ]
        }
    }
    yield {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_kb",
                        "arguments": {"query": "refund policy in the EU for Germany"},
                    }
                }
            ]
        }
    }
    # Simulate the model stalling with repeated "still checking" outputs.
    yield {"message": {"content": "I'm sorry, still checking that for you."}}
    yield {"message": {"content": "Apologies for the delay, still looking into it."}}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Fake local tool executor."""
    if name == "search_kb":
        return {"hits": [f"KB result for: {args.get('query', '')}"]}
    return {"error": f"unknown tool: {name}"}


def main() -> None:
    guard = AgentGuard(
        config=AuraGuardConfig(
            # Lower threshold to make the jitter loop intervention obvious in a tiny demo.
            arg_jitter_repeat_threshold=2,
            max_cost_per_run=1.0,
            cost_model=CostModel(tool_cost_by_name={"search_kb": 0.01}),
        )
    )

    print("=== Aura Guard + Ollama-style Loop (offline demo) ===\n")

    for i, response in enumerate(mock_ollama_turns(), start=1):
        print(f"--- Turn {i} ---")
        message = response.get("message", {})
        tool_calls: List[Dict[str, Any]] = message.get("tool_calls", [])

        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}

                decision = guard.check_tool(name, args=args)
                print(f"tool: {name}({args})")
                print(f"decision: {decision.action.value} — {decision.reason}")

                if decision.action == PolicyAction.ALLOW:
                    result = run_tool(name, args)
                    guard.record_result(ok=True, payload=result)
                    print(f"result: {result}")
                elif decision.action == PolicyAction.REWRITE:
                    print("Aura Guard rewrote the loop with injected guidance:")
                    print(decision.injected_system or "<no rewrite prompt>")
                    break
                else:
                    print("Aura Guard stopped the loop.")
                    break
        else:
            text = message.get("content", "")
            print(f"assistant: {text}")
            intervention = guard.check_output(text)
            if intervention:
                print(f"output intervention: {intervention.action.value} — {intervention.reason}")
                if intervention.injected_system:
                    print(intervention.injected_system)

        print()

    print("=== Stats ===")
    for k, v in guard.stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
