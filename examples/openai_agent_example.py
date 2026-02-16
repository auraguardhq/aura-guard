"""openai_agent_example.py

Shows how to integrate Aura Guard with a raw OpenAI ChatCompletions agent loop.

This is a demonstration — it uses mock responses instead of real API calls.

Usage (no API key needed):
    python examples/openai_agent_example.py
"""

from __future__ import annotations

import os
import sys

# Allow running this example from a repo clone without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Any, Dict, List, Optional

from aura_guard import AgentGuard, PolicyAction
from aura_guard.adapters.openai_adapter import (
    extract_tool_calls_from_chat_completion,
    extract_assistant_text,
    inject_system_message,
)


def mock_openai_response_with_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate an OpenAI response that includes a tool call."""
    import json
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }],
            },
        }],
    }


def mock_openai_response_text(text: str) -> Dict[str, Any]:
    """Simulate an OpenAI response with text content."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
            },
        }],
    }


def main() -> None:
    # Initialize Aura Guard with a cost budget
    guard = AgentGuard(
        secret_key=b"your-secret-key",
        max_cost_per_run=0.25,
        side_effect_tools={"refund"},
        tool_costs={"search_kb": 0.03, "refund": 0.05},
    )

    print("=== Aura Guard + OpenAI Integration Example ===\n")

    # Simulate agent loop
    responses = [
        mock_openai_response_with_tool_call("search_kb", {"query": "refund policy"}),
        mock_openai_response_with_tool_call("search_kb", {"query": "refund policy EU"}),
        mock_openai_response_with_tool_call("search_kb", {"query": "refund policy EU Germany"}),
        mock_openai_response_with_tool_call("refund", {"order_id": "o1", "amount": 25}),
        mock_openai_response_with_tool_call("refund", {"order_id": "o1", "amount": 25}),
        mock_openai_response_text("I apologize, let me check on that for you."),
        mock_openai_response_text("I apologize for the delay, checking now."),
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful support agent."},
        {"role": "user", "content": "I want a refund for order o1."},
    ]

    for i, resp in enumerate(responses):
        print(f"--- Turn {i + 1} ---")

        # Check for tool calls
        tool_calls = extract_tool_calls_from_chat_completion(resp, ticket_id="ticket-42")
        if tool_calls:
            for call in tool_calls:
                decision = guard.check_tool(
                    call.name, args=call.args, ticket_id=call.ticket_id,
                )
                print(f"  Tool: {call.name}({call.args})")
                print(f"  Decision: {decision.action.value} — {decision.reason}")

                if decision.action == PolicyAction.ALLOW:
                    # In a real app, you'd execute the tool here
                    guard.record_result(ok=True, payload={"status": "ok"})
                elif decision.action == PolicyAction.REWRITE:
                    print(f"  → Injecting system message")
                    messages = inject_system_message(messages, decision.injected_system or "")
                elif decision.action == PolicyAction.ESCALATE:
                    print(f"  → ESCALATING: {decision.escalation_packet}")
                    break
        else:
            # Text response — check for stalls
            text = extract_assistant_text(resp) or ""
            print(f"  Text: {text[:80]}...")
            stall = guard.check_output(text)
            if stall:
                print(f"  Stall: {stall.action.value} — {stall.reason}")

        print()

    # Final summary
    print("=== Run Summary ===")
    for k, v in guard.stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
