"""aura_guard.adapters.openai_adapter

Helpers for integrating Aura Guard with OpenAI-style message payloads.

This module is dependency-free: it operates on plain dicts/lists.
It does **not** call any OpenAI API.

Supported patterns:
- Chat Completions: response["choices"][0]["message"]["tool_calls"]
- Function calling: response["choices"][0]["message"]["function_call"]

Usage with AgentGuard:
    from aura_guard import AgentGuard
    from aura_guard.adapters.openai_adapter import (
        extract_tool_calls_from_chat_completion,
        inject_system_message,
    )

    guard = AgentGuard(secret_key=b"your-secret-key", max_cost_per_run=0.50)

    # After getting a response from OpenAI:
    tool_calls = extract_tool_calls_from_chat_completion(response, ticket_id="t1")
    for call in tool_calls:
        decision = guard.check_tool(call.name, args=call.args, ticket_id=call.ticket_id)
        if decision.action == "allow":
            result = execute_tool(call)
            guard.record_result(ok=True, payload=result)
        elif decision.action == "rewrite":
            messages = inject_system_message(messages, decision.injected_system)
            # Re-call the model with updated messages
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..types import ToolCall


def inject_system_message(
    messages: List[Dict[str, Any]],
    system_text: str,
) -> List[Dict[str, Any]]:
    """Return a new messages list with a system message prepended.

    If the first message is already role=system, it will be replaced.
    """
    msg = {"role": "system", "content": system_text}
    if not messages:
        return [msg]

    out = list(messages)
    if isinstance(out[0], dict) and out[0].get("role") == "system":
        out[0] = msg
    else:
        out.insert(0, msg)
    return out


def append_system_message(
    messages: List[Dict[str, Any]],
    system_text: str,
) -> List[Dict[str, Any]]:
    """Return a new messages list with a system message appended (after all other messages).

    Useful for injecting guard instructions without overwriting existing system prompts.
    """
    out = list(messages)
    out.append({"role": "system", "content": system_text})
    return out


def extract_tool_calls_from_chat_completion(
    resp: Dict[str, Any],
    *,
    ticket_id: Optional[str] = None,
) -> List[ToolCall]:
    """Extract ToolCall objects from a ChatCompletions-like response dict.

    Handles both the tool_calls format and legacy function_call format.
    Invalid entries are skipped.
    """
    try:
        choices = resp.get("choices") or []
        msg = choices[0].get("message") if choices else None
        tool_calls = (msg or {}).get("tool_calls") or []
    except Exception:
        return []

    calls: List[ToolCall] = []

    # Modern tool_calls format
    for tc in tool_calls:
        try:
            fn = tc.get("function") or {}
            name = fn.get("name")
            args_raw = fn.get("arguments")
            if not isinstance(name, str) or not name:
                continue

            args = _parse_args(args_raw)
            calls.append(ToolCall(name=name, args=args, ticket_id=ticket_id))
        except Exception:
            continue

    # Legacy function_call format (fallback)
    if not calls and msg:
        fc = (msg or {}).get("function_call")
        if isinstance(fc, dict):
            name = fc.get("name")
            if isinstance(name, str) and name:
                args = _parse_args(fc.get("arguments"))
                calls.append(ToolCall(name=name, args=args, ticket_id=ticket_id))

    return calls


def extract_assistant_text(resp: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text content from a ChatCompletions-like response."""
    try:
        choices = resp.get("choices") or []
        msg = choices[0].get("message") if choices else None
        content = (msg or {}).get("content")
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return None


def _parse_args(args_raw: Any) -> Dict[str, Any]:
    """Parse tool call arguments from various formats."""
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str) and args_raw.strip():
        try:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}
