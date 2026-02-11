"""aura_guard.serialization

State persistence for Aura Guard.

Serialize/deserialize GuardState to JSON for storage in Redis, DynamoDB,
Postgres JSONB, filesystem, or any other backend.

Security:
- result_cache and idempotency_ledger payloads are EXCLUDED from serialization
  as they may contain PII. Only signatures and metadata are persisted.
- To restore full cache capability after deserialization, the guard will
  rebuild the cache from new tool calls naturally.

Usage:
    from aura_guard.serialization import state_to_json, state_from_json

    # Save
    json_str = state_to_json(state)
    redis.set(f"guard:{run_id}", json_str, ex=3600)

    # Restore
    json_str = redis.get(f"guard:{run_id}")
    state = state_from_json(json_str)
"""

from __future__ import annotations

import json
from typing import Any, Dict
from uuid import uuid4

from .guard import GuardState
from .types import CostEvent, ToolCallSig


def state_to_json(state: GuardState) -> str:
    """Serialize GuardState to a JSON string.

    Excludes result_cache and idempotency_ledger payloads (PII risk).
    Persists only safe data: counts, reason codes, and HMAC-signed token signatures
    (no raw text, no raw args, no raw payloads).
    """
    data: Dict[str, Any] = {
        "version": 4,
        "run_id": state.run_id,

        # Tool stream (rolling signature history)
        "tool_stream": [
            {
                "name": s.name,
                "args_sig": s.args_sig,
                "ticket_sig": s.ticket_sig,
                "side_effect": s.side_effect,
            }
            for s in state.tool_stream
        ],

        # Arg jitter history (already HMAC'd, safe to persist)
        "tool_query_sigs": {
            tool: [sorted(sig_set) for sig_set in sigs[-12:]]
            for tool, sigs in state.tool_query_sigs.items()
        },

        # Quarantine and error state
        "quarantined_tools": state.quarantined_tools,
        "error_streaks": {
            f"{k[0]}||{k[1]}": v for k, v in state.error_streaks.items()
        },

        # Side-effect accounting
        "attempted_side_effect_calls": state.attempted_side_effect_calls,
        "executed_side_effect_calls": state.executed_side_effect_calls,

        # Per-tool call counts
        "tool_call_counts": dict(state.tool_call_counts),

        # Stall detection (full state)
        "stall_streak": state.stall_streak,
        "stall_pattern_streak": getattr(state, "stall_pattern_streak", 0),
        "stall_rewrite_attempts": state.stall_rewrite_attempts,
        "last_assistant_token_sigs": (
            sorted(state.last_assistant_token_sigs)
            if state.last_assistant_token_sigs is not None
            else None
        ),

        # Progress markers and unique sets (HMAC'd, safe to persist)
        "last_progress_marker": list(state.last_progress_marker),
        "unique_tool_calls_seen": sorted(state.unique_tool_calls_seen),
        "unique_tool_results_seen": sorted(state.unique_tool_results_seen),

        # Cost tracking
        "cumulative_cost": state.cumulative_cost,
        "reported_token_cost": state.reported_token_cost,
        "budget_warning_emitted": state.budget_warning_emitted,
        "cost_events": [
            {
                "event": e.event,
                "tool": e.tool,
                "amount": e.amount,
                "cumulative": e.cumulative,
                "limit": e.limit,
                "pct": e.pct,
            }
            for e in state.cost_events
        ],
    }

    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def state_from_json(data: str) -> GuardState:
    """Deserialize GuardState from a JSON string.

    Note: result_cache and idempotency_ledger payloads are NOT restored
    (privacy). The guard will naturally rebuild caches from new tool calls.
    HMAC-signed token signatures *are* restored (safe, and improves continuity
    for jitter/stall detection).
    """
    obj = json.loads(data)

    version = obj.get("version")
    if version is None or version < 4:
        raise ValueError(
            f"Incompatible state format: expected version >= 4, got {version!r}."
        )

    run_id = obj.get("run_id") or uuid4().hex
    state = GuardState(run_id=run_id)

    # Tool stream
    state.tool_stream = [
        ToolCallSig(
            name=s["name"],
            args_sig=s["args_sig"],
            ticket_sig=s.get("ticket_sig"),
            side_effect=s.get("side_effect", False),
        )
        for s in obj.get("tool_stream", [])
    ]

    # Quarantine and error state
    state.quarantined_tools = obj.get("quarantined_tools", {})
    state.error_streaks = {}
    for k_str, v in obj.get("error_streaks", {}).items():
        parts = k_str.split("||", 1)
        if len(parts) == 2:
            state.error_streaks[(parts[0], parts[1])] = v

    # Arg jitter history (restore HMAC'd token sig sets)
    state.tool_query_sigs = {
        tool: [set(sig_list) for sig_list in sigs]
        for tool, sigs in obj.get("tool_query_sigs", {}).items()
    }

    # Side-effect accounting
    state.attempted_side_effect_calls = obj.get("attempted_side_effect_calls", {})
    state.executed_side_effect_calls = obj.get("executed_side_effect_calls", {})
    state.tool_call_counts = obj.get("tool_call_counts", {})

    # Stall detection (full state)
    state.stall_streak = obj.get("stall_streak", 0)
    state.stall_pattern_streak = obj.get("stall_pattern_streak", 0)
    state.stall_rewrite_attempts = obj.get("stall_rewrite_attempts", 0)
    raw_sigs = obj.get("last_assistant_token_sigs")
    state.last_assistant_token_sigs = set(raw_sigs) if isinstance(raw_sigs, list) else None

    # Progress markers and unique sets
    lpm = obj.get("last_progress_marker", [0, 0])
    state.last_progress_marker = (lpm[0], lpm[1]) if isinstance(lpm, list) and len(lpm) == 2 else (0, 0)
    raw_calls = obj.get("unique_tool_calls_seen", [])
    state.unique_tool_calls_seen = (
        {tuple(item) for item in raw_calls} if isinstance(raw_calls, list) else set()
    )
    raw_results = obj.get("unique_tool_results_seen", [])
    state.unique_tool_results_seen = set(raw_results) if isinstance(raw_results, list) else set()

    # Cost tracking
    state.cumulative_cost = float(obj.get("cumulative_cost", 0.0))
    state.reported_token_cost = float(obj.get("reported_token_cost", 0.0))
    state.budget_warning_emitted = bool(obj.get("budget_warning_emitted", False))
    state.cost_events = [
        CostEvent(
            event=e["event"],
            tool=e["tool"],
            amount=float(e["amount"]),
            cumulative=float(e["cumulative"]),
            limit=e.get("limit"),
            pct=e.get("pct"),
        )
        for e in obj.get("cost_events", [])
    ]

    return state


def state_to_dict(state: GuardState) -> Dict[str, Any]:
    """Serialize GuardState to a Python dict (for embedding in larger structures)."""
    return json.loads(state_to_json(state))


def state_from_dict(data: Dict[str, Any]) -> GuardState:
    """Deserialize GuardState from a Python dict."""
    return state_from_json(json.dumps(data))
