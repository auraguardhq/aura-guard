"""aura_guard.types

Public types for Aura Guard.

This package is intentionally dependency-free. Integrations should adapt their
framework-specific tool/message formats into these dataclasses.

Security note:
- ToolCall.args and ToolResult.payload may contain PII.
- Aura Guard MUST NOT persist or emit those raw values.
- Aura Guard stores only keyed signatures (HMAC) of args/payload/ticket_id.

The types here keep raw fields because a tool executor needs them. The guard
engine is responsible for safe handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ================================
# Policy Actions
# ================================

class PolicyAction(str, Enum):
    """What the guard wants the orchestrator to do next."""

    ALLOW = "allow"        # Proceed with tool call / continue
    BLOCK = "block"        # Block the tool call (safe fail)
    CACHE = "cache"        # Return cached tool result (do not execute)
    REWRITE = "rewrite"    # Inject a system message and re-run the model
    ESCALATE = "escalate"  # Terminate the run with an escalation packet
    FINALIZE = "finalize"  # Terminate the run with a finalized output


# ================================
# Tool Call / Result
# ================================

@dataclass
class ToolCall:
    """A structured tool call request.

    `args` may contain PII.
    `ticket_id` may be PII.
    """

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    ticket_id: Optional[str] = None

    # If not set, Aura Guard can infer from config.side_effect_tools
    side_effect: Optional[bool] = None

    # Aura Guard may set this deterministically. Tool implementations may or may
    # not honor it; Aura Guard enforces idempotency regardless.
    idempotency_key: Optional[str] = None


@dataclass
class ToolResult:
    """A structured tool result.

    `payload` may contain PII.
    `payload_sig` is a safe keyed signature (HMAC) computed by Aura Guard.

    `error_code` should be a coarse classification (e.g. "429", "timeout",
    "5xx"), not a raw exception message. Error codes are persisted in
    serialized state and may be emitted in telemetry. Do not include PII,
    stack traces, or detailed error messages.
    """

    ok: bool
    payload: Any = None
    error_code: Optional[str] = None

    payload_sig: Optional[str] = None

    # If True, this result came from Aura Guard cache/ledger (no tool execution).
    cached: bool = False

    # For side-effect tools, True indicates the side effect was executed.
    side_effect_executed: Optional[bool] = None


# ================================
# Internal Signatures (PII-safe)
# ================================

@dataclass(frozen=True)
class ToolCallSig:
    """Safe signature representation of a tool call.

    Contains no raw args or raw ticket_id — only keyed HMAC hashes.
    """

    name: str
    args_sig: str
    ticket_sig: Optional[str] = None
    side_effect: bool = False


# ================================
# Policy Decision
# ================================

@dataclass
class PolicyDecision:
    """Return value of Aura Guard decision methods."""

    action: PolicyAction
    reason: str

    # For REWRITE
    injected_system: Optional[str] = None

    # For CACHE
    cached_result: Optional[ToolResult] = None

    # For FINALIZE / ESCALATE
    finalized_output: Optional[Dict[str, Any]] = None
    escalation_packet: Optional[Dict[str, Any]] = None


# ================================
# Cost Event
# ================================

@dataclass
class CostEvent:
    """A single cost tracking event."""

    event: str           # "cost_incurred", "budget_warning", "budget_exceeded"
    tool: str
    amount: float        # USD
    cumulative: float    # USD running total
    limit: Optional[float] = None
    pct: Optional[float] = None
