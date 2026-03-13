"""AuraGuard Standalone — single-file version for quick evaluation.

Download:
    curl -O https://raw.githubusercontent.com/auraguardhq/aura-guard/main/standalone/aura_guard_standalone.py

Usage:
    from aura_guard_standalone import AgentGuard, GuardDenied

    guard = AgentGuard(
        secret_key=b"your-secret-key",
        side_effect_tools={"refund", "cancel"},
        max_cost_per_run=1.00,
    )

    result = guard.run("search_kb", search_kb, query="refund policy")

For the full package (MCP, async, serialization, CLI, benchmarks):
    pip install aura-guard

GitHub: https://github.com/auraguardhq/aura-guard
"""

from __future__ import annotations

import copy
import functools
import hmac
import json
import logging
import re
import threading
import time as _time
import warnings
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
import sys
import types

if __name__ not in sys.modules:
    _standalone_module = types.ModuleType(__name__)
    _standalone_module.__dict__.update(globals())
    sys.modules[__name__] = _standalone_module

# ═══════════════════════════════════════════
# Types
# ═══════════════════════════════════════════

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

# ═══════════════════════════════════════════
# Config
# ═══════════════════════════════════════════

# ================================
# Cost Model
# ================================

@dataclass
class CostModel:
    """Cost estimation model used for budget enforcement and telemetry.

    This is intentionally crude. Real token accounting depends on your model,
    pricing, and prompt size. Set per-tool costs for more accurate tracking.
    Use record_tokens() on AgentGuard to report actual token spend.
    """

    default_tool_call_cost: float = 0.04       # USD per tool call
    tool_cost_by_name: Dict[str, float] = field(default_factory=dict)

    # Token cost estimation (used when you provide token counts)
    input_token_cost_per_1k: float = 0.003     # $/1K input tokens
    output_token_cost_per_1k: float = 0.015    # $/1K output tokens

    def tool_cost(self, tool_name: str) -> float:
        return float(self.tool_cost_by_name.get(tool_name, self.default_tool_call_cost))

    def token_cost(self, input_tokens: int = 0, output_tokens: int = 0) -> float:
        return (
            (input_tokens / 1000.0) * self.input_token_cost_per_1k
            + (output_tokens / 1000.0) * self.output_token_cost_per_1k
        )


# ================================
# Tool Policy
# ================================

class ToolAccess(Enum):
    """Access level for a tool in the policy layer."""
    ALLOW = "allow"           # Normal operation (default)
    DENY = "deny"             # Always block — tool is forbidden
    HUMAN_APPROVAL = "human"  # Escalate for human approval before execution


@dataclass
class ToolPolicy:
    """Per-tool policy configuration.

    Controls access, rate limits, and risk classification beyond
    the loop-detection primitives.

    Example:
        ToolPolicy(access=ToolAccess.ALLOW, max_calls=5, risk="high")
        ToolPolicy(access=ToolAccess.DENY)  # block tool entirely
    """

    access: ToolAccess = ToolAccess.ALLOW
    max_calls: Optional[int] = None          # per-run call limit for THIS tool (overrides global)
    risk: str = "default"                    # risk class tag for telemetry/filtering
    require_args: Optional[Set[str]] = None  # required arg keys (block if missing)
    deny_reason: str = ""                    # reason shown when denied


# ================================
# Guard Config
# ================================

@dataclass
class AuraGuardConfig:
    """Thresholds and policies for the guard.

    All primitives have sensible defaults. Override what you need.
    """

    # SECURITY
    # --------
    # Key used for keyed signatures (HMAC). Replace in production.
    secret_key: bytes = b"aura-guard-dev-key-CHANGE-ME"

    # LOOP WINDOW
    # -----------
    tool_loop_window: int = 12

    # Primitive 1: identical tool call repeats
    # Threshold includes the current call: threshold=3 means the 3rd identical
    # call is blocked/cached (first 2 are allowed).
    repeat_toolcall_threshold: int = 3

    # Primitive 2: argument jitter similarity (query-like tools)
    arg_jitter_similarity_threshold: float = 0.60   # Lowered from 0.85 — real LLMs reformulate
    arg_jitter_repeat_threshold: int = 3

    # Primitive 2b: per-tool call cap (catches smart reformulation loops)
    # If the same tool is *executed* (passes all prior checks) more than this
    # many times in a run, quarantine it regardless of argument similarity.
    # Calls blocked/cached by earlier primitives don't count toward this cap.
    max_calls_per_tool: Optional[int] = None   # None = no cap

    # Primitive 8: multi-tool sequence loop detection
    # Detects repeating sequences of tool calls (e.g. A→B→A→B or A→B→C→A→B→C).
    # Catches multi-agent ping-pong and circular delegation patterns.
    sequence_repeat_threshold: int = 3        # Number of times a pattern must repeat to trigger (3 = A→B→A→B→A→B)
    max_sequence_length: int = 4              # Max pattern length to check (2, 3, 4-tool patterns)
    sequence_detection_enabled: bool = True   # Set False to disable sequence detection

    # Primitive 3: error retry circuit breaker
    error_retry_threshold: int = 2
    # Error codes/classes that should trigger quarantine when repeated.
    quarantine_error_codes: Set[str] = field(
        default_factory=lambda: {"401", "403", "429", "timeout", "connection_error", "5xx"}
    )

    # Primitive 4: side-effect gating
    # Max executions of each side-effect tool per run (per-tool, not global).
    # E.g., with default 1: refund can execute once AND cancel can execute once.
    side_effect_max_executed_per_run: int = 1
    side_effect_tools: Set[str] = field(
        default_factory=lambda: {"send_reply", "refund", "cancel"}
    )

    # Primitive 5: no-state-change stall detection
    no_state_change_threshold: int = 4
    stall_text_similarity_threshold: float = 0.92
    stall_pattern_threshold: float = 0.6        # Stall pattern score (apology/filler detection)
    stall_pattern_streak_threshold: int = 2     # Consecutive high-pattern turns before intervention
    stall_rewrite_max_attempts: int = 1

    # Primitive 6: cost budget enforcement
    max_cost_per_run: Optional[float] = None   # USD; None = no limit
    cost_warning_threshold: float = 0.8        # Warn at 80% of budget

    # Query-like arg keys by tool name, used for argument-jitter detection.
    # Example: {"search_kb": ["query", "q"]}
    query_arg_keys_by_tool: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "search_kb": ["query", "q"],
            "search": ["query", "q"],
            "search_documents": ["query", "q"],
            "rag_retrieve": ["query", "q", "question"],
        }
    )

    # Cost model for budget enforcement and telemetry
    cost_model: CostModel = field(default_factory=CostModel)

    # CACHE CONTROLS
    # --------------
    # Tools whose results should NEVER be cached (e.g. tools with time-varying output)
    never_cache_tools: Set[str] = field(default_factory=set)
    # TTL for cached results in seconds. None = cache valid for entire run.
    cache_ttl_seconds: Optional[float] = None

    # ARG SIGNATURE CONTROLS
    # ----------------------
    # Arg keys to IGNORE when computing signatures (per-tool).
    # Use for keys that contain timestamps, request IDs, pagination cursors, etc.
    # Example: {"search_kb": {"request_id", "timestamp"}}
    arg_ignore_keys: Dict[str, Set[str]] = field(default_factory=dict)

    # STATE GROWTH CAPS
    # -----------------
    # Maximum entries in result_cache, unique_tool_calls_seen, etc.
    # Prevents unbounded memory growth in long-running agents.
    max_cache_entries: int = 1000
    max_unique_calls_tracked: int = 5000
    max_cost_events: int = 500

    # TOOL POLICIES
    # -------------
    # Per-tool access control: allow, deny, or require human approval.
    # Overrides default behavior for specific tools.
    # Example: {"delete_account": ToolPolicy(access=ToolAccess.DENY)}
    tool_policies: Dict[str, ToolPolicy] = field(default_factory=dict)

    # SHADOW / AUDIT MODE
    # -------------------
    # When True, the guard logs decisions but does NOT enforce them.
    # All tool calls return ALLOW. Use this to evaluate false-positive rates
    # before turning on enforcement in production.
    shadow_mode: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.secret_key, bytes):
            raise TypeError(
                f"secret_key must be bytes, got {type(self.secret_key).__name__}. "
                f"Use b\"your-key\" instead of \"your-key\"."
            )
        if self.repeat_toolcall_threshold < 1:
            raise ValueError("repeat_toolcall_threshold must be >= 1")
        if not 0.0 <= self.arg_jitter_similarity_threshold <= 1.0:
            raise ValueError("arg_jitter_similarity_threshold must be between 0.0 and 1.0")
        if self.arg_jitter_repeat_threshold < 1:
            raise ValueError("arg_jitter_repeat_threshold must be >= 1")
        if self.error_retry_threshold < 1:
            raise ValueError("error_retry_threshold must be >= 1")
        if self.side_effect_max_executed_per_run < 1:
            raise ValueError("side_effect_max_executed_per_run must be >= 1")
        if self.no_state_change_threshold < 1:
            raise ValueError("no_state_change_threshold must be >= 1")
        if not 0.0 <= self.stall_text_similarity_threshold <= 1.0:
            raise ValueError("stall_text_similarity_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.stall_pattern_threshold <= 1.0:
            raise ValueError("stall_pattern_threshold must be between 0.0 and 1.0")
        if self.max_cost_per_run is not None and self.max_cost_per_run <= 0:
            raise ValueError("max_cost_per_run must be > 0 when provided")
        if not 0.0 <= self.cost_warning_threshold <= 1.0:
            raise ValueError("cost_warning_threshold must be between 0.0 and 1.0")
        if self.max_cache_entries < 1:
            raise ValueError("max_cache_entries must be >= 1")
        if self.max_unique_calls_tracked < 1:
            raise ValueError("max_unique_calls_tracked must be >= 1")
        if self.tool_loop_window < 1:
            raise ValueError("tool_loop_window must be >= 1")
        if self.max_cost_events < 1:
            raise ValueError("max_cost_events must be >= 1")
        if self.sequence_repeat_threshold < 2:
            raise ValueError("sequence_repeat_threshold must be >= 2")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be >= 2")

    # --------------------------------
    # Helper methods
    # --------------------------------

    def is_side_effect_tool(self, tool_name: str, explicit_flag: Optional[bool] = None) -> bool:
        if explicit_flag is not None:
            return bool(explicit_flag)
        return tool_name in self.side_effect_tools

    def query_arg_keys(self, tool_name: str) -> List[str]:
        return list(self.query_arg_keys_by_tool.get(tool_name, []))

    def should_quarantine_error(self, error_code_class: str) -> bool:
        return error_code_class in self.quarantine_error_codes

    def is_cacheable(self, tool_name: str) -> bool:
        """Whether results from this tool can be cached."""
        return tool_name not in self.never_cache_tools

    def get_arg_ignore_keys(self, tool_name: str) -> Set[str]:
        """Arg keys to strip before computing signatures for this tool."""
        return self.arg_ignore_keys.get(tool_name, set())

    def get_tool_policy(self, tool_name: str) -> Optional[ToolPolicy]:
        """Get the policy for a specific tool, or None for default behavior."""
        return self.tool_policies.get(tool_name)

    def get_tool_max_calls(self, tool_name: str) -> Optional[int]:
        """Per-tool execution limit: tool policy override > global max_calls_per_tool.

        Counts executed calls only (calls blocked by earlier primitives are excluded).
        """
        policy = self.tool_policies.get(tool_name)
        if policy and policy.max_calls is not None:
            return policy.max_calls
        return self.max_calls_per_tool

# ═══════════════════════════════════════════
# Telemetry (minimal standalone version)
# ═══════════════════════════════════════════

class _StandaloneLogSink:
    """Minimal telemetry sink that logs to Python logging."""

    def emit(self, event: Dict[str, Any]) -> None:
        logging.getLogger("aura_guard").debug("event=%s", event.get("event", "?"))


@dataclass
class Telemetry:
    """Thin telemetry facade for the guard engine."""

    sink: Any = field(default_factory=_StandaloneLogSink)

    def emit(self, event_name: str, **fields: Any) -> None:
        evt: Dict[str, Any] = {"event": event_name, **fields}
        try:
            self.sink.emit(evt)
        except Exception:
            logging.getLogger("aura_guard").exception("Telemetry sink error")

# ═══════════════════════════════════════════
# Guard engine
# ═══════════════════════════════════════════

# ================================
# Canonicalization & Signatures
# ================================

def _canonicalize(obj: Any) -> Any:
    """Convert `obj` into a JSON-serializable structure with deterministic ordering.

    For stable signatures, tool args and payloads should use JSON-serializable
    primitives (str, int, float, bool, None, list, dict). Custom objects fall
    back to str(obj), which may be nondeterministic (e.g. memory addresses)
    and could weaken deduplication/caching.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"__bytes__": sha256(obj).hexdigest()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted((_canonicalize(x) for x in obj), key=lambda x: str(x))
    if isinstance(obj, dict):
        items = [(str(k), _canonicalize(v)) for k, v in obj.items()]
        return {k: v for k, v in sorted(items, key=lambda kv: kv[0])}
    return str(obj)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hmac_hex(secret_key: bytes, message: str) -> str:
    return hmac.new(secret_key, message.encode("utf-8"), sha256).hexdigest()


def _args_sig(cfg: AuraGuardConfig, args: Dict[str, Any], tool_name: str = "") -> str:
    filtered = args
    if tool_name:
        ignore = cfg.get_arg_ignore_keys(tool_name)
        if ignore:
            filtered = {k: v for k, v in args.items() if k not in ignore}
    return _hmac_hex(cfg.secret_key, "args:" + _stable_json_dumps(filtered))


def _ticket_sig(cfg: AuraGuardConfig, ticket_id: Optional[str], fallback_run_id: str) -> str:
    if ticket_id:
        return _hmac_hex(cfg.secret_key, "ticket:" + str(ticket_id))
    return _hmac_hex(cfg.secret_key, "run:" + fallback_run_id)


def _payload_sig(cfg: AuraGuardConfig, payload: Any) -> str:
    return _hmac_hex(cfg.secret_key, "payload:" + _stable_json_dumps(payload))


def _now() -> float:
    """Monotonic time for cache TTL."""
    return _time.monotonic()


# ================================
# Text Token Signatures
# ================================

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Pre-compiled stall pattern detectors (Primitive 5 enhancement)
# These catch "apology loops" and "still working" loops that token overlap misses
# because the phrasing varies but the intent is identical.
_STALL_APOLOGY_RE = re.compile(
    r"\b(i\s+apologize|i'?m\s+sorry|apologies|sorry\s+for)\b", re.I
)
_STALL_WORKING_RE = re.compile(
    r"\b(still\s+(looking|working|checking|searching|researching|investigating)"
    r"|let\s+me\s+(check|look|try|search|investigate)"
    r"|i'?m\s+(looking|working|checking|searching|researching|investigating)"
    r"|looking\s+into\s+(this|it|that)"
    r"|bear\s+with\s+me"
    r"|one\s+moment"
    r"|please\s+wait"
    r"|thank\s+you\s+for\s+(your\s+)?patience)\b", re.I
)
_STALL_FILLER_RE = re.compile(
    r"\b(inconvenience|delay|trouble|difficulty|issue|problem)\b", re.I
)


def _stall_pattern_score(text: str) -> float:
    """Score text for stall-like patterns (0.0 = no stall signals, 1.0 = maximum stall).

    Independent of token overlap — catches semantically similar but differently
    worded apology/filler loops.
    """
    score = 0.0
    if _STALL_APOLOGY_RE.search(text):
        score += 0.4
    if _STALL_WORKING_RE.search(text):
        score += 0.4
    if _STALL_FILLER_RE.search(text):
        score += 0.2
    return min(score, 1.0)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_sig_set(cfg: AuraGuardConfig, text: str) -> Set[str]:
    """Compute HMAC-signed token set for similarity without storing raw text."""
    toks = _tokenize(text)
    return {_hmac_hex(cfg.secret_key, "tok:" + t) for t in toks}


def _overlap_similarity(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|).

    Better than Jaccard for "argument jitter" where a query accumulates
    extra qualifiers (A is a subset of B).
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return inter / denom


# ================================
# Error Classification
# ================================

def _classify_error_code(error_code: Optional[str]) -> str:
    if not error_code:
        return "unknown"
    s = str(error_code).strip().lower()

    if s in {"timeout", "timed_out", "read_timeout", "write_timeout"}:
        return "timeout"
    if s in {"conn", "connection", "connection_error", "network_error"}:
        return "connection_error"
    if s.isdigit():
        if s.startswith("5"):
            return "5xx"
        return s
    if "timeout" in s:
        return "timeout"
    if "rate" in s or "429" in s:
        return "429"
    return s


# ================================
# Guard State
# ================================

@dataclass
class GuardState:
    """Per-run state. Create a new one for each agent run.

    .. warning:: NOT THREAD-SAFE

       GuardState is a plain dataclass with no locking. Do not share a single
       GuardState instance across threads or async tasks without external
       synchronization. For concurrent agent runs, create one GuardState per
       run. For distributed/multi-process agents, serialize and share state
       explicitly using :mod:`aura_guard.serialization`.

    .. warning:: run_id MUST NOT contain PII

       run_id is emitted in telemetry events and stored in serialized state
       as raw text. Use a random identifier (default: uuid4). Do not set
       run_id to user IDs, emails, ticket IDs, or any personally identifiable
       information.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)

    # Rolling history (request signatures)
    tool_stream: List[ToolCallSig] = field(default_factory=list)

    # Per-tool call counts (for max_calls_per_tool enforcement)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)

    # Tool → list of hashed token sets for query-like args (rolling)
    tool_query_sigs: Dict[str, List[Set[str]]] = field(default_factory=dict)

    # Caches
    result_cache: Dict[Tuple[str, str], ToolResult] = field(default_factory=dict)
    result_cache_ts: Dict[Tuple[str, str], float] = field(default_factory=dict)  # monotonic timestamps

    # Side-effect idempotency ledger
    idempotency_ledger: Dict[Tuple[str, str, str], ToolResult] = field(default_factory=dict)

    # Error tracking
    error_streaks: Dict[Tuple[str, str], int] = field(default_factory=dict)
    quarantined_tools: Dict[str, str] = field(default_factory=dict)

    # Side-effect accounting
    attempted_side_effect_calls: Dict[str, int] = field(default_factory=dict)
    executed_side_effect_calls: Dict[str, int] = field(default_factory=dict)

    # Progress tracking (primitive 5)
    unique_tool_calls_seen: Set[Tuple[str, str]] = field(default_factory=set)
    unique_tool_results_seen: Set[str] = field(default_factory=set)
    last_progress_marker: Tuple[int, int] = (0, 0)
    last_assistant_token_sigs: Optional[Set[str]] = None
    stall_streak: int = 0
    stall_pattern_streak: int = 0
    stall_rewrite_attempts: int = 0

    # Cost tracking (primitive 6)
    cumulative_cost: float = 0.0
    reported_token_cost: float = 0.0   # actual token costs reported by integrator
    cost_events: List[CostEvent] = field(default_factory=list)
    budget_warning_emitted: bool = False


# ================================
# Aura Guard Engine
# ================================

class AuraGuard:
    """Reliability middleware for tool-using agents: idempotency, circuit breaking, and loop detection.

    Call on_tool_call_request() before each tool execution,
    on_tool_result() after each tool execution,
    and on_llm_output() after each assistant response.
    """

    def __init__(
        self,
        config: Optional[AuraGuardConfig] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self.cfg = config or AuraGuardConfig()
        self.telemetry = telemetry

        if self.cfg.secret_key == b"aura-guard-dev-key-CHANGE-ME":
            if not self.cfg.shadow_mode:
                raise ValueError(
                    "AuraGuard default development secret_key is not allowed when "
                    "shadow_mode=False. Set AuraGuardConfig(secret_key=...) to a unique key."
                )
            warnings.warn(
                "AuraGuard is using the default development secret_key in shadow_mode. "
                "Set a unique secret_key before enabling enforcement.",
                stacklevel=2,
            )

    # -------------------------
    # Internal helpers
    # -------------------------

    def _emit(self, event: str, *, state: Optional[GuardState] = None, **fields: Any) -> None:
        if not self.telemetry:
            return
        if state is not None:
            fields["run_id"] = state.run_id
        self.telemetry.emit(event, **fields)

    def estimate_tool_cost(self, tool_name: str) -> float:
        return self.cfg.cost_model.tool_cost(tool_name)

    def _cache_valid(self, state: GuardState, cache_key: Tuple[str, str]) -> bool:
        """Check if a cached result is still valid (TTL not expired)."""
        if self.cfg.cache_ttl_seconds is None:
            return True  # no TTL = valid forever within run
        ts = state.result_cache_ts.get(cache_key)
        if ts is None:
            return True  # no timestamp = legacy entry, treat as valid
        return (_now() - ts) < self.cfg.cache_ttl_seconds

    def record_tokens(self, *, state: GuardState, input_tokens: int = 0,
                      output_tokens: int = 0, cost_override: Optional[float] = None) -> None:
        """Report actual token usage for accurate cost tracking.

        Call this after each LLM call with the real token counts from the API
        response. If cost_override is provided, it's used directly instead of
        computing from the cost model.

        Example:
            guard.record_tokens(state=state, input_tokens=resp.usage.input_tokens,
                                output_tokens=resp.usage.output_tokens)
        """
        if cost_override is not None:
            actual = cost_override
        else:
            actual = self.cfg.cost_model.token_cost(input_tokens, output_tokens)
        state.reported_token_cost += actual
        state.cumulative_cost += actual
        state.cost_events.append(CostEvent(
            event="token_cost_reported", tool="llm",
            amount=actual, cumulative=state.cumulative_cost,
        ))
        if len(state.cost_events) > self.cfg.max_cost_events:
            state.cost_events = state.cost_events[-self.cfg.max_cost_events:]
        self._emit(
            "token_cost_reported",
            state=state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(actual, 6),
            cumulative_cost=round(state.cumulative_cost, 4),
        )

    def _forced_outcome_system_prompt(self) -> str:
        """System instruction used when we detect a stall/no-state-change loop."""
        return (
            "SYSTEM ALERT: You are stuck in a loop. Do NOT call any tools.\n"
            "Output exactly ONE JSON object and nothing else.\n"
            "Schema:\n"
            "{\n"
            '  "action": "finalize" | "escalate",\n'
            '  "reason": string,\n'
            '  "reply_draft": string | null,\n'
            '  "escalation": {\n'
            '    "route": string | null,\n'
            '    "summary": string,\n'
            '    "missing_info": [string],\n'
            '    "tags": [string]\n'
            '  } | null\n'
            "}\n"
            "Rules:\n"
            "- If you have enough facts to answer safely, use action=finalize and write reply_draft.\n"
            "- Otherwise use action=escalate with a concise escalation packet.\n"
            "- Do not include markdown, commentary, or additional text."
        )

    def _parse_forced_outcome(self, text: str) -> Optional[PolicyDecision]:
        """Parse JSON output from forced outcome prompt."""
        t = text.strip()
        if not (t.startswith("{") and t.endswith("}")):
            return None

        try:
            obj = json.loads(t)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        action = obj.get("action")
        reason = obj.get("reason")
        if action not in {"finalize", "escalate"}:
            return None
        if not isinstance(reason, str) or not reason.strip():
            return None

        if action == "finalize":
            reply = obj.get("reply_draft")
            if not isinstance(reply, str) or not reply.strip():
                return None
            return PolicyDecision(
                action=PolicyAction.FINALIZE,
                reason=f"forced_finalize:{reason.strip()}",
                finalized_output={"reply_draft": reply, "reason": reason.strip()},
            )

        # Escalate
        esc = obj.get("escalation")
        if not isinstance(esc, dict):
            esc = {"route": None, "summary": "(missing)", "missing_info": [], "tags": []}

        packet = {
            "route": esc.get("route"),
            "summary": esc.get("summary") if isinstance(esc.get("summary"), str) else "(missing)",
            "missing_info": esc.get("missing_info") if isinstance(esc.get("missing_info"), list) else [],
            "tags": esc.get("tags") if isinstance(esc.get("tags"), list) else [],
            "reason": reason.strip(),
        }
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason=f"forced_escalate:{reason.strip()}",
            escalation_packet=packet,
        )

    # -------------------------
    # Public API
    # -------------------------

    def new_state(self, run_id: Optional[str] = None) -> GuardState:
        """Create a fresh GuardState for a new agent run.

        Args:
            run_id: Optional identifier for this run. Defaults to a random UUID.
                    Must not contain PII — it is emitted in telemetry and stored
                    in serialized state as raw text.
        """
        return GuardState(run_id=run_id or uuid4().hex)

    def on_tool_call_request(self, *, state: GuardState, call: ToolCall) -> PolicyDecision:
        """Evaluate a tool call before executing it.

        Returns a PolicyDecision telling the orchestrator what to do.
        In shadow_mode, non-ALLOW decisions are suppressed and returned as ALLOW.
        """
        decision = self._evaluate_tool_call(state=state, call=call)

        if self.cfg.shadow_mode and decision.action != PolicyAction.ALLOW:
            self._emit(
                "shadow_would_deny", state=state,
                tool=call.name,
                original_action=decision.action.value,
                original_reason=decision.reason,
            )
            return PolicyDecision(action=PolicyAction.ALLOW, reason="shadow_allow")

        return decision

    def _evaluate_tool_call(self, *, state: GuardState, call: ToolCall) -> PolicyDecision:
        """Internal: evaluate a tool call. Returns the raw decision (no shadow suppression)."""
        tool = call.name
        is_side_effect = self.cfg.is_side_effect_tool(tool, call.side_effect)

        args_sig = _args_sig(self.cfg, call.args, tool_name=tool)
        t_sig = _ticket_sig(self.cfg, call.ticket_id, state.run_id)

        sig = ToolCallSig(name=tool, args_sig=args_sig, ticket_sig=t_sig, side_effect=is_side_effect)

        # ──────────────────────────────────────────
        # Primitive 7: Tool policy layer (deny/approve)
        # ──────────────────────────────────────────
        policy = self.cfg.get_tool_policy(tool)
        if policy is not None:
            if policy.access == ToolAccess.DENY:
                reason = policy.deny_reason or f"tool_denied_by_policy"
                self._emit("tool_policy_denied", state=state, tool=tool, reason=reason)
                return PolicyDecision(
                    action=PolicyAction.BLOCK,
                    reason=f"policy_deny:{reason}",
                )
            if policy.access == ToolAccess.HUMAN_APPROVAL:
                self._emit("tool_policy_human_approval", state=state, tool=tool, risk=policy.risk)
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="policy_human_approval_required",
                    escalation_packet={
                        "route": "human_approval",
                        "summary": f"Tool '{tool}' requires human approval (risk: {policy.risk})",
                        "tool": tool,
                        "args_sig": args_sig,
                        "risk": policy.risk,
                        "tags": ["human_approval", f"risk:{policy.risk}"],
                    },
                )
            # Check policy-level required args
            if policy.require_args:
                missing = policy.require_args - set(call.args.keys())
                if missing:
                    self._emit(
                        "tool_policy_missing_args", state=state,
                        tool=tool, missing_args=sorted(missing),
                    )
                    return PolicyDecision(
                        action=PolicyAction.BLOCK,
                        reason=f"policy_missing_args:{','.join(sorted(missing))}",
                    )

        # ──────────────────────────────────────────
        # Early cache checks (must run before budget)
        # ──────────────────────────────────────────
        if is_side_effect:
            key = (t_sig, tool, args_sig)
            if key in state.idempotency_ledger:
                cached = state.idempotency_ledger[key]
                cached2 = ToolResult(
                    ok=cached.ok,
                    payload=cached.payload,
                    error_code=cached.error_code,
                    payload_sig=cached.payload_sig,
                    cached=True,
                    side_effect_executed=cached.side_effect_executed,
                )
                est = self.estimate_tool_cost(tool)
                self._emit(
                    "idempotent_replay_blocked",
                    state=state,
                    tool=tool,
                    ticket_sig=t_sig,
                    args_sig=args_sig,
                    estimated_cost_avoided=round(est, 4),
                )
                return PolicyDecision(
                    action=PolicyAction.CACHE,
                    reason="idempotent_replay",
                    cached_result=cached2,
                )

        # +1 because the current call is not yet in tool_stream (appended only on ALLOW)
        repeats = sum(
            1 for s in state.tool_stream if s.name == tool and s.args_sig == args_sig
        ) + 1

        # Generic repeat cache is for non-side-effect tools only.
        # Side-effect tools use the ticket-scoped idempotency ledger (above).
        if not is_side_effect:
            if repeats >= self.cfg.repeat_toolcall_threshold:
                cache_key = (tool, args_sig)
                if cache_key in state.result_cache and self._cache_valid(state, cache_key):
                    cached = state.result_cache[cache_key]
                    cached2 = ToolResult(
                        ok=cached.ok,
                        payload=cached.payload,
                        error_code=cached.error_code,
                        payload_sig=cached.payload_sig,
                        cached=True,
                        side_effect_executed=cached.side_effect_executed,
                    )
                    est = self.estimate_tool_cost(tool)
                    self._emit(
                        "tool_call_cache_hit",
                        state=state,
                        tool=tool,
                        args_sig=args_sig,
                        repeats=repeats,
                        estimated_cost_avoided=round(est, 4),
                    )
                    return PolicyDecision(
                        action=PolicyAction.CACHE,
                        reason="identical_toolcall_loop_cache",
                        cached_result=cached2,
                    )

        # ──────────────────────────────────────────
        # Primitive 6: Cost budget enforcement
        # ──────────────────────────────────────────
        if self.cfg.max_cost_per_run is not None:
            estimated = self.estimate_tool_cost(tool)
            projected = state.cumulative_cost + estimated

            if round(projected, 8) >= round(self.cfg.max_cost_per_run, 8):
                evt = CostEvent(
                    event="budget_exceeded", tool=tool,
                    amount=estimated, cumulative=state.cumulative_cost,
                    limit=self.cfg.max_cost_per_run,
                    pct=round(projected / self.cfg.max_cost_per_run * 100, 1),
                )
                state.cost_events.append(evt)
                if len(state.cost_events) > self.cfg.max_cost_events:
                    state.cost_events = state.cost_events[-self.cfg.max_cost_events:]
                self._emit(
                    "budget_exceeded_escalate",
                    state=state,
                    tool=tool,
                    cumulative_cost=round(state.cumulative_cost, 4),
                    projected_cost=round(projected, 4),
                    limit=self.cfg.max_cost_per_run,
                    estimated_cost_avoided=round(estimated, 4),
                )
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="cost_budget_exceeded",
                    escalation_packet={
                        "route": None,
                        "summary": (
                            f"Agent run would exceed cost budget "
                            f"(${state.cumulative_cost:.2f} spent + "
                            f"${estimated:.2f} projected / "
                            f"${self.cfg.max_cost_per_run:.2f} limit)"
                        ),
                        "tags": ["budget_exceeded"],
                        "cumulative_cost": round(state.cumulative_cost, 4),
                        "projected_cost": round(projected, 4),
                        "limit": self.cfg.max_cost_per_run,
                    },
                )

            # Budget warning (emit once)
            warn_threshold = self.cfg.max_cost_per_run * self.cfg.cost_warning_threshold
            if round(projected, 8) >= round(warn_threshold, 8) and not state.budget_warning_emitted:
                state.budget_warning_emitted = True
                pct = round(projected / self.cfg.max_cost_per_run * 100, 1)
                evt = CostEvent(
                    event="budget_warning", tool=tool,
                    amount=estimated, cumulative=state.cumulative_cost,
                    limit=self.cfg.max_cost_per_run, pct=pct,
                )
                state.cost_events.append(evt)
                if len(state.cost_events) > self.cfg.max_cost_events:
                    state.cost_events = state.cost_events[-self.cfg.max_cost_events:]
                self._emit(
                    "budget_warning",
                    state=state,
                    tool=tool,
                    cumulative_cost=round(state.cumulative_cost, 4),
                    projected_cost=round(projected, 4),
                    limit=self.cfg.max_cost_per_run,
                    pct=pct,
                )

        # ──────────────────────────────────────────
        # Primitive 3 (enforcement): Quarantine check
        # ──────────────────────────────────────────
        if tool in state.quarantined_tools:
            reason = state.quarantined_tools[tool]
            injected = (
                f"SYSTEM ALERT: Tool '{tool}' is unavailable ({reason}). "
                "Do not call it again in this run. If you cannot proceed safely, "
                "output an escalation packet."
            )
            self._emit(
                "tool_quarantined_block",
                state=state,
                tool=tool, reason=reason, args_sig=args_sig,
                estimated_cost_avoided=round(self.estimate_tool_cost(tool), 4),
            )
            return PolicyDecision(
                action=PolicyAction.REWRITE,
                reason=f"tool_quarantined:{reason}",
                injected_system=injected,
            )

        # ──────────────────────────────────────────
        # Primitive 4: Side-effect gating + idempotency
        # ──────────────────────────────────────────
        if is_side_effect:
            state.attempted_side_effect_calls[tool] = (
                state.attempted_side_effect_calls.get(tool, 0) + 1
            )

            # Max execution limit
            executed = state.executed_side_effect_calls.get(tool, 0)
            if executed >= self.cfg.side_effect_max_executed_per_run:
                self._emit(
                    "side_effect_limit_block",
                    state=state,
                    tool=tool, ticket_sig=t_sig, args_sig=args_sig,
                    executed=executed,
                    limit=self.cfg.side_effect_max_executed_per_run,
                    estimated_cost_avoided=round(self.estimate_tool_cost(tool), 4),
                )
                return PolicyDecision(
                    action=PolicyAction.BLOCK,
                    reason="side_effect_limit_exceeded",
                )

            # Set deterministic idempotency key
            call.idempotency_key = _hmac_hex(
                self.cfg.secret_key, f"idem:{t_sig}:{tool}:{args_sig}"
            )[:32]

        # ──────────────────────────────────────────
        # Primitive 1: Identical tool-call loop
        # ──────────────────────────────────────────
        if not is_side_effect and repeats >= self.cfg.repeat_toolcall_threshold:
            self._emit(
                "identical_toolcall_loop_block",
                state=state,
                tool=tool, args_sig=args_sig,
                repeats=repeats,
                estimated_cost_avoided=round(self.estimate_tool_cost(tool), 4),
            )
            return PolicyDecision(
                action=PolicyAction.BLOCK,
                reason="identical_toolcall_loop",
            )

        # ──────────────────────────────────────────
        # Primitive 2: Argument-jitter similarity
        # ──────────────────────────────────────────
        query_keys = self.cfg.query_arg_keys(tool)
        q_val: Optional[str] = None
        for k in query_keys:
            v = call.args.get(k)
            if isinstance(v, str) and v.strip():
                q_val = v
                break

        if q_val is not None:
            q_sig_set = _token_sig_set(self.cfg, q_val)
            hist = state.tool_query_sigs.setdefault(tool, [])

            similar = sum(
                1
                for prev in hist[-self.cfg.tool_loop_window:]
                if _overlap_similarity(prev, q_sig_set) >= self.cfg.arg_jitter_similarity_threshold
            )

            hist.append(q_sig_set)
            if len(hist) > self.cfg.tool_loop_window:
                state.tool_query_sigs[tool] = hist[-self.cfg.tool_loop_window:]

            if similar >= self.cfg.arg_jitter_repeat_threshold:
                # Quarantine the tool for the remainder of the run
                state.quarantined_tools[tool] = "arg_jitter_loop"

                injected = (
                    f"SYSTEM ALERT: You are repeatedly calling '{tool}' with near-identical queries. "
                    "Do not vary the query further and do not call this tool again. "
                    "Use the best available prior result to proceed. "
                    "If you cannot proceed safely, output an escalation packet."
                )
                self._emit(
                    "arg_jitter_loop_quarantine",
                    state=state,
                    tool=tool, args_sig=args_sig, similar=similar,
                    estimated_cost_avoided=round(self.estimate_tool_cost(tool), 4),
                )
                return PolicyDecision(
                    action=PolicyAction.REWRITE,
                    reason="arg_jitter_loop",
                    injected_system=injected,
                )

        # ──────────────────────────────────────────
        # Primitive 2b: Per-tool call cap
        # ──────────────────────────────────────────
        cap_decision = self._check_tool_call_cap(state, tool)
        if cap_decision is not None:
            return cap_decision

        # ──────────────────────────────────────────
        # Primitive 8: Multi-tool sequence loop detection
        # ──────────────────────────────────────────
        seq_decision = self._check_sequence_loop(state, tool)
        if seq_decision is not None:
            return seq_decision

        # All checks passed — count the executed call
        state.tool_call_counts[tool] = state.tool_call_counts.get(tool, 0) + 1

        # Store request signature in rolling window (only for calls that pass all checks)
        state.tool_stream.append(sig)
        if len(state.tool_stream) > self.cfg.tool_loop_window:
            state.tool_stream = state.tool_stream[-self.cfg.tool_loop_window:]

        return PolicyDecision(action=PolicyAction.ALLOW, reason="allow")

    def _check_tool_call_cap(self, state: GuardState, tool: str) -> Optional[PolicyDecision]:
        """Check per-tool call cap (catches smart reformulation loops).

        Uses per-tool policy limit if set, otherwise global max_calls_per_tool.
        Returns a REWRITE decision if the cap is exceeded, else None.

        Note: This counts calls that pass all prior checks (identical repeat,
        jitter, circuit breaker, side-effect gating, budget). Calls blocked by
        earlier primitives are not counted toward the cap. This means
        max_calls_per_tool=3 allows 3 *executed* calls, not 3 *attempted* calls.
        """
        cap = self.cfg.get_tool_max_calls(tool)
        if cap is None:
            return None

        count = state.tool_call_counts.get(tool, 0)
        if count >= cap:
            state.quarantined_tools[tool] = "max_calls_per_tool"
            injected = (
                f"SYSTEM ALERT: You have called '{tool}' {count} times this run. "
                "You must now use the information you have already gathered. "
                "Do not call this tool again. Synthesize your answer from prior results."
            )
            self._emit(
                "tool_call_cap_quarantine",
                state=state,
                tool=tool, count=count,
                limit=cap,
            )
            return PolicyDecision(
                action=PolicyAction.REWRITE,
                reason="max_calls_per_tool",
                injected_system=injected,
            )

        return None

    def _check_sequence_loop(self, state: GuardState, tool: str) -> Optional[PolicyDecision]:
        """Check for repeating multi-tool sequences (Primitive 8).

        Detects patterns like A→B→A→B (ping-pong) or A→B→C→A→B→C (circular).
        Uses tool names from the rolling tool_stream.

        Returns a REWRITE decision if a repeating sequence is detected, else None.
        """
        if not self.cfg.sequence_detection_enabled:
            return None

        # Extract tool names from executed history + current candidate call
        names = [s.name for s in state.tool_stream] + [tool]
        n = len(names)

        # Check each pattern length from 2 up to max_sequence_length
        for pattern_len in range(2, self.cfg.max_sequence_length + 1):
            needed = pattern_len * self.cfg.sequence_repeat_threshold
            if n < needed:
                continue

            # Extract the last `needed` tool names
            recent = names[-needed:]
            pattern = recent[:pattern_len]

            # Primitive 8 targets multi-tool loops, not single-tool repeats.
            if len(set(pattern)) < 2:
                continue

            # Check if the entire slice is the pattern repeated
            is_repeating = True
            for i in range(needed):
                if recent[i] != pattern[i % pattern_len]:
                    is_repeating = False
                    break

            if is_repeating:
                # Quarantine the current tool to break the cycle
                state.quarantined_tools[tool] = "sequence_loop"
                pattern_str = " → ".join(pattern)

                injected = (
                    f"SYSTEM ALERT: You are stuck in a repeating tool-call loop: "
                    f"{pattern_str} (repeated {self.cfg.sequence_repeat_threshold} times). "
                    f"Do not continue this pattern. Use the information you already have "
                    f"to proceed. If you cannot proceed safely, output an escalation packet."
                )
                self._emit(
                    "sequence_loop_detected",
                    state=state,
                    tool=tool,
                    pattern=pattern,
                    pattern_length=pattern_len,
                    repeats=self.cfg.sequence_repeat_threshold,
                    estimated_cost_avoided=round(self.estimate_tool_cost(tool), 4),
                )
                return PolicyDecision(
                    action=PolicyAction.REWRITE,
                    reason=f"sequence_loop:{pattern_str}",
                    injected_system=injected,
                )

        return None

    def on_tool_result(self, *, state: GuardState, call: ToolCall, result: ToolResult) -> None:
        """Update state after a tool call completes (or is served from cache)."""
        tool = call.name
        side_effect = self.cfg.is_side_effect_tool(tool, call.side_effect)

        args_sig = _args_sig(self.cfg, call.args, tool_name=tool)
        t_sig = _ticket_sig(self.cfg, call.ticket_id, state.run_id)

        # Defensive copy: prevent caller mutation from affecting cached entries.
        # We do NOT deep-copy payload (could be large); only the ToolResult wrapper.
        from dataclasses import replace as _dc_replace
        result = _dc_replace(result)

        # Fill payload signature if missing
        if result.payload_sig is None and result.payload is not None:
            result.payload_sig = _payload_sig(self.cfg, result.payload)

        # Infer side_effect_executed when possible
        if result.side_effect_executed is None:
            result.side_effect_executed = bool(side_effect and result.ok and not result.cached)

        # Cache successful non-cached results (respecting never_cache_tools).
        # Side-effect results go ONLY into the idempotency ledger (below),
        # not into the generic cache — prevents cross-ticket false suppression.
        if result.ok and not result.cached and self.cfg.is_cacheable(tool) and not side_effect:
            # Enforce state growth cap on result_cache
            if len(state.result_cache) < self.cfg.max_cache_entries:
                state.result_cache[(tool, args_sig)] = result
                state.result_cache_ts[(tool, args_sig)] = _now()

            # Update "unique" progress markers (bounded)
            if len(state.unique_tool_calls_seen) < self.cfg.max_unique_calls_tracked:
                state.unique_tool_calls_seen.add((tool, args_sig))
            if result.payload_sig and len(state.unique_tool_results_seen) < self.cfg.max_unique_calls_tracked:
                state.unique_tool_results_seen.add(result.payload_sig)

        # Side-effect ledger and executed counts
        if side_effect:
            key = (t_sig, tool, args_sig)
            if result.ok or result.side_effect_executed:
                state.idempotency_ledger[key] = result

            if result.side_effect_executed and not result.cached:
                state.executed_side_effect_calls[tool] = (
                    state.executed_side_effect_calls.get(tool, 0) + 1
                )

        # Reset error streaks on success (streaks must be consecutive)
        if result.ok:
            keys_to_reset = [k for k in state.error_streaks if k[0] == tool]
            for k in keys_to_reset:
                state.error_streaks[k] = 0

        # Primitive 3: Error retry circuit breaker
        if not result.ok:
            err_class = _classify_error_code(result.error_code)
            streak_key = (tool, err_class)
            state.error_streaks[streak_key] = state.error_streaks.get(streak_key, 0) + 1

            if (
                state.error_streaks[streak_key] >= self.cfg.error_retry_threshold
                and self.cfg.should_quarantine_error(err_class)
            ):
                state.quarantined_tools[tool] = f"error_retry:{err_class}"
                self._emit(
                    "tool_quarantined_error_retry",
                    state=state,
                    tool=tool, error_class=err_class,
                    streak=state.error_streaks[streak_key],
                )

        # Primitive 6: Cost tracking
        # Every executed tool call costs money — failures are NOT free.
        # Only cached results (never sent to the API) are free.
        if not result.cached:
            est = self.estimate_tool_cost(tool)
            state.cumulative_cost += est
            state.cost_events.append(CostEvent(
                event="cost_incurred", tool=tool,
                amount=est, cumulative=state.cumulative_cost,
            ))
            if len(state.cost_events) > self.cfg.max_cost_events:
                state.cost_events = state.cost_events[-self.cfg.max_cost_events:]

    def on_llm_output(self, *, state: GuardState, text: Any) -> Optional[PolicyDecision]:
        """Inspect an assistant output to detect stall/no-state-change.

        In shadow_mode, non-None decisions are suppressed and None is returned.
        """
        decision = self._evaluate_llm_output(state=state, text=text)

        if decision is not None and self.cfg.shadow_mode:
            self._emit(
                "shadow_would_intervene", state=state,
                original_action=decision.action.value,
                original_reason=decision.reason,
            )
            return None

        return decision

    def _evaluate_llm_output(self, *, state: GuardState, text: Any) -> Optional[PolicyDecision]:
        """Internal: evaluate LLM output. Returns the raw decision (no shadow suppression)."""

        if not isinstance(text, str):
            return None

        if text == "":
            return None

        # If we previously asked for a forced outcome, try to parse it.
        if state.stall_rewrite_attempts > 0:
            parsed = self._parse_forced_outcome(text)
            if parsed is not None:
                self._emit(
                    "forced_outcome_parsed",
                    state=state,
                    action=parsed.action.value,
                    reason=parsed.reason,
                )
                return parsed

            # Exhausted rewrite attempts → deterministic escalation
            if state.stall_rewrite_attempts >= self.cfg.stall_rewrite_max_attempts:
                packet = {
                    "route": None,
                    "summary": "Agent failed to comply with forced finalize/escalate after stall detection.",
                    "missing_info": [],
                    "tags": ["loop_detected", "stall", "noncompliant"],
                    "reason": "stall",
                }
                self._emit("stall_deterministic_escalate_noncompliant", state=state)
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason="stall_noncompliant_forced_outcome",
                    escalation_packet=packet,
                )

        # Primitive 5: Stall detection (dual approach)
        #
        # Two independent signals, either can trigger intervention:
        # A) Token overlap: identical or near-identical text repeated (threshold 0.92)
        # B) Stall patterns: apology/filler language with no tool progress (catches
        #    "I apologize" → "Sorry for the delay" → "I'm still looking" which have
        #    LOW token overlap but HIGH stall intent)
        #
        cur_marker = (len(state.unique_tool_calls_seen), len(state.unique_tool_results_seen))
        if cur_marker != state.last_progress_marker:
            # Progress happened via tools; reset ALL stall counters.
            state.last_progress_marker = cur_marker
            state.stall_streak = 0
            state.stall_pattern_streak = 0
            state.last_assistant_token_sigs = _token_sig_set(self.cfg, text)
            return None

        # No new tool progress. Check both signals.

        # Signal A: Token overlap similarity
        cur_tokens = _token_sig_set(self.cfg, text)
        prev_tokens = state.last_assistant_token_sigs

        if prev_tokens is None:
            state.last_assistant_token_sigs = cur_tokens
            state.stall_streak = 0
            # Still check pattern on first message (no overlap baseline yet)
        else:
            sim = _overlap_similarity(prev_tokens, cur_tokens)
            if sim >= self.cfg.stall_text_similarity_threshold:
                state.stall_streak += 1
            else:
                state.stall_streak = 0
            state.last_assistant_token_sigs = cur_tokens

        # Signal B: Stall pattern detection (apology/filler/working language)
        pattern_score = _stall_pattern_score(text)
        if pattern_score >= self.cfg.stall_pattern_threshold:
            state.stall_pattern_streak += 1
        else:
            state.stall_pattern_streak = 0

        # Trigger intervention if EITHER signal exceeds threshold
        should_intervene = (
            state.stall_streak >= self.cfg.no_state_change_threshold
            or state.stall_pattern_streak >= self.cfg.stall_pattern_streak_threshold
        )

        if should_intervene:
            # First attempt: force the model to produce a final outcome.
            if state.stall_rewrite_attempts < self.cfg.stall_rewrite_max_attempts:
                state.stall_rewrite_attempts += 1
                # Capture trigger reason BEFORE resetting streaks
                trigger = "pattern" if state.stall_pattern_streak >= self.cfg.stall_pattern_streak_threshold else "similarity"
                state.stall_streak = 0
                state.stall_pattern_streak = 0

                injected = self._forced_outcome_system_prompt()
                self._emit(
                    "stall_forced_rewrite",
                    state=state,
                    trigger=trigger,
                    pattern_score=round(pattern_score, 3),
                    attempts=state.stall_rewrite_attempts,
                )
                return PolicyDecision(
                    action=PolicyAction.REWRITE,
                    reason="no_state_change_loop",
                    injected_system=injected,
                )

            # Deterministic fallback escalation.
            packet = {
                "route": None,
                "summary": "Agent stalled (no-state-change loop detected).",
                "missing_info": [],
                "tags": ["loop_detected", "stall"],
                "reason": "stall",
            }
            self._emit("stall_deterministic_escalate", state=state)
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="stall_fallback_escalate",
                escalation_packet=packet,
            )

        return None

    def get_run_summary(self, state: GuardState) -> Dict[str, Any]:
        """Return a summary of guard activity for this run."""
        return {
            "run_id": state.run_id,
            "cumulative_cost_usd": round(state.cumulative_cost, 4),
            "cost_limit_usd": self.cfg.max_cost_per_run,
            "tools_quarantined": dict(state.quarantined_tools),
            "side_effects_attempted": dict(state.attempted_side_effect_calls),
            "side_effects_executed": dict(state.executed_side_effect_calls),
            "unique_tool_calls": len(state.unique_tool_calls_seen),
            "unique_tool_results": len(state.unique_tool_results_seen),
            "stall_streak": state.stall_streak,
            "stall_rewrite_attempts": state.stall_rewrite_attempts,
            "cost_events": len(state.cost_events),
        }

# ═══════════════════════════════════════════
# Middleware
# ═══════════════════════════════════════════

logger = logging.getLogger("aura_guard")

class GuardDenied(Exception):
    """Raised by guard.run() and @guard.protect when a tool call is denied.

    Attributes:
        action: The PolicyAction (BLOCK, REWRITE, ESCALATE, FINALIZE)
        reason: Human-readable reason for the denial
        decision: The full PolicyDecision object
    """

    def __init__(self, decision: PolicyDecision):
        self.action = decision.action
        self.reason = decision.reason
        self.decision = decision
        super().__init__(f"GuardDenied: {decision.action.value} — {decision.reason}")


class AgentGuard:
    """Simplified guard wrapper for custom agent loops.

    WARNING: AgentGuard holds per-run state and is NOT thread-safe.
    Do not share a single AgentGuard instance across threads, async tasks,
    or concurrent requests. Create one AgentGuard per agent run.
    For web servers, create a new instance per request.

    Wraps AuraGuard into three methods:
    - check_tool()     → call before each tool execution
    - record_result()  → call after each tool execution
    - check_output()   → call after each LLM output

    Args:
        max_cost_per_run: Optional USD spending cap per agent run.
        tool_costs: Optional dict mapping tool names to per-call costs (USD).
        default_tool_cost: Default per-call cost if not in tool_costs.
        side_effect_tools: Set of tool names that perform mutations.
        max_calls_per_tool: Max times any single tool can be called per run.
        secret_key: HMAC key for signatures. Change in production.
        telemetry: Optional Telemetry instance for event emission.
        config: Optional full AuraGuardConfig (overrides other params).
    """

    def __init__(
        self,
        *,
        max_cost_per_run: Optional[float] = None,
        tool_costs: Optional[Dict[str, float]] = None,
        default_tool_cost: float = 0.04,
        side_effect_tools: Optional[set] = None,
        max_calls_per_tool: Optional[int] = None,
        secret_key: Optional[bytes] = None,
        telemetry: Optional[Telemetry] = None,
        config: Optional[AuraGuardConfig] = None,
        shadow_mode: bool = False,
        strict_mode: bool = False,
    ):
        self._creator_thread = threading.get_ident()

        if config is not None:
            self._cfg = config
        else:
            kwargs: Dict[str, Any] = {}
            if max_cost_per_run is not None:
                kwargs["max_cost_per_run"] = max_cost_per_run
            if side_effect_tools is not None:
                kwargs["side_effect_tools"] = side_effect_tools
            if max_calls_per_tool is not None:
                kwargs["max_calls_per_tool"] = max_calls_per_tool
            if secret_key is not None:
                kwargs["secret_key"] = secret_key
            if shadow_mode:
                kwargs["shadow_mode"] = True

            cost_model = CostModel(
                default_tool_call_cost=default_tool_cost,
                tool_cost_by_name=dict(tool_costs or {}),
            )
            kwargs["cost_model"] = cost_model
            self._cfg = AuraGuardConfig(**kwargs)

        self._shadow = self._cfg.shadow_mode
        self._strict_mode = strict_mode
        self._guard = AuraGuard(config=self._cfg, telemetry=telemetry)
        self._state = self._guard.new_state()
        self._last_call: Optional[ToolCall] = None

        # Public counters
        self.blocks: int = 0
        self.cache_hits: int = 0
        self.rewrite_decisions: int = 0   # guard returned REWRITE (tool denied with instruction)
        self.escalations: int = 0
        self.tool_calls_executed: int = 0  # actually sent to the tool (success + failure)
        self.tool_calls_failed: int = 0    # executed but returned error
        self.tool_calls_denied: int = 0    # guard prevented execution (block + cache + rewrite + escalate)
        self.shadow_would_deny: int = 0    # shadow mode: would have denied but allowed
        self.missed_results: int = 0       # check_tool called before prior ALLOW had record_result
        self._interventions: List[Dict[str, str]] = []

    # ─────────────────────────────────────────
    # 3-Method API
    # ─────────────────────────────────────────

    def _assert_same_thread(self) -> None:
        if threading.get_ident() != self._creator_thread:
            raise RuntimeError(
                "AgentGuard is not thread-safe. Create one instance per agent run. See docs."
            )

    def check_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
    ) -> PolicyDecision:
        """Check whether a tool call should proceed.

        Returns a PolicyDecision. Inspect decision.action:
        - "allow"    → execute the tool, then call record_result()
        - "cache"    → use decision.cached_result, skip execution
        - "block"    → skip execution
        - "rewrite"  → inject decision.injected_system into next model prompt
        - "escalate" → stop the agent run, use decision.escalation_packet
        - "finalize" → stop the agent run, use decision.finalized_output
        """
        self._assert_same_thread()
        if self._last_call is not None:
            if self._strict_mode:
                raise RuntimeError(
                    "check_tool() called without a preceding record_result() for "
                    f"tool '{self._last_call.name}'."
                )

            logger.warning(
                "check_tool() called without a preceding record_result() for tool '%s'. "
                "The previous result will not be recorded. This may cause the guard to "
                "undercount tool calls.",
                self._last_call.name,
            )
            self.missed_results += 1

        call = ToolCall(
            name=name,
            args=args or {},
            ticket_id=ticket_id,
            side_effect=side_effect,
        )
        if self._shadow and hasattr(self._guard, "_evaluate_tool_call"):
            decision = self._guard._evaluate_tool_call(state=self._state, call=call)
        else:
            decision = self._guard.on_tool_call_request(state=self._state, call=call)

        # Shadow mode: log the decision but override to ALLOW
        if self._shadow and decision.action != PolicyAction.ALLOW:
            self.shadow_would_deny += 1
            self._interventions.append({"tool": name, "action": "shadow_deny", "reason": decision.reason})
            self._guard._emit(
                "shadow_would_deny", state=self._state,
                tool=name, original_action=decision.action.value,
                original_reason=decision.reason,
            )
            # Let the tool execute — return ALLOW
            self._last_call = call
            return PolicyDecision(action=PolicyAction.ALLOW, reason="shadow_allow")

        # Only set _last_call on ALLOW — prevents sloppy integrators from
        # calling record_result() after a block and corrupting state.
        if decision.action == PolicyAction.ALLOW:
            self._last_call = call
        elif decision.action == PolicyAction.BLOCK:
            self._last_call = None
            self.blocks += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "block", "reason": decision.reason})
        elif decision.action == PolicyAction.CACHE:
            self._last_call = None
            self.cache_hits += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "cache", "reason": decision.reason})
            # Auto-record cached results
            if decision.cached_result:
                self._guard.on_tool_result(
                    state=self._state, call=call, result=decision.cached_result,
                )
        elif decision.action == PolicyAction.REWRITE:
            self._last_call = None
            self.rewrite_decisions += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": "rewrite", "reason": decision.reason})
        elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
            self._last_call = None
            self.escalations += 1
            self.tool_calls_denied += 1
            self._interventions.append({"tool": name, "action": decision.action.value, "reason": decision.reason})

        return decision

    def record_result(
        self,
        ok: bool = True,
        payload: Any = None,
        error_code: Optional[str] = None,
        side_effect_executed: Optional[bool] = None,
    ) -> None:
        """Record the result of the most recent tool call.

        Call this after executing a tool that was allowed by check_tool().

        Args:
            ok: Whether the tool call succeeded.
            payload: The tool's return value (may contain PII — cached in memory only).
            error_code: A coarse error classification (e.g. "429", "timeout", "ValueError").
                        Do not pass raw exception messages — error_code is persisted in
                        serialized state. Use type(exc).__name__ or a short category string.
            side_effect_executed: Set to True if the side effect was executed (even on timeout).
        """
        self._assert_same_thread()
        if self._last_call is None:
            if self._strict_mode:
                raise RuntimeError(
                    "record_result() called without a preceding check_tool(). "
                    "In strict_mode, every record_result() must follow a check_tool() "
                    "that returned ALLOW."
                )
            return
        result = ToolResult(ok=ok, payload=payload, error_code=error_code, side_effect_executed=side_effect_executed)
        self._guard.on_tool_result(state=self._state, call=self._last_call, result=result)
        self._last_call = None
        self.tool_calls_executed += 1
        if not ok:
            self.tool_calls_failed += 1

    def check_output(self, text: str) -> Optional[PolicyDecision]:
        """Check an assistant's text output for stall/loop behavior.

        Returns None if no intervention needed, or a PolicyDecision if the
        agent should be rewritten, escalated, or finalized.
        """
        self._assert_same_thread()
        if self._shadow and hasattr(self._guard, "_evaluate_llm_output"):
            decision = self._guard._evaluate_llm_output(state=self._state, text=text)
        else:
            decision = self._guard.on_llm_output(state=self._state, text=text)
        if decision is not None:
            # Shadow mode: log but don't enforce
            if self._shadow:
                self.shadow_would_deny += 1
                self._guard._emit(
                    "shadow_would_intervene", state=self._state,
                    original_action=decision.action.value,
                    original_reason=decision.reason,
                )
                return None

            if decision.action == PolicyAction.REWRITE:
                self.rewrite_decisions += 1
                self._interventions.append({"tool": "_llm_output", "action": "rewrite", "reason": decision.reason})
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                self.escalations += 1
                self._interventions.append({"tool": "_llm_output", "action": "escalate", "reason": decision.reason})
        return decision

    def record_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0,
                      cost_override: Optional[float] = None) -> None:
        """Report actual LLM token usage for accurate cost tracking.

        Call after each LLM API response with the real usage numbers.

        Example:
            guard.record_tokens(input_tokens=resp.usage.input_tokens,
                                output_tokens=resp.usage.output_tokens)
        """
        self._guard.record_tokens(
            state=self._state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_override=cost_override,
        )

    def run(
        self,
        tool_name: str,
        fn: Callable[..., Any],
        *,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool function with full guard protection in one call.

        Handles the complete check_tool → execute → record_result cycle.

        Returns the tool's return value on ALLOW.
        Returns cached payload on CACHE (may be None if restored from serialization).
        Raises GuardDenied on BLOCK, REWRITE, ESCALATE, or FINALIZE.

        All tool arguments MUST be passed as keyword arguments. This ensures the
        guard's signature tracker sees the same arguments the function receives.
        Positional arguments are not supported — use the 3-method API
        (check_tool / record_result) if your tool requires positional args.

        Example:
            result = guard.run("search_kb", search_kb, query="refund policy")
            result = guard.run("refund", refund, order_id="o1", ticket_id="t1", side_effect=True)

        Args:
            tool_name: Name of the tool (used for guard tracking).
            fn: The callable to execute if the guard allows it.
            ticket_id: Optional ticket/session ID for idempotency.
            side_effect: Override whether this tool is a side-effect.
            **kwargs: Keyword arguments passed to fn AND used as tool args for guard signature tracking.

        .. note:: Reserved keyword arguments

           ``ticket_id`` and ``side_effect`` are consumed by the guard wrapper
           and are NOT forwarded to ``fn``. If your tool function has parameters
           named ``ticket_id`` or ``side_effect``, use the 3-method API instead.

        .. warning:: Side-effect timeout handling

           If a side-effect tool succeeds server-side but times out locally,
           run() cannot mark side_effect_executed=True on the error path.
           For tools where ambiguous execution is possible (payments, emails,
           cancellations), use the 3-method API (check_tool / record_result)
           so you can set side_effect_executed=True on timeout.
        """
        decision = self.check_tool(
            tool_name, args=kwargs, ticket_id=ticket_id, side_effect=side_effect,
        )

        if decision.action == PolicyAction.ALLOW:
            try:
                result = fn(**kwargs)
                self.record_result(ok=True, payload=result)
                return result
            except Exception as exc:
                self.record_result(ok=False, error_code=type(exc).__name__)
                raise

        if decision.action == PolicyAction.CACHE:
            return decision.cached_result.payload if decision.cached_result else None

        raise GuardDenied(decision)

    def protect(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        tool_name: Optional[str] = None,
        ticket_id: Optional[str] = None,
        side_effect: Optional[bool] = None,
    ) -> Any:
        """Decorator that wraps a function with guard protection.

        Every call to the decorated function goes through check_tool → execute → record_result.

        IMPORTANT: The decorated function must be called with keyword arguments only.
        Positional arguments are not tracked by the guard and will cause a TypeError.
        If your tool requires positional args, use the 3-method API instead.

        Can be used with or without arguments:

            @guard.protect
            def search_kb(query): ...

            @guard.protect(tool_name="custom_name", side_effect=True)
            def charge_card(customer_id, amount): ...

        Args:
            fn: The function to wrap (when used without parentheses).
            tool_name: Override the tool name (defaults to fn.__name__).
            ticket_id: Optional ticket/session ID for idempotency.
            side_effect: Override whether this tool is a side-effect.

        .. note:: Reserved keyword arguments

           ``ticket_id`` and ``side_effect`` are consumed by the guard wrapper
           and are NOT forwarded to ``fn``. If your tool function has parameters
           named ``ticket_id`` or ``side_effect``, use the 3-method API instead.

        .. warning:: Side-effect timeout handling

           If a side-effect tool succeeds server-side but times out locally,
           run() cannot mark side_effect_executed=True on the error path.
           For tools where ambiguous execution is possible (payments, emails,
           cancellations), use the 3-method API (check_tool / record_result)
           so you can set side_effect_executed=True on timeout.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            name = tool_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if args:
                    raise TypeError(
                        f"@guard.protect requires keyword arguments only. "
                        f"Call {func.__name__}() with keyword args so the guard "
                        f"can track them for deduplication. "
                        f"Got {len(args)} positional arg(s)."
                    )
                return self.run(
                    name, func,
                    ticket_id=ticket_id, side_effect=side_effect,
                    **kwargs,
                )

            return wrapper

        if fn is not None:
            # Called without parentheses: @guard.protect
            return decorator(fn)
        # Called with parentheses: @guard.protect(tool_name="...", ...)
        return decorator


    # ─────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────

    @property
    def cost_spent(self) -> float:
        """Cumulative cost spent this run (USD) — includes tool + token costs."""
        return round(self._state.cumulative_cost, 4)

    @property
    def reported_token_cost(self) -> float:
        """Actual token costs reported by the integrator (USD)."""
        return round(self._state.reported_token_cost, 4)

    @property
    def cost_limit(self) -> Optional[float]:
        """Cost budget limit for this run (USD), or None."""
        return self._cfg.max_cost_per_run

    @property
    def cost_remaining(self) -> Optional[float]:
        """Remaining cost budget (USD), or None if no limit."""
        if self._cfg.max_cost_per_run is None:
            return None
        return round(self._cfg.max_cost_per_run - self._state.cumulative_cost, 4)

    @property
    def quarantined_tools(self) -> Dict[str, str]:
        """Tools that have been quarantined in this run, with reasons."""
        return dict(self._state.quarantined_tools)

    @property
    def rewrites(self) -> int:
        """Backward-compat alias for rewrite_decisions."""
        return self.rewrite_decisions

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics for this run."""
        return {
            "tool_calls_executed": self.tool_calls_executed,
            "tool_calls_failed": self.tool_calls_failed,
            "tool_calls_cached": self.cache_hits,
            "tool_calls_denied": self.tool_calls_denied,
            "blocks": self.blocks,
            "cache_hits": self.cache_hits,
            "rewrite_decisions": self.rewrite_decisions,
            "escalations": self.escalations,
            "cost_spent_usd": self.cost_spent,
            "reported_token_cost_usd": self.reported_token_cost,
            "cost_limit_usd": self.cost_limit,
            "cost_remaining_usd": self.cost_remaining,
            "quarantined_tools": self.quarantined_tools,
            "unique_tool_calls": len(self._state.unique_tool_calls_seen),
            "stall_streak": self._state.stall_streak,
            "shadow_mode": self._shadow,
            "shadow_would_deny": self.shadow_would_deny,
            "missed_results": self.missed_results,
            "interventions_count": len(self._interventions),
        }

    def report_data(self) -> Dict[str, Any]:
        """Structured run report for programmatic use / JSON export.

        Returns a dict with:
        - summary: overall call counts and efficiency
        - cost: spending breakdown
        - interventions_by_primitive: which enforcement rules triggered
        - interventions_by_tool: per-tool breakdown
        - side_effects: attempted vs executed
        - quarantined_tools: which tools were quarantined and why
        - top_interventions: most frequent intervention reasons
        - interventions: full intervention log
        """
        total_calls = self.tool_calls_executed + self.tool_calls_denied
        efficiency = round(self.tool_calls_executed / total_calls * 100, 1) if total_calls > 0 else 100.0

        by_primitive: Dict[str, int] = {}
        by_tool: Dict[str, Dict[str, int]] = {}
        reason_counts: Dict[str, int] = {}

        for iv in self._interventions:
            reason = iv["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            tool = iv["tool"]
            if tool not in by_tool:
                by_tool[tool] = {"block": 0, "cache": 0, "rewrite": 0, "escalate": 0, "finalize": 0, "shadow_deny": 0}
            action = iv["action"]
            if action in by_tool[tool]:
                by_tool[tool][action] += 1

            primitive = self._classify_primitive(reason)
            by_primitive[primitive] = by_primitive.get(primitive, 0) + 1

        top = sorted(reason_counts.items(), key=lambda x: -x[1])


        return {
            "summary": {
                "total_calls": total_calls,
                "executed": self.tool_calls_executed,
                "denied": self.tool_calls_denied,
                "efficiency_pct": efficiency,
                "blocks": self.blocks,
                "cache_hits": self.cache_hits,
                "rewrites": self.rewrite_decisions,
                "escalations": self.escalations,
            },
            "cost": {
                "spent_usd": self.cost_spent,
                "limit_usd": self.cost_limit,
                "remaining_usd": self.cost_remaining,
                "token_cost_usd": self.reported_token_cost,
            },
            "interventions_by_primitive": by_primitive,
            "interventions_by_tool": by_tool,
            "side_effects": {
                "attempted": dict(self._state.attempted_side_effect_calls),
                "executed": dict(self._state.executed_side_effect_calls),
            },
            "quarantined_tools": self.quarantined_tools,
            "top_interventions": [{"reason": r, "count": c} for r, c in top[:10]],
            "stall_streak": self._state.stall_streak,
            "shadow_mode": self._shadow,
            "shadow_would_deny": self.shadow_would_deny,
            "interventions": list(self._interventions),
        }

    @staticmethod
    def _classify_primitive(reason: str) -> str:
        """Map a decision reason to its enforcement primitive name."""
        if reason.startswith("identical_toolcall"):
            return "repeat_detection"
        if reason.startswith("arg_jitter") or reason.startswith("query_jitter"):
            return "jitter_detection"
        if reason.startswith("error_") or reason.startswith("quarantine"):
            return "circuit_breaker"
        if reason.startswith("idempotent") or reason.startswith("side_effect"):
            return "side_effect_gating"
        if reason.startswith("stall") or reason.startswith("no_state_change"):
            return "stall_detection"
        if reason.startswith("cost_") or reason.startswith("budget"):
            return "cost_budget"
        if reason.startswith("policy_") or reason.startswith("tool_denied"):
            return "tool_policy"
        if reason.startswith("sequence_loop"):
            return "sequence_detection"
        if reason.startswith("max_calls"):
            return "per_tool_cap"
        if reason == "shadow_allow":
            return "shadow_mode"
        return "other"

    def report(self) -> str:
        """Formatted text report of guard activity for this run.

        Returns a human-readable string ready to print or log.
        """
        d = self.report_data()
        s = d["summary"]
        c = d["cost"]
        lines = []

        lines.append("=" * 50)
        lines.append("AURAGUARD RUN REPORT")
        lines.append("=" * 50)
        lines.append("")

        lines.append(f"Run efficiency: {s['efficiency_pct']}% productive ({s['executed']} executed, {s['denied']} denied)")
        if s["denied"] > 0:
            parts = []
            if s["cache_hits"]:
                parts.append(f"Cached: {s['cache_hits']}")
            if s["blocks"]:
                parts.append(f"Blocked: {s['blocks']}")
            if s["rewrites"]:
                parts.append(f"Rewritten: {s['rewrites']}")
            if s["escalations"]:
                parts.append(f"Escalated: {s['escalations']}")
            lines.append(f"  {', '.join(parts)}")
        lines.append("")

        if c["limit_usd"] is not None:
            pct_remaining = round(c["remaining_usd"] / c["limit_usd"] * 100, 0) if c["limit_usd"] > 0 else 0
            lines.append(f"Cost: ${c['spent_usd']:.2f} spent / ${c['limit_usd']:.2f} budget ({pct_remaining:.0f}% remaining)")
        else:
            lines.append(f"Cost: ${c['spent_usd']:.2f} spent (no budget limit)")
        if c["token_cost_usd"] > 0:
            lines.append(f"  Token cost: ${c['token_cost_usd']:.2f}")
        lines.append("")

        se = d["side_effects"]
        if se["attempted"] or se["executed"]:
            executed_total = sum(se["executed"].values())
            attempted_total = sum(se["attempted"].values())
            blocked_total = attempted_total - executed_total
            lines.append(f"Side-effects: {executed_total} executed, {blocked_total} blocked")
            for tool, count in se["executed"].items():
                attempted = se["attempted"].get(tool, count)
                if attempted > count:
                    lines.append(f"  {tool}: {count} executed, {attempted - count} blocked")
                else:
                    lines.append(f"  {tool}: {count} executed")
            lines.append("")

        by_p = d["interventions_by_primitive"]
        if by_p:
            lines.append("Interventions by primitive:")
            for prim, count in sorted(by_p.items(), key=lambda x: -x[1]):
                lines.append(f"  [{count:>3}x] {prim}")
            lines.append("")

        top = d["top_interventions"]
        if top:
            lines.append("Top interventions:")
            for item in top[:5]:
                lines.append(f"  [{item['count']:>3}x] {item['reason']}")
            lines.append("")

        qt = d["quarantined_tools"]
        if qt:
            lines.append("Quarantined tools:")
            for tool, reason in qt.items():
                lines.append(f"  {tool} ({reason})")
            lines.append("")

        if d["stall_streak"] > 0:
            lines.append(f"Stall streak: {d['stall_streak']} consecutive stall detections")
            lines.append("")

        if d["shadow_mode"]:
            lines.append(f"Shadow mode: ON ({d['shadow_would_deny']} calls would have been denied)")
            lines.append("")

        return "\n".join(lines)

    @property
    def summary(self) -> Dict[str, Any]:
        """Alias for stats (for LangChain adapter compatibility)."""
        return self.stats

    def reset(self) -> None:
        """Reset state for a new run (same config)."""
        self._state = self._guard.new_state()
        self._last_call = None
        self.blocks = 0
        self.cache_hits = 0
        self.rewrite_decisions = 0
        self.escalations = 0
        self.tool_calls_executed = 0
        self.tool_calls_failed = 0
        self.tool_calls_denied = 0
        self.shadow_would_deny = 0
        self.missed_results = 0
        self._interventions = []

# ═══════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════

__all__ = [
    "AgentGuard",
    "GuardDenied",
    "AuraGuard",
    "AuraGuardConfig",
    "GuardState",
    "CostModel",
    "CostEvent",
    "PolicyAction",
    "PolicyDecision",
    "ToolAccess",
    "ToolPolicy",
    "ToolCall",
    "ToolCallSig",
    "ToolResult",
    "Telemetry",
]
