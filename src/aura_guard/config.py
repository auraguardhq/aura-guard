"""aura_guard.config

Configuration for Aura Guard.

All thresholds and policies are explicit with conservative defaults.

Important: provide a unique secret_key per environment/tenant if you emit
telemetry. The default key is for local development/testing only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


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
    # If an agent calls the same tool more than this many times in a run,
    # quarantine it regardless of argument similarity.
    max_calls_per_tool: Optional[int] = None   # None = no cap

    # Primitive 3: error retry circuit breaker
    error_retry_threshold: int = 2
    # Error codes/classes that should trigger quarantine when repeated.
    quarantine_error_codes: Set[str] = field(
        default_factory=lambda: {"401", "403", "429", "timeout", "connection_error", "5xx"}
    )

    # Primitive 4: side-effect gating
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

    # Telemetry redaction behavior
    # ----------------------------
    # If True, Aura Guard will avoid storing any raw text in guard state.
    # It will still compute token signatures for similarity.
    redact_text_in_state: bool = True

    def __post_init__(self) -> None:
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
        """Per-tool call limit: tool policy override > global max_calls_per_tool."""
        policy = self.tool_policies.get(tool_name)
        if policy and policy.max_calls is not None:
            return policy.max_calls
        return self.max_calls_per_tool
