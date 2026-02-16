"""aura_guard.middleware

Simplified wrapper for Aura Guard — the "3-method API".

For developers who want minimal integration overhead.
Reduces the full guard API to: check_tool(), record_result(), check_output().

Usage:
    from aura_guard import AgentGuard

    guard = AgentGuard(secret_key=b"your-secret-key", max_cost_per_run=0.50)

    # Before each tool call
    decision = guard.check_tool("search_kb", args={"query": "refund policy"})
    if decision.action == "allow":
        result = execute_tool(...)
        guard.record_result(ok=True, payload=result)
    else:
        handle_decision(decision)

    # After each LLM output
    stall = guard.check_output("I apologize for the inconvenience...")
    if stall:
        handle_stall(stall)
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .config import AuraGuardConfig, CostModel
from .guard import AuraGuard, GuardState
from .telemetry import Telemetry
from .types import PolicyAction, PolicyDecision, ToolCall, ToolResult

import logging

logger = logging.getLogger("aura_guard")


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
        decision = self._guard.on_tool_call_request(state=self._state, call=call)

        # Shadow mode: log the decision but override to ALLOW
        if self._shadow and decision.action != PolicyAction.ALLOW:
            self.shadow_would_deny += 1
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
        elif decision.action == PolicyAction.CACHE:
            self._last_call = None
            self.cache_hits += 1
            self.tool_calls_denied += 1
            # Auto-record cached results
            if decision.cached_result:
                self._guard.on_tool_result(
                    state=self._state, call=call, result=decision.cached_result,
                )
        elif decision.action == PolicyAction.REWRITE:
            self._last_call = None
            self.rewrite_decisions += 1
            self.tool_calls_denied += 1
        elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
            self._last_call = None
            self.escalations += 1
            self.tool_calls_denied += 1

        return decision

    def record_result(
        self,
        ok: bool = True,
        payload: Any = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Record the result of the most recent tool call.

        Call this after executing a tool that was allowed by check_tool().
        """
        self._assert_same_thread()
        if self._last_call is None:
            return
        result = ToolResult(ok=ok, payload=payload, error_code=error_code)
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
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                self.escalations += 1
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
        }

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
