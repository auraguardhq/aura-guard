"""aura_guard.adapters.langchain_adapter

Drop-in Aura Guard integration for LangChain agents.

Requires: pip install langchain-core (or pip install aura-guard[langchain])

Usage:
    from aura_guard.adapters.langchain_adapter import AuraCallbackHandler

    handler = AuraCallbackHandler(secret_key=b"your-secret-key", max_cost_per_run=1.00)
    agent = initialize_agent(tools=tools, llm=llm, callbacks=[handler])

    # After the agent runs:
    print(handler.summary)
    # {"cost_spent_usd": 0.12, "cost_saved_usd": 0.40, "blocks": 3, ...}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Union

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Provide a stub so the module can be imported without langchain
    BaseCallbackHandler = object  # type: ignore

from ..config import AuraGuardConfig, CostModel
from ..guard import AuraGuard, GuardState
from ..telemetry import Telemetry
from ..types import PolicyAction, PolicyDecision, ToolCall, ToolResult

logger = logging.getLogger("aura_guard.langchain")


class AuraToolBlocked(Exception):
    """Raised when Aura Guard blocks a tool call.

    Catch this in your error handler to gracefully handle blocked tools.

    Attributes:
        action: The PolicyAction (BLOCK, CACHE, REWRITE, ESCALATE, FINALIZE)
        reason: Human-readable reason for the block
        decision: The full PolicyDecision object
    """

    def __init__(self, decision: PolicyDecision):
        self.action = decision.action
        self.reason = decision.reason
        self.decision = decision
        super().__init__(f"Aura Guard {decision.action.value}: {decision.reason}")


class AuraCallbackHandler(BaseCallbackHandler):
    """LangChain callback that wraps Aura Guard enforcement.

    Drop this into your agent's callbacks list. It intercepts tool calls
    before execution and LLM outputs after generation.

    Args:
        max_cost_per_run: Optional USD spending cap per agent run.
        tool_costs: Dict mapping tool names to per-call costs (USD).
        default_tool_cost: Default per-call cost.
        side_effect_tools: Set of tool names that perform mutations.
        secret_key: Optional signing key for AuraGuardConfig.
        config: Optional full AuraGuardConfig (overrides other params).
        telemetry: Optional Telemetry instance.
        raise_on_block: If True (default), raise AuraToolBlocked on non-ALLOW
                        decisions. If False, log warnings instead.
    """

    # LangChain callback handler properties
    raise_error: bool = True

    def __init__(
        self,
        *,
        max_cost_per_run: Optional[float] = None,
        tool_costs: Optional[Dict[str, float]] = None,
        default_tool_cost: float = 0.04,
        side_effect_tools: Optional[Set[str]] = None,
        secret_key: Optional[bytes] = None,
        config: Optional[AuraGuardConfig] = None,
        telemetry: Optional[Telemetry] = None,
        raise_on_block: bool = True,
    ):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for the LangChain adapter. "
                "Install it with: pip install langchain-core\n"
                "Or: pip install aura-guard[langchain]"
            )

        if config is not None:
            self._cfg = config
        else:
            kwargs: Dict[str, Any] = {}
            if max_cost_per_run is not None:
                kwargs["max_cost_per_run"] = max_cost_per_run
            if side_effect_tools is not None:
                kwargs["side_effect_tools"] = side_effect_tools
            if secret_key is not None:
                kwargs["secret_key"] = secret_key
            cost_model = CostModel(
                default_tool_call_cost=default_tool_cost,
                tool_cost_by_name=dict(tool_costs or {}),
            )
            kwargs["cost_model"] = cost_model
            self._cfg = AuraGuardConfig(**kwargs)

        self._guard = AuraGuard(config=self._cfg, telemetry=telemetry)
        self._state = self._guard.new_state()
        self._raise_on_block = raise_on_block

        # Track decisions and costs
        self._decisions: List[Dict[str, Any]] = []
        self._cost_saved: float = 0.0
        self._blocks: int = 0
        self._cache_hits: int = 0
        self._rewrites: int = 0
        self._escalations: int = 0

        # Pending tool call for result tracking
        self._pending_call: Optional[ToolCall] = None

    # ─────────────────────────────────────────
    # LangChain Callback Hooks
    # ─────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Intercept tool call before execution."""
        tool_name = serialized.get("name") or serialized.get("id", "unknown")

        # Parse args
        args: Dict[str, Any] = {}
        if isinstance(input_str, str):
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, TypeError):
                args = {"raw_input": input_str}
        elif isinstance(input_str, dict):
            args = input_str

        call = ToolCall(name=tool_name, args=args)
        self._pending_call = call
        decision = self._guard.on_tool_call_request(state=self._state, call=call)

        self._decisions.append({
            "tool": tool_name,
            "action": decision.action.value,
            "reason": decision.reason,
        })

        if decision.action == PolicyAction.ALLOW:
            return

        # Track cost saved
        est = self._guard.estimate_tool_cost(tool_name)
        self._cost_saved += est

        if decision.action == PolicyAction.BLOCK:
            self._blocks += 1
            logger.warning("Aura Guard BLOCK tool '%s': %s", tool_name, decision.reason)
            if self._raise_on_block:
                raise AuraToolBlocked(decision)

        elif decision.action == PolicyAction.CACHE:
            self._cache_hits += 1
            logger.info("Aura Guard CACHE tool '%s': %s", tool_name, decision.reason)
            if self._raise_on_block:
                raise AuraToolBlocked(decision)

        elif decision.action == PolicyAction.REWRITE:
            self._rewrites += 1
            logger.warning("Aura Guard REWRITE for tool '%s': %s", tool_name, decision.reason)
            if self._raise_on_block:
                raise AuraToolBlocked(decision)

        elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
            self._escalations += 1
            logger.warning("Aura Guard %s: %s", decision.action.value, decision.reason)
            if self._raise_on_block:
                raise AuraToolBlocked(decision)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record tool result for progress tracking."""
        if self._pending_call is not None:
            result = ToolResult(ok=True, payload=output)
            self._guard.on_tool_result(
                state=self._state, call=self._pending_call, result=result,
            )
            self._pending_call = None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record tool error."""
        if self._pending_call is not None:
            # Don't record AuraToolBlocked as an error
            if isinstance(error, AuraToolBlocked):
                self._pending_call = None
                return
            result = ToolResult(ok=False, error_code=type(error).__name__)
            self._guard.on_tool_result(
                state=self._state, call=self._pending_call, result=result,
            )
            self._pending_call = None

    def on_llm_end(
        self,
        response: Any,  # LLMResult
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Check for stall/no-state-change after LLM output."""
        try:
            text = response.generations[0][0].text
        except (IndexError, AttributeError):
            return

        if not text:
            return

        decision = self._guard.on_llm_output(state=self._state, text=text)
        if decision is not None:
            if decision.action == PolicyAction.REWRITE:
                self._rewrites += 1
                logger.warning("Aura Guard stall detected, requesting rewrite: %s", decision.reason)
            elif decision.action in (PolicyAction.ESCALATE, PolicyAction.FINALIZE):
                self._escalations += 1
                logger.warning("Aura Guard forcing %s: %s", decision.action.value, decision.reason)

    # ─────────────────────────────────────────
    # Summary & Metrics
    # ─────────────────────────────────────────

    @property
    def cost_spent(self) -> float:
        return round(self._state.cumulative_cost, 4)

    @property
    def cost_saved(self) -> float:
        return round(self._cost_saved, 4)

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a summary of guard activity for this run."""
        return {
            "cost_spent_usd": self.cost_spent,
            "cost_saved_usd": self.cost_saved,
            "cost_limit_usd": self._cfg.max_cost_per_run,
            "blocks": self._blocks,
            "cache_hits": self._cache_hits,
            "rewrites": self._rewrites,
            "escalations": self._escalations,
            "decisions": list(self._decisions),
            "quarantined_tools": dict(self._state.quarantined_tools),
            "stall_streak": self._state.stall_streak,
        }

    def reset(self) -> None:
        """Reset for a new agent run (same config)."""
        self._state = self._guard.new_state()
        self._decisions.clear()
        self._cost_saved = 0.0
        self._blocks = 0
        self._cache_hits = 0
        self._rewrites = 0
        self._escalations = 0
        self._pending_call = None
