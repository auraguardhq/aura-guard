"""tests/test_guard.py

Tests for Aura Guard core engine — enforcement primitives and policy layer.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aura_guard import (
    AgentGuard,
    AuraGuard,
    AuraGuardConfig,
    CostModel,
    GuardState,
    PolicyAction,
    PolicyDecision,
    ToolCall,
    ToolResult,
)
from aura_guard.telemetry import InMemoryTelemetry, Telemetry


# ─────────────────────────────────────
# Fixtures
# ─────────────────────────────────────

@pytest.fixture
def config():
    return AuraGuardConfig(secret_key=b"test-secret-key", 
        cost_model=CostModel(default_tool_call_cost=0.04),
        max_cost_per_run=1.00,
        side_effect_tools={"refund", "send_reply", "cancel"},
    )


@pytest.fixture
def telemetry():
    return InMemoryTelemetry()


@pytest.fixture
def guard(config, telemetry):
    return AuraGuard(config=config, telemetry=Telemetry(sink=telemetry))


@pytest.fixture
def state(guard):
    return guard.new_state(run_id="test-run")


# ─────────────────────────────────────
# Primitive 1: Identical tool-call repeat
# ─────────────────────────────────────

class TestIdenticalRepeat:
    def test_allows_first_two_calls(self, guard, state):
        call = ToolCall(name="search_kb", args={"query": "test"})
        for _ in range(2):
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="ok"))

    def test_caches_after_threshold(self, guard, state):
        call = ToolCall(name="get_order", args={"order_id": "o1"})
        # Execute 3 times (threshold = 3)
        for _ in range(3):
            d = guard.on_tool_call_request(state=state, call=call)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="order_data"))

        # 4th call should be cached or blocked
        d = guard.on_tool_call_request(state=state, call=call)
        assert d.action in (PolicyAction.CACHE, PolicyAction.BLOCK)

    def test_different_args_not_blocked(self, guard, state):
        for i in range(5):
            call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload=f"data_{i}"))


# ─────────────────────────────────────
# Primitive 2: Argument jitter
# ─────────────────────────────────────

class TestArgumentJitter:
    def test_detects_jitter(self, guard, state):
        queries = [
            "refund policy",
            "refund policy EU",
            "refund policy EU Germany",
            "refund policy EU Germany 2024",
        ]
        decisions = []
        for q in queries:
            call = ToolCall(name="search_kb", args={"query": q})
            d = guard.on_tool_call_request(state=state, call=call)
            decisions.append(d)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload=f"results:{q}"))

        # At least one should be REWRITE (quarantine)
        actions = [d.action for d in decisions]
        assert PolicyAction.REWRITE in actions or PolicyAction.BLOCK in actions

    def test_non_query_tool_not_affected(self, guard, state):
        for i in range(5):
            call = ToolCall(name="get_order", args={"order_id": f"order_{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"repeat_toolcall_threshold": 0}, "repeat_toolcall_threshold must be >= 1"),
            (
                {"arg_jitter_similarity_threshold": -0.1},
                "arg_jitter_similarity_threshold must be between 0.0 and 1.0",
            ),
            ({"arg_jitter_similarity_threshold": 1.1}, "arg_jitter_similarity_threshold must be between 0.0 and 1.0"),
            ({"arg_jitter_repeat_threshold": 0}, "arg_jitter_repeat_threshold must be >= 1"),
            ({"error_retry_threshold": 0}, "error_retry_threshold must be >= 1"),
            ({"side_effect_max_executed_per_run": 0}, "side_effect_max_executed_per_run must be >= 1"),
            ({"no_state_change_threshold": 0}, "no_state_change_threshold must be >= 1"),
            ({"stall_text_similarity_threshold": -0.1}, "stall_text_similarity_threshold must be between 0.0 and 1.0"),
            ({"stall_text_similarity_threshold": 1.1}, "stall_text_similarity_threshold must be between 0.0 and 1.0"),
            ({"stall_pattern_threshold": -0.1}, "stall_pattern_threshold must be between 0.0 and 1.0"),
            ({"stall_pattern_threshold": 1.1}, "stall_pattern_threshold must be between 0.0 and 1.0"),
            ({"max_cost_per_run": 0.0}, "max_cost_per_run must be > 0 when provided"),
            ({"max_cost_per_run": -1.0}, "max_cost_per_run must be > 0 when provided"),
            ({"cost_warning_threshold": -0.1}, "cost_warning_threshold must be between 0.0 and 1.0"),
            ({"cost_warning_threshold": 1.1}, "cost_warning_threshold must be between 0.0 and 1.0"),
            ({"max_cache_entries": 0}, "max_cache_entries must be >= 1"),
            ({"max_unique_calls_tracked": 0}, "max_unique_calls_tracked must be >= 1"),
            ({"tool_loop_window": 0}, "tool_loop_window must be >= 1"),
        ],
    )
    def test_rejects_invalid_thresholds(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            AuraGuardConfig(secret_key=b"test-secret-key", **kwargs)

    def test_rejects_string_secret_key(self):
        with pytest.raises(TypeError, match="secret_key must be bytes"):
            AuraGuardConfig(secret_key="not-bytes")


# ─────────────────────────────────────
# Primitive 3: Error retry circuit breaker
# ─────────────────────────────────────

class TestErrorCircuitBreaker:
    def test_quarantines_after_errors(self, guard, state):
        call = ToolCall(name="search_kb", args={"query": "test"})

        # First call succeeds
        d1 = guard.on_tool_call_request(state=state, call=call)
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=False, error_code="429"))

        # Different args to avoid identical-repeat detection
        call2 = ToolCall(name="search_kb", args={"query": "test2"})
        d2 = guard.on_tool_call_request(state=state, call=call2)
        guard.on_tool_result(state=state, call=call2, result=ToolResult(ok=False, error_code="429"))

        # Tool should now be quarantined
        call3 = ToolCall(name="search_kb", args={"query": "test3"})
        d3 = guard.on_tool_call_request(state=state, call=call3)
        assert d3.action == PolicyAction.REWRITE
        assert "quarantined" in d3.reason


# ─────────────────────────────────────
# Primitive 4: Side-effect gating
# ─────────────────────────────────────

class TestSideEffectGating:
    def test_allows_first_side_effect(self, guard, state):
        call = ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1")
        d = guard.on_tool_call_request(state=state, call=call)
        assert d.action == PolicyAction.ALLOW
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="refunded"))

    def test_blocks_second_side_effect(self, guard, state):
        call = ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1")

        # First execution
        d1 = guard.on_tool_call_request(state=state, call=call)
        assert d1.action == PolicyAction.ALLOW
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="refunded"))

        # Second attempt — same args, same ticket → idempotent replay
        d2 = guard.on_tool_call_request(state=state, call=call)
        assert d2.action == PolicyAction.CACHE
        assert d2.cached_result is not None

    def test_blocks_different_refund_after_limit(self, guard, state):
        # First refund
        c1 = ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1")
        d1 = guard.on_tool_call_request(state=state, call=c1)
        assert d1.action == PolicyAction.ALLOW
        guard.on_tool_result(state=state, call=c1, result=ToolResult(ok=True))

        # Second refund, different args — should be blocked (limit=1 per run)
        c2 = ToolCall(name="refund", args={"order_id": "o2", "amount": 20}, ticket_id="t2")
        d2 = guard.on_tool_call_request(state=state, call=c2)
        assert d2.action == PolicyAction.BLOCK
        assert "side_effect_limit" in d2.reason

    def test_cached_idempotent_replay_does_not_increment_executed_count(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
            side_effect_tools={"refund"},
            side_effect_max_executed_per_run=2,
        )
        guard = AuraGuard(config=cfg)
        state = guard.new_state(run_id="test-run")

        first_call = ToolCall(name="refund", args={"order": 123}, ticket_id="t1")
        first_decision = guard.on_tool_call_request(state=state, call=first_call)
        assert first_decision.action == PolicyAction.ALLOW
        guard.on_tool_result(state=state, call=first_call, result=ToolResult(ok=True, payload="refunded-123"))

        replay_decision = guard.on_tool_call_request(state=state, call=first_call)
        assert replay_decision.action == PolicyAction.CACHE
        assert replay_decision.cached_result is not None
        guard.on_tool_result(state=state, call=first_call, result=replay_decision.cached_result)

        assert state.executed_side_effect_calls["refund"] == 1

        second_call = ToolCall(name="refund", args={"order": 456}, ticket_id="t2")
        second_decision = guard.on_tool_call_request(state=state, call=second_call)
        assert second_decision.action == PolicyAction.ALLOW


# ─────────────────────────────────────
# Primitive 5: Stall detection
# ─────────────────────────────────────

class TestStallDetection:
    def test_no_stall_on_varied_text(self, guard, state):
        texts = [
            "Your order was shipped on January 15.",
            "The tracking number is XY123456789.",
            "Expected delivery is February 1.",
        ]
        for t in texts:
            d = guard.on_llm_output(state=state, text=t)
            assert d is None

    def test_on_llm_output_returns_none_for_empty_text(self, guard, state):
        assert guard.on_llm_output(state=state, text="") is None
        assert guard.on_llm_output(state=state, text=None) is None

    def test_detects_stall_on_repeated_text(self, guard, state):
        text = "I apologize for the inconvenience. We're looking into it."
        decisions = []
        for _ in range(10):
            d = guard.on_llm_output(state=state, text=text)
            if d is not None:
                decisions.append(d)

        # Should eventually trigger REWRITE or ESCALATE
        assert len(decisions) > 0
        actions = {d.action for d in decisions}
        assert actions & {PolicyAction.REWRITE, PolicyAction.ESCALATE}


# ─────────────────────────────────────
# Primitive 6: Cost budget enforcement
# ─────────────────────────────────────

class TestCostBudget:
    def test_escalates_on_budget_exceeded(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
            max_cost_per_run=0.10,
            cost_model=CostModel(default_tool_call_cost=0.04),
        )
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        # Execute 2 calls = $0.08
        for i in range(2):
            call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        # 3rd call would bring total to $0.12 > $0.10 limit
        call3 = ToolCall(name="get_order", args={"order_id": "o3"})
        d3 = guard.on_tool_call_request(state=state, call=call3)
        assert d3.action == PolicyAction.ESCALATE
        assert "budget" in d3.reason

    def test_no_limit_when_none(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_per_run=None)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        for i in range(20):
            call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

    def test_budget_warning_emitted(self):
        sink = InMemoryTelemetry()
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
            max_cost_per_run=0.20,
            cost_model=CostModel(default_tool_call_cost=0.04),
            cost_warning_threshold=0.8,
        )
        guard = AuraGuard(config=cfg, telemetry=Telemetry(sink=sink))
        state = guard.new_state()

        # $0.04 * 4 = $0.16, which is 80% of $0.20
        for i in range(4):
            call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        warnings = sink.find("budget_warning")
        assert len(warnings) >= 1

    def test_identical_repeat_cache_hit_bypasses_budget_projection(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
            max_cost_per_run=0.10,
            cost_model=CostModel(default_tool_call_cost=0.04),
        )
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        call = ToolCall(name="get_order", args={"order_id": "o1"})

        # Execute 2 calls = $0.08
        for _ in range(2):
            decision = guard.on_tool_call_request(state=state, call=call)
            assert decision.action == PolicyAction.ALLOW
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="order_data"))

        assert state.cumulative_cost == pytest.approx(0.08)

        # Third identical call should come from cache, not budget escalation.
        decision3 = guard.on_tool_call_request(state=state, call=call)
        assert decision3.action == PolicyAction.CACHE
        assert decision3.reason == "identical_toolcall_loop_cache"
        assert state.cumulative_cost == pytest.approx(0.08)


# ─────────────────────────────────────
# AgentGuard middleware
# ─────────────────────────────────────

class TestAgentGuard:
    def test_basic_flow(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        d = g.check_tool("search_kb", args={"query": "test"})
        assert d.action == PolicyAction.ALLOW
        g.record_result(ok=True, payload="results")
        assert g.cost_spent > 0

    def test_stats(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        g.check_tool("search_kb", args={"query": "test"})
        g.record_result(ok=True)
        stats = g.stats
        assert "cost_spent_usd" in stats
        assert "blocks" in stats
        assert stats["cost_limit_usd"] == 1.00

    def test_reset(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        g.check_tool("search_kb", args={"query": "test"})
        g.record_result(ok=True)
        assert g.cost_spent > 0
        g.reset()
        assert g.cost_spent == 0
        assert g.blocks == 0

    def test_cost_remaining(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=0.50, default_tool_cost=0.10)
        g.check_tool("search_kb", args={"query": "test"})
        g.record_result(ok=True)
        assert g.cost_remaining is not None
        assert g.cost_remaining < 0.50

    def test_check_tool_without_record_result_strict_mode_raises(self):
        g = AgentGuard(secret_key=b"test-secret-key", strict_mode=True)
        d = g.check_tool("search_kb", args={"query": "one"})
        assert d.action == PolicyAction.ALLOW

        with pytest.raises(RuntimeError, match="without a preceding record_result"):
            g.check_tool("search_kb", args={"query": "two"})

    def test_check_tool_without_record_result_warns_and_increments_counter(self, caplog):
        g = AgentGuard(secret_key=b"test-secret-key", strict_mode=False)
        d = g.check_tool("search_kb", args={"query": "one"})
        assert d.action == PolicyAction.ALLOW

        with caplog.at_level("WARNING", logger="aura_guard"):
            g.check_tool("search_kb", args={"query": "two"})

        assert "without a preceding record_result" in caplog.text
        assert g.missed_results == 1
        assert g.stats["missed_results"] == 1

    @pytest.mark.parametrize("strict_mode", [False, True])
    def test_normal_flow_works_in_both_modes(self, strict_mode):
        g = AgentGuard(secret_key=b"test-secret-key", strict_mode=strict_mode)

        d1 = g.check_tool("search_kb", args={"query": "one"})
        assert d1.action == PolicyAction.ALLOW
        g.record_result(ok=True, payload="ok")

        d2 = g.check_tool("search_kb", args={"query": "two"})
        assert d2.action == PolicyAction.ALLOW
        g.record_result(ok=True, payload="ok2")

        assert g.tool_calls_executed == 2
        assert g.missed_results == 0

    def test_thread_safety_assert_allows_single_thread_usage(self):
        g = AgentGuard(secret_key=b"test-secret-key")

        d = g.check_tool("search_kb", args={"query": "single-thread"})
        assert d.action == PolicyAction.ALLOW
        g.record_result(ok=True, payload="ok")
        assert g.check_output("All good") is None

    def test_thread_safety_assert_raises_when_check_tool_called_on_different_thread(self):
        g = AgentGuard(secret_key=b"test-secret-key")
        result = {}

        def _call_check_tool():
            try:
                g.check_tool("search_kb", args={"query": "cross-thread"})
            except Exception as exc:  # pragma: no cover - validated below
                result["error"] = exc

        worker = threading.Thread(target=_call_check_tool)
        worker.start()
        worker.join()

        assert isinstance(result.get("error"), RuntimeError)
        assert str(result["error"]) == (
            "AgentGuard is not thread-safe. Create one instance per agent run. See docs."
        )


class TestConvenienceAPI:
    def test_run_allows_and_returns_result(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

        def my_tool(query):
            return {"results": [query]}

        result = g.run("search_kb", my_tool, query="test")
        assert result == {"results": ["test"]}
        assert g.tool_calls_executed == 1

    def test_run_returns_cached_on_repeat(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        call_count = 0

        def my_tool(query):
            nonlocal call_count
            call_count += 1
            return {"results": [query]}

        # Execute 3 times to hit repeat threshold, then 4th should cache
        for _ in range(3):
            g.run("get_order", my_tool, query="same")
        g.run("get_order", my_tool, query="same")
        assert g.cache_hits > 0

    def test_run_raises_guard_denied_on_block(self):
        from aura_guard import GuardDenied
        from aura_guard.config import ToolPolicy, ToolAccess

        g = AgentGuard(
            secret_key=b"test-secret-key",
            config=AuraGuardConfig(
                secret_key=b"test-secret-key",
                tool_policies={"forbidden": ToolPolicy(access=ToolAccess.DENY, deny_reason="not allowed")},
            ),
        )

        def forbidden():
            return "should not run"

        with pytest.raises(GuardDenied) as exc_info:
            g.run("forbidden", forbidden)

        assert exc_info.value.action == PolicyAction.BLOCK
        assert "not allowed" in exc_info.value.reason

    def test_run_records_error_on_exception(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

        def failing_tool(query):
            raise ValueError("tool broke")

        with pytest.raises(ValueError, match="tool broke"):
            g.run("bad_tool", failing_tool, query="test")

        assert g.tool_calls_failed == 1
        assert g.tool_calls_executed == 1

    def test_run_keyword_only_enforcement(self):
        """run() enforces keyword-only args via the * in its signature."""
        g = AgentGuard(secret_key=b"test-secret-key")

        def my_tool(query):
            return query

        # This should raise TypeError because positional args after fn are not allowed
        with pytest.raises(TypeError):
            g.run("search", my_tool, "positional_arg")

    def test_protect_decorator_basic(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

        @g.protect
        def search_kb(query):
            return {"hits": [query]}

        result = search_kb(query="refund policy")
        assert result == {"hits": ["refund policy"]}
        assert g.tool_calls_executed == 1

    def test_protect_decorator_with_options(self):
        g = AgentGuard(
            secret_key=b"test-secret-key",
            max_cost_per_run=1.00,
            side_effect_tools={"charge"},
        )

        @g.protect(tool_name="charge", side_effect=True)
        def charge_card(customer_id, amount):
            return {"charged": amount}

        result = charge_card(customer_id="cus_42", amount=49)
        assert result == {"charged": 49}
        assert g.tool_calls_executed == 1

    def test_protect_rejects_positional_args(self):
        g = AgentGuard(secret_key=b"test-secret-key")

        @g.protect
        def search_kb(query):
            return query

        with pytest.raises(TypeError, match="keyword arguments only"):
            search_kb("positional_value")

    def test_protect_preserves_function_name(self):
        g = AgentGuard(secret_key=b"test-secret-key")

        @g.protect
        def my_special_tool(x):
            return x

        assert my_special_tool.__name__ == "my_special_tool"

    def test_guard_denied_attributes(self):
        from aura_guard import GuardDenied

        decision = PolicyDecision(action=PolicyAction.BLOCK, reason="test_reason")
        exc = GuardDenied(decision)
        assert exc.action == PolicyAction.BLOCK
        assert exc.reason == "test_reason"
        assert exc.decision is decision
        assert "block" in str(exc)
        assert "test_reason" in str(exc)


class TestAsyncConvenienceAPI:
    def test_async_run(self):
        import asyncio
        from aura_guard import AsyncAgentGuard

        async def _test():
            g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

            async def async_search(query):
                return {"results": [query]}

            result = await g.run("search_kb", async_search, query="test")
            assert result == {"results": ["test"]}
            assert g.stats["tool_calls_executed"] == 1

        asyncio.run(_test())

    def test_async_run_sync_fn(self):
        import asyncio
        from aura_guard import AsyncAgentGuard

        async def _test():
            g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

            def sync_search(query):
                return {"results": [query]}

            result = await g.run("search_kb", sync_search, query="test")
            assert result == {"results": ["test"]}

        asyncio.run(_test())

    def test_async_protect_decorator(self):
        import asyncio
        from aura_guard import AsyncAgentGuard

        async def _test():
            g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

            @g.protect
            async def search_kb(query):
                return {"hits": [query]}

            result = await search_kb(query="test")
            assert result == {"hits": ["test"]}

        asyncio.run(_test())

    def test_async_protect_rejects_positional_args(self):
        import asyncio
        from aura_guard import AsyncAgentGuard

        async def _test():
            g = AsyncAgentGuard(secret_key=b"test-secret-key")

            @g.protect
            async def search_kb(query):
                return query

            with pytest.raises(TypeError, match="keyword arguments only"):
                await search_kb("positional")

        asyncio.run(_test())


# ─────────────────────────────────────
# Serialization
# ─────────────────────────────────────

class TestSerialization:
    def test_roundtrip(self, guard, state):
        from aura_guard.serialization import state_to_json, state_from_json

        # Do some operations
        call = ToolCall(name="search_kb", args={"query": "test"})
        guard.on_tool_call_request(state=state, call=call)
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        # Serialize
        json_str = state_to_json(state)
        assert isinstance(json_str, str)
        assert "test-run" in json_str

        # Deserialize
        restored = state_from_json(json_str)
        assert restored.run_id == state.run_id
        assert restored.cumulative_cost == state.cumulative_cost
        assert len(restored.tool_stream) == len(state.tool_stream)

    def test_state_from_json_generates_run_id_when_missing(self):
        from aura_guard.serialization import state_from_json

        payload = {
            "version": 4,
            "tool_stream": [],
        }

        restored = state_from_json(json.dumps(payload))
        assert isinstance(restored.run_id, str)
        assert len(restored.run_id) == 32

    def test_state_from_json_rejects_missing_or_old_version(self):
        from aura_guard.serialization import state_from_json

        with pytest.raises(ValueError, match="Incompatible state format"):
            state_from_json(json.dumps({"run_id": "abc123"}))

        with pytest.raises(ValueError, match="Incompatible state format"):
            state_from_json(json.dumps({"version": 3, "run_id": "abc123"}))

    def test_dict_roundtrip(self, guard, state):
        from aura_guard.serialization import state_to_dict, state_from_dict

        call = ToolCall(name="search_kb", args={"query": "test"})
        guard.on_tool_call_request(state=state, call=call)
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        d = state_to_dict(state)
        assert isinstance(d, dict)
        assert d["run_id"] == "test-run"

        restored = state_from_dict(d)
        assert restored.run_id == state.run_id


class TestIdempotencyLedgerSerialization:
    def test_idempotency_survives_serialization(self):
        from aura_guard.serialization import state_from_json, state_to_json

        cfg = AuraGuardConfig(secret_key=b"test-secret-key", side_effect_tools={"refund"})
        guard = AuraGuard(config=cfg)
        state = guard.new_state(run_id="idempotency-serialize")

        call = ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1")
        decision = guard.on_tool_call_request(state=state, call=call)
        assert decision.action == PolicyAction.ALLOW
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="refunded"))

        restored = state_from_json(state_to_json(state))

        replay = guard.on_tool_call_request(state=restored, call=call)
        assert replay.action == PolicyAction.CACHE
        assert replay.cached_result is not None
        assert replay.cached_result.payload is None

    def test_idempotency_serialization_backward_compat(self):
        from aura_guard.serialization import state_from_json

        payload = {
            "version": 4,
            "run_id": "compat-v4",
            "tool_stream": [],
            "tool_query_sigs": {},
            "quarantined_tools": {},
            "error_streaks": {},
            "attempted_side_effect_calls": {},
            "executed_side_effect_calls": {},
            "tool_call_counts": {},
            "stall_streak": 0,
            "stall_pattern_streak": 0,
            "stall_rewrite_attempts": 0,
            "last_assistant_token_sigs": None,
            "last_progress_marker": [0, 0],
            "unique_tool_calls_seen": [],
            "unique_tool_results_seen": [],
            "cumulative_cost": 0.0,
            "reported_token_cost": 0.0,
            "budget_warning_emitted": False,
            "cost_events": [],
        }

        state = state_from_json(json.dumps(payload))
        assert state.run_id == "compat-v4"
        assert state.idempotency_ledger == {}


class TestSerializationFidelity:
    def test_state_roundtrip_preserves_decision_behavior(self):
        from aura_guard.serialization import state_from_json, state_to_json

        cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
            cost_model=CostModel(default_tool_call_cost=0.01),
            max_cost_per_run=None,
            side_effect_tools={"refund"},
            side_effect_max_executed_per_run=1,
            repeat_toolcall_threshold=2,
            error_retry_threshold=2,
        )
        guard = AuraGuard(config=cfg)
        original_state = guard.new_state(run_id="serialization-fidelity")

        def run_sequence(sequence, state):
            decisions = []
            for call, result in sequence:
                decision = guard.on_tool_call_request(state=state, call=call)
                decisions.append(decision)
                if decision.action == PolicyAction.ALLOW and result is not None:
                    guard.on_tool_result(state=state, call=call, result=result)
            return decisions

        initial_sequence = [
            (ToolCall(name="get_order", args={"order_id": "o1"}), ToolResult(ok=True, payload="order:o1:v1")),
            (ToolCall(name="get_order", args={"order_id": "o1"}), ToolResult(ok=True, payload="order:o1:v2")),
            (ToolCall(name="get_order", args={"order_id": "o1"}), None),
            (ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1"), ToolResult(ok=True, payload="refunded:o1")),
            (ToolCall(name="refund", args={"order_id": "o1", "amount": 10}, ticket_id="t1"), None),
            (ToolCall(name="refund", args={"order_id": "o2", "amount": 20}, ticket_id="t2"), None),
            (ToolCall(name="flaky_api", args={"req": "a"}), ToolResult(ok=False, error_code="429")),
            (ToolCall(name="flaky_api", args={"req": "b"}), ToolResult(ok=False, error_code="429")),
            (ToolCall(name="flaky_api", args={"req": "c"}), None),
            (ToolCall(name="get_order", args={"order_id": "o2"}), ToolResult(ok=True, payload="order:o2:v1")),
            (ToolCall(name="get_order", args={"order_id": "o2"}), ToolResult(ok=True, payload="order:o2:v2")),
        ]
        run_sequence(initial_sequence, original_state)

        restored_state = state_from_json(state_to_json(original_state))

        additional_sequence = [
            (ToolCall(name="flaky_api", args={"req": "d"}), None),
            (ToolCall(name="refund", args={"order_id": "o3", "amount": 30}, ticket_id="t3"), None),
            (ToolCall(name="refund", args={"order_id": "o4", "amount": 40}, ticket_id="t4"), None),
            (ToolCall(name="get_order", args={"order_id": "o4"}), ToolResult(ok=True, payload="order:o4:v1")),
            (ToolCall(name="get_order", args={"order_id": "o4"}), None),
            (ToolCall(name="get_order", args={"order_id": "o5"}), ToolResult(ok=True, payload="order:o5:v1")),
            (ToolCall(name="get_order", args={"order_id": "o5"}), None),
        ]

        original_decisions = run_sequence(additional_sequence, original_state)
        restored_decisions = run_sequence(additional_sequence, restored_state)

        assert len(original_decisions) == len(restored_decisions)
        for original, restored in zip(original_decisions, restored_decisions):
            assert original.action == restored.action
            assert original.reason == restored.reason


# ─────────────────────────────────────
# Telemetry
# ─────────────────────────────────────

class TestTelemetry:
    def test_inmemory_telemetry(self, guard, state, telemetry):
        call = ToolCall(name="refund", args={"order_id": "o1"}, ticket_id="t1")
        d1 = guard.on_tool_call_request(state=state, call=call)
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        # Second refund should be blocked
        d2 = guard.on_tool_call_request(state=state, call=call)

        # Should have telemetry events
        assert len(telemetry.events) > 0

    def test_cost_saved_tracking(self, telemetry):
        telemetry.emit({"event": "test", "estimated_cost_avoided": 0.04})
        telemetry.emit({"event": "test", "estimated_cost_avoided": 0.08})
        assert telemetry.cost_saved == 0.12

    def test_find_events(self, telemetry):
        telemetry.emit({"event": "foo"})
        telemetry.emit({"event": "bar"})
        telemetry.emit({"event": "foo"})
        assert len(telemetry.find("foo")) == 2
        assert len(telemetry.find("bar")) == 1


class TestSequenceLoopDetection:
    """Tests for Primitive 8: multi-tool sequence loop detection."""

    def test_detects_ping_pong_loop(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=2, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        decisions = []
        for t in ["agent_a", "agent_b", "agent_a", "agent_b"]:
            call = ToolCall(name=t, args={"task": f"do {t} work"})
            d = guard.on_tool_call_request(state=state, call=call)
            decisions.append(d)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload=f"result_{t}"))

        assert PolicyAction.REWRITE in [d.action for d in decisions]
        rewrite_decisions = [d for d in decisions if d.action == PolicyAction.REWRITE]
        assert any("sequence_loop" in d.reason for d in rewrite_decisions)

    def test_detects_three_tool_cycle(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=2, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        decisions = []
        for t in ["triage", "research", "action", "triage", "research", "action"]:
            call = ToolCall(name=t, args={"task": "work"})
            d = guard.on_tool_call_request(state=state, call=call)
            decisions.append(d)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload="ok"))

        assert PolicyAction.REWRITE in [d.action for d in decisions]

    def test_no_false_positive_on_varied_tools(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=2, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        for t in ["search", "get_order", "check_status", "send_email", "update_ticket"]:
            call = ToolCall(name=t, args={"id": "123"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW, f"Tool {t} should be allowed, got {d.action}"
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

    def test_disabled_when_config_false(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_detection_enabled=False)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        i = 0
        for _ in range(3):
            for t in ["agent_a", "agent_b"]:
                i += 1
                call = ToolCall(name=t, args={"task": f"work-{i}"})
                d = guard.on_tool_call_request(state=state, call=call)
                assert d.action == PolicyAction.ALLOW
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

    def test_quarantines_tool_on_detection(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=2, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        for t in ["agent_a", "agent_b", "agent_a", "agent_b"]:
            call = ToolCall(name=t, args={"task": "work"})
            d = guard.on_tool_call_request(state=state, call=call)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        assert any("sequence_loop" in v for v in state.quarantined_tools.values())

    def test_threshold_3_requires_three_repeats(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=3, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        i = 0
        for t in ["agent_a", "agent_b", "agent_a", "agent_b"]:
            i += 1
            call = ToolCall(name=t, args={"task": f"work-{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            assert d.action == PolicyAction.ALLOW, "2x repeat should not trigger with threshold=3"
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        triggered = False
        for t in ["agent_a", "agent_b"]:
            i += 1
            call = ToolCall(name=t, args={"task": f"work-{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            if d.action == PolicyAction.REWRITE:
                triggered = True
                break
            guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        assert triggered, "3rd repeat should trigger with threshold=3"

    def test_rewrite_includes_pattern_in_reason(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=2, max_sequence_length=4)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        decisions = []
        for t in ["coordinator", "worker", "coordinator", "worker"]:
            call = ToolCall(name=t, args={"task": "work"})
            d = guard.on_tool_call_request(state=state, call=call)
            decisions.append(d)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        rewrite_decisions = [d for d in decisions if d.action == PolicyAction.REWRITE]
        assert rewrite_decisions
        assert any("coordinator" in d.reason or "worker" in d.reason for d in rewrite_decisions)

    def test_config_validation(self):
        with pytest.raises(ValueError, match="sequence_repeat_threshold must be >= 2"):
            AuraGuardConfig(secret_key=b"test-secret-key", sequence_repeat_threshold=1)

        with pytest.raises(ValueError, match="max_sequence_length must be >= 2"):
            AuraGuardConfig(secret_key=b"test-secret-key", max_sequence_length=1)


# ─────────────────────────────────────
# OpenAI Adapter
# ─────────────────────────────────────

class TestOpenAIAdapter:
    def test_extract_tool_calls(self):
        from aura_guard.adapters.openai_adapter import extract_tool_calls_from_chat_completion

        resp = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "search_kb",
                            "arguments": '{"query": "test"}',
                        }
                    }]
                }
            }]
        }
        calls = extract_tool_calls_from_chat_completion(resp, ticket_id="t1")
        assert len(calls) == 1
        assert calls[0].name == "search_kb"
        assert calls[0].args["query"] == "test"
        assert calls[0].ticket_id == "t1"

    def test_extract_assistant_text(self):
        from aura_guard.adapters.openai_adapter import extract_assistant_text

        resp = {"choices": [{"message": {"content": "Hello world"}}]}
        assert extract_assistant_text(resp) == "Hello world"

    def test_inject_system_message(self):
        from aura_guard.adapters.openai_adapter import inject_system_message

        messages = [{"role": "user", "content": "hi"}]
        result = inject_system_message(messages, "You are a bot")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a bot"
        assert len(result) == 2

    def test_inject_replaces_existing_system(self):
        from aura_guard.adapters.openai_adapter import inject_system_message

        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "hi"},
        ]
        result = inject_system_message(messages, "new")
        assert result[0]["content"] == "new"
        assert len(result) == 2


# ─────────────────────────────────────
# Run Summary
# ─────────────────────────────────────

class TestRunSummary:
    def test_get_run_summary(self, guard, state):
        call = ToolCall(name="search_kb", args={"query": "test"})
        guard.on_tool_call_request(state=state, call=call)
        guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True))

        summary = guard.get_run_summary(state)
        assert summary["run_id"] == "test-run"
        assert summary["cumulative_cost_usd"] > 0
        assert summary["unique_tool_calls"] == 1


# ─────────────────────────────────────
# Shadow Mode
# ─────────────────────────────────────

class TestShadowMode:
    def test_shadow_allows_everything(self):
        g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=0.10, shadow_mode=True, default_tool_cost=0.04)
        # Execute 5 identical calls — normally would trigger block/cache
        for _ in range(5):
            d = g.check_tool("search_kb", args={"query": "test"})
            assert d.action == PolicyAction.ALLOW
            g.record_result(ok=True, payload="results")

        assert g.shadow_would_deny > 0
        assert g.blocks == 0
        assert g.cache_hits == 0
        assert g.stats["shadow_mode"] is True

    def test_shadow_counts_would_deny(self):
        g = AgentGuard(secret_key=b"test-secret-key", 
            max_cost_per_run=0.50,
            side_effect_tools={"refund"},
            shadow_mode=True,
        )
        # First refund — allowed
        d1 = g.check_tool("refund", args={"order_id": "o1"}, ticket_id="t1")
        assert d1.action == PolicyAction.ALLOW
        g.record_result(ok=True, payload="refunded")

        # Second refund, same args — would be CACHE in enforcement mode
        d2 = g.check_tool("refund", args={"order_id": "o1"}, ticket_id="t1")
        assert d2.action == PolicyAction.ALLOW  # shadow: still allowed
        assert g.shadow_would_deny == 1

    def test_shadow_stall_not_enforced(self):
        g = AgentGuard(secret_key=b"test-secret-key", shadow_mode=True)
        text = "I apologize for the inconvenience. We're looking into it."
        for _ in range(10):
            d = g.check_output(text)
            assert d is None  # shadow mode: never intervenes on output
        assert g.shadow_would_deny > 0


# ─────────────────────────────────────
# Async Guard
# ─────────────────────────────────────

class TestAsyncGuard:
    def test_async_import(self):
        from aura_guard import AsyncAgentGuard
        g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        assert g.cost_spent == 0

    def test_async_sync_parity(self):
        """AsyncAgentGuard should produce identical results to AgentGuard."""
        import asyncio
        from aura_guard import AsyncAgentGuard

        async def _run():
            g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
            d = await g.check_tool("search_kb", args={"query": "test"})
            assert d.action == PolicyAction.ALLOW
            await g.record_result(ok=True, payload="results")
            assert g.cost_spent > 0
            stall = await g.check_output("Normal response text.")
            assert stall is None
            return g.stats

        stats = asyncio.run(_run())
        assert stats["cost_spent_usd"] > 0


# ─────────────────────────────────────
# Secret key enforcement
# ─────────────────────────────────────

class TestSecretKeyEnforcement:
    def test_default_key_allowed_in_shadow_mode(self):
        AuraGuard(config=AuraGuardConfig(shadow_mode=True))

    def test_default_key_rejected_in_enforcement_mode(self):
        with pytest.raises(ValueError, match="default development secret_key"):
            AuraGuard(config=AuraGuardConfig(shadow_mode=False))

    @pytest.mark.parametrize("shadow_mode", [True, False])
    def test_custom_key_allowed_in_both_modes(self, shadow_mode):
        AuraGuard(config=AuraGuardConfig(secret_key=b"my-production-key", shadow_mode=shadow_mode))


class TestCoreShadowMode:
    """Test that shadow_mode works in the core AuraGuard engine, not just the AgentGuard wrapper."""

    def test_shadow_mode_suppresses_block_in_core(self):
        """on_tool_call_request returns ALLOW when shadow_mode=True, even if the tool would be blocked."""
        from aura_guard.config import ToolPolicy, ToolAccess

        cfg = AuraGuardConfig(
            secret_key=b"test-secret-key",
            shadow_mode=True,
            tool_policies={"forbidden": ToolPolicy(access=ToolAccess.DENY, deny_reason="not allowed")},
        )
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        call = ToolCall(name="forbidden", args={"x": 1})
        decision = guard.on_tool_call_request(state=state, call=call)
        assert decision.action == PolicyAction.ALLOW
        assert decision.reason == "shadow_allow"

    def test_shadow_mode_suppresses_stall_in_core(self):
        """on_llm_output returns None when shadow_mode=True, even if stall is detected."""
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", shadow_mode=True)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        text = "I apologize for the inconvenience. We're looking into it."
        for _ in range(10):
            decision = guard.on_llm_output(state=state, text=text)
            assert decision is None  # shadow mode: never intervenes

    def test_shadow_mode_false_still_enforces_in_core(self):
        """Verify enforcement still works when shadow_mode=False."""
        from aura_guard.config import ToolPolicy, ToolAccess

        cfg = AuraGuardConfig(
            secret_key=b"test-secret-key",
            shadow_mode=False,
            tool_policies={"forbidden": ToolPolicy(access=ToolAccess.DENY)},
        )
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        call = ToolCall(name="forbidden", args={"x": 1})
        decision = guard.on_tool_call_request(state=state, call=call)
        assert decision.action == PolicyAction.BLOCK


class TestCostEventsCap:
    def test_cost_events_bounded_by_max(self):
        cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_events=5)
        guard = AuraGuard(config=cfg)
        state = guard.new_state()

        for i in range(10):
            call = ToolCall(name="search", args={"q": f"query_{i}"})
            d = guard.on_tool_call_request(state=state, call=call)
            if d.action == PolicyAction.ALLOW:
                guard.on_tool_result(state=state, call=call, result=ToolResult(ok=True, payload=f"r{i}"))

        assert len(state.cost_events) <= 5


class TestMCPAdapter:
    """Tests for the MCP adapter (without requiring the mcp package)."""

    def test_import_without_mcp_raises(self):
        """GuardedMCP should raise ImportError if mcp is not installed."""
        # We test the adapter's guard logic by mocking, not by importing mcp.
        # This test verifies the error message when mcp is missing.
        from aura_guard.adapters import mcp_adapter

        if not mcp_adapter.HAS_MCP:
            with pytest.raises(ImportError, match="mcp"):
                mcp_adapter.GuardedMCP("test", secret_key=b"test-key")

    def test_guard_wrapping_logic_sync(self):
        """Test that the guard check/execute/record cycle works correctly."""
        # Simulate what GuardedMCP does internally without needing FastMCP
        from aura_guard import AgentGuard, PolicyAction

        guard = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)

        def search_kb(query):
            return {"results": [query]}

        # Simulate the guarded tool call pattern used in mcp_adapter
        tool_name = "search_kb"
        kwargs = {"query": "refund policy"}

        decision = guard.check_tool(tool_name, args=kwargs)
        assert decision.action == PolicyAction.ALLOW

        result = search_kb(**kwargs)
        guard.record_result(ok=True, payload=result)
        assert result == {"results": ["refund policy"]}
        assert guard.tool_calls_executed == 1

    def test_guard_blocks_duplicate_side_effect(self):
        """Test that side-effect dedup works as it would in MCP context."""
        from aura_guard import AgentGuard, PolicyAction

        guard = AgentGuard(
            secret_key=b"test-secret-key",
            side_effect_tools={"refund"},
        )

        # First refund — allowed
        d1 = guard.check_tool("refund", args={"order_id": "o1", "amount": 50}, ticket_id="t1")
        assert d1.action == PolicyAction.ALLOW
        guard.record_result(ok=True, payload={"status": "refunded"})

        # Same refund — should be cached (idempotent replay)
        d2 = guard.check_tool("refund", args={"order_id": "o1", "amount": 50}, ticket_id="t1")
        assert d2.action == PolicyAction.CACHE

    def test_guard_denied_produces_error_string(self):
        """Test that GuardDenied can be formatted as an MCP error message."""
        from aura_guard.middleware import GuardDenied
        from aura_guard.types import PolicyAction, PolicyDecision

        decision = PolicyDecision(
            action=PolicyAction.BLOCK,
            reason="side_effect_limit_exceeded",
        )
        exc = GuardDenied(decision)

        # This is what the MCP adapter returns to the LLM
        error_msg = f"[AURA GUARD] {exc.action.value.upper()}: {exc.reason}"
        assert "[AURA GUARD] BLOCK: side_effect_limit_exceeded" == error_msg

    def test_guard_stats_accessible(self):
        """Test that guard stats are accessible (as they would be via mcp.guard_stats)."""
        from aura_guard import AgentGuard

        guard = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        guard.check_tool("search_kb", args={"query": "test"})
        guard.record_result(ok=True, payload="results")

        stats = guard.stats
        assert stats["tool_calls_executed"] == 1
        assert stats["cost_spent_usd"] > 0

    def test_guard_reset(self):
        """Test that guard can be reset between MCP sessions."""
        from aura_guard import AgentGuard

        guard = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
        guard.check_tool("search_kb", args={"query": "test"})
        guard.record_result(ok=True, payload="results")
        assert guard.tool_calls_executed == 1

        guard.reset()
        assert guard.tool_calls_executed == 0
        assert guard.cost_spent == 0
