"""Run all tests without pytest dependency."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator for test functions."""
    def wrapper(fn):
        global PASS, FAIL
        try:
            fn()
            PASS += 1
            print(f"  ✓ {name}")
        except Exception as e:
            FAIL += 1
            ERRORS.append((name, e))
            print(f"  ✗ {name}: {e}")
        return fn
    return wrapper


# ─── Imports ───
from aura_guard import (
    AgentGuard, AuraGuard, AuraGuardConfig, CostModel,
    PolicyAction, ToolCall, ToolResult,
)
from aura_guard.telemetry import InMemoryTelemetry, Telemetry
from aura_guard.serialization import state_to_json, state_from_json, state_to_dict, state_from_dict
from aura_guard.adapters.openai_adapter import (
    extract_tool_calls_from_chat_completion,
    extract_assistant_text,
    inject_system_message,
)

cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
    cost_model=CostModel(default_tool_call_cost=0.04),
    max_cost_per_run=1.00,
    side_effect_tools={"refund", "send_reply", "cancel"},
)


print("\n=== Primitive 1: Identical Repeat ===")

@test("allows first calls")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="search_kb", args={"query": "test"})
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.ALLOW

@test("caches after threshold")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="get_order", args={"order_id": "o1"})
    for _ in range(3):
        d = g.on_tool_call_request(state=s, call=c)
        if d.action == PolicyAction.ALLOW:
            g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload="data"))
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action in (PolicyAction.CACHE, PolicyAction.BLOCK), f"Got {d.action}"

@test("different args not blocked")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for i in range(5):
        c = ToolCall(name="get_order", args={"order_id": f"o{i}"})
        d = g.on_tool_call_request(state=s, call=c)
        assert d.action == PolicyAction.ALLOW
        g.on_tool_result(state=s, call=c, result=ToolResult(ok=True))


print("\n=== Primitive 2: Argument Jitter ===")

@test("detects jitter on search queries")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    queries = ["refund policy", "refund policy EU", "refund policy EU Germany", "refund policy EU Germany 2024"]
    actions = []
    for q in queries:
        c = ToolCall(name="search_kb", args={"query": q})
        d = g.on_tool_call_request(state=s, call=c)
        actions.append(d.action)
        if d.action == PolicyAction.ALLOW:
            g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload=f"r:{q}"))
    assert PolicyAction.REWRITE in actions or PolicyAction.BLOCK in actions, f"Actions: {actions}"


print("\n=== Primitive 3: Error Circuit Breaker ===")

@test("quarantines tool after repeated errors")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for i in range(3):
        c = ToolCall(name="search_kb", args={"query": f"test{i}"})
        g.on_tool_call_request(state=s, call=c)
        g.on_tool_result(state=s, call=c, result=ToolResult(ok=False, error_code="429"))
    c = ToolCall(name="search_kb", args={"query": "test_final"})
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.REWRITE
    assert "quarantined" in d.reason


print("\n=== Primitive 4: Side-Effect Gating ===")

@test("allows first side effect")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="refund", args={"order_id": "o1"}, ticket_id="t1")
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.ALLOW

@test("caches idempotent replay")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="refund", args={"order_id": "o1"}, ticket_id="t1")
    d1 = g.on_tool_call_request(state=s, call=c)
    g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload="done"))
    d2 = g.on_tool_call_request(state=s, call=c)
    assert d2.action == PolicyAction.CACHE

@test("blocks second different refund")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c1 = ToolCall(name="refund", args={"order_id": "o1"}, ticket_id="t1")
    g.on_tool_call_request(state=s, call=c1)
    g.on_tool_result(state=s, call=c1, result=ToolResult(ok=True))
    c2 = ToolCall(name="refund", args={"order_id": "o2"}, ticket_id="t2")
    d2 = g.on_tool_call_request(state=s, call=c2)
    assert d2.action == PolicyAction.BLOCK


print("\n=== Primitive 5: Stall Detection ===")

@test("no stall on varied text")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for t in ["Order shipped Jan 15.", "Tracking: XY12345.", "ETA: Feb 1."]:
        d = g.on_llm_output(state=s, text=t)
        assert d is None

@test("detects stall on repeated text")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state()
    text = "I apologize for the inconvenience. We're looking into it."
    decisions = []
    for _ in range(10):
        d = g.on_llm_output(state=s, text=text)
        if d:
            decisions.append(d)
    assert len(decisions) > 0
    actions = {d.action for d in decisions}
    assert actions & {PolicyAction.REWRITE, PolicyAction.ESCALATE}


print("\n=== Primitive 6: Cost Budget ===")

@test("escalates on budget exceeded")
def _():
    c = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_per_run=0.10, cost_model=CostModel(default_tool_call_cost=0.04))
    g = AuraGuard(config=c)
    s = g.new_state()
    for i in range(2):
        call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
        d = g.on_tool_call_request(state=s, call=call)
        assert d.action == PolicyAction.ALLOW
        g.on_tool_result(state=s, call=call, result=ToolResult(ok=True))
    call3 = ToolCall(name="get_order", args={"order_id": "o3"})
    d3 = g.on_tool_call_request(state=s, call=call3)
    assert d3.action == PolicyAction.ESCALATE
    assert "budget" in d3.reason

@test("no limit when None")
def _():
    c = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_per_run=None)
    g = AuraGuard(config=c)
    s = g.new_state()
    for i in range(15):
        call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
        d = g.on_tool_call_request(state=s, call=call)
        assert d.action == PolicyAction.ALLOW
        g.on_tool_result(state=s, call=call, result=ToolResult(ok=True))

@test("budget warning emitted")
def _():
    sink = InMemoryTelemetry()
    # Budget $0.50, cost $0.10/call, warning at 50% ($0.25). 3 calls = $0.30 projected on 3rd.
    c = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_per_run=0.50, cost_model=CostModel(default_tool_call_cost=0.10), cost_warning_threshold=0.5)
    g = AuraGuard(config=c, telemetry=Telemetry(sink=sink))
    s = g.new_state()
    for i in range(4):
        call = ToolCall(name="get_order", args={"order_id": f"o{i}"})
        d = g.on_tool_call_request(state=s, call=call)
        if d.action == PolicyAction.ALLOW:
            g.on_tool_result(state=s, call=call, result=ToolResult(ok=True))
    warnings = sink.find("budget_warning")
    assert len(warnings) >= 1, f"Expected warnings, got {len(warnings)}. Events: {[e['event'] for e in sink.events]}"


@test("failed tool calls still incur cost")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    g = AuraGuard(config=cfg)
    s = g.new_state()
    call = ToolCall(name="get_order", args={"order_id": "o1"})
    g.on_tool_call_request(state=s, call=call)
    g.on_tool_result(state=s, call=call, result=ToolResult(ok=False, error_code="429"))
    assert s.cumulative_cost > 0, f"Failed call should still cost money, got {s.cumulative_cost}"
    cost_after_fail = s.cumulative_cost
    # A second successful call should also add cost
    call2 = ToolCall(name="get_order", args={"order_id": "o2"})
    g.on_tool_call_request(state=s, call=call2)
    g.on_tool_result(state=s, call=call2, result=ToolResult(ok=True, payload="ok"))
    assert s.cumulative_cost > cost_after_fail, "Successful call should add more cost"
    # But a cached result should NOT add cost
    cost_before_cache = s.cumulative_cost
    call3 = ToolCall(name="get_order", args={"order_id": "o1"})
    g.on_tool_call_request(state=s, call=call3)
    g.on_tool_result(state=s, call=call3, result=ToolResult(ok=True, cached=True))
    assert s.cumulative_cost == cost_before_cache, "Cached result should be free"


@test("per-tool call cap quarantines after N calls")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_calls_per_tool=3)
    g = AuraGuard(config=cfg)
    s = g.new_state()
    # First 3 calls should be allowed
    for i in range(3):
        call = ToolCall(name="search_kb", args={"query": f"completely unique query {i}"})
        d = g.on_tool_call_request(state=s, call=call)
        assert d.action == PolicyAction.ALLOW, f"Call {i} should be allowed, got {d.action}"
        g.on_tool_result(state=s, call=call, result=ToolResult(ok=True, payload=f"result {i}"))
    # 4th call should be quarantined
    call4 = ToolCall(name="search_kb", args={"query": "totally different query"})
    d4 = g.on_tool_call_request(state=s, call=call4)
    assert d4.action == PolicyAction.REWRITE, f"Call 4 should be REWRITE, got {d4.action}"
    assert "max_calls_per_tool" in d4.reason
    assert "search_kb" in s.quarantined_tools
    # A different tool should still be allowed
    call5 = ToolCall(name="get_order", args={"order_id": "o1"})
    d5 = g.on_tool_call_request(state=s, call=call5)
    assert d5.action == PolicyAction.ALLOW, f"Different tool should be allowed, got {d5.action}"


print("\n=== AgentGuard Middleware ===")

@test("basic flow")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    d = g.check_tool("search_kb", args={"query": "test"})
    assert d.action == PolicyAction.ALLOW
    g.record_result(ok=True, payload="results")
    assert g.cost_spent > 0

@test("stats and summary")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    g.check_tool("search_kb", args={"query": "test"})
    g.record_result(ok=True)
    s = g.stats
    assert "cost_spent_usd" in s
    assert s["cost_limit_usd"] == 1.00

@test("reset clears state")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    g.check_tool("search_kb", args={"query": "test"})
    g.record_result(ok=True)
    g.reset()
    assert g.cost_spent == 0

@test("cost_remaining tracks correctly")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=0.50, default_tool_cost=0.10)
    g.check_tool("search_kb", args={"query": "test"})
    g.record_result(ok=True)
    assert g.cost_remaining is not None
    assert g.cost_remaining < 0.50


print("\n=== Serialization ===")

@test("JSON roundtrip")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state(run_id="test-ser")
    c = ToolCall(name="search_kb", args={"query": "test"})
    g.on_tool_call_request(state=s, call=c)
    g.on_tool_result(state=s, call=c, result=ToolResult(ok=True))
    j = state_to_json(s)
    r = state_from_json(j)
    assert r.run_id == "test-ser"
    assert r.cumulative_cost == s.cumulative_cost
    assert len(r.tool_stream) == len(s.tool_stream)

@test("dict roundtrip")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state(run_id="test-dict")
    c = ToolCall(name="search_kb", args={"query": "test"})
    g.on_tool_call_request(state=s, call=c)
    g.on_tool_result(state=s, call=c, result=ToolResult(ok=True))
    d = state_to_dict(s)
    r = state_from_dict(d)
    assert r.run_id == "test-dict"


print("\n=== OpenAI Adapter ===")

@test("extract tool calls")
def _():
    resp = {"choices": [{"message": {"tool_calls": [{"function": {"name": "search_kb", "arguments": '{"query": "test"}'}}]}}]}
    calls = extract_tool_calls_from_chat_completion(resp, ticket_id="t1")
    assert len(calls) == 1
    assert calls[0].name == "search_kb"
    assert calls[0].args["query"] == "test"

@test("extract assistant text")
def _():
    resp = {"choices": [{"message": {"content": "Hello world"}}]}
    assert extract_assistant_text(resp) == "Hello world"

@test("inject system message")
def _():
    msgs = [{"role": "user", "content": "hi"}]
    result = inject_system_message(msgs, "You are a bot")
    assert result[0]["role"] == "system"
    assert len(result) == 2


print("\n=== Run Summary ===")

@test("get_run_summary")
def _():
    g = AuraGuard(config=cfg)
    s = g.new_state(run_id="summary-test")
    c = ToolCall(name="search_kb", args={"query": "test"})
    g.on_tool_call_request(state=s, call=c)
    g.on_tool_result(state=s, call=c, result=ToolResult(ok=True))
    summary = g.get_run_summary(s)
    assert summary["run_id"] == "summary-test"
    assert summary["cumulative_cost_usd"] > 0


# ===========================================
#   Fix 3: Cache Controls
# ===========================================

@test("never_cache_tools prevents caching")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", never_cache_tools={"get_time"})
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="get_time", args={"tz": "UTC"})
    g.on_tool_call_request(state=s, call=c)
    g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload={"time": "12:00"}))
    assert len(s.result_cache) == 0, "get_time should not be cached when in never_cache_tools"


@test("arg_ignore_keys strips keys from signature")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        arg_ignore_keys={"search_kb": {"request_id", "timestamp"}},
        repeat_toolcall_threshold=2,
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    # Two calls with different request_ids but same query — should be treated as identical
    c1 = ToolCall(name="search_kb", args={"query": "test", "request_id": "aaa", "timestamp": "t1"})
    c2 = ToolCall(name="search_kb", args={"query": "test", "request_id": "bbb", "timestamp": "t2"})
    d1 = g.on_tool_call_request(state=s, call=c1)
    g.on_tool_result(state=s, call=c1, result=ToolResult(ok=True, payload="r1"))
    d2 = g.on_tool_call_request(state=s, call=c2)
    # With ignored keys, c2 is seen as a repeat of c1
    assert d1.action == PolicyAction.ALLOW
    # d2 should be allowed since threshold=2, but after 2 more it should be caught
    c3 = ToolCall(name="search_kb", args={"query": "test", "request_id": "ccc", "timestamp": "t3"})
    g.on_tool_result(state=s, call=c2, result=ToolResult(ok=True, payload="r2"))
    d3 = g.on_tool_call_request(state=s, call=c3)
    assert d3.action in (PolicyAction.CACHE, PolicyAction.BLOCK), \
        f"Expected CACHE/BLOCK for repeat after threshold, got {d3.action}"


@test("cache_ttl_seconds expires old entries")
def _():
    import time as _t
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        cache_ttl_seconds=0.05,  # 50ms TTL
        repeat_toolcall_threshold=2,
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="get_order", args={"id": "1"})
    # Build up repeats
    for _ in range(3):
        d = g.on_tool_call_request(state=s, call=c)
        if d.action == PolicyAction.ALLOW:
            g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload="x"))
    # At this point should have hit cache. Let TTL expire.
    _t.sleep(0.06)
    # Reset tool stream so repeat count doesn't block
    s.tool_stream.clear()
    d = g.on_tool_call_request(state=s, call=c)
    # After TTL, cache is stale — if threshold not met with fresh stream, should ALLOW
    assert d.action == PolicyAction.ALLOW, f"Expected ALLOW after TTL expiry, got {d.action}"


# ===========================================
#   Fix 4: Real Cost Accounting
# ===========================================

@test("record_tokens adds to cumulative cost")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        cost_model=CostModel(input_token_cost_per_1k=0.003, output_token_cost_per_1k=0.015),
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    g.record_tokens(state=s, input_tokens=1000, output_tokens=500)
    # Expected: 1000/1000*0.003 + 500/1000*0.015 = 0.003 + 0.0075 = 0.0105
    assert abs(s.reported_token_cost - 0.0105) < 0.0001, f"Got {s.reported_token_cost}"
    assert abs(s.cumulative_cost - 0.0105) < 0.0001, f"Got {s.cumulative_cost}"
    assert len(s.cost_events) == 1
    assert s.cost_events[0].event == "token_cost_reported"


@test("record_tokens with cost_override")
def _():
    g = AuraGuard(config=AuraGuardConfig(secret_key=b"test-secret-key", ))
    s = g.new_state()
    g.record_tokens(state=s, cost_override=0.05)
    assert abs(s.reported_token_cost - 0.05) < 0.0001
    assert abs(s.cumulative_cost - 0.05) < 0.0001


@test("middleware record_tokens")
def _():
    from aura_guard.middleware import AgentGuard
    guard = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    guard.record_tokens(input_tokens=2000, output_tokens=1000)
    assert guard.reported_token_cost > 0
    assert guard.cost_spent > 0


# ===========================================
#   Fix 5: State Growth Caps
# ===========================================

@test("result_cache bounded by max_cache_entries")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_cache_entries=3)
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for i in range(5):
        c = ToolCall(name="search", args={"q": f"query_{i}"})
        g.on_tool_call_request(state=s, call=c)
        g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload=f"r{i}"))
    assert len(s.result_cache) <= 3, f"Cache grew to {len(s.result_cache)}, expected max 3"


@test("unique_tool_calls_seen bounded")
def _():
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", max_unique_calls_tracked=3)
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for i in range(5):
        c = ToolCall(name="search", args={"q": f"q{i}"})
        g.on_tool_call_request(state=s, call=c)
        g.on_tool_result(state=s, call=c, result=ToolResult(ok=True, payload=f"r{i}"))
    assert len(s.unique_tool_calls_seen) <= 3, f"Set grew to {len(s.unique_tool_calls_seen)}, expected max 3"


# ===========================================
#   Fix 6: Policy Layer
# ===========================================

@test("tool policy DENY blocks tool")
def _():
    from aura_guard.config import ToolPolicy, ToolAccess
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        tool_policies={"delete_account": ToolPolicy(access=ToolAccess.DENY, deny_reason="forbidden")}
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="delete_account", args={"user_id": "123"})
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.BLOCK, f"Expected BLOCK, got {d.action}"
    assert "policy_deny" in d.reason


@test("tool policy HUMAN_APPROVAL escalates")
def _():
    from aura_guard.config import ToolPolicy, ToolAccess
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        tool_policies={"refund_large": ToolPolicy(access=ToolAccess.HUMAN_APPROVAL, risk="high")}
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="refund_large", args={"amount": 5000})
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.ESCALATE, f"Expected ESCALATE, got {d.action}"
    assert "human_approval" in d.reason


@test("tool policy require_args blocks when missing")
def _():
    from aura_guard.config import ToolPolicy, ToolAccess
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        tool_policies={"transfer": ToolPolicy(require_args={"account_id", "amount"})}
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    c = ToolCall(name="transfer", args={"amount": 100})  # missing account_id
    d = g.on_tool_call_request(state=s, call=c)
    assert d.action == PolicyAction.BLOCK, f"Expected BLOCK, got {d.action}"
    assert "missing_args" in d.reason


@test("tool policy per-tool max_calls overrides global")
def _():
    from aura_guard.config import ToolPolicy
    cfg = AuraGuardConfig(secret_key=b"test-secret-key", 
        max_calls_per_tool=10,  # global
        tool_policies={"search_kb": ToolPolicy(max_calls=2)}  # per-tool override
    )
    g = AuraGuard(config=cfg)
    s = g.new_state()
    for i in range(3):
        c = ToolCall(name="search_kb", args={"query": f"q{i}"})
        d = g.on_tool_call_request(state=s, call=c)
        if d.action == PolicyAction.ALLOW:
            g.on_tool_result(state=s, call=c, result=ToolResult(ok=True))
    # 3rd call should be denied by per-tool policy (max_calls=2)
    assert d.action == PolicyAction.REWRITE, f"Expected REWRITE on 3rd call, got {d.action}"
    assert "max_calls_per_tool" in d.reason


@test("serialization preserves reported_token_cost")
def _():
    from aura_guard.serialization import state_to_json, state_from_json
    g = AuraGuard(config=AuraGuardConfig(secret_key=b"test-secret-key", ))
    s = g.new_state()
    g.record_tokens(state=s, input_tokens=1000, output_tokens=500)
    original_cost = s.reported_token_cost
    json_str = state_to_json(s)
    restored = state_from_json(json_str)
    assert abs(restored.reported_token_cost - original_cost) < 0.0001, \
        f"Expected {original_cost}, got {restored.reported_token_cost}"


# ===========================================
#   Shadow Mode
# ===========================================

print("\n=== Shadow Mode ===")

@test("shadow mode allows everything")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", max_cost_per_run=0.10, shadow_mode=True, default_tool_cost=0.04)
    for _ in range(5):
        d = g.check_tool("search_kb", args={"query": "test"})
        assert d.action == PolicyAction.ALLOW, f"Shadow mode should ALLOW, got {d.action}"
        g.record_result(ok=True, payload="results")
    assert g.shadow_would_deny > 0, "Shadow mode should track would-deny count"
    assert g.blocks == 0
    assert g.stats["shadow_mode"] is True

@test("shadow mode stall not enforced")
def _():
    g = AgentGuard(secret_key=b"test-secret-key", shadow_mode=True)
    text = "I apologize for the inconvenience. We're looking into it."
    for _ in range(10):
        d = g.check_output(text)
        assert d is None, f"Shadow mode should not intervene on output, got {d}"
    assert g.shadow_would_deny > 0


# ===========================================
#   Async Guard
# ===========================================

print("\n=== Async Guard ===")

@test("async guard import and basic properties")
def _():
    from aura_guard import AsyncAgentGuard
    g = AsyncAgentGuard(secret_key=b"test-secret-key", max_cost_per_run=1.00)
    assert g.cost_spent == 0

@test("async guard sync parity")
def _():
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


# ===========================================
#   Default Key Warning
# ===========================================

print("\n=== Secret Key Enforcement ===")

@test("default key allowed in shadow mode")
def _():
    AuraGuard(config=AuraGuardConfig(shadow_mode=True))

@test("default key rejected in enforcement mode")
def _():
    try:
        AuraGuard(config=AuraGuardConfig(shadow_mode=False))
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "default development secret_key" in str(e)

@test("custom key allowed")
def _():
    AuraGuard(config=AuraGuardConfig(secret_key=b"my-production-key-1234"))


# ─── Report ───
print("\n" + "=" * 50)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 50)

if ERRORS:
    print("\nFailures:")
    for name, e in ERRORS:
        print(f"\n  {name}:")
        traceback.print_exception(type(e), e, e.__traceback__)

sys.exit(1 if FAIL else 0)
