# Getting Started with AuraGuard (5 minutes)

## What AuraGuard does

Agents break in production. AuraGuard prevents the three most common failures:

| Failure | What happens | AuraGuard response |
|---------|-------------|-------------------|
| **Tool loops** | Agent searches the same thing 20 times with slightly different wording | Detects jitter, returns cached result |
| **Duplicate side-effects** | Agent issues a refund, gets a timeout, retries — customer refunded twice | Idempotency ledger blocks the replay |
| **Retry storms** | API returns 429, agent retries immediately, gets 429 again, loops | Circuit breaker quarantines the tool |

Think of it as: **database → transaction guard, API → rate limiter, agent → AuraGuard.**

## Install
```bash
pip install aura-guard
```

## Protect a tool in one line
```python
from aura_guard import AgentGuard, GuardDenied

guard = AgentGuard(
    secret_key=b"your-secret-key",
    side_effect_tools={"refund", "cancel"},
)

# guard.run() checks the call, executes the function, and records the result
result = guard.run("search_kb", search_kb, query="refund policy")
```

That's it. If the agent calls `search_kb` with the same query three times, the third call returns the cached result instead of executing again.

## Protect with a decorator
```python
@guard.protect
def refund(order_id, amount):
    return payment_api.refund(order_id=order_id, amount=amount)

# First call: executes normally
refund(order_id="ORD-123", amount=50)

# Second call (same args, same ticket): blocked — idempotent replay
try:
    refund(order_id="ORD-123", amount=50)
except GuardDenied as e:
    print(f"Blocked: {e.reason}")
    # e.decision has the full PolicyDecision object
```

**Important:** `guard.run()` and `@guard.protect` require keyword arguments only. This ensures the guard sees the same arguments the function receives.

## Add a cost budget
```python
guard = AgentGuard(
    secret_key=b"your-secret-key",
    side_effect_tools={"refund"},
    max_cost_per_run=1.00,          # USD cap
    tool_costs={"search_kb": 0.03}, # per-call cost estimates
)
```

When the agent approaches the budget, the guard escalates instead of allowing the call.

## Add per-tool limits
```python
guard = AgentGuard(
    secret_key=b"your-secret-key",
    max_calls_per_tool=5,  # max 5 calls to any single tool
)
```

After 5 calls to the same tool, the guard quarantines it and tells the agent to use the information it already has.

## Run the built-in demo
```bash
aura-guard demo
```

This runs a simulated agent that triggers loops, retries, and duplicate side-effects — then shows how the guard catches each one.

## Run the benchmarks
```bash
aura-guard bench --all
```

14 scenarios covering every failure mode. Shows cost savings per scenario.

## Evaluate before enforcing (shadow mode)
```python
guard = AgentGuard(
    secret_key=b"your-secret-key",
    shadow_mode=True,  # log decisions but don't enforce
)
```

Shadow mode lets you measure what would be blocked without actually blocking anything. Check `guard.stats["shadow_would_deny"]` to see the false-positive rate on your real traffic before turning enforcement on.

## Full control (3-method API)

For complex agent loops where you need full control over the check → execute → record cycle:
```python
from aura_guard import AgentGuard, PolicyAction

guard = AgentGuard(secret_key=b"your-secret-key")

decision = guard.check_tool("search_kb", args={"query": "refund policy"})

if decision.action == PolicyAction.ALLOW:
    result = execute_tool("search_kb", {"query": "refund policy"})
    guard.record_result(ok=True, payload=result)
elif decision.action == PolicyAction.CACHE:
    result = decision.cached_result.payload
elif decision.action == PolicyAction.REWRITE:
    # Inject decision.injected_system into your next LLM prompt
    pass
else:
    # BLOCK / ESCALATE / FINALIZE
    stop_or_escalate(decision)
```

Use the 3-method API when:
- Your tool function takes positional arguments
- You need to handle side-effect timeouts with `side_effect_executed=True`
- You're integrating with a framework that manages its own tool execution

## MCP integration
```python
from aura_guard.adapters.mcp_adapter import GuardedMCP

mcp = GuardedMCP(
    "Customer Support",
    secret_key=b"your-secret-key",
    side_effect_tools={"refund"},
    max_cost_per_run=1.00,
)

@mcp.tool()
def search_kb(query: str) -> str:
    return db.search(query)

@mcp.tool(side_effect=True)
def refund(order_id: str, amount: float) -> str:
    return payments.refund(order_id, amount)

mcp.run(transport="stdio")
```

Works with Claude Desktop, Cursor, Windsurf, OpenAI Agents SDK, or any MCP client.

Install: `pip install aura-guard[mcp]`

## What to read next

- [README](../README.md) — full configuration reference, enforcement primitives, all examples
- [Architecture](ARCHITECTURE.md) — how the engine works internally
- [Evaluation plan](EVALUATION_PLAN.md) — how to measure AuraGuard on your real traffic
- [Live A/B results](LIVE_AB_EXAMPLE.md) — real Claude Sonnet test results with cost breakdowns
