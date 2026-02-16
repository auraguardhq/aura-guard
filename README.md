# Aura Guard
Reliability middleware for tool-using AI agents. Prevents tool loops, duplicate side-effects, and retry storms.

```python
from aura_guard import AgentGuard, PolicyAction

guard = AgentGuard(
    secret_key=b"your-secret-key",
    side_effect_tools={"refund", "cancel"},
    max_calls_per_tool=3,
    max_cost_per_run=1.00,
)

decision = guard.check_tool("search_kb", args={"query": "refund policy"})

if decision.action == PolicyAction.ALLOW:
    result = execute_tool(...)
    guard.record_result(ok=True, payload=result)
elif decision.action == PolicyAction.CACHE:
    result = decision.cached_result.payload
elif decision.action == PolicyAction.REWRITE:
    inject_into_prompt(decision.injected_system)
else:
    stop_agent(decision.reason)
```

Aura Guard sits between your agent and its tools. Before each tool call, it returns a deterministic decision: ALLOW, CACHE, BLOCK, REWRITE, or ESCALATE. No LLM calls, sub-millisecond overhead. Core engine makes no network requests; optional webhook telemetry performs HTTP calls.

Python 3.10+ · Zero dependencies · Apache-2.0


## Demo

### Without Aura Guard
Two agents (Coordinator + Analyst) told to consult each other. No termination condition. The loop runs forever.

[![Without Aura Guard](https://img.youtube.com/vi/FkBsRK6OS-4/maxresdefault.jpg)](https://youtu.be/FkBsRK6OS-4)

60 rounds. 79,525 tokens. $0.16 in 3.6 minutes. Never stops on its own.

Extrapolated: $2.68/hour → $706 over 11 days → $5,650 at GPT-4o pricing.

### With Aura Guard
Same agents. Same prompts. Same task. Guard detects the loop automatically.

[![With Aura Guard](https://img.youtube.com/vi/6U-YWF-w7wY/maxresdefault.jpg)](https://youtu.be/6U-YWF-w7wY)

7 rounds. 8,540 tokens. $0.017. Caught by identical_toolcall_loop_cache.

## Table of contents

- [Install](#install)
- [The problem](#the-problem)
- [Integration](#integration)
- [Configuration](#configuration-the-knobs-that-matter)
- [Shadow mode](#shadow-mode-evaluate-before-enforcing)
- [Thread Safety](#thread-safety)
- [Async support](#async-support)
- [Status & limitations](#status--limitations)
- [Docs](#docs)
- [License](#license)

## Install

### Option A (recommended):

```bash
pip install aura-guard

# or with uv
uv pip install aura-guard
```

Try the built-in demo: `aura-guard demo`

### Option B (from source / dev): install from a cloned repo

```bash
git clone https://github.com/auraguardhq/aura-guard.git
cd aura-guard
pip install -e .
```

### Optional: LangChain adapter

```bash
pip install langchain-core
```

---

### Benchmarks (synthetic)

```bash
aura-guard bench --all
```

These simulate common agent failure modes (tool loops, retry storms, duplicate side-effects). Costs are estimated — the important signal is the relative difference. See `docs/EVALUATION_PLAN.md` for real-model evaluation.

### Real-model evaluation

Tested with Claude Sonnet 4 (`claude-sonnet-4-20250514`), 5 scenarios × 5 runs per variant, real LLM tool-use calls with rigged tool implementations.

| Scenario | No Guard | With Guard | Result |
|----------|----------|------------|--------|
| A: Jitter Loop | $0.2778 | $0.1446 | 48% saved |
| B: Double Refund | $0.1397 | $0.1456 | Prevented duplicate refund at +$0.006 overhead |
| C: Error Retry Spiral | $0.1345 | $0.0953 | 29% saved |
| D: Smart Reformulation | $0.8607 | $0.1465 | 83% saved |
| E: Flagship | $0.3494 | $0.1446 | 59% saved |

> All costs are p50 (median) across 5 runs. Scenario B costs slightly more because the guard adds an intervention turn but prevents the duplicate side-effect (the refund only executes once). In Scenario B guard runs, 2 of 5 completed in fewer turns ($0.10), while 3 of 5 required the extra intervention turn ($0.145).

64 guard interventions across 25 runs. No false positives observed in manual review (expected — see caveat below). Task completion maintained or improved in all scenarios.

Full results, per-run data, and screenshots: [docs/LIVE_AB_EXAMPLE.md](docs/LIVE_AB_EXAMPLE.md) | [JSON report](reports/2026-02-09_claude-sonnet-4_ab.json)

---

## The problem
Agent run without guard:

1. search_kb("refund policy")              → 3 results
2. search_kb("refund policy EU")           → 2 results
3. search_kb("refund policy EU Germany")   → 2 results
4. search_kb("refund policy EU Germany 2024") → 1 result
5. search_kb("refund policy EU returns")   → 2 results
6. refund(order="ORD-123", amount=50)      → success
7. refund(order="ORD-123", amount=50)      → success (DUPLICATE!)
8. search_kb("refund confirmation")        → 1 result
... 14 tool calls, $0.56, customer refunded twice


## With Aura Guard
Agent run with guard:

1. search_kb("refund policy")              → ALLOW → 3 results
2. search_kb("refund policy EU")           → ALLOW → 2 results
3. search_kb("refund policy EU Germany")   → ALLOW → 2 results
4. search_kb("refund policy EU Germany 2024") → REWRITE (jitter loop detected)
5. refund(order="ORD-123", amount=50)      → ALLOW → success
6. refund(order="ORD-123", amount=50)      → CACHE (idempotent replay)
... 4 tool calls executed, $0.16, one refund

max_steps doesn't distinguish productive calls from loops. Retry libraries don't prevent duplicate side-effects. Idempotency keys protect writes but don't stop search spirals or stalled outputs. Aura Guard handles all of these with a single middleware layer.

---

## Integration

Aura Guard does **not** call your LLM and does **not** execute tools.  
You keep your agent loop. You just add 3 hook calls:

1) `check_tool(...)` **before** you execute a tool  
2) `record_result(...)` **after** the tool finishes (success or error)  
3) `check_output(...)` **after** the model produces text (optional but recommended)

### Minimal example

```python
from aura_guard import AgentGuard, PolicyAction

guard = AgentGuard(
    secret_key=b"your-secret-key",
    max_calls_per_tool=3,                 # stop “search forever”
    side_effect_tools={"refund", "cancel"},
    max_cost_per_run=1.00,                # optional budget (USD)
    tool_costs={"search_kb": 0.03},        # optional; improves cost reporting
)

def run_tool(tool_name: str, args: dict):
    decision = guard.check_tool(tool_name, args=args, ticket_id="ticket-123")

    if decision.action == PolicyAction.ALLOW:
        try:
            result = execute_tool(tool_name, args)  # <-- your tool function
            guard.record_result(ok=True, payload=result)
            return result
        except Exception as e:
            # classify errors however you want ("429", "timeout", "5xx", ...)
            guard.record_result(ok=False, error_code=type(e).__name__)
            raise

    if decision.action == PolicyAction.CACHE:
        # Aura Guard tells you “reuse the previous result”
        return decision.cached_result.payload if decision.cached_result else None

    if decision.action == PolicyAction.REWRITE:
        # You should inject decision.injected_system into your next prompt
        # and re-run the model.
        raise RuntimeError(f"Rewrite requested: {decision.reason}")

    # BLOCK / ESCALATE / FINALIZE
    raise RuntimeError(f"Stopped: {decision.action.value} — {decision.reason}")
```

Framework-specific adapters for OpenAI and LangChain are included. See examples/ for integration patterns.

<details>
<summary>Framework examples (Anthropic, OpenAI, LangChain)</summary>

### Anthropic (Claude)

```python
import anthropic
from aura_guard import AgentGuard, PolicyAction

client = anthropic.Anthropic()
guard = AgentGuard(secret_key=b"your-secret-key", max_cost_per_run=1.00, side_effect_tools={"refund", "send_email"})

# In your agent loop, after the model returns tool_use blocks:
for block in response.content:
    if block.type == "tool_use":
        decision = guard.check_tool(block.name, args=block.input)

        if decision.action == PolicyAction.ALLOW:
            result = execute_tool(block.name, block.input)
            guard.record_result(ok=True, payload=result)
        elif decision.action == PolicyAction.CACHE:
            result = decision.cached_result.payload  # reuse previous result
        else:
            # BLOCK / REWRITE / ESCALATE — handle accordingly
            break

# After each assistant text response:
guard.check_output(assistant_text)

# Track real token spend:
guard.record_tokens(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
)
```

### OpenAI

```python
from aura_guard import AgentGuard, PolicyAction
from aura_guard.adapters.openai_adapter import (
    extract_tool_calls_from_chat_completion,
    inject_system_message,
)

guard = AgentGuard(secret_key=b"your-secret-key", max_cost_per_run=1.00)

# After each OpenAI response:
tool_calls = extract_tool_calls_from_chat_completion(response)
for call in tool_calls:
    decision = guard.check_tool(call.name, args=call.args)

    if decision.action == PolicyAction.ALLOW:
        result = execute_tool(call.name, call.args)
        guard.record_result(ok=True, payload=result)
    elif decision.action == PolicyAction.REWRITE:
        messages = inject_system_message(messages, decision.injected_system)
        # Re-call the model with updated messages
```

### LangChain

```python
from aura_guard.adapters.langchain_adapter import AuraCallbackHandler

handler = AuraCallbackHandler(
    secret_key=b"your-secret-key",
    max_cost_per_run=1.00,
    side_effect_tools={"refund", "send_email"},
)

# Pass as a callback — Aura Guard intercepts tool calls automatically:
agent = initialize_agent(tools=tools, llm=llm, callbacks=[handler])
agent.run("Process refund for order ORD-123")

# After the run:
print(handler.summary)
# {"cost_spent_usd": 0.12, "cost_saved_usd": 0.40, "blocks": 3, ...}
```

</details>

### Recommended: record real token usage (more accurate costs)

After each LLM call, report usage:

```python
guard.record_tokens(
    input_tokens=resp.usage.input_tokens,
    output_tokens=resp.usage.output_tokens,
)
```

---

## Configuration (the knobs that matter)

Most teams start here:

- **Mark side-effect tools**  
  e.g. `{"refund", "cancel", "send_email"}`

- **Cap expensive tools**  
  e.g. `max_calls_per_tool=3` for search/retrieval

- **Set a max budget per run**  
  e.g. `max_cost_per_run=1.00`

- **Tell Aura Guard your tool costs**  
  so reports are meaningful

For advanced options, see `AuraGuardConfig` in `src/aura_guard/config.py`.

## Shadow mode (evaluate before enforcing)

Shadow mode lets you measure what Aura Guard would block without actually blocking anything. Every decision that would be BLOCK, CACHE, REWRITE, or ESCALATE is logged and counted, but the agent receives ALLOW instead.

This lets you evaluate false positive rates before turning enforcement on in production.
```python
from aura_guard import AgentGuard

guard = AgentGuard(
    max_cost_per_run=1.00,
    max_calls_per_tool=8,
    shadow_mode=True,
)

# Agent runs normally — nothing is blocked
decision = guard.check_tool("search_kb", args={"query": "refund policy"})
# decision.action is always ALLOW in shadow mode

# After the run, check what would have been blocked:
print(guard.stats)
# {"shadow_mode": True, "shadow_would_deny": 3, ...}
```

Use shadow mode to:
- Tune thresholds on real traffic before enforcing
- Compare guard behavior across config changes
- Build confidence that enforcement won't break working agents

When ready, remove `shadow_mode=True` to start enforcing.

### Example: per-tool policies (deny / human approval)

```python
from aura_guard import AgentGuard, AuraGuardConfig, ToolPolicy, ToolAccess

guard = AgentGuard(
    config=AuraGuardConfig(
        secret_key=b"your-secret-key",
        tool_policies={
            "delete_account": ToolPolicy(access=ToolAccess.DENY, deny_reason="Too risky"),
            "large_refund": ToolPolicy(access=ToolAccess.HUMAN_APPROVAL, risk="high"),
            "search_kb": ToolPolicy(max_calls=5),
        },
    ),
)
```

---

## Telemetry & persistence (optional)

### Telemetry
Aura Guard can emit structured events (counts + signatures, not raw args/payloads).  
See `src/aura_guard/telemetry.py`.

### Persist state (optional)
You can serialize guard state to JSON and store it in Redis / Postgres / etc.

```python
from aura_guard.serialization import state_to_json, state_from_json

json_str = state_to_json(state)
state = state_from_json(json_str)
```

---

## Thread safety

Aura Guard is **not** thread-safe. Each `AgentGuard` instance stores per-run state and must be used from the thread that created it. Sharing a guard across threads will raise `RuntimeError`. Create one guard per agent run.

## Async support

Use `AsyncAgentGuard` for async agent loops. It calls the synchronous engine directly (no I/O, sub-millisecond), safe for the event loop.
```python
from aura_guard.async_middleware import AsyncAgentGuard, PolicyAction

guard = AsyncAgentGuard(secret_key=b"your-secret-key", max_cost_per_run=0.50)
decision = await guard.check_tool("search_kb", args={"query": "test"})
```

## Status & limitations

Aura Guard is v0.3 — the API is stabilizing but may change before v1.0.

**Stable:** The 3-method API (check_tool / record_result / check_output), the 6 PolicyAction values, and AuraGuardConfig.

**May change:** Default threshold values, serialization format (versioned — old state will error, not silently corrupt), telemetry event names.

**Limitations:**
- In-memory state only. Not thread-safe. Create one guard per agent run.
- Side-effect enforcement is at-most-once within a single process. Not exactly-once across restarts.
- Argument jitter detection uses token overlap, not semantic similarity. English-biased.
- Cost estimates are configurable approximations, not actual billing data.
- Serialized state and telemetry contain HMAC signatures only. In-memory caches hold tool payloads for caching and idempotency during a run.

For architecture details, see docs/ARCHITECTURE.md.

---

## Docs

- `docs/ARCHITECTURE.md` — how the engine is structured
- `docs/EVALUATION_PLAN.md` — how to evaluate credibly
- `docs/LIVE_AB_EXAMPLE.md` — live A/B walkthrough and sample output
- `docs/RESULTS.md` — how to publish results (recommended format)

---

## Contributing

See `CONTRIBUTING.md`.

---

## License

Apache-2.0
