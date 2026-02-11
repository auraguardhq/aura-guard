# Aura Guard

Stop AI agents from **looping on tools** and accidentally doing the same action twice (double refunds, duplicate emails, endless retries).

Aura Guard is **reliability middleware for tool-using agents**. It sits **between your agent and its tools** (search, refund, get_order, send_email, etc.) and focuses on three controls: **idempotency, circuit breaking, and loop detection**.  
Before a tool runs, Aura Guard answers:

- ✅ **ALLOW** → run the tool  
- ♻️ **CACHE** → reuse the last result (don’t call the tool again)  
- ⛔ **BLOCK** → stop a risky / repetitive call  
- ✍️ **REWRITE** → tell the model “stop looping, do this instead”  
- 🧑‍💼 **ESCALATE / FINALIZE** → stop the run safely  

**Core goals**
- Prevent duplicate side-effects (refund twice, email twice, cancel twice)
- Contain retry storms (429 / timeouts / 5xx)
- Detect loops early and stop runaways
- Provide deterministic, inspectable decisions

✅ Python 3.10+  
✅ Dependency‑free core (optional LangChain adapter)  
✅ Framework‑agnostic (works with your custom loop)

---

## Table of contents

- [30-second demo (no API key)](#30-second-demo-no-api-key)
- [How it works (1 minute explanation)](#how-it-works-1-minute-explanation)
- [Install](#install)
- [Benchmarks](#benchmarks)
- [What problem does this solve](#what-problem-does-this-solve)
- [Why not just max_steps / retries / idempotency keys?](#why-not-just-max_steps--retries--idempotency-keys)
- [2-minute integration (copy/paste)](#2-minute-integration-copypaste)
- [Live A/B (real model) — optional](#live-ab-real-model--optional)
- [Configuration (the knobs that matter)](#configuration-the-knobs-that-matter)
- [LangChain (optional)](#langchain-optional)
- [Telemetry & persistence (optional)](#telemetry--persistence-optional)
- [Non-goals & limitations](#non-goals--limitations)
- [Security & privacy](#security--privacy)
- [Shadow mode](#shadow-mode-evaluate-before-enforcing)
- [Async support](#async-support)
- [Quick integration examples](#quick-integration-examples)
- [Docs](#docs)
- [Contributing](#contributing)
- [License](#license)

---

## 30-second demo (no API key)

This is the fastest way to “feel” what Aura Guard does.

```bash
pip install aura-guard
aura-guard demo
```

You should see output like:

```text
================================================================
  Aura Guard — Triage Simulation Demo
================================================================
  Assumed tool-call cost: $0.04 per call

  Variant                   Calls  SideFX  Blocks   Cache     Cost  Terminated
  ────────────────────────────────────────────────────────────────────────
  no_guard                     11       3       0       0    $0.44  -
  call_limit(5)                 5       3       0       0    $0.20  call_limit
  aura_guard                    4       1       0       2    $0.16  escalate

  Cost saved vs no_guard:     $0.28 (64%)
  Side-effects prevented:     2
  Rewrites issued:            6
```

For source installs and benchmarks, see Install and Benchmarks below.

---

## How it works (1 minute explanation)

![How Aura Guard sits between your agent loop and tools](docs/assets/how-it-works.png)

Aura Guard keeps run-scoped state and makes deterministic decisions from it:
- repeated/near-repeated calls (loop detection)
- repeated tool errors (circuit breaker behavior)
- side-effect replay risk (idempotency protection)
- output stall signals and run cost

---

## Install

### Option A (recommended):

```bash
pip install aura-guard
```

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

## Benchmarks

Run the full synthetic benchmark suite:

```bash
aura-guard bench --all
```

Run one scenario:

```bash
aura-guard bench --scenario <ID>
```

Save JSON output:

```bash
aura-guard bench --all --json-out reports/bench.json
```

Note: benchmark costs are estimated; the most important signal is the **relative difference** under the same config.

---

## What problem does this solve?

Agents that can call tools are powerful — but they fail in very predictable ways:

- They **repeat the same tool call** (or almost the same call) over and over.
- They “try different keywords” and **spiral**.
- They hit an error (429 / timeout) and **retry forever**.
- They see a tool response like “pending” and **do the side-effect twice**.
- They produce “sorry, still checking…” text and **stall**.

## Why not just max_steps / retries / idempotency keys?

- **`max_steps` is blunt**: it does not distinguish productive steps from runaway loops.
- **Backoff/retry libraries don’t prevent duplicate side effects**: they retry transport failures, not business-level replay risk.
- **Idempotency keys help side effects but not loops/cost/quarantine**: they protect write operations, but they do not stop search spirals, repeated failed reads, or stalled agent outputs.

---

## 2-minute integration (copy/paste)

Aura Guard does **not** call your LLM and does **not** execute tools.  
You keep your agent loop. You just add 3 hook calls:

1) `check_tool(...)` **before** you execute a tool  
2) `record_result(...)` **after** the tool finishes (success or error)  
3) `check_output(...)` **after** the model produces text (optional but recommended)

### Minimal example

```python
from aura_guard import AgentGuard, PolicyAction

guard = AgentGuard(
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

### Recommended: record real token usage (more accurate costs)

After each LLM call, report usage:

```python
guard.record_tokens(
    input_tokens=resp.usage.input_tokens,
    output_tokens=resp.usage.output_tokens,
)
```

---

## Live A/B (real model) — optional

If you want “real model behavior” (not just the synthetic benchmark), run the live A/B harness.

### Anthropic example

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python examples/live_test.py --ab --runs 5 --json-out ab.json
```

This produces a JSON report (recommended: commit it under `reports/`).

**Tip:** Prefer **reproducible commands + JSON artifacts** over screenshots.  
See `docs/RESULTS.md` and `reports/README.md`.

<details>
<summary><b>Example A/B snapshot (table)</b></summary>

Example numbers (5 runs per scenario):

| Scenario | No Guard (avg) | Aura Guard (avg) | Saved (avg) |
|---|---:|---:|---:|
| A: Jitter Loop (reformulation trap) | $0.2778 | $0.1447 | $0.1331 |
| B: Double Refund (ambiguous response trap) | $0.1396 | $0.1275 | $0.0120 |
| C: Error Retry Spiral | $0.1345 | $0.0952 | $0.0393 |
| D: Smart Reformulation (cap enforcement) | $0.8093 | $0.1464 | $0.6629 |
| E: Flagship — Guard + Good Answer | $0.3497 | $0.1420 | $0.2077 |

</details>

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

### Example: per-tool policies (deny / human approval)

```python
from aura_guard import AgentGuard, AuraGuardConfig, ToolPolicy, ToolAccess

guard = AgentGuard(
    config=AuraGuardConfig(
        tool_policies={
            "delete_account": ToolPolicy(access=ToolAccess.DENY, deny_reason="Too risky"),
            "large_refund": ToolPolicy(access=ToolAccess.HUMAN_APPROVAL, risk="high"),
            "search_kb": ToolPolicy(max_calls=5),
        },
    ),
)
```

---

## LangChain (optional)

Aura Guard includes a small LangChain callback adapter.

```python
from aura_guard.adapters.langchain_adapter import AuraCallbackHandler

handler = AuraCallbackHandler(max_cost_per_run=1.00)
# pass handler in your callbacks=[handler]
```

Install: `pip install langchain-core` (see Install section).

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

## Non-goals & limitations

Aura Guard is **not**:
- a content moderation system
- a factuality/accuracy verifier
- a prompt library that “makes agents smarter”
- an observability product (it can emit telemetry, but that’s not the focus)

It **is**:
- a deterministic enforcement layer for tool loops, retries, side-effects, and budgets

---

## Security & privacy

- Guard state is designed to store **signatures** (HMAC hashes), not raw tool args or payloads.
- If you persist state or emit telemetry in production, set a unique `secret_key`.
- Don’t turn on raw-text persistence unless you understand the privacy impact.

### Privacy by design

Aura Guard’s state management uses **HMAC-SHA256 signatures exclusively**. Raw PII — arguments, result payloads, ticket IDs — is **never persisted to disk or emitted in telemetry**. Only keyed hashes are stored.

This means:
- Guard state can be safely written to Redis, Postgres, or log aggregators without leaking customer data.
- Telemetry events contain tool names, reason codes, and cost counters — never raw input or output.
- If your application handles EU personal data, Aura Guard is **GDPR-friendly by design**: no personal data in the guard’s own persistence layer.

> **Note:** Your tool *executors* still handle raw data — Aura Guard’s privacy guarantee covers only the guard’s own state and telemetry, not your application’s tool implementations.

---

## Shadow mode (evaluate before enforcing)

Run Aura Guard in **shadow mode** to see what it *would* block without actually blocking anything. Use this to measure false-positive rates before turning on enforcement in production.

```python
guard = AgentGuard(
    max_cost_per_run=0.50,
    shadow_mode=True,  # log decisions, don't enforce
)

# Your agent loop runs normally — all tools execute.
# After the run, check what the guard would have done:
print(guard.stats["shadow_would_deny"])  # number of would-have-been denials
```

When you’re confident in the false-positive rate, remove `shadow_mode=True` to activate enforcement.

---

## Async support

For async agent loops (FastAPI, LangGraph, etc.), use `AsyncAgentGuard`:

```python
from aura_guard import AsyncAgentGuard, PolicyAction

guard = AsyncAgentGuard(max_cost_per_run=0.50)

decision = await guard.check_tool("search_kb", args={"query": "test"})
if decision.action == PolicyAction.ALLOW:
    result = await execute_tool(...)
    await guard.record_result(ok=True, payload=result)

stall = await guard.check_output(assistant_text)
```

The async wrapper calls the same deterministic engine (no I/O, sub-millisecond) — safe to run directly on the event loop.

---

## Quick integration examples

### Anthropic (Claude)

```python
import anthropic
from aura_guard import AgentGuard, PolicyAction

client = anthropic.Anthropic()
guard = AgentGuard(max_cost_per_run=1.00, side_effect_tools={"refund", "send_email"})

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

guard = AgentGuard(max_cost_per_run=1.00)

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

---

## Docs

- `docs/ARCHITECTURE.md` — how the engine is structured
- `docs/EVALUATION_PLAN.md` — how to evaluate credibly
- `docs/RESULTS.md` — how to publish results (recommended format)

---

## Contributing

See `CONTRIBUTING.md`.

---

## License

Apache-2.0
