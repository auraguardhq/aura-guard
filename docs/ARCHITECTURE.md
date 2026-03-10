# Aura Guard — Architecture & Design Decisions

## What this is (in one sentence)

Aura Guard is a small, deterministic reliability middleware layer that watches an agent’s tool calls and stops common failure modes like **tool loops**, **retry storms**, and **duplicate side-effects**.

## Origin

Aura Guard started as an internal reliability layer during agent testing. In red-team style runs, the biggest repeated failure mode was **tool-call runtime enforcement**: agents looping on search, retrying failing tools forever, or repeating side-effect tools (refund twice, email twice, cancel twice).

This repo extracts those controls into a focused, standalone SDK.

## Design Principles

1. **Zero dependencies.** The core library uses only Python stdlib. No numpy,
   no ML models, no external services required.

2. **Deterministic heuristics over ML.** All enforcement uses exact-match
   signatures, token-set overlap, and counters. This makes behavior
   predictable, explainable, and auditable.

3. **Fail-safe defaults.** For normal tools, when uncertain, the guard prefers
   to allow the call (to avoid breaking workflows). For side-effect tools,
   operators should configure stricter rules (idempotency, human approval,
   or deny) depending on risk.

4. **Composable primitives.** Each enforcement primitive (repeat detection,
   jitter detection, circuit breaker, side-effect gating, stall detection,
   cost budget, per-tool cap) operates independently. They can be enabled
   or disabled individually.

5. **Framework-agnostic.** The 3-method API (check_tool → record_result →
   check_output) works with any agent loop. Adapters for OpenAI and
   LangChain are provided as convenience layers.

## Known Limitations

- **Cost model is estimated.** Tool call costs use configurable per-tool
  estimates, not actual provider billing. Integrators can report real
  token counts via `record_tokens()` for more accurate tracking.

- **Similarity is token-overlap, not semantic.** Query jitter detection
  uses exact token intersection, not embeddings. Synonym-based
  reformulation may not be caught. The per-tool call cap serves as
  a catch-all for this case.

- **English-biased tokenization.** The regex tokenizer (`[a-z0-9]+`)
  works well for English and code identifiers. Non-Latin scripts
  will have reduced jitter detection accuracy.

- **In-memory state.** Guard state lives in-process. For multi-process
  or distributed agents, state must be serialized/shared explicitly
  using the provided serialization API.

## Primitive 8: Multi-Tool Sequence Loop Detection

Added in v0.4.0.

Detects repeating sequences of tool calls that indicate circular delegation
or ping-pong patterns between agents/tools.

**How it works:**
- After each tool call passes all existing checks, the guard examines the
  recent tool name history (from `tool_stream`).
- For each possible pattern length (2, 3, 4 tools), it checks whether the
  last N entries form the same pattern repeated K times.
- Example: stream [..., A, B, A, B] with threshold=2 detects a 2-tool
  ping-pong (pattern [A,B] repeated twice).

**Configuration:**
- `sequence_repeat_threshold` (default 3): How many times a pattern must
  repeat before triggering. 3 means A→B→A→B→A→B (6 consecutive alternating
  calls). Threshold 2 was found to cause false positives on normal workflows
  where two tools alternate legitimately.
- `max_sequence_length` (default 4): Maximum pattern length to check.
  Checks 2-tool, 3-tool, and 4-tool patterns.
- `sequence_detection_enabled` (default True): Set False to disable.

**When triggered:**
- The last tool in the detected pattern is quarantined.
- A REWRITE decision is returned with a system prompt describing the loop.
- The agent is told to use existing information instead of continuing the pattern.

**Limitations:**
- Detects exact tool-name sequences only. If agents use different tool
  names for equivalent operations, the pattern won't be detected.
- The pattern must repeat consecutively — interleaved patterns with
  unrelated calls in between are not detected.

## File Structure

```
src/aura_guard/
  guard.py          Core engine — all 8 enforcement primitives
  config.py         Configuration, cost model, and tool policy layer
  middleware.py      AgentGuard wrapper (3-method API)
  types.py          ToolCall, ToolResult, PolicyDecision, PolicyAction
  serialization.py  JSON/dict roundtrip for guard state
  telemetry.py      Event emission (pluggable sinks)
  cli.py            CLI entrypoint
  bench/            Benchmark suite + demo
  adapters/         OpenAI + LangChain adapters
```
