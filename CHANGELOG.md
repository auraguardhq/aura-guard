# Changelog

## Unreleased

## 0.7.1 — 2026-03-13

### New Features
- **Single-file standalone version.** `standalone/aura_guard_standalone.py` contains the full guard engine, middleware, and convenience API in one file. No install needed: `curl -O` and import. Includes all 8 primitives, `guard.run()`, `@guard.protect`, `guard.report()`.
- **README rewrite.** New "Not a budget tool" comparison table at the top. Report output shown above the fold. `curl` command for instant evaluation.

## 0.7.0 — 2026-03-13

### New Features
- **Run diagnostics and reporting.** `guard.report()` returns a formatted text report of guard activity: call efficiency, cost breakdown, interventions by primitive, per-tool breakdown, side-effect summary, quarantined tools, and top intervention reasons. `guard.report_data()` returns the same data as a JSON-serializable dict.
- Intervention log tracks every non-ALLOW decision with tool name, action, and reason. Accessible via `report_data()["interventions"]`.
- Shadow mode interventions (would-deny) are also tracked in the intervention log.
- `AsyncAgentGuard` exposes `report()` and `report_data()` via delegation.

## 0.6.0 — 2026-03-12

### New Features
- **MCP adapter v2 with session isolation.** `GuardedMCP` now supports `session_mode="per_session"` for HTTP/SSE transport. Each connected client gets an independent `AgentGuard` with its own loop detection, cost budgets, idempotency ledger, and quarantine state. No more cross-client state bleed.
- **Session registry** with configurable `max_sessions` (default 100) and `session_ttl_seconds` (default 3600). Idle sessions are cleaned up lazily. Oldest sessions are evicted when at capacity.
- **Custom session ID function** via `session_id_fn` parameter for advanced session scoping.
- `session_mode="single"` (default) preserves backward-compatible behavior for stdio transport.
- Updated example with `--http` flag to demonstrate multi-client mode.

## 0.5.1 — 2026-03-12

### Fixes (from ChatGPT Pro deep audit)
- **P0:** Side-effect tool results no longer enter the generic repeat cache. Previously, a side-effect call (e.g. refund) with identical args but a different ticket could be falsely suppressed by the generic `identical_toolcall_loop_cache` instead of being evaluated by the ticket-scoped idempotency ledger. Side-effect tools now exclusively use the idempotency ledger for dedup.
- **P1:** MCP stdio example no longer prints to stdout (which corrupts JSON-RPC). Output redirected to stderr.
- **P1:** Serialization no longer hardcodes `tool_query_sigs` truncation to 12 entries. Previously, users with `tool_loop_window > 12` would silently lose jitter history on state restore.
- **P1:** MCP adapter now includes `injected_system` for REWRITE, `escalation_packet` for ESCALATE, and `finalized_output` for FINALIZE in returned messages. CACHE `json.dumps` now handles non-serializable payloads gracefully.
- **P1:** Documented that `run()` / `@protect` cannot express `side_effect_executed=True` on exceptions — use 3-method API for ambiguous side-effect timeouts.
- **P1:** Documented that `ticket_id` and `side_effect` are reserved kwargs in `run()` / `@protect` and not forwarded to the wrapped function.
- **P1:** Fixed README privacy claim — serialized state contains HMAC signatures plus metadata (tool names, reasons, costs), not signatures only.
- **P2:** `tool_call_counts` now increments only after all checks pass. Previously, a call blocked by sequence detection still incremented the per-tool cap counter.
- **P2:** `tool_stream` now records only executed calls (calls that pass all checks), not attempted calls. Fixes sequence detection and repeat counting accuracy.
- **P2:** Cached `ToolResult` entries are now shallow-copied on write to prevent caller mutation from affecting cached replays.

## 0.5.0 — 2026-03-12

### New Features
- **MCP (Model Context Protocol) adapter.** `GuardedMCP` wraps FastMCP servers with automatic AuraGuard protection on every tool call. Loop detection, side-effect dedup, cost budgets — all enforced transparently. Denied calls return structured error messages to the LLM instead of crashing the server. Install: `pip install aura-guard[mcp]`.
- New example: `examples/mcp_guarded_server.py` — ready-to-use MCP server with guarded tools for Claude Desktop, Cursor, or any MCP client.
- `mcp` added as optional dependency in pyproject.toml.

## 0.4.0 — 2026-03-09

### New Features
- **Primitive 8: Multi-tool sequence loop detection.** Detects repeating sequences of tool calls (A→B→A→B→A→B ping-pong, A→B→C→A→B→C→A→B→C circular delegation). Quarantines the current tool and issues REWRITE when a pattern repeats. Configurable via `sequence_repeat_threshold` (default 3), `max_sequence_length` (default 4), and `sequence_detection_enabled` (default True). Default threshold is 3 (not 2) to avoid false positives on normal alternating workflows.
- 3 new benchmark scenarios: ping-pong delegation, circular 3-agent, and mixed normal + sequence loop.
- Updated `docs/ARCHITECTURE.md` with Primitive 8 design and limitations.

## 0.3.11 — 2026-03-09

### Fixes
- **P0:** `shadow_mode` now works in the core `AuraGuard` engine, not just the `AgentGuard` wrapper. `on_tool_call_request()` and `on_llm_output()` now suppress non-ALLOW decisions when `shadow_mode=True`, returning ALLOW / None instead. Previously, using `AuraGuard` directly with `shadow_mode=True` would still return BLOCK/REWRITE/ESCALATE decisions.
- **P1:** Updated README serialization warning — idempotency ledger keys survive serialization since v0.3.9. The old warning incorrectly stated they did not.
- **P1:** Documented that `run_id` must not contain PII — it is emitted in telemetry and stored in serialized state as raw text.
- **P1:** Added latency warning to `WebhookTelemetry` and `SlackWebhookTelemetry` docstrings — synchronous HTTP calls add network latency to every guard decision.
- **P1:** `cost_events` list is now bounded by `max_cost_events` (default 500). Previously grew unbounded.
- **P2:** Budget escalation message now shows projected cost, not just spent.
- **P2:** Missing `_emit()` added for `policy_missing_args` blocks.
- **P2:** `secret_key` type is now validated early — passing a string instead of bytes raises `TypeError` with a clear message.
- **P2:** Documented that `side_effect_max_executed_per_run` is per-tool, not global.
- **P2:** Documented that `error_code` should be coarse classifications, not raw exception messages.
- **P2:** Documented that `_canonicalize` expects JSON-serializable types for stable signatures.

## 0.3.10 — 2026-03-07

### New Features
- **Convenience API:** `guard.run(name, fn, **kwargs)` — one-liner that handles the full check_tool → execute → record_result cycle. Returns the result on ALLOW, returns cached payload on CACHE, raises `GuardDenied` on BLOCK/REWRITE/ESCALATE/FINALIZE.
- **Decorator API:** `@guard.protect` — wraps any function with automatic guard protection. Tool name inferred from function name (overridable). Supports both `@guard.protect` and `@guard.protect(tool_name="...", side_effect=True)` syntax.
- **`GuardDenied` exception:** Raised by `run()` and `@guard.protect` when a tool call is denied. Carries the full `PolicyDecision` for inspection.
- **Async support:** `AsyncAgentGuard.run()` and `@async_guard.protect` support both sync and async tool functions.
- **Keyword-only enforcement:** Both `run()` and `@guard.protect` require keyword arguments for tool calls. This ensures the guard's signature tracker sees the same arguments the function receives. Positional args raise `TypeError` with a clear message.
- The 3-method API (`check_tool`, `record_result`, `check_output`) remains unchanged and is still recommended for complex integrations or tools that require positional arguments.

## 0.3.9 — 2026-03-06

### Fixes
- **P0:** Idempotency ledger keys now survive serialization. Previously, deserializing guard state would lose the idempotency ledger entirely, allowing duplicate side-effects on the first call after restore. Keys (HMAC signatures) and safe metadata are now persisted; raw payloads remain excluded (PII safety). Serialization version bumped to 5 (backward compatible with version 4).
- **P1:** Clarified `max_calls_per_tool` / `_check_tool_call_cap` documentation — the cap counts *executed* calls (those that pass all prior checks), not *attempted* calls. Calls blocked/cached by earlier primitives don't count toward the cap.
- **P1:** Added `aura-guard bench --all` to CI workflow — catches regressions in scenario loading, benchmark runner, and report generation.
- **P2:** Version is now read from package metadata (`importlib.metadata`) instead of being hardcoded in two places. `pyproject.toml` is the single source of truth.

## 0.3.8 — 2026-02-16

### Fixes
- **P2:** `record_result()` now raises `RuntimeError` in `strict_mode` when called without a preceding `check_tool()` — previously silently no-oped
- **P1:** Documented `strict_mode` in README (feature existed since v0.3.4 but was undocumented)
- **P1:** Fixed LIVE_AB_EXAMPLE.md version note — v0.3.4 included code changes (secret-key enforcement, strict mode), not just docs
- Added serialization caveat to README — `result_cache` and `idempotency_ledger` are excluded (PII risk)

## 0.3.7 — 2026-02-16

### Fixes
- **P0:** Synthetic benchmark (`aura-guard bench --all`) now runs in enforcement mode instead of shadow mode — previously showed 0% savings across all scenarios
- **P0:** Idempotency ledger now stores results when `side_effect_executed=True` even if `ok=False` — previously allowed duplicate side-effects after timeouts
- **P1:** Added `side_effect_executed` parameter to `AgentGuard.record_result()` and `AsyncAgentGuard.record_result()` — enables correct handling of "side effect succeeded but call timed out" scenarios
- **P1:** Clarified quality claim to "scored scenarios (B–E)" — Scenario A was not quality-scored

### Improvements
- Error retry streaks are now consecutive (reset on success) — prevents quarantining tools with rare intermittent failures
- Promoted `estimate_tool_cost()` to public API (was `_estimate_tool_cost`)
- Added version note to LIVE_AB_EXAMPLE.md explaining report was generated on v0.3.1

## 0.3.6 — 2026-02-16

### Docs & Accuracy Fixes

- Fixed all code examples to include required `secret_key` parameter (README, docstrings, examples, bench runner).
- Clarified that "no network requests" applies to core engine only; optional webhook telemetry performs HTTP calls.
- Updated HMAC claim: serialized state and telemetry use signatures only; in-memory caches hold payloads during a run.
- Softened "0 false positives" to "No false positives observed in manual review".
- Fixed LIVE_AB_EXAMPLE.md cost table to match JSON report data.
- Added Thread Safety and Async Support sections to README.
- Added secret_key parameter to LangChain AuraCallbackHandler.
- Fixed async_middleware docstring about blocking behavior.

## 0.3.5 — 2026-02-15

### Docs & Accuracy Fixes

- Fixed all README code examples to include required `secret_key` parameter.
- Clarified that "no network requests" applies to core engine only; optional webhook telemetry performs HTTP calls.
- Updated HMAC claim: serialized state and telemetry use signatures only; in-memory caches hold payloads during a run.
- Softened "0 false positives" to "No false positives observed in manual review".

## 0.3.4 — 2026-02-15

### Security & Reliability (P1 fixes)

- Default secret_key now raises ValueError in production mode (shadow_mode=False). Set a unique key for production use.
- Added strict_mode to AgentGuard. When enabled, raises RuntimeError if record_result() is skipped between check_tool() calls. Tracks missed_results count in stats.
- Added thread safety assertion. AgentGuard now detects cross-thread usage and raises RuntimeError with clear guidance.

### Docs

- Added Shadow Mode section to README (was missing despite being in table of contents).
- Added blog post demo scripts to /examples (Demo 1: ping-pong loop, Demo 2: retry storm).

- License changed from MIT to Apache-2.0.

## 0.3.1 — 2026-02-08

### Improvements

- Live A/B harness (`examples/live_test.py`) now reports **token usage**, **token cost**, **tool cost**, and **total cost** per run and stores them in `ab.json`.
- A/B JSON output now includes per-run `final_response` plus per-variant aggregate quality stats (mean + distribution).
- A/B console summary now shows **NG vs AG quality** side-by-side (keyword rubric).
- Updated `examples/live_test.py` scenario list in the header comment to match scenarios A–E.

### Release hygiene

- Version bump to **0.3.1** across package metadata.
- Tests: 44 total (added shadow mode / async / default-key warning coverage)

## 0.3.0 — 2026-02-07

### New Features

**Tool Policy Layer (Primitive 7)**
- Per-tool access control: `ToolAccess.ALLOW`, `ToolAccess.DENY`, `ToolAccess.HUMAN_APPROVAL`
- Risk classification tags per tool for telemetry/filtering
- Required args enforcement (block calls missing required parameters)
- Per-tool call limits that override the global `max_calls_per_tool`

**Cache Controls**
- `never_cache_tools` — set of tools whose results should never be cached
- `cache_ttl_seconds` — TTL for cached results (expires old entries)
- `arg_ignore_keys` — per-tool arg keys stripped before computing signatures
  (handles timestamps, request IDs, pagination cursors)

**Real Cost Accounting**
- `record_tokens()` method on both `AuraGuard` and `AgentGuard`
- Report actual input/output token counts from LLM API responses
- `cost_override` parameter for direct cost reporting
- `reported_token_cost` tracked separately from estimated tool costs

**State Growth Caps**
- `max_cache_entries` (default 1000) — bounds result_cache size
- `max_unique_calls_tracked` (default 5000) — bounds unique_tool_calls_seen and unique_tool_results_seen
- Prevents unbounded memory growth in long-running agents

### Bug Fixes

- **Fixed float comparison bug in budget warning.** `0.04 * 4 = 0.16` was not
  triggering warning at threshold `0.20 * 0.8 = 0.16000000000000003`. Now uses
  `round(x, 8)` for all cost comparisons.
- **Fixed stall detection telemetry bug.** `stall_pattern_streak` was reset to 0
  before checking its value, so trigger always reported "similarity" instead of
  "pattern". Now captures the trigger reason before resetting.

### Cleanup

- Removed stale `RED_TEAM_AUDIT.md` and `SURGERY_PLAN.md` (referenced TQG module
  that no longer exists in this repo)
- Added `docs/ARCHITECTURE.md` with design decisions and known limitations
- Added `docs/EVALUATION_PLAN.md` — credible evaluation methodology for
  engineers and investors (shadow mode, A/B design, quality metrics)
- Replaced README with founder-grade version (honest framing, no hand-waving)
- Documented threshold semantics (`repeat_toolcall_threshold=3` means "block
  on the 3rd identical call, first 2 are allowed")
- Updated docstring: 7 enforcement primitives (was 6), plus tool policy layer
- Serialization version bumped to 4 (new `reported_token_cost` field)

### Tests

- 38 tests (was 25): added coverage for all new features
- Cache controls: never_cache, arg_ignore_keys, TTL expiry
- Cost accounting: record_tokens, cost_override, middleware integration
- State caps: bounded cache, bounded unique sets
- Policy layer: deny, human_approval, require_args, per-tool max_calls
- Serialization roundtrip for reported_token_cost

## 0.2.2 — 2026-02-06

- A/B baseline mode (`--ab` flag)
- Honest metrics: tool_calls_executed/failed/denied/cached
- Quarantine reason passthrough
- Flagship scenario E
- Log formatting with ellipsis truncation
- JSON output for all CLI commands
- Multi-run statistics (p50/p90)

## 0.2.0 — 2026-02-05

- Initial release with 6 enforcement primitives
- Zero dependencies, framework-agnostic
- OpenAI and LangChain adapters
- Benchmark suite with 10 scenarios
