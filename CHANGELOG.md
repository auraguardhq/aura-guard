# Changelog

## Unreleased

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
