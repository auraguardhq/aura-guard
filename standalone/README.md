# AuraGuard Standalone

Single-file version of AuraGuard for quick evaluation. No install needed.

## Download

```bash
curl -O https://raw.githubusercontent.com/auraguardhq/aura-guard/main/standalone/aura_guard_standalone.py
```

## Usage

```python
from aura_guard_standalone import AgentGuard, GuardDenied

guard = AgentGuard(
    secret_key=b"your-secret-key",
    side_effect_tools={"refund", "cancel"},
    max_cost_per_run=1.00,
)

result = guard.run("search_kb", search_kb, query="refund policy")

print(guard.report())
```

## What's included

All 8 enforcement primitives, `AgentGuard`, `guard.run()`, `@guard.protect`,
`GuardDenied`, `guard.report()`, `guard.report_data()`.

## What's NOT included

Async wrapper, serialization, MCP/OpenAI/LangChain adapters, CLI, benchmarks.
For those, install the full package:

```bash
pip install aura-guard
```
