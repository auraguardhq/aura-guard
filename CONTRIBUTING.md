# Contributing

Thanks for your interest in contributing to Aura Guard.

## Quick start

```bash
git clone <your-fork-url>
cd auraguard-dev
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python tests/run_tests.py
pytest -q
```

## Project layout

- `src/aura_guard/guard.py` — core policy engine
- `src/aura_guard/middleware.py` — ergonomic wrapper (`AgentGuard`)
- `src/aura_guard/bench/` — synthetic benchmark harness
- `examples/` — live and local demos

## Guidelines

- Keep the core library dependency-free.
- Prefer deterministic behavior and tests over “prompt-only” fixes.
- Avoid persisting raw tool payloads by default (privacy).
