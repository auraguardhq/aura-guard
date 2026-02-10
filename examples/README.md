# Examples

This folder contains small runnable scripts that show what Aura Guard does.

## 1) Offline demo (no API key)

**triage_simulation.py**  
Shows a “broken agent” that:
- retries a refund (side-effect) more than once
- spams a search tool with slightly different queries
- stalls by repeating the same apology

Run:

```bash
python examples/triage_simulation.py
```

## 2) Live A/B with Anthropic (real model)

**live_test.py**  
Connects to a real LLM + rigged tools and runs multiple scenarios (loops, retries, idempotency).
It can output a JSON report you can commit under `reports/`.

Run:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python examples/live_test.py --ab --runs 5 --json-out ab.json
```

## 3) OpenAI integration (no API calls)

**openai_agent_example.py**  
Shows how to wire Aura Guard into an OpenAI-style agent loop **using mock responses** (safe + free).

Run:

```bash
python examples/openai_agent_example.py
```

## 4) OpenAI live demo (real API calls, costs tokens)

**real_agent_test.py**  
Runs a real tool-using agent twice:
- once without Aura Guard
- once with Aura Guard

⚠️ This connects to OpenAI and will cost tokens.

Run:

```bash
pip install openai
export OPENAI_API_KEY=...
python examples/real_agent_test.py --model gpt-4o-mini --max-turns 20
```

Tip: you can also set `OPENAI_MODEL` instead of passing `--model`.

## 5) Ollama/local-model loop demo (no API key)

**ollama_agent_loop.py**  
Shows a minimal Ollama-style agent loop with mock tool-call responses and Aura Guard's 3-method API:
- `check_tool(...)` before tool execution
- `record_result(...)` after tool execution
- `check_output(...)` on assistant text

The demo intentionally jitters search queries and then repeats stall text so you can see Aura Guard rewrite/stop the loop.

Run:

```bash
python examples/ollama_agent_loop.py
```
