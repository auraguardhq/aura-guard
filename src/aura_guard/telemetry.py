"""aura_guard.telemetry

Structured event emission for Aura Guard.

Aura Guard itself does not depend on any observability backend. Instead, it
emits structured events via a tiny sink interface. Plug in any backend:
Python logging, webhooks, OpenTelemetry, Langfuse, Datadog, etc.

Security:
- Telemetry events must never include raw tool args, raw tool payloads, or
  raw ticket IDs.
- Emit only keyed signatures and counts.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol


# ================================
# Sink Protocol
# ================================

class TelemetrySink(Protocol):
    """Interface for telemetry backends."""

    def emit(self, event: Dict[str, Any]) -> None: ...


# ================================
# Built-in Sinks
# ================================

@dataclass
class LoggingTelemetry:
    """Default telemetry sink using Python logging."""

    logger_name: str = "aura_guard"
    level: int = logging.INFO

    def emit(self, event: Dict[str, Any]) -> None:
        logging.getLogger(self.logger_name).log(self.level, "%s", event)


@dataclass
class InMemoryTelemetry:
    """Stores events in-memory. Useful for tests, benchmarks, and harnesses."""

    events: List[Dict[str, Any]] = field(default_factory=list)

    def emit(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    def find(self, event_name: str) -> List[Dict[str, Any]]:
        """Return all events matching the given event name."""
        return [e for e in self.events if e.get("event") == event_name]

    @property
    def cost_saved(self) -> float:
        """Sum of all estimated_cost_avoided values across events."""
        total = 0.0
        for e in self.events:
            v = e.get("estimated_cost_avoided")
            if v is not None:
                total += float(v)
        return total


@dataclass
class WebhookTelemetry:
    """Send guard events to an HTTP webhook (Slack, PagerDuty, custom dashboard).

    .. warning:: LATENCY IMPACT

       This sink performs synchronous HTTP calls (urllib) on every guard event.
       Each call blocks for up to ``timeout_seconds`` (default 2.0s). This
       adds network latency to the guard decision path and violates the
       "sub-millisecond overhead" guarantee.

       For production use, prefer:
       - LoggingTelemetry + a log shipper (Fluentd, Vector, etc.)
       - A custom async sink that buffers and ships events out-of-band
       - CompositeTelemetry with InMemoryTelemetry for local + async shipping

    Events are fire-and-forget with a short timeout.
    Failed deliveries are silently dropped (guard must not block on telemetry).
    """

    url: str
    timeout_seconds: float = 2.0
    auth_header: Optional[str] = None          # e.g., "Bearer sk-..."
    include_timestamp: bool = True

    def emit(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        if self.include_timestamp:
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header

        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers=headers, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_seconds)
        except Exception:
            pass  # fire and forget — telemetry must never break the guard


@dataclass
class SlackWebhookTelemetry:
    """Format guard events as Slack messages and send via incoming webhook.

    Formats events into human-readable Slack messages with emoji indicators.

    .. warning:: LATENCY IMPACT — same as WebhookTelemetry. See its docstring.
    """

    webhook_url: str
    channel: Optional[str] = None
    timeout_seconds: float = 2.0

    _EMOJI_MAP: Dict[str, str] = field(default_factory=lambda: {
        "tool_call_cache_hit": "🔄",
        "identical_toolcall_loop_block": "🛑",
        "arg_jitter_loop_quarantine": "🔒",
        "idempotent_replay_blocked": "🔁",
        "side_effect_limit_block": "⛔",
        "tool_quarantined_block": "🚫",
        "tool_quarantined_error_retry": "💥",
        "budget_warning": "⚠️",
        "budget_exceeded_escalate": "🚨",
        "stall_forced_rewrite": "🌀",
        "stall_deterministic_escalate": "⏹️",
    }, repr=False)

    def emit(self, event: Dict[str, Any]) -> None:
        event_name = event.get("event", "unknown")
        emoji = self._EMOJI_MAP.get(event_name, "🛡️")
        tool = event.get("tool", "")
        reason = event.get("reason", event_name)
        cost = event.get("estimated_cost_avoided")

        lines = [f"{emoji} *Aura Guard* — `{event_name}`"]
        if tool:
            lines.append(f"Tool: `{tool}`")
        if reason and reason != event_name:
            lines.append(f"Reason: {reason}")
        if cost is not None:
            lines.append(f"Cost avoided: ${cost:.4f}")

        text = "\n".join(lines)
        payload: Dict[str, Any] = {"text": text}
        if self.channel:
            payload["channel"] = self.channel

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_seconds)
        except Exception:
            pass


@dataclass
class CompositeTelemetry:
    """Fan-out to multiple sinks (e.g., logging + webhook + Langfuse)."""

    sinks: List[Any] = field(default_factory=list)  # List[TelemetrySink]

    def emit(self, event: Dict[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                pass

    def add(self, sink: Any) -> "CompositeTelemetry":
        self.sinks.append(sink)
        return self


# ================================
# Telemetry Facade
# ================================

@dataclass
class Telemetry:
    """Thin facade that the guard uses.

    Provide a custom sink to forward events to your observability backend.
    """

    sink: Any = field(default_factory=LoggingTelemetry)  # TelemetrySink

    def emit(self, event_name: str, **fields: Any) -> None:
        evt: Dict[str, Any] = {"event": event_name, **fields}
        try:
            self.sink.emit(evt)
        except Exception:
            # Telemetry must not break the guard.
            logging.getLogger("aura_guard").exception("Telemetry sink error")
