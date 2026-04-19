"""Audit trail for context-management events.

Parallel to ``agent_tool_kit.observability.AuditLog`` but for context events
instead of tool calls. When something in this package makes a decision —
loaded a skill, summarized history, compacted a tool result, spun up a sub-
agent — an event is written here. Later you can replay the log to debug
"why did the agent lose the customer's name at turn 15?" without re-running
the LLM.

Events are append-only and optionally durable per record (JSONL), same
contract as ``AuditLog``. Thread-safe: the lock covers both in-memory append
and file write so records never interleave on disk.

Common event types you'll see:

- ``skill_loaded`` — a skill's Level-2 content was pulled into the window
- ``skill_unloaded`` — a previously loaded skill was dropped to free budget
- ``history_summarized`` — older turns were compressed into a summary
- ``tool_result_compacted`` — a tool result moved full → summary → reference
- ``budget_exceeded`` — total context exceeded the budget; compression ran
- ``delegated`` — a sub-agent was spawned
- ``memory_compacted`` — working memory was compressed

Event type is a free-form string so callers can emit custom events
without this module having to enumerate them.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class ContextEvent(BaseModel):
    """One entry in the context-management audit trail."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str
    # e.g., "budget_allocator", "skill_router", "history_manager"
    component: str
    # Free-form event payload — structure depends on event_type.
    details: dict[str, Any] = Field(default_factory=dict)
    # Current window occupancy at time of event. Useful for "why did compression
    # run?" debugging. Optional — not every event needs it.
    tokens_before: int | None = None
    tokens_after: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None


class ContextEventLog:
    """Append-only log of context-management events.

    Mirror of ``agent_tool_kit.observability.AuditLog`` for context decisions.
    Records are kept in memory and (if ``path`` is set) flushed to a JSONL file
    per record so a crash mid-conversation still leaves a complete trail.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.session_id = session_id
        self._records: list[ContextEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        event_type: str,
        component: str,
        details: dict[str, Any] | None = None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
    ) -> ContextEvent:
        event = ContextEvent(
            event_type=event_type,
            component=component,
            details=details or {},
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            session_id=self.session_id,
        )
        with self._lock:
            self._records.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(event.model_dump_json() + "\n")
        return event

    def records(self, event_type: str | None = None) -> list[ContextEvent]:
        """Return a snapshot of recorded events, optionally filtered by type."""
        with self._lock:
            recs = list(self._records)
        if event_type is None:
            return recs
        return [r for r in recs if r.event_type == event_type]

    def summary(self) -> dict[str, int]:
        """Count events by type — useful as a quick debugging signal."""
        with self._lock:
            recs = list(self._records)
        counts: dict[str, int] = {}
        for r in recs:
            counts[r.event_type] = counts.get(r.event_type, 0) + 1
        return counts

    @classmethod
    def replay(cls, path: str | Path) -> Iterator[ContextEvent]:
        """Iterate events from a previously written JSONL log."""
        p = Path(path)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield ContextEvent.model_validate_json(line)
