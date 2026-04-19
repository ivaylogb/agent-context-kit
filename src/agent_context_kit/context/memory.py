"""Structured working memory across turns.

This is a superset of ``agent_eval_loop.agent.scratchpad.Scratchpad``. Drop-in
compatible for the bits of code that already use the Scratchpad interface
(``set`` / ``get`` / ``append_to`` / ``render`` / ``clear`` / ``compact`` /
``entries`` / ``history``) but adds the capabilities the simpler scratchpad
was missing:

- **Typed entries** — each entry carries a type tag (``fact``, ``decision``,
  ``note``, ``customer_statement``, etc.) so the renderer can group them
  and the compactor can prioritize.
- **Priority entries** — entries marked ``priority=True`` (customer-stated
  constraints, verified facts) never get compacted away. The compaction
  strategy respects this.
- **Relevance decay** — each entry tracks the turn it was written on. Older
  entries are candidates for compaction first.
- **Auto-compaction** — when entry count exceeds ``max_entries`` or total
  token count exceeds ``max_tokens``, older non-priority entries are rolled
  up into a summary entry. Threshold configurable; disabled by default so
  the simple case (short conversations) doesn't pay the cost.

The rendered output keeps the same shape Scratchpad produced for simple
key=value entries, so existing prompt templates that reference
``<scratchpad>`` keep working.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_context_kit.tokens import HeuristicTokenCounter, TokenCounter


class MemoryEntry(BaseModel):
    """One typed entry in working memory.

    ``turn_added`` is used by compaction to decide which entries are eligible
    to roll up. ``priority`` entries are pinned: a customer-stated constraint
    or a verified fact should survive compaction regardless of age.
    """

    key: str
    value: Any
    entry_type: str = "note"
    priority: bool = False
    turn_added: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemory:
    """Cross-turn structured memory for an agent.

    API surface compatible with ``Scratchpad``: ``set`` / ``get`` /
    ``append_to`` / ``render`` / ``clear`` / ``compact`` / ``entries`` /
    ``history`` all behave the same for basic usage. New methods
    (``set_typed``, ``pin``, ``entries_by_type``, etc.) enable the richer
    capabilities without breaking existing callers.

    Usage::

        mem = WorkingMemory()
        mem.set("customer_name", "Alex")                 # legacy scratchpad API
        mem.set_typed(
            "order_id", "ORD-123",
            entry_type="fact", priority=True,
        )
        mem.advance_turn()                                # call per turn
        system_prompt += f"<scratchpad>\\n{mem.render()}\\n</scratchpad>"
    """

    def __init__(
        self,
        counter: TokenCounter | None = None,
        max_tokens: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        # Ordered dict keeps insertion order — render order is insertion order
        # unless the caller explicitly groups by type.
        self._entries: dict[str, MemoryEntry] = {}
        self._history: list[dict[str, Any]] = []
        self._turn: int = 0
        self._counter = counter or HeuristicTokenCounter()
        self.max_tokens = max_tokens
        self.max_entries = max_entries

    # ------------------------------------------------------- turn management

    def advance_turn(self) -> int:
        """Bump the turn counter. Call once per conversation turn.

        Relevance decay and auto-compaction both key off of turn numbers.
        If you never call this, memory still works — you just lose the
        age-aware features.
        """
        self._turn += 1
        return self._turn

    @property
    def current_turn(self) -> int:
        return self._turn

    # ------------------------------------------------------- Scratchpad API

    def set(self, key: str, value: Any) -> None:
        """Set a memory entry. Overwrites if key exists.

        Scratchpad-compatible: entries created via ``set`` are untyped
        (``entry_type="note"``) and non-priority. Use ``set_typed`` when
        the entry is a fact that should survive compaction.
        """
        old = self._entries.get(key)
        old_value = old.value if old is not None else None
        # Preserve priority if the entry was already pinned, so a benign
        # ``set`` doesn't accidentally unpin a critical entry.
        priority = old.priority if old is not None else False
        entry_type = old.entry_type if old is not None else "note"
        self._entries[key] = MemoryEntry(
            key=key,
            value=value,
            entry_type=entry_type,
            priority=priority,
            turn_added=self._turn,
        )
        self._history.append(
            {
                "action": "set",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "turn": self._turn,
            }
        )

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._entries.get(key)
        return entry.value if entry is not None else default

    def append_to(self, key: str, value: Any) -> None:
        """Append to a list-type entry. Creates the list if it doesn't exist.

        Scratchpad-compatible. If the existing entry isn't a list, it's
        promoted to one (``x`` → ``[x]``) before the append, matching
        Scratchpad's behavior.
        """
        existing = self._entries.get(key)
        if existing is None:
            new_value: list[Any] = [value]
        elif isinstance(existing.value, list):
            new_value = list(existing.value)
            new_value.append(value)
        else:
            new_value = [existing.value, value]
        # Preserve type + priority flags from the prior entry.
        old_priority = existing.priority if existing is not None else False
        old_type = existing.entry_type if existing is not None else "note"
        self._entries[key] = MemoryEntry(
            key=key,
            value=new_value,
            entry_type=old_type,
            priority=old_priority,
            turn_added=self._turn,
        )
        self._history.append(
            {
                "action": "append",
                "key": key,
                "appended": value,
                "turn": self._turn,
            }
        )

    def clear(self) -> None:
        self._entries.clear()
        self._history.clear()
        self._turn = 0

    # ------------------------------------------------ typed / priority API

    def set_typed(
        self,
        key: str,
        value: Any,
        *,
        entry_type: str = "note",
        priority: bool = False,
    ) -> None:
        """Set an entry with an explicit type and priority.

        Use this when the entry is a fact that should never be compacted
        (``priority=True``) or when you want grouped rendering by type.
        """
        old = self._entries.get(key)
        old_value = old.value if old is not None else None
        self._entries[key] = MemoryEntry(
            key=key,
            value=value,
            entry_type=entry_type,
            priority=priority,
            turn_added=self._turn,
        )
        self._history.append(
            {
                "action": "set_typed",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "entry_type": entry_type,
                "priority": priority,
                "turn": self._turn,
            }
        )

    def pin(self, key: str) -> None:
        """Mark an existing entry as priority (exempt from compaction).

        No-op if the key doesn't exist — we don't raise here because the
        agent loop shouldn't crash on a benign misuse.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        if not entry.priority:
            entry.priority = True
            self._history.append(
                {"action": "pin", "key": key, "turn": self._turn}
            )

    def unpin(self, key: str) -> None:
        """Unpin an entry so future compaction can claim it."""
        entry = self._entries.get(key)
        if entry is None or not entry.priority:
            return
        entry.priority = False
        self._history.append(
            {"action": "unpin", "key": key, "turn": self._turn}
        )

    def delete(self, key: str) -> bool:
        """Remove an entry. Returns True if it existed, False if it didn't."""
        if key not in self._entries:
            return False
        del self._entries[key]
        self._history.append({"action": "delete", "key": key, "turn": self._turn})
        return True

    def entries_by_type(self, entry_type: str) -> list[MemoryEntry]:
        """All entries matching a type tag, in insertion order."""
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    # ------------------------------------------------------------- render

    def render(self, group_by_type: bool = False) -> str:
        """Render memory for injection into the system prompt.

        Returns an empty string when memory is empty (so we don't waste
        tokens on an empty ``<scratchpad>`` tag).

        ``group_by_type=True`` prints entries grouped under type headers
        instead of in insertion order. Useful once memory is rich enough
        that grouped output is more readable (working hypothesis: ~5+
        entries across 3+ types).
        """
        if not self._entries:
            return ""

        if group_by_type:
            return self._render_grouped()
        return self._render_flat()

    def _render_flat(self) -> str:
        lines: list[str] = []
        for entry in self._entries.values():
            lines.append(self._render_line(entry))
        return "\n".join(lines)

    def _render_grouped(self) -> str:
        buckets: dict[str, list[MemoryEntry]] = {}
        for entry in self._entries.values():
            buckets.setdefault(entry.entry_type, []).append(entry)
        sections: list[str] = []
        for entry_type, bucket in buckets.items():
            section = [f"[{entry_type}]"]
            for entry in bucket:
                section.append("  " + self._render_line(entry))
            sections.append("\n".join(section))
        return "\n\n".join(sections)

    def _render_line(self, entry: MemoryEntry) -> str:
        """Format a single entry the way Scratchpad did for API compatibility."""
        prefix = "* " if entry.priority else ""
        value = entry.value
        if isinstance(value, list):
            items = "\n".join(f"  - {item}" for item in value)
            return f"{prefix}{entry.key}:\n{items}"
        if isinstance(value, dict):
            return f"{prefix}{entry.key}: {json.dumps(value, indent=2, default=str)}"
        return f"{prefix}{entry.key}: {value}"

    def render_tokens(self) -> int:
        """Token count of the rendered output."""
        return self._counter.count(self.render())

    # ---------------------------------------------------------- compaction

    def compact(
        self,
        summarizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Compact memory to reduce token usage.

        Scratchpad-compatible signature: if a summarizer is provided, it
        receives the current (untyped) entries-dict and should return a
        condensed version. Priority entries are always preserved through
        the summarizer: the summarizer is given the non-priority subset,
        and its output is merged back with the priority entries intact.

        With no summarizer, this is a no-op for API compatibility — use
        ``rollup_old(...)`` for the turn-based automatic compaction.
        """
        if summarizer is None:
            return
        # Isolate priority entries so they survive compaction untouched.
        pinned: dict[str, MemoryEntry] = {
            k: e for k, e in self._entries.items() if e.priority
        }
        compactible: dict[str, Any] = {
            k: e.value for k, e in self._entries.items() if not e.priority
        }
        summarized = summarizer(compactible)
        # Rebuild: pinned first (preserves their metadata), then the
        # summarizer's output as plain notes at the current turn.
        new_entries: dict[str, MemoryEntry] = dict(pinned)
        for k, v in summarized.items():
            if k in new_entries:
                # Don't overwrite pinned entries.
                continue
            new_entries[k] = MemoryEntry(
                key=k,
                value=v,
                entry_type="summary",
                priority=False,
                turn_added=self._turn,
            )
        self._entries = new_entries
        self._history.append({"action": "compact", "turn": self._turn})

    def rollup_old(
        self,
        older_than_turns: int,
        summarizer: Callable[[list[MemoryEntry]], str] | None = None,
    ) -> int:
        """Fold entries added more than N turns ago into a ``memory_summary``.

        Priority entries are exempt. Returns the number of entries folded.

        Without a summarizer, old entries are replaced with a one-line
        bullet list of their keys. Pass a summarizer (often an LLM-backed
        one) for higher-fidelity compaction.
        """
        if self._turn == 0:
            return 0
        cutoff = self._turn - older_than_turns
        old_entries: list[MemoryEntry] = [
            e for e in self._entries.values()
            if not e.priority and e.turn_added < cutoff
        ]
        if not old_entries:
            return 0

        if summarizer is not None:
            summary_text = summarizer(old_entries)
        else:
            summary_text = "; ".join(
                f"{e.key}={_short_repr(e.value)}" for e in old_entries
            )

        # Remove the folded entries.
        for entry in old_entries:
            self._entries.pop(entry.key, None)

        # Attach (or extend) a single ``memory_summary`` entry.
        existing_summary = self._entries.get("memory_summary")
        if existing_summary is None:
            self._entries["memory_summary"] = MemoryEntry(
                key="memory_summary",
                value=summary_text,
                entry_type="summary",
                priority=False,
                turn_added=self._turn,
            )
        else:
            merged = f"{existing_summary.value}; {summary_text}"
            existing_summary.value = merged
            existing_summary.turn_added = self._turn

        self._history.append(
            {
                "action": "rollup",
                "folded": [e.key for e in old_entries],
                "turn": self._turn,
            }
        )
        return len(old_entries)

    def auto_compact_if_needed(
        self,
        summarizer: Callable[[list[MemoryEntry]], str] | None = None,
    ) -> bool:
        """Trigger rollup if limits are exceeded. Returns True if anything changed.

        Respects ``max_entries`` and ``max_tokens`` configured at construction.
        Fold threshold starts at "older than 3 turns" and relaxes to "older
        than 1 turn" if the first pass didn't free enough space. Priority
        entries are always exempt.
        """
        if self.max_entries is None and self.max_tokens is None:
            return False

        def over_limit() -> bool:
            if self.max_entries is not None and len(self._entries) > self.max_entries:
                return True
            if self.max_tokens is not None and self.render_tokens() > self.max_tokens:
                return True
            return False

        if not over_limit():
            return False

        changed = False
        for threshold in (3, 2, 1):
            folded = self.rollup_old(threshold, summarizer=summarizer)
            changed = changed or folded > 0
            if not over_limit():
                break
        return changed

    # ---------------------------------------------------------- properties

    @property
    def entries(self) -> dict[str, Any]:
        """Untyped view: ``{key: value}``. Scratchpad-compatible.

        Returns a shallow copy so callers can't mutate internal state.
        """
        return {k: e.value for k, e in self._entries.items()}

    @property
    def typed_entries(self) -> dict[str, MemoryEntry]:
        """Full typed view. Returns a shallow copy."""
        return dict(self._entries)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Append-only mutation log. Scratchpad-compatible."""
        return list(self._history)


def _short_repr(value: Any, limit: int = 40) -> str:
    """Short repr for entries being summarized — keeps the summary line readable."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{...{len(value)} keys}}"
    s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."
