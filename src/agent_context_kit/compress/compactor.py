"""Tool-result compaction.

Tool results are usually the noisiest thing in the window — they're the
JSON the agent got back from its last action. A verbose API response (an
order lookup with 40 fields, a search result with 20 hits) is 80% noise
by the time the model has extracted what it needs. Leaving the full
result around for 10 more turns is how context windows starve.

The compactor progressively degrades a result through three states:

1. ``full`` — verbatim. Keep until it's older than ``full_ttl_turns``.
2. ``summary`` — one-line synopsis ("Order ORD-78234: shipped, ETA
   Thursday"). Keep until ``summary_ttl_turns``.
3. ``reference`` — just a handle ("see tool_result:lookup_order:turn7").
   Retrievable from the audit log or a side store if the agent ever needs
   the original content.

The summarizer is pluggable. The default ``truncate_summarizer`` just cuts
the content to a character budget — cheap and good enough for homogeneous
result shapes (all the same tool). For heterogeneous tools or semantic
compression, pass an LLM-backed summarizer.

This module is read-after-write consistent: running the compactor twice
in a row is idempotent as long as no new results arrived between calls.
Each slot tracks its state explicitly, so we don't mis-transition.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent_context_kit.compress.summarizer import truncate_summarizer
from agent_context_kit.context.window import ContextWindow, ToolResultSlot

# Callable receiving ``(tool_name, content)`` and returning the summary text.
SummaryFn = Callable[[str, str], str]


class ToolResultCompactor:
    """Ages tool results from full → summary → reference over time.

    Usage::

        compactor = ToolResultCompactor(
            full_ttl_turns=2,
            summary_ttl_turns=5,
            summarizer=my_summarizer,
        )
        # At the end of each turn:
        compactor.compact(window, current_turn=len(window.history.turns))

    ``current_turn`` is passed explicitly so the caller can drive the
    compaction clock; the compactor doesn't assume the window's own turn
    count is authoritative.
    """

    def __init__(
        self,
        *,
        full_ttl_turns: int = 2,
        summary_ttl_turns: int = 5,
        summarizer: SummaryFn | None = None,
        summary_char_limit: int = 160,
    ) -> None:
        if full_ttl_turns < 0 or summary_ttl_turns < 0:
            raise ValueError("TTL values must be non-negative")
        if summary_ttl_turns < full_ttl_turns:
            raise ValueError(
                "summary_ttl_turns must be >= full_ttl_turns (summaries "
                "shouldn't expire before the full versions they replace)"
            )
        self.full_ttl_turns = full_ttl_turns
        self.summary_ttl_turns = summary_ttl_turns
        self.summary_char_limit = summary_char_limit
        self.summarizer = summarizer or truncate_summarizer(summary_char_limit)

    def compact(
        self,
        window: ContextWindow,
        *,
        current_turn: int,
    ) -> dict[str, int]:
        """Walk every tool result slot and transition it as appropriate.

        Returns a dict summarizing what happened: ``{"summarized": N,
        "referenced": M, "unchanged": K}``. Useful for budget-event logging
        at the caller.
        """
        out = {"summarized": 0, "referenced": 0, "unchanged": 0}
        slots = window.tool_results()
        for idx, slot in enumerate(slots):
            age = current_turn - slot.turn_index
            if slot.state == "full" and age >= self.full_ttl_turns:
                summary = self.summarizer(slot.tool_name, slot.content)
                window.update_tool_result(idx, content=summary, state="summary")
                out["summarized"] += 1
            elif slot.state == "summary" and age >= self.summary_ttl_turns:
                ref = self._reference(slot)
                window.update_tool_result(idx, content=ref, state="reference")
                out["referenced"] += 1
            else:
                out["unchanged"] += 1
        return out

    def _reference(self, slot: ToolResultSlot) -> str:
        return f"see tool_result:{slot.tool_name}:turn{slot.turn_index}"


def llm_summarizer(
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 150,
) -> SummaryFn:
    """Summarizer backed by a cheap LLM.

    Usage::

        from anthropic import Anthropic
        summarizer = llm_summarizer(Anthropic())
        compactor = ToolResultCompactor(summarizer=summarizer)

    The prompt is deliberately specific: we want one concise line of the
    *actionable* summary — the fields the agent would re-read if it had
    to reference this result again. Not a paragraph of prose.
    """
    system = (
        "You compress tool-call results into one actionable line for the "
        "agent's context. Keep the key IDs and status; drop metadata, "
        "audit fields, and verbose text. Output the single line only."
    )

    def summarize(tool_name: str, content: str) -> str:
        try:
            prompt = f"Tool: {tool_name}\nResult:\n{content}\n\nOne-line summary:"
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            summary = "".join(text_blocks).strip()
            if not summary:
                return truncate_summarizer(160)(tool_name, content)
            return summary.splitlines()[0][:300]
        except Exception:  # noqa: BLE001 — fall back to truncation
            return truncate_summarizer(160)(tool_name, content)

    return summarize


def extract_key_fields(
    tool_name: str,
    content: str,
    fields: list[str],
    char_limit: int = 160,
) -> str:
    """Rule-based summarizer: pull out named fields from a JSON result.

    Usage::

        def summarize(tool_name, content):
            if tool_name == "lookup_order":
                return extract_key_fields(
                    tool_name, content, ["order_id", "status", "eta"]
                )
            return truncate_summarizer(160)(tool_name, content)

    Falls back to truncation if the content isn't JSON or the fields
    aren't present — so this is safe to apply blindly to "mostly-JSON"
    tool outputs. The fallback is why we accept ``char_limit`` here
    instead of hard-coding 160.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return truncate_summarizer(char_limit)(tool_name, content)
    if not isinstance(data, dict):
        return truncate_summarizer(char_limit)(tool_name, content)
    parts: list[str] = []
    for field in fields:
        if field in data:
            parts.append(f"{field}={data[field]}")
    if not parts:
        return truncate_summarizer(char_limit)(tool_name, content)
    return f"{tool_name}: " + ", ".join(parts)
