"""Pluggable compression strategies for the context window.

The window's default strategies (defined inline in ``context/window.py``)
are the minimum viable policy. This module provides richer strategies you
can register via ``window.set_compaction_strategy``:

- ``ProgressiveCompactionStrategy`` for tool results: wires up a
  ``ToolResultCompactor`` to run every time the window asks tool results
  to free tokens.
- ``RelevanceBasedHistoryStrategy``: instead of rolling all-older-than-N
  into a summary, sort turns by a relevance score and drop the bottom
  half. Requires a scoring function.
- ``SummaryHistoryStrategy``: wraps a ``TurnSummaryFn`` (LLM-backed
  typically) so history rollover produces schema-bound summaries, not
  just silent drops.

The strategies share the contract from ``ContextWindow.set_compaction_strategy``:
``(window, tokens_to_free) -> int`` where the return is how many tokens
were freed. Each strategy is a Python callable — either bind the class
(which exposes ``__call__``) or pull off ``.run`` for equivalent wiring.

Thresholds (claude-code-style cascading compression) are also here:
``CascadingThreshold`` fires progressively more aggressive strategies as
window usage climbs past 80/90/95/98%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from agent_context_kit.compress.compactor import ToolResultCompactor
from agent_context_kit.compress.summarizer import TurnSummaryFn
from agent_context_kit.context.budget import ComponentCategory
from agent_context_kit.context.history import Turn
from agent_context_kit.context.window import ContextWindow

# ---------------------------------------------------------------------------
# Tool-result progressive compaction
# ---------------------------------------------------------------------------


class ProgressiveCompactionStrategy:
    """Tool-result strategy that wires in the ``ToolResultCompactor``.

    Usage::

        compactor = ToolResultCompactor(summarizer=my_summarizer)
        strategy = ProgressiveCompactionStrategy(compactor)
        window.set_compaction_strategy(
            ComponentCategory.TOOL_RESULTS, strategy,
        )

    When the window needs to free tool-result tokens, it calls us; we
    drive the compactor to advance every slot's state by one tier and
    return the tokens freed.
    """

    def __init__(self, compactor: ToolResultCompactor) -> None:
        self.compactor = compactor

    def __call__(self, window: ContextWindow, tokens_to_free: int) -> int:
        counter = window._counter  # noqa: SLF001 — same module space
        before = sum(counter.count(t.content) for t in window.tool_results())
        # Use the current turn from the history as the reference clock so the
        # compactor ages slots the same way the budget enforcer sees them.
        current_turn = len(window.history.turns)
        self.compactor.compact(window, current_turn=current_turn)
        after = sum(counter.count(t.content) for t in window.tool_results())
        return max(0, before - after)


# ---------------------------------------------------------------------------
# Summary-backed history rollover
# ---------------------------------------------------------------------------


class SummaryHistoryStrategy:
    """History strategy that plumbs a TurnSummaryFn into rollover.

    The ``HistoryManager`` already supports a summarizer, but only if one
    was passed at construction. This strategy lets you retrofit a
    summarizer onto an existing manager — useful when the summarizer
    depends on a client that isn't available until later, or when you
    want to swap summarizers at runtime (cheap → expensive) as the
    conversation gets more complex.
    """

    def __init__(self, summarizer: TurnSummaryFn) -> None:
        self.summarizer = summarizer

    def __call__(self, window: ContextWindow, tokens_to_free: int) -> int:
        # Swap the summarizer onto the history manager; the manager's
        # rollover will use it for the compaction pass.
        original = window.history.summarizer
        window.history.summarizer = self.summarizer
        try:
            before = window.history.tokens()
            window.history.rollover()
            return max(0, before - window.history.tokens())
        finally:
            window.history.summarizer = original


# ---------------------------------------------------------------------------
# Relevance-based history keeper
# ---------------------------------------------------------------------------


TurnScoreFn = Callable[[Turn], float]


class RelevanceBasedHistoryStrategy:
    """Keep the highest-scoring turns, drop the rest.

    The scoring function decides what "relevant" means:
    - Recency-weighted: ``turn.index / total``
    - Tag-weighted: critical tags score higher
    - Semantic: embedding similarity to the current message

    Turns marked ``critical`` always survive regardless of score — the
    strategy combines with ``HistoryManager``'s pin mechanism, not
    against it.
    """

    def __init__(
        self,
        score_fn: TurnScoreFn,
        keep_minimum: int = 3,
    ) -> None:
        self.score_fn = score_fn
        self.keep_minimum = keep_minimum

    def __call__(self, window: ContextWindow, tokens_to_free: int) -> int:
        history = window.history
        turns = history.turns
        if len(turns) <= self.keep_minimum:
            return 0
        # Score each turn. Critical turns get +inf so they always survive.
        scored: list[tuple[float, Turn]] = []
        for turn in turns:
            if turn.critical:
                scored.append((float("inf"), turn))
            else:
                scored.append((float(self.score_fn(turn)), turn))
        # Sort by score desc; keep the top ones.
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(self.keep_minimum, int(len(scored) * 0.5))
        survivors = {id(t) for _, t in scored[:keep]}
        dropped = [t for t in turns if id(t) not in survivors]
        if not dropped:
            return 0
        # Rebuild the history manager's turn list to retain only survivors
        # in original conversation order.
        surviving_turns = [t for t in turns if id(t) in survivors]
        counter = window._counter  # noqa: SLF001
        freed = sum(counter.count(t.content) for t in dropped)
        history._turns = surviving_turns  # noqa: SLF001
        return freed


# ---------------------------------------------------------------------------
# Cascading thresholds (Claude-Code-style escalation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadingThreshold:
    """One tier of the auto-compression cascade.

    The window's fraction of capacity triggers one or more strategies.
    Ordered low → high by ``trigger_fraction``. Tiers are additive: at
    98% the 80%, 90%, 95%, and 98% strategies all apply.
    """

    trigger_fraction: float  # 0.80, 0.90, etc.
    categories: tuple[ComponentCategory, ...]
    # Free-form label shown in the context-event log so debugging shows
    # which tier fired.
    label: str = ""


DEFAULT_CASCADE: tuple[CascadingThreshold, ...] = (
    CascadingThreshold(
        trigger_fraction=0.80,
        categories=(ComponentCategory.TOOL_RESULTS,),
        label="80%_compact_tool_results",
    ),
    CascadingThreshold(
        trigger_fraction=0.90,
        categories=(ComponentCategory.HISTORY,),
        label="90%_summarize_history",
    ),
    CascadingThreshold(
        trigger_fraction=0.95,
        categories=(ComponentCategory.WORKING_MEMORY, ComponentCategory.SKILLS),
        label="95%_compact_memory_and_skills",
    ),
    CascadingThreshold(
        trigger_fraction=0.98,
        categories=(
            ComponentCategory.TOOL_RESULTS,
            ComponentCategory.HISTORY,
            ComponentCategory.WORKING_MEMORY,
            ComponentCategory.SKILLS,
        ),
        label="98%_emergency_compress_all",
    ),
)


def apply_cascade(
    window: ContextWindow,
    cascade: Sequence[CascadingThreshold] = DEFAULT_CASCADE,
) -> list[str]:
    """Fire every cascade tier whose trigger has been hit.

    Returns the labels of tiers that fired, in order. Each tier calls
    each of its categories' registered strategies with a ``tokens_to_free``
    of "everything above the tier's trigger". The strategies decide how
    much they can actually free.

    Use this as the canonical "auto-compact at 80/90/95/98%" wiring in
    place of (or alongside) the budget's fixed per-category overrun
    calculation.
    """
    capacity = window.budget.allocation.input_budget
    if capacity <= 0:
        return []
    fired: list[str] = []
    for tier in sorted(cascade, key=lambda t: t.trigger_fraction):
        current = window.tokens()
        if current < tier.trigger_fraction * capacity:
            continue
        target_ceiling = int(tier.trigger_fraction * capacity)
        over = max(0, current - target_ceiling)
        if over == 0:
            continue
        for cat in tier.categories:
            strategy = window._strategies.get(cat)  # noqa: SLF001 — same package
            if strategy is None:
                continue
            strategy(window, over)
        fired.append(tier.label)
    return fired
