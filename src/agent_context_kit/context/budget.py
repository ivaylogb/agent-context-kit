"""Token budget allocation across context components.

A context window is a finite resource. If you don't explicitly allocate it,
the noisiest component (usually tool results or conversation history) crowds
out the quiet ones (instructions, active routine) that actually steer
behavior. A budget makes the allocation a product decision instead of an
accident.

The allocator owns two things:

1. **Per-component budgets** — how many tokens each component is allowed.
   Instructions get a hard allocation; compressible components (history,
   tool results, working memory) get a share of the remainder.
2. **Compression priority order** — which components get compressed first
   when the total exceeds the window limit. Instructions are never
   compressed; history goes first; working memory last. Pluggable.

The window itself (``ContextWindow``) calls ``allocate`` before assembly
and ``over_budget`` after it. Both are pure functions of the component
sizes — the allocator doesn't own the components, just the policy.

Default allocation for a 100K-token context model (matching the reference
doc's table):

| Component          | Budget  | Priority | Compressible |
|--------------------|---------|----------|--------------|
| Instructions       | 5K      | Critical | No           |
| Active routine     | 3K      | High     | No           |
| Tool descriptions  | 5K      | High     | Partial      |
| Working memory     | 5K      | Medium   | Yes          |
| Recent tool results| 10K     | Medium   | Yes          |
| Conversation hist. | 15K     | Low      | Yes          |
| Headroom           | 57K     | Reserved | —            |

The headroom reservation matters. Reserve at least ``max_tokens`` (the
model's ``max_tokens`` reply budget) plus a safety margin — otherwise a
long model response is the thing that tips you over the real window limit.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field


class ComponentCategory(str, Enum):
    """The components the budget knows about.

    New categories can be added here without breaking existing callers —
    unknown categories fall into ``OTHER`` for allocation purposes, which
    means they share the remainder with other compressible components.
    """

    # Never compressed — role, constraints, guardrails.
    INSTRUCTIONS = "instructions"
    # Never compressed — the current procedure the agent is following.
    ROUTINE = "routine"
    # Partially compressible — inactive tools can be dropped from the menu.
    TOOL_DESCRIPTIONS = "tool_descriptions"
    # Compressible — working memory entries roll up by age.
    WORKING_MEMORY = "working_memory"
    # Compressible — full → summary → reference.
    TOOL_RESULTS = "tool_results"
    # Compressible — recent in full, older summarized.
    HISTORY = "history"
    # Progressive-disclosure skills currently loaded into the window.
    SKILLS = "skills"
    # Catch-all. Treated as compressible.
    OTHER = "other"


# Default compression priority order: first in list is compressed first.
# Instructions and routine are absent — they are never compressed.
DEFAULT_COMPRESSION_ORDER: tuple[ComponentCategory, ...] = (
    ComponentCategory.HISTORY,
    ComponentCategory.TOOL_RESULTS,
    ComponentCategory.WORKING_MEMORY,
    ComponentCategory.OTHER,
    ComponentCategory.SKILLS,
    ComponentCategory.TOOL_DESCRIPTIONS,
)


class BudgetAllocation(BaseModel):
    """Per-component budget allocations and total window size.

    All values are in tokens. ``component_limits`` is sparse — only
    allocated categories appear; everything else draws from the shared
    remainder via ``default_share``.
    """

    total_window: int = Field(gt=0)
    reply_reservation: int = Field(ge=0)
    component_limits: dict[ComponentCategory, int] = Field(default_factory=dict)
    # Per-category share of the remainder pool, 0..1. Shares are normalized
    # so they always sum to 1.0; if the caller passes weights that don't sum
    # to 1, they get proportionally scaled.
    default_share: dict[ComponentCategory, float] = Field(default_factory=dict)

    @property
    def input_budget(self) -> int:
        """Tokens available for the request (window minus reply reservation)."""
        return max(0, self.total_window - self.reply_reservation)

    def limit_for(self, category: ComponentCategory) -> int:
        """Hard limit for a category if explicitly allocated; else 0 (share pool)."""
        return self.component_limits.get(category, 0)

    def share_for(self, category: ComponentCategory) -> float:
        """Fractional share of the remainder pool for a category."""
        return self.default_share.get(category, 0.0)


class TokenBudget:
    """Compute per-component budgets for a configured context window.

    The allocator takes the total window size, a reply reservation (so the
    model always has room to emit its response), hard limits for critical
    components, and fractional shares for compressible components. It
    produces a ``BudgetAllocation`` that the window can consult when
    deciding whether to compress.

    Typical production configuration::

        budget = TokenBudget.for_window(
            total_window=200_000,
            reply_reservation=4096,
            hard_limits={
                ComponentCategory.INSTRUCTIONS: 5_000,
                ComponentCategory.ROUTINE: 3_000,
                ComponentCategory.TOOL_DESCRIPTIONS: 5_000,
            },
            shares={
                ComponentCategory.HISTORY: 0.50,
                ComponentCategory.TOOL_RESULTS: 0.30,
                ComponentCategory.WORKING_MEMORY: 0.15,
                ComponentCategory.SKILLS: 0.05,
            },
        )
    """

    def __init__(
        self,
        allocation: BudgetAllocation,
        compression_order: Iterable[ComponentCategory] = DEFAULT_COMPRESSION_ORDER,
    ) -> None:
        self.allocation = allocation
        # Validate compression order: no unknown-to-enum values, no
        # non-compressible components.
        self._compression_order = tuple(compression_order)
        for cat in self._compression_order:
            if cat in (ComponentCategory.INSTRUCTIONS, ComponentCategory.ROUTINE):
                raise ValueError(
                    f"{cat.value!r} is never compressed and must not appear "
                    "in compression_order."
                )

    # -------------------------------------------------------- constructors

    @classmethod
    def for_window(
        cls,
        total_window: int,
        *,
        reply_reservation: int = 4096,
        hard_limits: dict[ComponentCategory, int] | None = None,
        shares: dict[ComponentCategory, float] | None = None,
        compression_order: Iterable[ComponentCategory] | None = None,
    ) -> TokenBudget:
        """Convenience constructor that normalizes shares to sum to 1.0.

        If ``shares`` is empty, all compressible categories get equal share
        of the remainder. If the shares don't sum to 1, they're rescaled.
        """
        if total_window <= 0:
            raise ValueError("total_window must be positive")
        if reply_reservation < 0:
            raise ValueError("reply_reservation must be non-negative")
        if reply_reservation >= total_window:
            raise ValueError("reply_reservation must be smaller than total_window")

        limits = dict(hard_limits) if hard_limits else {}
        raw_shares = dict(shares) if shares else {}
        # Normalize shares to sum to 1.0 if any are provided.
        if raw_shares:
            total = sum(raw_shares.values())
            if total <= 0:
                raise ValueError("share values must be positive and sum > 0")
            normalized = {k: v / total for k, v in raw_shares.items()}
        else:
            normalized = {}

        allocation = BudgetAllocation(
            total_window=total_window,
            reply_reservation=reply_reservation,
            component_limits=limits,
            default_share=normalized,
        )
        return cls(
            allocation=allocation,
            compression_order=(
                compression_order
                if compression_order is not None
                else DEFAULT_COMPRESSION_ORDER
            ),
        )

    # --------------------------------------------------------- computation

    def target_for(
        self,
        category: ComponentCategory,
    ) -> int:
        """Return the target token budget for a given component category.

        Hard limits win over shares. If a category has no hard limit and no
        share, it gets zero — the caller's error if they wanted it to show
        up. (We'd rather surface a "zero budget" bug than silently route to
        a default share, which makes allocation opaque.)
        """
        hard = self.allocation.limit_for(category)
        if hard > 0:
            return hard
        share = self.allocation.share_for(category)
        if share <= 0:
            return 0
        remainder = self._remainder_after_hard_limits()
        return int(remainder * share)

    def _remainder_after_hard_limits(self) -> int:
        """Tokens left after the hard-limit components are subtracted."""
        hard_total = sum(self.allocation.component_limits.values())
        remainder = self.allocation.input_budget - hard_total
        return max(0, remainder)

    def snapshot(self) -> dict[ComponentCategory, int]:
        """All per-category target budgets as a dict. Useful for logging."""
        out: dict[ComponentCategory, int] = {}
        for cat in ComponentCategory:
            out[cat] = self.target_for(cat)
        return out

    def over_budget(
        self,
        usage: dict[ComponentCategory, int],
    ) -> dict[ComponentCategory, int]:
        """Return per-category overrun: how many tokens to free from each.

        Positive values = shrink by that many tokens. Zero or negative means
        within budget. Used by ``ContextWindow`` to decide what to compress
        and by how much.
        """
        out: dict[ComponentCategory, int] = {}
        for cat, used in usage.items():
            target = self.target_for(cat)
            delta = used - target
            if delta > 0:
                out[cat] = delta
        return out

    def compression_order(self) -> tuple[ComponentCategory, ...]:
        """The order in which compressible components are shrunk when over budget."""
        return self._compression_order
