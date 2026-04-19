"""Turn-aware conversation history.

Unmanaged conversation history is where context windows go to die. A 50-turn
conversation is 30K+ tokens of mostly-redundant exchanges, and the oldest
turns — which are often the most important (customer's original problem,
early decisions) — get shoved further from the model's attention with each
new turn.

This module gives you three policies, composable through ``HistoryManager``:

1. **Sliding window** — keep the last N turns in full. Simple; loses long-
   horizon context.
2. **Summarize-and-keep** — turns older than N are summarized into a single
   block; recent turns stay verbatim. Summarizer is pluggable (LLM, rule-
   based, or your own).
3. **Critical-moment preservation** — turns tagged as critical survive
   regardless of age. Tag turns where: a tool was called, a decision was
   made, the customer stated a constraint, or an escalation happened.

The hybrid default — recent turns in full, older turns summarized, critical
moments always preserved — is what Claude Code and production agents
actually use.

The manager produces an Anthropic-API-compatible ``messages`` list via
``build_api_messages()``. Wire it into ``AgentRunner`` by replacing
``conversation_history`` with the manager's output each turn.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_context_kit.tokens import HeuristicTokenCounter, TokenCounter


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    """One turn in the conversation.

    A turn is a single user or assistant message. Tool-call blocks and their
    results are attached to the assistant turn they belong to via
    ``tool_blocks``, so a round-trip (user → assistant-with-tools → tool-
    results → assistant-text) is represented as two turns, not four messages.
    That mirrors how humans think about conversation shape, which makes
    summarization prompts easier to write.
    """

    role: TurnRole
    content: str
    # Optional structured blocks for tool-call round-trips. When present,
    # these override ``content`` for API message construction. Shape matches
    # Anthropic's content blocks (``{"type": "text"|"tool_use"|"tool_result", ...}``).
    tool_blocks: list[dict[str, Any]] | None = None
    # ``critical`` turns are never dropped or summarized, regardless of age.
    critical: bool = False
    # Optional free-form tags for categorization (``"escalation"``,
    # ``"customer_constraint"``, ``"payment"``, etc.). Not consulted by the
    # manager directly; useful for replay and per-tag eval filtering.
    tags: list[str] = Field(default_factory=list)
    # Turn index within the conversation, assigned by the manager at append.
    index: int = 0


class HistoryManager:
    """Manages conversation history with configurable retention policy.

    Default policy: keep the last ``keep_recent`` turns in full, roll older
    turns into a rolling summary, always preserve critical turns. Summarizer
    is pluggable — pass ``summarizer=None`` to get sliding-window-only
    behavior (older turns are dropped, not summarized).

    Typical usage::

        history = HistoryManager(keep_recent=8, summarizer=my_summarizer)
        history.add_user("Where's my order?")
        history.add_assistant("Let me check — could you share the order number?")
        ...
        api_messages = history.build_api_messages()
        # Feed api_messages into AgentRunner's message construction.
    """

    def __init__(
        self,
        keep_recent: int = 8,
        summarizer: Callable[[list[Turn]], str] | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        self.keep_recent = keep_recent
        self.summarizer = summarizer
        self._counter = counter or HeuristicTokenCounter()
        self._turns: list[Turn] = []
        # When older turns have been summarized, the summary text lives here
        # and is prepended to the API messages as a synthetic ``user`` note.
        # None means no summary yet (fresh conversation).
        self._summary: str | None = None
        self._next_index: int = 0

    # ---------------------------------------------------------------- write

    def add_user(
        self,
        content: str,
        *,
        critical: bool = False,
        tags: list[str] | None = None,
    ) -> Turn:
        """Append a user turn."""
        turn = Turn(
            role=TurnRole.USER,
            content=content,
            critical=critical,
            tags=tags or [],
            index=self._next_index,
        )
        self._turns.append(turn)
        self._next_index += 1
        return turn

    def add_assistant(
        self,
        content: str,
        *,
        tool_blocks: list[dict[str, Any]] | None = None,
        critical: bool = False,
        tags: list[str] | None = None,
    ) -> Turn:
        """Append an assistant turn, optionally with tool-use/result blocks."""
        turn = Turn(
            role=TurnRole.ASSISTANT,
            content=content,
            tool_blocks=tool_blocks,
            critical=critical,
            tags=tags or [],
            index=self._next_index,
        )
        self._turns.append(turn)
        self._next_index += 1
        return turn

    def mark_critical(self, index: int) -> None:
        """Pin a turn so future summarization leaves it untouched."""
        for turn in self._turns:
            if turn.index == index:
                turn.critical = True
                return

    def tag_turn(self, index: int, *tags: str) -> None:
        """Attach categorization tags to a turn — useful for replay/eval."""
        for turn in self._turns:
            if turn.index == index:
                for t in tags:
                    if t not in turn.tags:
                        turn.tags.append(t)
                return

    # -------------------------------------------------- retention / compaction

    def rollover(self) -> bool:
        """Apply the retention policy: summarize turns older than ``keep_recent``.

        Critical turns are kept verbatim. Non-critical older turns are
        handed to the summarizer and then dropped. If no summarizer is
        configured, older non-critical turns are simply dropped (sliding
        window). Returns True if anything changed.
        """
        if len(self._turns) <= self.keep_recent:
            return False

        overflow_count = len(self._turns) - self.keep_recent
        to_compact: list[Turn] = []
        to_keep: list[Turn] = []
        for i, turn in enumerate(self._turns):
            is_recent = i >= len(self._turns) - self.keep_recent
            if is_recent or turn.critical:
                to_keep.append(turn)
            else:
                to_compact.append(turn)
        # If nothing is compactable (everything older is critical), we're done.
        if not to_compact:
            return False

        if self.summarizer is not None:
            summary_text = self.summarizer(to_compact)
            if self._summary is None:
                self._summary = summary_text
            else:
                # Append to the existing summary. The summarizer can be
                # called again on the merged summary if it grows too large —
                # see ``compress_summary``.
                self._summary = f"{self._summary}\n\n{summary_text}"
        # Else: drop silently. Sliding window.

        self._turns = to_keep
        # ``overflow_count`` is the number we inspected for compaction; the
        # return value communicates that work happened, not how much.
        return overflow_count > 0

    def compress_summary(
        self,
        summarizer: Callable[[str], str],
    ) -> bool:
        """Re-summarize the existing summary block if it's grown unwieldy.

        When long conversations roll over repeatedly, the summary itself can
        grow to thousands of tokens. This lets the caller re-compress it —
        typically calling the same LLM summarizer on the summary text to
        produce a tighter version. Returns True if a summary existed to
        compress.
        """
        if self._summary is None:
            return False
        self._summary = summarizer(self._summary)
        return True

    # ---------------------------------------------------------------- read

    @property
    def turns(self) -> list[Turn]:
        """A shallow copy of the currently retained turns."""
        return list(self._turns)

    @property
    def summary(self) -> str | None:
        """The current rolled-up summary, or ``None`` if nothing compacted yet."""
        return self._summary

    def tokens(self) -> int:
        """Token count of the retained turns + summary combined."""
        total = 0
        if self._summary:
            total += self._counter.count(self._summary)
        for turn in self._turns:
            total += self._counter.count(turn.content)
            if turn.tool_blocks:
                for block in turn.tool_blocks:
                    for v in block.values():
                        if isinstance(v, str):
                            total += self._counter.count(v)
                        else:
                            total += self._counter.count(str(v))
        return total

    def build_api_messages(self) -> list[dict[str, Any]]:
        """Emit an Anthropic-API-style ``messages`` list.

        If a summary exists, it's prepended as a synthetic ``user`` message
        with an explicit marker so the model knows it's a retrospective
        recap, not a direct customer statement. Pairing this marker with a
        system-prompt line like "history_summary reflects earlier turns,
        not direct user input" keeps the model from treating the summary as
        a pending user ask.
        """
        messages: list[dict[str, Any]] = []
        if self._summary:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "<history_summary>\n"
                        f"{self._summary}\n"
                        "</history_summary>"
                    ),
                }
            )
        for turn in self._turns:
            content: str | list[dict[str, Any]]
            if turn.tool_blocks:
                content = turn.tool_blocks
            else:
                content = turn.content
            messages.append({"role": turn.role.value, "content": content})
        return messages

    def clear(self) -> None:
        self._turns.clear()
        self._summary = None
        self._next_index = 0
