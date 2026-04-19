"""Managed context window.

``ContextWindow`` is the runtime owner of what goes into the LLM on each
turn. It composes the other primitives in this package (budget, history,
memory, skills, tool-result compactor) and produces the two things the
Anthropic API actually takes: a system-prompt string and a ``messages``
list.

The design goals:

- **Assembly order is deterministic.** Instructions → routine → skill
  context → tool descriptions → working memory. The order matters to the
  model; this module enforces it so examples and prod agents don't drift.

- **Budget is enforced at the end, not during assembly.** Each component
  is free to produce its natural render; the window then measures the
  total, asks the budget what's over, and runs compression strategies.
  This lets components stay simple — they don't each need to know the
  current budget.

- **Compression is pluggable but has a sensible default.** Call
  ``enforce_budget()`` and the window compresses the highest-priority
  overruns first, in the order the budget's compression list specifies.

- **Emits observability events at every interesting step.** Pass a
  ``ContextEventLog`` at construction to trace what the window did each
  turn. Default is no-op.

Integration with ``agent_eval_loop.agent.runner.AgentRunner``::

    window = ContextWindow(...)
    window.set_instructions("You are a support agent. ...")
    window.load_skill(billing_skill)
    window.add_user_turn("Where's my order?")

    runner.system_prompt = window.build_system_prompt()
    runner.conversation_history = []  # we own history now
    # Patch the API call to use window.build_api_messages() instead.

See ``examples/`` for the full wiring pattern and the ``ManagedAgentRunner``
subclass in ``agent_context_kit.context.managed_runner`` (optional).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_context_kit.context.budget import ComponentCategory, TokenBudget
from agent_context_kit.context.history import HistoryManager
from agent_context_kit.context.memory import WorkingMemory
from agent_context_kit.observability import ContextEventLog
from agent_context_kit.tokens import HeuristicTokenCounter, TokenCounter


@dataclass
class SkillSlot:
    """A skill currently loaded into the window.

    The window stores just what it needs to render: the skill's name (so
    unloading can find it) and its body text. The Skill object itself
    lives in the skill registry; we don't hold a reference to it, so the
    registry can version-swap skills without stale references here.
    """

    name: str
    body: str


@dataclass
class ToolResultSlot:
    """A tool result in the window.

    ``state`` tracks how compacted this result currently is — full
    (verbatim), summary (handler emitted a short summary), or reference
    (just a file path or ID pointing to the full content). The compactor
    transitions slots through these states as they age out.
    """

    tool_name: str
    # Turn index this result came back on. Used for staleness decisions.
    turn_index: int
    content: str
    state: str = "full"  # "full" | "summary" | "reference"


@dataclass
class _WindowState:
    """Holds the mutable state for the window.

    Kept as a plain dataclass so it's cheap to snapshot for debugging
    (``dataclasses.asdict``) and so the window's top-level API stays free
    of the boring fields.
    """

    instructions: str = ""
    routine: str = ""
    tool_descriptions: str = ""
    skills: list[SkillSlot] = field(default_factory=list)
    tool_results: list[ToolResultSlot] = field(default_factory=list)
    # Static notes the caller wants prepended to the system prompt
    # (e.g., compliance macros). Rendered in the order added.
    static_sections: list[tuple[str, str]] = field(default_factory=list)


class ContextWindow:
    """Assembles the system prompt and messages for each turn.

    Use this as the single source of truth for what the agent sees. Other
    primitives (``WorkingMemory``, ``HistoryManager``, compactors, skill
    routers) plug into it; the window orchestrates assembly and budget
    enforcement.

    The window does NOT own the Anthropic client — construct and call
    it yourself, or plug into ``AgentRunner``. The window's job is to
    produce the inputs (system prompt + messages); the caller does the API
    call and then feeds the response back in as an assistant turn.
    """

    def __init__(
        self,
        budget: TokenBudget,
        *,
        memory: WorkingMemory | None = None,
        history: HistoryManager | None = None,
        counter: TokenCounter | None = None,
        event_log: ContextEventLog | None = None,
    ) -> None:
        self.budget = budget
        self._counter = counter or HeuristicTokenCounter()
        # Defaults: a fresh working memory and a history manager with an 8-
        # turn sliding window and no summarizer. Callers who want summary-
        # backed history should construct their own ``HistoryManager``
        # with a summarizer and pass it in.
        self.memory = memory or WorkingMemory(counter=self._counter)
        self.history = history or HistoryManager(counter=self._counter)
        self.events = event_log
        self._state = _WindowState()
        # Compaction strategies per category. Each strategy takes the
        # current bucket and the number of tokens to free, and returns a
        # new bucket. Set via ``set_compaction_strategy``.
        self._strategies: dict[
            ComponentCategory,
            Callable[[ContextWindow, int], int],
        ] = {}
        self._install_default_strategies()

    # ----------------------------------------------------- component setters

    def set_instructions(self, text: str) -> None:
        """Set the top-level instructions (role, constraints, guardrails)."""
        self._state.instructions = text

    def set_routine(self, text: str) -> None:
        """Set the current routine (procedure the agent is following)."""
        self._state.routine = text

    def set_tool_descriptions(self, text: str) -> None:
        """Set the tool descriptions section.

        This is the narrative description ("when to use X, when NOT to use
        Y") that lives in the system prompt. The actual tool schemas go to
        the API via ``AgentRunner.tool_schemas`` — these two sources are
        separate by design: the prose is for the model's reasoning, the
        schemas are for the API dispatcher.
        """
        self._state.tool_descriptions = text

    def add_static_section(self, tag: str, body: str) -> None:
        """Add an arbitrary labelled section (compliance macros, etc.).

        Rendered verbatim, wrapped in ``<{tag}>...</{tag}>``, in the order
        added. Use sparingly — most content belongs in one of the typed
        components. Static sections are not budget-tracked; they're assumed
        small and always required.
        """
        self._state.static_sections.append((tag, body))

    # ---------------------------------------------------- skills (on demand)

    def load_skill(self, name: str, body: str) -> None:
        """Load a skill's body into the window (Level-2 content).

        Called by the skill router after it has decided which skills are
        relevant for the current turn. Idempotent: re-loading the same name
        replaces the existing body (useful when the loader re-reads the
        file and the content changed between turns).
        """
        tokens_before = self.tokens()
        for slot in self._state.skills:
            if slot.name == name:
                slot.body = body
                self._emit(
                    "skill_reloaded",
                    "context_window",
                    {"skill": name},
                    tokens_before,
                    self.tokens(),
                )
                return
        self._state.skills.append(SkillSlot(name=name, body=body))
        self._emit(
            "skill_loaded",
            "context_window",
            {"skill": name},
            tokens_before,
            self.tokens(),
        )

    def unload_skill(self, name: str) -> bool:
        """Drop a previously loaded skill. Returns True if anything was removed."""
        tokens_before = self.tokens()
        before = len(self._state.skills)
        self._state.skills = [s for s in self._state.skills if s.name != name]
        removed = len(self._state.skills) < before
        if removed:
            self._emit(
                "skill_unloaded",
                "context_window",
                {"skill": name},
                tokens_before,
                self.tokens(),
            )
        return removed

    def loaded_skill_names(self) -> list[str]:
        return [s.name for s in self._state.skills]

    # ------------------------------------------------- tool results (dynamic)

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        turn_index: int | None = None,
    ) -> None:
        """Record a tool result in the window.

        The result starts in the ``full`` state. As it ages or the window
        runs over budget, the compactor transitions it through ``summary``
        and eventually ``reference`` (see ``compress/compactor.py``).

        ``turn_index`` defaults to the current history turn count; passing
        it explicitly is useful when the caller owns the turn numbering.
        """
        if turn_index is None:
            turn_index = len(self.history.turns)
        self._state.tool_results.append(
            ToolResultSlot(
                tool_name=tool_name,
                turn_index=turn_index,
                content=content,
                state="full",
            )
        )

    def tool_results(self) -> list[ToolResultSlot]:
        return list(self._state.tool_results)

    def update_tool_result(
        self,
        index: int,
        *,
        content: str | None = None,
        state: str | None = None,
    ) -> None:
        """Mutate a tool-result slot in place. Used by compactors."""
        if index < 0 or index >= len(self._state.tool_results):
            raise IndexError(f"tool result index out of range: {index}")
        slot = self._state.tool_results[index]
        if content is not None:
            slot.content = content
        if state is not None:
            slot.state = state

    # ----------------------------------------------------------- turn ops

    def add_user_turn(
        self,
        content: str,
        *,
        critical: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        """Append a user turn and advance working-memory's turn counter.

        The turn counter bump happens here (not in ``add_assistant_turn``)
        because conceptually a conversation turn is initiated by the user;
        the assistant's reply is the back-half of the same turn. Working
        memory's relevance decay keys off this counter.
        """
        self.history.add_user(content, critical=critical, tags=tags)
        self.memory.advance_turn()

    def add_assistant_turn(
        self,
        content: str,
        *,
        tool_blocks: list[dict[str, Any]] | None = None,
        critical: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        self.history.add_assistant(
            content, tool_blocks=tool_blocks, critical=critical, tags=tags
        )

    # ---------------------------------------------------------- assembly

    def build_system_prompt(self) -> str:
        """Assemble the final system prompt from all components.

        Order: static sections → instructions → routine → skills → tool
        descriptions → tool results (compacted) → working memory. Empty
        components are skipped.
        """
        sections: list[str] = []
        for tag, body in self._state.static_sections:
            sections.append(f"<{tag}>\n{body.strip()}\n</{tag}>")
        if self._state.instructions:
            sections.append(
                f"<instructions>\n{self._state.instructions.strip()}\n</instructions>"
            )
        if self._state.routine:
            sections.append(f"<routine>\n{self._state.routine.strip()}\n</routine>")
        if self._state.skills:
            skill_body = "\n\n".join(
                f"## {slot.name}\n{slot.body.strip()}"
                for slot in self._state.skills
            )
            sections.append(f"<skills>\n{skill_body}\n</skills>")
        if self._state.tool_descriptions:
            sections.append(
                f"<tools>\n{self._state.tool_descriptions.strip()}\n</tools>"
            )
        if self._state.tool_results:
            sections.append(self._render_tool_results())
        memory_text = self.memory.render()
        if memory_text:
            sections.append(f"<scratchpad>\n{memory_text}\n</scratchpad>")
        return "\n\n".join(sections)

    def build_api_messages(self) -> list[dict[str, Any]]:
        """Anthropic-API-compatible ``messages`` list for the current turn."""
        return self.history.build_api_messages()

    def _render_tool_results(self) -> str:
        lines = []
        for slot in self._state.tool_results:
            marker = slot.state.upper()
            lines.append(
                f"[{marker}] {slot.tool_name} (turn {slot.turn_index}): {slot.content}"
            )
        body = "\n".join(lines)
        return f"<tool_results>\n{body}\n</tool_results>"

    # --------------------------------------------------------- accounting

    def usage(self) -> dict[ComponentCategory, int]:
        """Per-category token usage right now. Drives budget decisions."""
        c = self._counter
        out: dict[ComponentCategory, int] = {
            ComponentCategory.INSTRUCTIONS: c.count(self._state.instructions),
            ComponentCategory.ROUTINE: c.count(self._state.routine),
            ComponentCategory.TOOL_DESCRIPTIONS: c.count(self._state.tool_descriptions),
            ComponentCategory.WORKING_MEMORY: self.memory.render_tokens(),
            ComponentCategory.HISTORY: self.history.tokens(),
            ComponentCategory.SKILLS: sum(
                c.count(s.body) for s in self._state.skills
            ),
            ComponentCategory.TOOL_RESULTS: sum(
                c.count(t.content) for t in self._state.tool_results
            ),
        }
        return out

    def tokens(self) -> int:
        """Total tokens across all tracked components plus static sections."""
        usage = self.usage()
        total = sum(usage.values())
        for _, body in self._state.static_sections:
            total += self._counter.count(body)
        return total

    # ---------------------------------------------------------- strategies

    def set_compaction_strategy(
        self,
        category: ComponentCategory,
        strategy: Callable[[ContextWindow, int], int],
    ) -> None:
        """Register a compaction callable for a category.

        The strategy receives ``(window, tokens_to_free)`` and should mutate
        the window to free at least that many tokens. It returns the number
        of tokens actually freed (may be less if the bucket is empty).
        """
        self._strategies[category] = strategy

    def _install_default_strategies(self) -> None:
        """Default strategies that don't depend on the compress/ submodule.

        The compress/ submodule provides richer strategies (schema-based
        summarization, progressive compaction); those can be plugged in
        via ``set_compaction_strategy``. The defaults here are simple
        fallbacks so the window works out of the box.
        """
        self._strategies[ComponentCategory.HISTORY] = _default_history_strategy
        self._strategies[ComponentCategory.WORKING_MEMORY] = _default_memory_strategy
        self._strategies[ComponentCategory.TOOL_RESULTS] = _default_tool_results_strategy
        self._strategies[ComponentCategory.SKILLS] = _default_skills_strategy

    def enforce_budget(self) -> dict[ComponentCategory, int]:
        """Run compression until usage fits the budget. Returns freed-per-category.

        Strategies run in the order the budget specifies. Each pass frees
        as many tokens as it can; if the total is still over budget after
        every strategy has run once, we run them again. Two full passes is
        the ceiling — more than that usually means the budget is just too
        small for the conversation, and we surface the remaining overrun
        via the ``budget_exhausted`` event.
        """
        tokens_before = self.tokens()
        freed: dict[ComponentCategory, int] = {}
        max_passes = 2
        for pass_idx in range(max_passes):
            overruns = self.budget.over_budget(self.usage())
            # Also check if total exceeds input_budget — a category may be
            # under its limit but the sum still too big.
            total_over = self.tokens() - self.budget.allocation.input_budget
            if not overruns and total_over <= 0:
                break
            for category in self.budget.compression_order():
                # How much to free: whichever is bigger, the category's
                # overrun or a proportional share of the global overrun.
                deficit = max(
                    overruns.get(category, 0),
                    total_over if total_over > 0 else 0,
                )
                if deficit <= 0:
                    continue
                strategy = self._strategies.get(category)
                if strategy is None:
                    continue
                freed_here = strategy(self, deficit)
                if freed_here > 0:
                    freed[category] = freed.get(category, 0) + freed_here
                    self._emit(
                        "compaction",
                        f"strategy:{category.value}",
                        {"freed": freed_here, "pass": pass_idx + 1},
                        tokens_before=self.tokens() + freed_here,
                        tokens_after=self.tokens(),
                    )
                # Re-check after each strategy; we may already be within
                # budget and can skip later ones.
                total_over = self.tokens() - self.budget.allocation.input_budget
                if total_over <= 0:
                    break

        tokens_after = self.tokens()
        if freed:
            self._emit(
                "budget_enforced",
                "context_window",
                {"freed_total": tokens_before - tokens_after, "by_category": {
                    k.value: v for k, v in freed.items()
                }},
                tokens_before,
                tokens_after,
            )
        remaining_over = self.tokens() - self.budget.allocation.input_budget
        if remaining_over > 0:
            self._emit(
                "budget_exhausted",
                "context_window",
                {"remaining_over": remaining_over},
                tokens_before,
                tokens_after,
            )
        return freed

    # --------------------------------------------------------- utility

    def _emit(
        self,
        event_type: str,
        component: str,
        details: dict[str, Any] | None = None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
    ) -> None:
        if self.events is None:
            return
        self.events.record(
            event_type=event_type,
            component=component,
            details=details or {},
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )


# ---------------------------------------------------------------------------
# Default compaction strategies. Simple, dependency-free; richer strategies
# live in ``agent_context_kit.compress.strategies``.
# ---------------------------------------------------------------------------


def _default_history_strategy(window: ContextWindow, tokens_to_free: int) -> int:
    """Trigger a history rollover. Returns tokens actually freed."""
    before = window.history.tokens()
    changed = window.history.rollover()
    if not changed:
        return 0
    return max(0, before - window.history.tokens())


def _default_memory_strategy(window: ContextWindow, tokens_to_free: int) -> int:
    """Fold old memory entries into a summary line."""
    before = window.memory.render_tokens()
    window.memory.rollup_old(older_than_turns=3)
    return max(0, before - window.memory.render_tokens())


def _default_tool_results_strategy(window: ContextWindow, tokens_to_free: int) -> int:
    """Replace the oldest full tool result with a ``[REF]`` marker.

    Ages the oldest result first — if it's still ``full``, drop to
    ``summary`` by truncation; if already ``summary``, drop to ``reference``
    (just tool_name and turn_index). Frees tokens proportional to the
    content length; we don't try to hit ``tokens_to_free`` exactly — the
    budget enforcer will re-check and call us again if needed.
    """
    slots = window.tool_results()
    if not slots:
        return 0
    # Sort by turn_index ascending so the oldest gets compacted first.
    indexed = sorted(range(len(slots)), key=lambda i: slots[i].turn_index)
    freed = 0
    counter = window._counter  # noqa: SLF001 — within the module
    for idx in indexed:
        slot = slots[idx]
        before = counter.count(slot.content)
        if slot.state == "full":
            new_content = _truncate(slot.content, 120)
            window.update_tool_result(idx, content=new_content, state="summary")
        elif slot.state == "summary":
            new_content = f"see tool_result:{slot.tool_name}:turn{slot.turn_index}"
            window.update_tool_result(idx, content=new_content, state="reference")
        else:
            continue
        after = counter.count(window.tool_results()[idx].content)
        freed += max(0, before - after)
        if freed >= tokens_to_free:
            break
    return freed


def _default_skills_strategy(window: ContextWindow, tokens_to_free: int) -> int:
    """Last-resort: unload a loaded skill. Drops one at a time until the
    deficit is covered.

    In practice the skill router should have scoped loading tightly enough
    that this never runs; it's here as a safety net. We drop skills in
    reverse load order (most recently loaded first) on the assumption that
    the earliest-loaded skill is the one we've been working on longest.
    """
    skills = window._state.skills  # noqa: SLF001 — within module
    if not skills:
        return 0
    counter = window._counter  # noqa: SLF001
    freed = 0
    # Copy the list so we can mutate while iterating its reverse.
    for skill in reversed(list(skills)):
        before = counter.count(skill.body)
        window.unload_skill(skill.name)
        freed += before
        if freed >= tokens_to_free:
            break
    return freed


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
