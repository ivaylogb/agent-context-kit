"""Core context-management primitives.

The building blocks that every other submodule plugs into:

- ``TokenBudget`` / ``BudgetAllocation`` / ``ComponentCategory`` — how the
  window is divided across components.
- ``ContextWindow`` — assembles the system prompt and messages; enforces the
  budget; mediates compaction.
- ``HistoryManager`` / ``Turn`` / ``TurnRole`` — turn-aware conversation
  history with summarization and critical-moment preservation.
- ``WorkingMemory`` / ``MemoryEntry`` — structured cross-turn memory; a
  superset of ``agent_eval_loop.agent.scratchpad.Scratchpad``.
"""

from agent_context_kit.context.budget import (
    DEFAULT_COMPRESSION_ORDER,
    BudgetAllocation,
    ComponentCategory,
    TokenBudget,
)
from agent_context_kit.context.history import HistoryManager, Turn, TurnRole
from agent_context_kit.context.memory import MemoryEntry, WorkingMemory
from agent_context_kit.context.window import ContextWindow, SkillSlot, ToolResultSlot

__all__ = [
    "BudgetAllocation",
    "ComponentCategory",
    "ContextWindow",
    "DEFAULT_COMPRESSION_ORDER",
    "HistoryManager",
    "MemoryEntry",
    "SkillSlot",
    "TokenBudget",
    "ToolResultSlot",
    "Turn",
    "TurnRole",
    "WorkingMemory",
]
