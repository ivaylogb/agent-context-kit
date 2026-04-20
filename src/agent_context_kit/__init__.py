"""agent-context-kit: runtime context engineering for LLM agents.

The LLM is the CPU. The context window is RAM. Context engineering is the
operating system that manages what goes in.

This package gives you the runtime machinery: token budgets across
component types, dynamic skill loading with progressive disclosure,
tool-result compaction, schema-based conversation summarization, and
sub-agent isolation.

Public surface (grouped by the write/select/compress/isolate taxonomy):

**WRITE — what enters context**
- ``ContextWindow`` — assembles the system prompt and messages each turn.
- ``WorkingMemory`` — structured cross-turn state (superset of Scratchpad).
- ``HistoryManager`` — turn-aware history with summarization and critical-
  moment preservation.

**SELECT — choosing what's relevant**
- ``SkillRegistry`` / ``Skill`` — two-level progressive disclosure.
- ``RuleBasedRouter`` / ``EmbeddingRouter`` / ``LLMRouter`` — multi-label
  classifiers that load only the skills relevant to each turn.

**COMPRESS — keeping context lean**
- ``TokenBudget`` — per-component budget allocator.
- ``ToolResultCompactor`` — full → summary → reference aging.
- ``llm_summarizer`` / ``build_schema_summarizer`` — schema-bound
  conversation summarization.
- ``apply_cascade`` — 80/90/95/98% threshold-driven auto-compression.

**ISOLATE — containing complexity**
- ``SubAgent`` / ``delegate`` — isolated sub-agent with structured return.
- ``StaticPlanner`` / ``LLMPlanner`` — planner → executor orchestration.

**OBSERVABILITY**
- ``ContextEventLog`` — audit trail for context-management decisions.
"""

from agent_context_kit.compress import (
    DEFAULT_CASCADE,
    DEFAULT_SCHEMA,
    CascadingThreshold,
    ConversationSummary,
    ProgressiveCompactionStrategy,
    RelevanceBasedHistoryStrategy,
    SummaryHistoryStrategy,
    ToolResultCompactor,
    apply_cascade,
    build_schema_summarizer,
    extract_key_fields,
    extractive_summarizer,
    llm_summarizer,
    truncate_summarizer,
)
from agent_context_kit.context import (
    BudgetAllocation,
    ComponentCategory,
    ContextWindow,
    HistoryManager,
    MemoryEntry,
    TokenBudget,
    Turn,
    TurnRole,
    WorkingMemory,
)
from agent_context_kit.isolate import (
    ExecutionResult,
    LLMPlanner,
    PlanResult,
    PlanStep,
    StaticPlanner,
    SubAgent,
    delegate,
)
from agent_context_kit.observability import ContextEvent, ContextEventLog
from agent_context_kit.skills import (
    EmbeddingRouter,
    LLMRouter,
    RouterMatch,
    RuleBasedRouter,
    Skill,
    SkillLoader,
    SkillMetadata,
    SkillRegistry,
    SkillRouter,
)
from agent_context_kit.tokens import (
    CallableCounter,
    HeuristicTokenCounter,
    TiktokenCounter,
    TokenCounter,
    count_messages,
    count_tokens,
    default_counter,
)

__all__ = [
    "BudgetAllocation",
    "CallableCounter",
    "CascadingThreshold",
    "ComponentCategory",
    "ContextEvent",
    "ContextEventLog",
    "ContextWindow",
    "ConversationSummary",
    "DEFAULT_CASCADE",
    "DEFAULT_SCHEMA",
    "EmbeddingRouter",
    "ExecutionResult",
    "HeuristicTokenCounter",
    "HistoryManager",
    "LLMPlanner",
    "LLMRouter",
    "MemoryEntry",
    "PlanResult",
    "PlanStep",
    "ProgressiveCompactionStrategy",
    "RelevanceBasedHistoryStrategy",
    "RouterMatch",
    "RuleBasedRouter",
    "Skill",
    "SkillLoader",
    "SkillMetadata",
    "SkillRegistry",
    "SkillRouter",
    "StaticPlanner",
    "SubAgent",
    "SummaryHistoryStrategy",
    "TiktokenCounter",
    "TokenBudget",
    "TokenCounter",
    "ToolResultCompactor",
    "Turn",
    "TurnRole",
    "WorkingMemory",
    "apply_cascade",
    "build_schema_summarizer",
    "count_messages",
    "count_tokens",
    "default_counter",
    "delegate",
    "extract_key_fields",
    "extractive_summarizer",
    "llm_summarizer",
    "truncate_summarizer",
]

__version__ = "0.1.0"
