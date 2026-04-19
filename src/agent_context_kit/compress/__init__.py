"""Compression strategies for keeping the context window lean."""

from agent_context_kit.compress.compactor import (
    SummaryFn,
    ToolResultCompactor,
    extract_key_fields,
)
from agent_context_kit.compress.compactor import (
    llm_summarizer as llm_tool_result_summarizer,
)
from agent_context_kit.compress.strategies import (
    DEFAULT_CASCADE,
    CascadingThreshold,
    ProgressiveCompactionStrategy,
    RelevanceBasedHistoryStrategy,
    SummaryHistoryStrategy,
    TurnScoreFn,
    apply_cascade,
)
from agent_context_kit.compress.summarizer import (
    DEFAULT_SCHEMA,
    ConversationSummary,
    TurnSummaryFn,
    build_schema_summarizer,
    extractive_summarizer,
    llm_summarizer,
    truncate_summarizer,
    turns_as_plain_text,
)

__all__ = [
    "CascadingThreshold",
    "ConversationSummary",
    "DEFAULT_CASCADE",
    "DEFAULT_SCHEMA",
    "ProgressiveCompactionStrategy",
    "RelevanceBasedHistoryStrategy",
    "SummaryFn",
    "SummaryHistoryStrategy",
    "ToolResultCompactor",
    "TurnScoreFn",
    "TurnSummaryFn",
    "apply_cascade",
    "build_schema_summarizer",
    "extract_key_fields",
    "extractive_summarizer",
    "llm_summarizer",
    "llm_tool_result_summarizer",
    "truncate_summarizer",
    "turns_as_plain_text",
]
