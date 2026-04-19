"""Dynamic skill loading with progressive disclosure.

A skill is a bundle of instructions, routines, and examples that teach the
agent how to handle one specific domain. This submodule provides:

- ``Skill`` / ``SkillMetadata`` — the in-memory shape of a skill.
- ``SkillLoader`` — filesystem loader for Markdown+YAML-frontmatter skill files.
- ``SkillRegistry`` — a catalogue with two-level disclosure (menu vs body).
- ``RuleBasedRouter`` / ``EmbeddingRouter`` / ``LLMRouter`` — routers that
  pick the relevant subset for each turn.

Parallel in design to ``agent_tool_kit``'s ``CapabilityRegistry`` +
``ToolClassifier`` — same progressive-disclosure principle, applied to
instructions/routines instead of tool schemas.
"""

from agent_context_kit.skills.loader import (
    SkillLoader,
    SkillParseError,
    SkillRegistry,
    parse_skill_file,
)
from agent_context_kit.skills.router import (
    DEFAULT_ROUTER_MODEL,
    EmbeddingRouter,
    EmbedFn,
    LLMRouter,
    RouterMatch,
    RuleBasedRouter,
    SkillRouter,
)
from agent_context_kit.skills.skill import (
    Skill,
    SkillMetadata,
    discover_skill_paths,
)

__all__ = [
    "DEFAULT_ROUTER_MODEL",
    "EmbedFn",
    "EmbeddingRouter",
    "LLMRouter",
    "RouterMatch",
    "RuleBasedRouter",
    "Skill",
    "SkillLoader",
    "SkillMetadata",
    "SkillParseError",
    "SkillRegistry",
    "SkillRouter",
    "discover_skill_paths",
    "parse_skill_file",
]
