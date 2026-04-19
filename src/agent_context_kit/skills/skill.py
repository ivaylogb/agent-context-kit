"""Skill definition — Level-1 metadata + Level-2 body.

A skill is the context-loading counterpart to a tool: a bundle of
instructions, routines, and examples that teach the agent how to handle
one specific domain (billing, returns, password reset, ...). Skills
follow the same progressive-disclosure principle as tools:

- **Level 1 (metadata, always loaded)** — name, one-line description,
  tags, optional trigger keywords. ~50-100 tokens per skill. The router
  sees the full catalogue cheaply.
- **Level 2 (body, loaded on demand)** — full instructions, routines,
  examples. Loaded only when the router selects this skill for a turn.
- **Level 3 (reference data, loaded as the skill needs it)** — not part
  of the skill itself; the skill's body references files, knowledge-base
  IDs, or tool calls that fetch deeper data only when required.

Skills live on disk as Markdown files with YAML front-matter. Loading
them is the ``SkillLoader``'s job; the ``Skill`` class is just the in-
memory representation.

File format::

    ---
    name: billing
    description: Invoice and payment questions — plans, refunds, overcharges.
    tags: [billing, payments]
    trigger_keywords: [invoice, bill, refund, charge, payment]
    when_not_to_use: |
      Don't use for shipping/delivery, returns, or account changes —
      those have their own skills.
    ---

    # Billing

    You handle customer billing questions. Always pull the invoice
    before answering. ...

The ``when_not_to_use`` field mirrors the tool pattern: models reach
for the most semantically similar skill by default, and a scoped
negative constraint is cheaper than rewriting the positive one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """Level-1 metadata. Always loaded into the context window.

    Budget for the whole skill catalogue's metadata: ``~50-100 tokens per
    skill × number of skills``. With 20 skills that's 1-2K tokens for the
    catalogue, which the model can skim in one pass. The router then loads
    only the one or two bodies it actually needs.
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    # Keywords used by the ``RuleBasedRouter``. Optional; LLM/embedding
    # routers ignore these.
    trigger_keywords: list[str] = Field(default_factory=list)
    # Same contract as the tool-kit's ``when_not_to_use``: a negative
    # constraint the router (and the model itself) can use to avoid
    # misrouting. Free-form text, not parsed.
    when_not_to_use: str | None = None
    # Optional explicit path to the body file. Normally derived by the
    # loader from the skill's on-disk location; this field is for the
    # in-memory construction path.
    body_path: str | None = None

    def menu_entry(self) -> dict[str, Any]:
        """Compact representation the router renders into its prompt.

        Includes the negative constraint so router prompts can cite it
        when deciding. Deliberately excludes ``body_path`` — the router
        should never see filesystem paths.
        """
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
        }
        if self.when_not_to_use:
            out["when_not_to_use"] = self.when_not_to_use
        return out


class Skill(BaseModel):
    """A skill: metadata + body content.

    Construct directly for in-memory skills, or via ``SkillLoader`` for
    filesystem-backed skills. The body is typically loaded lazily — it
    lives on disk and is only read when the router selects this skill.
    """

    metadata: SkillMetadata
    body: str = ""

    # Allow empty body to survive validation — it's legitimate for a
    # freshly-discovered skill whose body hasn't been loaded yet.
    model_config = {"frozen": False}

    @property
    def name(self) -> str:
        return self.metadata.name

    def token_estimate(self, counter: Any = None) -> int:
        """Estimate the body's token count. ``counter`` must expose ``count``."""
        if counter is None:
            # Fallback: char/4 heuristic so callers don't have to import
            # the token module just for a rough estimate.
            return max(1, len(self.body) // 4)
        return counter.count(self.body)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """Construct from a plain dict (e.g., decoded YAML)."""
        body = data.pop("body", "")
        return cls(metadata=SkillMetadata(**data), body=body)


def discover_skill_paths(root: str | Path, pattern: str = "*.md") -> list[Path]:
    """List skill file paths under a directory. Sorted for determinism.

    Deterministic ordering matters for routers that rank by position as a
    tiebreaker — the same skill set should produce the same ranking across
    runs.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    if root_path.is_file():
        return [root_path]
    return sorted(root_path.glob(pattern))
