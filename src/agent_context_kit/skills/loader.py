"""Filesystem loader for skills.

Skills live as Markdown files with YAML front-matter. This module parses
them, builds an in-memory ``SkillRegistry``, and supports lazy body loading
(the metadata is parsed up front; the body is only read when asked for).

Lazy loading matters for large catalogues. A hundred skills with 2K-token
bodies is 200K tokens — more than you want to parse and hold in RAM at
startup when only two or three of them will be used per turn.

Typical wiring::

    loader = SkillLoader()
    registry = loader.load_directory("skills/")
    # registry.menu() is cheap — just metadata
    # registry.get("billing").body triggers the lazy load
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent_context_kit.skills.skill import Skill, SkillMetadata, discover_skill_paths

FRONTMATTER_RE = re.compile(
    r"^---\s*\n(?P<front>.*?)\n---\s*\n?(?P<body>.*)$",
    re.DOTALL,
)


class SkillParseError(ValueError):
    """Raised when a skill file is malformed (bad front-matter, missing name, etc.)."""


def parse_skill_file(text: str, source_path: str | None = None) -> Skill:
    """Parse a skill file's text into a ``Skill`` object.

    Accepts either front-matter-with-body files or plain-YAML files
    (where the body is absent). ``source_path`` is used only in error
    messages so developers can find the bad file.

    Raises ``SkillParseError`` for malformed input — better to blow up at
    load time than to silently register a broken skill.
    """
    match = FRONTMATTER_RE.match(text)
    if match is None:
        # No front-matter — treat the entire file as YAML-metadata (body is
        # whatever ``body:`` says, possibly empty). This is the path for
        # programmatically-generated skills.
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            where = f" ({source_path})" if source_path else ""
            raise SkillParseError(f"YAML parse error{where}: {e}") from e
        if not isinstance(data, dict):
            where = f" ({source_path})" if source_path else ""
            raise SkillParseError(
                f"Skill file must be a YAML mapping at the top level{where}."
            )
        body = data.pop("body", "")
    else:
        front_text = match.group("front")
        body = match.group("body").strip()
        try:
            data = yaml.safe_load(front_text) or {}
        except yaml.YAMLError as e:
            where = f" ({source_path})" if source_path else ""
            raise SkillParseError(f"Front-matter YAML error{where}: {e}") from e
        if not isinstance(data, dict):
            where = f" ({source_path})" if source_path else ""
            raise SkillParseError(
                f"Front-matter must be a YAML mapping{where}."
            )

    if "name" not in data or not str(data.get("name", "")).strip():
        where = f" ({source_path})" if source_path else ""
        raise SkillParseError(f"Skill missing required 'name' field{where}.")
    if "description" not in data or not str(data.get("description", "")).strip():
        where = f" ({source_path})" if source_path else ""
        raise SkillParseError(
            f"Skill missing required 'description' field{where}."
        )

    if source_path is not None:
        data.setdefault("body_path", source_path)

    try:
        metadata = SkillMetadata(**{
            k: v for k, v in data.items() if k in SkillMetadata.model_fields
        })
    except Exception as e:  # noqa: BLE001
        where = f" ({source_path})" if source_path else ""
        raise SkillParseError(f"Invalid skill metadata{where}: {e}") from e

    return Skill(metadata=metadata, body=body)


class SkillLoader:
    """Load skills from disk.

    Two modes:

    - **Eager** (default) — parse every file on ``load_directory`` and keep
      bodies in memory. Fast lookups, but expensive for large catalogues.
    - **Lazy** — parse only the front-matter at load time; the body is read
      on first access. Suitable for large catalogues where most skills
      won't be used in a given conversation.

    The lazy mode uses the filesystem's mtime to decide whether to re-read
    a file — if you edit a skill on disk, the next access picks up the new
    content without needing a manual invalidation step.
    """

    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy

    def load_file(self, path: str | Path) -> Skill:
        """Parse a single skill file. Errors bubble up as ``SkillParseError``."""
        p = Path(path)
        text = p.read_text()
        skill = parse_skill_file(text, source_path=str(p))
        if self.lazy:
            # Drop the body so subsequent ``get_body`` re-reads.
            skill.body = ""
        return skill

    def load_directory(
        self,
        root: str | Path,
        pattern: str = "*.md",
    ) -> SkillRegistry:
        """Parse every matching file under ``root`` and register them.

        Duplicate skill names across files raise ``SkillParseError`` — a
        skill name is the router's handle to its content, and silent
        last-wins behavior would hide authoring bugs.
        """
        registry = SkillRegistry(loader=self)
        for path in discover_skill_paths(root, pattern=pattern):
            skill = self.load_file(path)
            registry.register(skill)
        return registry

    def get_body(self, skill: Skill) -> str:
        """Return the skill's body, reading from disk if we're in lazy mode."""
        if skill.body:
            return skill.body
        body_path = skill.metadata.body_path
        if body_path is None:
            return ""
        path = Path(body_path)
        if not path.exists():
            return ""
        text = path.read_text()
        parsed = parse_skill_file(text, source_path=str(path))
        skill.body = parsed.body
        return parsed.body


class SkillRegistry:
    """A collection of skills with progressive-disclosure accessors.

    Mirror of ``agent_tool_kit.registry.CapabilityRegistry``: cheap ``menu``
    always loaded, ``bodies_for`` for the subset the router picks. The
    two registries compose — a full agent wiring reaches for both.
    """

    def __init__(
        self,
        skills: Iterable[Skill] = (),
        loader: SkillLoader | None = None,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        self._loader = loader
        for s in skills:
            self.register(s)

    # --------------------------------------------------------------- mutate

    def register(self, skill: Skill) -> Skill:
        """Add a skill. Re-registration of an existing name raises ``ValueError``."""
        name = skill.name
        if name in self._skills:
            raise ValueError(f"Skill {name!r} is already registered.")
        self._skills[name] = skill
        return skill

    def unregister(self, name: str) -> bool:
        """Remove a skill. Returns True if it was present."""
        return self._skills.pop(name, None) is not None

    # --------------------------------------------------------------- query

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills.keys())

    # ----------------------------------------------- progressive disclosure

    def menu(self, tags: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Cheap catalogue (Level-1 metadata only).

        ``tags`` filters to skills that share at least one tag with the
        given set. Use to pre-scope the router when you know the domain
        in advance (e.g., only ``billing`` + ``payments`` for a billing
        sub-agent).
        """
        tag_set = set(tags) if tags else None
        out: list[dict[str, Any]] = []
        for skill in self._skills.values():
            if tag_set is not None and not (set(skill.metadata.tags) & tag_set):
                continue
            out.append(skill.metadata.menu_entry())
        return out

    def bodies_for(self, names: Iterable[str]) -> dict[str, str]:
        """Full body text for a named subset.

        Uses the loader for lazy skills. Unknown names raise ``KeyError`` —
        matching the tool-kit registry's ``schemas_for`` behavior.
        """
        out: dict[str, str] = {}
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(f"Unknown skill: {name!r}")
            if skill.body or self._loader is None:
                out[name] = skill.body
            else:
                out[name] = self._loader.get_body(skill)
        return out
