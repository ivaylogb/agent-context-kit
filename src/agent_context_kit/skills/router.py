"""Routers that decide which skills to load for a given turn.

Three implementations share a common ``SkillRouter`` interface so the rest
of the system doesn't need to care which one is wired in:

- ``RuleBasedRouter`` — keyword matching against each skill's
  ``trigger_keywords``. Fast, cheap, brittle. Good for the first version
  of a system or for high-confidence keyword domains (product IDs, opcodes).
- ``EmbeddingRouter`` — compare a message embedding to each skill's
  description embedding. Much more robust to paraphrases. Requires an
  embedding function (supplied by the caller — any callable that turns
  text into a vector works).
- ``LLMRouter`` — a cheap classifier model (Haiku by default) reads the
  skill menu and the message, returns a ranked list of names. Most
  flexible, small latency cost (~200ms at Haiku). Mirrors
  ``agent_tool_kit.classifier.ToolClassifier``.

All routers:
- Take the full conversation history, not just the current message —
  "compare fees" has no referent without context.
- Return a ranked list of ``RouterMatch`` objects (name + confidence +
  matched terms), not just names. Callers can threshold on confidence.
- Fall back to "select everything" on errors, never to "select nothing".

Typical usage::

    router = LLMRouter(registry, client=anthropic_client)
    matches = router.route(conversation_history=turns, threshold=0.3)
    loaded = skill_registry.bodies_for([m.name for m in matches])
    for name, body in loaded.items():
        window.load_skill(name, body)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import anthropic
from pydantic import BaseModel, Field

from agent_context_kit.context.history import Turn, TurnRole
from agent_context_kit.skills.loader import SkillRegistry

DEFAULT_ROUTER_MODEL = "claude-haiku-4-5-20251001"


class RouterMatch(BaseModel):
    """One skill match with a confidence score.

    Confidence is 0..1. The scale is router-specific — a 0.3 from the
    rule-based router means "one or two keywords hit"; a 0.3 from the
    embedding router means "cosine similarity 0.3" (which is pretty low).
    Calibrate thresholds per router type, not globally.
    """

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    # Free-form explanation — populated by the LLM router, useful for
    # debugging "why did this skill get selected?".
    reasoning: str | None = None


class SkillRouter(ABC):
    """Abstract base for skill routers.

    Implementations override ``route``; the base class provides common
    helpers: flatten history into a single string, clamp to top-K.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    @abstractmethod
    def route(
        self,
        message: str | None = None,
        *,
        conversation_history: Sequence[Turn] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
    ) -> list[RouterMatch]:
        """Return ranked matches. Must be overridden by subclasses.

        ``message`` and ``conversation_history`` are both optional but at
        least one must be provided. If both are given, the message is
        treated as the latest user turn (not duplicated into the history).
        """

    # ------------------------------------------------------------- helpers

    def _flatten(
        self,
        message: str | None,
        history: Sequence[Turn] | None,
    ) -> str:
        """Concatenate history + the latest message into a single string.

        Order: oldest → newest. Matches how the model would read them.
        Empty inputs raise — a router with no input has no signal.
        """
        parts: list[str] = []
        if history:
            for turn in history:
                label = "User" if turn.role == TurnRole.USER else "Assistant"
                parts.append(f"{label}: {turn.content}")
        if message:
            parts.append(f"User: {message}")
        if not parts:
            raise ValueError("route() requires at least a message or history")
        return "\n".join(parts)

    def _clamp(
        self,
        matches: list[RouterMatch],
        threshold: float,
        top_k: int | None,
    ) -> list[RouterMatch]:
        """Apply threshold and top-K in that order. Preserves ranking."""
        filtered = [m for m in matches if m.confidence >= threshold]
        if top_k is not None and top_k >= 0:
            filtered = filtered[:top_k]
        return filtered


# ---------------------------------------------------------------------------
# Rule-based router
# ---------------------------------------------------------------------------


class RuleBasedRouter(SkillRouter):
    """Keyword matcher against each skill's ``trigger_keywords``.

    Case-insensitive word-boundary matching. Confidence is ``matches /
    total_keywords`` — so a skill with 3 keywords, 2 of which hit, gets
    confidence 0.67. A skill with no keywords at all gets confidence 0.0
    regardless of content (it opts out of rule-based routing).

    Use this when your domains have clear keyword signals (product IDs,
    error codes, operation names). Don't use it as the primary router for
    natural-language flows — paraphrases will miss.
    """

    def route(
        self,
        message: str | None = None,
        *,
        conversation_history: Sequence[Turn] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
    ) -> list[RouterMatch]:
        text = self._flatten(message, conversation_history).lower()
        matches: list[RouterMatch] = []
        for skill in self.registry:
            keywords = skill.metadata.trigger_keywords
            if not keywords:
                continue
            hits: list[str] = []
            for kw in keywords:
                if not kw:
                    continue
                # Word-boundary match, case insensitive. Avoids "bill" matching
                # "billion" or "billing" matching "kiln-building".
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, text):
                    hits.append(kw)
            if hits:
                confidence = len(hits) / len(keywords)
                matches.append(
                    RouterMatch(
                        name=skill.name,
                        confidence=confidence,
                        matched_terms=hits,
                    )
                )
        matches.sort(key=lambda m: (-m.confidence, m.name))
        return self._clamp(matches, threshold, top_k)


# ---------------------------------------------------------------------------
# Embedding-based router
# ---------------------------------------------------------------------------


EmbedFn = Callable[[str], Sequence[float]]


class EmbeddingRouter(SkillRouter):
    """Cosine-similarity matcher against skill description embeddings.

    The ``embed_fn`` is caller-supplied — any function that turns text
    into a fixed-length vector works. We deliberately don't bundle a
    specific embedding backend: OpenAI, Voyage, local sentence-transformers,
    whatever you already have in the rest of your stack.

    Skill embeddings are computed eagerly at router construction and
    cached. If you register new skills after construction, call
    ``rebuild_index`` to pick them up.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        embed_fn: EmbedFn,
    ) -> None:
        super().__init__(registry)
        self._embed = embed_fn
        self._skill_vectors: dict[str, Sequence[float]] = {}
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Re-embed every skill's description. Call after registry changes.

        We embed ``name + description + tags`` concatenated — the description
        alone is too sparse a signal. Adding tags and the name anchors the
        embedding in the domain vocabulary.
        """
        vectors: dict[str, Sequence[float]] = {}
        for skill in self.registry:
            meta = skill.metadata
            text = f"{meta.name}. {meta.description}"
            if meta.tags:
                text += " tags: " + ", ".join(meta.tags)
            vectors[skill.name] = self._embed(text)
        self._skill_vectors = vectors

    def route(
        self,
        message: str | None = None,
        *,
        conversation_history: Sequence[Turn] | None = None,
        threshold: float = 0.3,
        top_k: int | None = 3,
    ) -> list[RouterMatch]:
        # Default thresholds are tuned for normalized cosine similarity:
        # >0.3 is typically meaningful, >0.5 is strong. Tune for your
        # embedding model — they're not universal.
        text = self._flatten(message, conversation_history)
        query_vec = self._embed(text)
        matches: list[RouterMatch] = []
        for name, skill_vec in self._skill_vectors.items():
            sim = _cosine(query_vec, skill_vec)
            if sim <= 0.0:
                continue
            # Clamp to [0, 1] — cosine can be negative for orthogonal-ish
            # embeddings, and we're using this as a confidence score.
            confidence = max(0.0, min(1.0, sim))
            matches.append(RouterMatch(name=name, confidence=confidence))
        matches.sort(key=lambda m: (-m.confidence, m.name))
        return self._clamp(matches, threshold, top_k)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors. Zero on length mismatch or zero norms."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    import math
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# LLM-based router
# ---------------------------------------------------------------------------


ROUTER_SYSTEM_PROMPT = (
    "You are a skill routing classifier. Given a conversation and a menu of "
    "skills, select the subset of skills whose context should be loaded to "
    "respond to the latest user message.\n\n"
    "Rules:\n"
    "- Consider the full conversation, not just the last message. Paraphrases, "
    "follow-ups, and pronouns may refer to prior context.\n"
    "- Be selective — excluding irrelevant skills reduces noise for the executor.\n"
    "- If multiple domains apply, return all of them with confidence scores.\n"
    "- Output strict JSON only. No prose, no markdown fencing.\n\n"
    "Output format (each score is 0.0-1.0):\n"
    '{{"matches": [{{"name": "...", "confidence": 0.0, "reasoning": "..."}}]}}\n\n'
    "Available skills:\n{menu}\n"
)


class LLMRouter(SkillRouter):
    """Classifier-model router. Cheap model reads the menu, ranks skills.

    Mirrors ``agent_tool_kit.classifier.ToolClassifier``: the classifier
    sees just enough (skill menu + conversation), a cheap model is enough
    for the decision, the executor model then runs with a clean context.

    Degrades gracefully: JSON parse errors, unknown skill names, and empty
    selections all fall back to returning all skills with a low confidence
    score — better to over-load than to block the request.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_ROUTER_MODEL,
        max_tokens: int = 500,
    ) -> None:
        super().__init__(registry)
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def route(
        self,
        message: str | None = None,
        *,
        conversation_history: Sequence[Turn] | None = None,
        threshold: float = 0.3,
        top_k: int | None = 3,
    ) -> list[RouterMatch]:
        menu = self.registry.menu()
        if not menu:
            return []
        text = self._flatten(message, conversation_history)
        menu_lines = []
        for m in menu:
            tags = ", ".join(m.get("tags", [])) or "(no tags)"
            line = f"- {m['name']}: {m['description']} [tags: {tags}]"
            if m.get("when_not_to_use"):
                line += f"\n    NOT FOR: {m['when_not_to_use']}"
            menu_lines.append(line)
        system = ROUTER_SYSTEM_PROMPT.format(menu="\n".join(menu_lines))

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
        except Exception:  # noqa: BLE001 — fall back on any client failure
            return self._fallback(threshold, top_k)

        text_blocks = [
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ]
        raw = "".join(text_blocks)
        matches = self._parse(raw)
        if not matches:
            return self._fallback(threshold, top_k)
        matches.sort(key=lambda m: (-m.confidence, m.name))
        return self._clamp(matches, threshold, top_k)

    def _parse(self, text: str) -> list[RouterMatch]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data: Any = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        raw_matches = data.get("matches")
        if not isinstance(raw_matches, list):
            return []
        out: list[RouterMatch] = []
        for m in raw_matches:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not isinstance(name, str) or name not in self.registry:
                continue
            try:
                confidence = float(m.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            confidence = max(0.0, min(1.0, confidence))
            reasoning = m.get("reasoning")
            out.append(
                RouterMatch(
                    name=name,
                    confidence=confidence,
                    reasoning=reasoning if isinstance(reasoning, str) else None,
                )
            )
        return out

    def _fallback(
        self,
        threshold: float,
        top_k: int | None,
    ) -> list[RouterMatch]:
        """Return every skill with low confidence — load-everything mode.

        The confidence is set slightly above the default 0.3 threshold so
        the same downstream filtering still returns results, but the
        ``reasoning`` marks it as a fallback for observability.
        """
        fallback_conf = max(0.35, threshold)
        matches = [
            RouterMatch(
                name=skill.name,
                confidence=fallback_conf,
                reasoning="fallback: router failed, loading catalogue",
            )
            for skill in self.registry
        ]
        return self._clamp(matches, threshold, top_k)
