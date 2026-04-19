"""Schema-based conversation summarization.

When tool-result compaction and working-memory rollup hit diminishing
returns (the window is still too big), the next tool is a full-
conversation summary. The trick is making the summary useful downstream:

- **Free-form prose** works but isn't machine-reusable. The model produces
  a different shape each time. Eval gets flaky.
- **Schema-bound JSON** produces a uniform shape every time. The agent can
  reason about specific fields ("check current_state", "check unresolved")
  instead of re-reading prose. This is what we do.

The default schema matches the reference doc — ``customer_intent``,
``key_facts``, ``actions_taken``, ``current_state``, ``unresolved`` — but
callers can pass their own. The only requirement is that the schema is a
dict with string keys and primitive / list / dict values.

A summary is produced by a callable: either the LLM-backed
``llm_summarizer`` (recommended for production) or one of the helpers
(``truncate_summarizer``, ``extractive_summarizer``) for offline /
deterministic scenarios.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from agent_context_kit.context.history import Turn, TurnRole

# A conversation summarizer transforms a list of turns into a summary
# string (ready for ``HistoryManager.summarizer``) or a schema'd dict
# (for programmatic use).
TurnSummaryFn = Callable[[Sequence[Turn]], str]


# Default summary schema keys — matches the reference doc.
DEFAULT_SCHEMA: dict[str, str] = {
    "customer_intent": "What the user is ultimately trying to accomplish.",
    "key_facts": "Bullet list of concrete facts established (IDs, constraints, preferences).",
    "actions_taken": "Bullet list of what the agent has already done or verified.",
    "current_state": "The current step in the workflow.",
    "unresolved": "Open questions, pending confirmations, or stated asks not yet answered.",
}


class ConversationSummary(BaseModel):
    """Schema-bound conversation summary.

    Fields map to the default schema keys. Additional custom keys land in
    ``extras`` — we don't lose them but they aren't typed. If you've got a
    stable custom schema, define your own BaseModel and use it directly
    with ``build_schema_summarizer(your_model)``.
    """

    customer_intent: str = ""
    key_facts: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    current_state: str = ""
    unresolved: list[str] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Render as prose suitable for HistoryManager's summary block."""
        sections: list[str] = []
        if self.customer_intent:
            sections.append(f"Intent: {self.customer_intent}")
        if self.key_facts:
            bullets = "\n".join(f"- {f}" for f in self.key_facts)
            sections.append(f"Key facts:\n{bullets}")
        if self.actions_taken:
            bullets = "\n".join(f"- {a}" for a in self.actions_taken)
            sections.append(f"Actions taken:\n{bullets}")
        if self.current_state:
            sections.append(f"Current state: {self.current_state}")
        if self.unresolved:
            bullets = "\n".join(f"- {u}" for u in self.unresolved)
            sections.append(f"Unresolved:\n{bullets}")
        if self.extras:
            for k, v in self.extras.items():
                sections.append(f"{k}: {v}")
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Deterministic summarizers (no LLM required)
# ---------------------------------------------------------------------------


def truncate_summarizer(char_limit: int) -> Callable[[str, str], str]:
    """Build a tool-result summarizer that just truncates.

    Used as the default for ``ToolResultCompactor``. Signature matches
    ``SummaryFn`` from ``compactor.py``: ``(tool_name, content) -> str``.
    Kept here alongside the other summarizers so callers have one import
    for all the stock options.
    """

    def summarize(tool_name: str, content: str) -> str:
        if len(content) <= char_limit:
            return content
        return content[: char_limit - 3].rstrip() + "..."

    return summarize


def turns_as_plain_text(turns: Sequence[Turn]) -> str:
    """Flatten a sequence of turns into a plain transcript.

    Used as input to LLM summarizers; also a fine standalone "summarizer"
    for tests and baselines (it just returns the transcript itself —
    useful to compare against compressed versions).
    """
    lines: list[str] = []
    for turn in turns:
        label = "User" if turn.role == TurnRole.USER else "Agent"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def extractive_summarizer(max_lines: int = 5) -> TurnSummaryFn:
    """Keep the first and last few lines of the conversation verbatim.

    Heuristic: the opening usually has the customer's original ask, the
    tail usually has the most recent commitments. Mid-conversation
    clarifications are what we usually compress. Not a great summary — use
    as a baseline, not a production default.
    """

    def summarize(turns: Sequence[Turn]) -> str:
        if not turns:
            return ""
        if len(turns) <= max_lines:
            return turns_as_plain_text(turns)
        half = max_lines // 2
        head = turns[:half]
        tail = turns[-(max_lines - half):]
        snippet = turns_as_plain_text(list(head) + list(tail))
        return f"{snippet}\n[{len(turns) - max_lines} intermediate turns omitted]"

    return summarize


# ---------------------------------------------------------------------------
# LLM-backed summarizer
# ---------------------------------------------------------------------------


SUMMARIZER_SYSTEM_PROMPT = (
    "You are a conversation compressor. Given a transcript, produce a "
    "structured summary using the schema below. Keep every concrete fact, "
    "commitment, and open question — drop pleasantries, filler, and any "
    "content a successor agent wouldn't need to continue the conversation.\n\n"
    "Schema:\n{schema}\n\n"
    "Output strict JSON matching the schema. No prose, no markdown."
)


def llm_summarizer(
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 800,
    schema: dict[str, str] | None = None,
) -> TurnSummaryFn:
    """Produce a ``TurnSummaryFn`` backed by a cheap LLM.

    On parse failure, falls back to ``extractive_summarizer`` so the
    HistoryManager never receives an empty summary and silently loses
    content.
    """
    schema_effective = schema or DEFAULT_SCHEMA
    schema_text = "\n".join(f"- {k}: {v}" for k, v in schema_effective.items())
    system = SUMMARIZER_SYSTEM_PROMPT.format(schema=schema_text)
    fallback = extractive_summarizer(max_lines=6)

    def summarize(turns: Sequence[Turn]) -> str:
        if not turns:
            return ""
        transcript = turns_as_plain_text(turns)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": transcript}],
            )
            text_blocks = [
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            raw = "".join(text_blocks).strip()
        except Exception:  # noqa: BLE001
            return fallback(turns)
        parsed = _parse_summary(raw, schema_effective)
        if parsed is None:
            return fallback(turns)
        return parsed.render()

    return summarize


def _parse_summary(
    text: str,
    schema: dict[str, str],
) -> ConversationSummary | None:
    """Best-effort JSON extraction. Returns ``None`` on unrecoverable parse errors."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data: Any = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    known = {"customer_intent", "key_facts", "actions_taken", "current_state",
             "unresolved"}
    kwargs: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for k, v in data.items():
        if k in known:
            kwargs[k] = v
        else:
            extras[k] = v
    if extras:
        kwargs["extras"] = extras
    try:
        return ConversationSummary(**kwargs)
    except Exception:  # noqa: BLE001 — validation hiccup falls back.
        return None


def build_schema_summarizer(
    model_class: type[BaseModel],
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 800,
    fallback: TurnSummaryFn | None = None,
) -> TurnSummaryFn:
    """Summarizer for a caller-defined schema (arbitrary Pydantic model).

    The model's JSON schema is embedded in the system prompt so the LLM
    knows the exact field names and types to produce. On parse failure
    the ``fallback`` is used (default: extractive).

    Use this when the default schema doesn't fit — e.g., a research
    assistant wants ``["findings", "hypotheses", "open_questions"]``,
    not a customer-intent field.
    """
    fallback_fn = fallback or extractive_summarizer(max_lines=6)
    schema_json = json.dumps(model_class.model_json_schema(), indent=2)
    system = (
        "You are a conversation compressor. Produce a structured summary "
        "that populates the schema below. Keep concrete facts and open "
        "questions; drop filler. Output strict JSON matching the schema.\n\n"
        f"Schema:\n{schema_json}"
    )

    def summarize(turns: Sequence[Turn]) -> str:
        if not turns:
            return ""
        transcript = turns_as_plain_text(turns)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": transcript}],
            )
            raw = "".join(
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            ).strip()
        except Exception:  # noqa: BLE001
            return fallback_fn(turns)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            parsed = model_class.model_validate(data)
        except Exception:  # noqa: BLE001
            return fallback_fn(turns)
        return json.dumps(parsed.model_dump(), indent=2, default=str)

    return summarize
