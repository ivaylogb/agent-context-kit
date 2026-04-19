"""Token counting primitives.

Everything in this package that cares about context-window usage — budgets,
compaction, window assembly — needs a way to ask "how many tokens is this
string?". The counter is deliberately pluggable because different deployments
have different needs:

- **Development / offline** — a char/4 heuristic is accurate enough for
  budget decisions and has zero dependencies. Default.
- **tiktoken** — closer-to-real counts for OpenAI-style BPE tokenizers. Still
  not exactly Anthropic's tokenizer, but within a few percent on English text
  and much better than char/4. Enabled via the ``tiktoken`` extra.
- **Anthropic API** — the real thing, for when you need exact counts. Makes
  a network call per invocation, so unsuitable for the hot path.

The ``TokenCounter`` protocol makes all three interchangeable. Pass whichever
you need into ``TokenBudget``, ``ContextWindow``, etc. The heuristic is the
default because zero-dependency, zero-latency, and "approximately right" is
what budget enforcement actually needs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can count tokens in a string.

    Implementations should be cheap — the budget enforcer calls this
    on every component on every turn.
    """

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Char-based approximation. Default counter — no external dependencies.

    The 4-chars-per-token ratio is a rough average across English prose,
    code, and JSON. It's good enough for budget allocation: an error band
    of +/- 15% is smaller than the safety headroom you should be keeping
    anyway. Use a real tokenizer if you're trying to squeeze the last few
    percent of utilization out of the window.
    """

    CHARS_PER_TOKEN = 4.0

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.CHARS_PER_TOKEN))


class TiktokenCounter:
    """tiktoken-backed counter. Use when you have the extra installed.

    Defaults to ``cl100k_base`` which is the OpenAI tokenizer. It is not
    Anthropic's tokenizer, but empirically tracks within ~5% on English
    prose — good enough for budget decisions, and much better than char/4.

    Construct lazily: ``TiktokenCounter()`` only imports tiktoken at init
    time, so a deployment that doesn't install the extra can still import
    this module without blowing up.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "TiktokenCounter requires the 'tiktoken' extra. "
                "Install with: pip install 'agent-context-kit[tiktoken]'"
            ) from e
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


class CallableCounter:
    """Wrap an arbitrary ``(text) -> int`` callable as a ``TokenCounter``.

    Used mostly in tests and when you already have a counter function
    from some other system you want to re-use.
    """

    def __init__(self, fn: Callable[[str], int]) -> None:
        self._fn = fn

    def count(self, text: str) -> int:
        return self._fn(text)


def default_counter() -> TokenCounter:
    """Return the default counter: tiktoken if importable, else the heuristic."""
    try:
        return TiktokenCounter()
    except ImportError:
        return HeuristicTokenCounter()


def count_tokens(text: str, counter: TokenCounter | None = None) -> int:
    """Convenience wrapper. Use ``HeuristicTokenCounter`` when no counter is given."""
    c = counter or HeuristicTokenCounter()
    return c.count(text)


def count_messages(
    messages: Iterable[dict[str, Any]],
    counter: TokenCounter | None = None,
) -> int:
    """Count tokens across a sequence of Anthropic-API-style messages.

    Each message's ``content`` may be a string, a list of blocks, or an
    arbitrary dict (for tool results etc.). We serialize non-string content
    via ``str(...)`` — this is approximate by design, not exact.
    """
    c = counter or HeuristicTokenCounter()
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += c.count(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total += c.count(v)
                        else:
                            total += c.count(str(v))
                else:
                    total += c.count(str(block))
        elif content is not None:
            total += c.count(str(content))
    return total
