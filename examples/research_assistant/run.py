"""Research assistant: planner → executor with context isolation.

The user asks a multi-part research question. The planner breaks it into
steps; each step runs in its own sub-agent with a clean context and only
the tools it needs; the planner synthesizes the results.

What this demonstrates:

- **Context isolation.** Each sub-agent sees only its step's context, not
  the others' internal state. Side effects stay contained.
- **Structured results.** Sub-agents return typed ``ExecutionResult``
  objects, not prose blobs. The synthesizer reads them reliably.
- **Dependencies.** Step 2 depends on step 1's output via the
  ``{step_NAME}`` placeholder. The planner resolves the dependency before
  dispatching step 2.

Run with ``--dry-run`` to use a scripted fake client instead of the API.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from agent_context_kit import (
    ContextEventLog,
    PlanStep,
    StaticPlanner,
)

# ---------------------------------------------------------------------------
# Stub tools for the demo. Real deployments would wire in
# agent_tool_kit.Tool / CapabilityRegistry instances and use their
# ``tool_schema`` / ``handlers`` accessors.
# ---------------------------------------------------------------------------


SEARCH_DOCS_SCHEMA: dict[str, Any] = {
    "name": "search_docs",
    "description": (
        "Search internal documentation for a given topic. Returns a "
        "list of document IDs with title and a one-paragraph snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

FETCH_DOC_SCHEMA: dict[str, Any] = {
    "name": "fetch_doc",
    "description": "Fetch full text of a document by ID.",
    "input_schema": {
        "type": "object",
        "properties": {"doc_id": {"type": "string"}},
        "required": ["doc_id"],
    },
}

# In-memory "knowledge base" keyed by doc id.
DOCS = {
    "DOC-001": {
        "title": "Context window management in large language models",
        "body": (
            "Context windows are finite. Effective agents use three "
            "strategies: progressive disclosure (load only relevant "
            "instructions), token budgeting (per-component allocations), "
            "and compression (tool results decay full→summary→reference). "
            "Production deployments typically reserve 30-50% headroom."
        ),
    },
    "DOC-002": {
        "title": "Sub-agent isolation patterns",
        "body": (
            "Sub-agents spawned with clean contexts outperform in-context "
            "tool chains on tasks requiring >10 steps. The planner pattern "
            "(high-level plan, per-step executor) is a common production shape."
        ),
    },
    "DOC-003": {
        "title": "Tool result compaction tradeoffs",
        "body": (
            "Aggressive compaction frees tokens but can lose referential "
            "accuracy. A three-tier aging scheme (full → summary → reference) "
            "tends to balance freshness and recall better than binary keep/drop."
        ),
    },
}


def search_docs(query: str) -> list[dict[str, Any]]:
    q = (query or "").lower()
    hits = []
    for doc_id, doc in DOCS.items():
        if any(word in (doc["title"] + " " + doc["body"]).lower() for word in q.split()):
            hits.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "snippet": doc["body"][:120],
            })
    return hits


def fetch_doc(doc_id: str) -> dict[str, Any]:
    doc = DOCS.get(doc_id)
    if doc is None:
        return {"error": f"Unknown doc {doc_id}"}
    return {"doc_id": doc_id, "title": doc["title"], "body": doc["body"]}


# ---------------------------------------------------------------------------
# Dry-run fake client. Produces deterministic scripted responses so the
# example runs end-to-end offline.
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, btype: str, **kwargs: Any) -> None:
        self.type = btype
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, blocks: list[_Block]) -> None:
        self.content = blocks


class _FakeMessages:
    def __init__(self, script: list[list[_Block]]) -> None:
        self._script = list(script)

    def create(self, **kwargs: Any) -> _Resp:
        if not self._script:
            return _Resp([_Block("text", text="done")])
        return _Resp(self._script.pop(0))


class FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages(
            [
                # ---- Step 1: 'search' sub-agent
                [_Block("tool_use", id="tu1", name="search_docs", input={"query": "context window"})],
                [_Block("text", text="found: DOC-001, DOC-003")],
                # ---- Step 2: 'extract' sub-agent uses search output ----
                [_Block("tool_use", id="tu2", name="fetch_doc", input={"doc_id": "DOC-001"})],
                [_Block("text", text="DOC-001 discusses progressive disclosure, budgeting, and compression")],
                # ---- Synthesis
                [_Block("text", text="context engineering has three pillars: progressive disclosure, budgeting, compression")],
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use scripted fake client instead of Anthropic API.",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Re-run with --dry-run for offline demo.")
        return 1

    event_log = ContextEventLog()
    client = FakeClient() if args.dry_run else None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()

    planner = StaticPlanner(
        client=client,
        event_log=event_log,
        tool_schemas={
            "search_docs": SEARCH_DOCS_SCHEMA,
            "fetch_doc": FETCH_DOC_SCHEMA,
        },
        tool_handlers={
            "search_docs": search_docs,
            "fetch_doc": fetch_doc,
        },
    )

    task = (
        "Summarize what the internal docs say about managing the "
        "context window for LLM agents. Cite the doc IDs."
    )

    plan = [
        PlanStep(
            name="search",
            task=(
                "Search the docs for 'context window' and return the matching "
                "doc IDs and snippets."
            ),
            tools=["search_docs"],
            max_turns=3,
        ),
        PlanStep(
            name="extract",
            task=(
                "Given these search results: {step_search}\n\n"
                "Fetch the first matching doc (DOC-001) and summarize what it "
                "says in one sentence."
            ),
            tools=["fetch_doc"],
            depends_on=["search"],
            max_turns=3,
        ),
    ]

    result = planner.execute(task, plan)

    print(f"plan ok: {result.ok}")
    print(f"step outputs:")
    for name, output in result.step_outputs.items():
        print(f"  {name}: {str(output)[:140]}")
    print(f"\nsynthesis:\n  {result.synthesis[:500]}")
    print(f"\nevent log:")
    for event_type, count in event_log.summary().items():
        print(f"  {event_type}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
