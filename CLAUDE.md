# CLAUDE.md

Guidance for Claude Code and future contributors working in this repo.

## What this repo is

`agent-context-kit` is the third in a portfolio of patterns for production LLM agents. Positioning:

- `agent-eval-loop` — the offline improvement loop (simulate → evaluate → improve).
- `agent-tool-kit` — production-grade tool design (schemas, fat tools, registry, observability).
- **`agent-context-kit`** — what happens at runtime when the agent is actually running: how context enters the window, how it's compressed across turns, how complex tasks get isolated into sub-agents.

All three are meant to be used together. Integration points:

- `WorkingMemory` is a superset of `agent_eval_loop.agent.scratchpad.Scratchpad` — same interface, more features.
- `ManagedAgentRunner` wraps `agent_eval_loop.agent.runner.AgentRunner` and replaces its static context with a managed window.
- `SkillRegistry` mirrors `agent_tool_kit.registry.CapabilityRegistry` — same progressive-disclosure principle, applied to instructions instead of tool schemas.
- `ContextEventLog` mirrors `agent_tool_kit.observability.AuditLog` — same append-only JSONL contract, for context decisions instead of tool calls.

If you change an integration point, update the corresponding module in the sibling repo too. Drift between the three is the most common source of bugs.

## Layout

```
src/agent_context_kit/
  tokens.py            # token counting primitives
  observability.py     # ContextEventLog
  managed_runner.py    # adapter for agent-eval-loop's AgentRunner
  context/
    budget.py          # TokenBudget, ComponentCategory
    window.py          # ContextWindow (assembles system prompt + messages)
    history.py         # HistoryManager (turn-aware conversation history)
    memory.py          # WorkingMemory (Scratchpad superset)
  skills/
    skill.py           # Skill + SkillMetadata
    loader.py          # SkillLoader, SkillRegistry
    router.py          # RuleBasedRouter, EmbeddingRouter, LLMRouter
  compress/
    compactor.py       # ToolResultCompactor (full → summary → reference)
    summarizer.py      # schema-bound conversation summarization
    strategies.py      # pluggable strategies, CascadingThreshold
  isolate/
    delegate.py        # SubAgent, ExecutionResult, delegate()
    planner.py         # StaticPlanner, LLMPlanner, PlanStep

examples/              # three runnable examples
docs/                  # best-practices.md + reference docs
tests/                 # smoke tests (no API calls required)
```

## Design invariants

Read before changing architecture.

- **Every LLM-backed component has a deterministic fallback.** Routers fall back to "load everything" on parse errors. Summarizers fall back to extractive. This is non-negotiable — a flaky classifier must not block requests.

- **Components don't know about each other's internals.** The budget allocator doesn't own the window; the window doesn't reach into the history manager's private state (except `_turns` in the relevance-based strategy, which is documented). Keep it that way — the whole point is that each piece can be swapped.

- **Priority entries always survive.** Working memory's `priority=True` entries and history's `critical=True` turns are pinned against every compaction strategy. A bug here corrupts customer-stated facts.

- **Token counting is pluggable.** Every budget-aware component accepts a `TokenCounter`. Default is `HeuristicTokenCounter` (chars/4) — good enough for budget decisions, zero-dependency. tiktoken and real API counters are opt-in.

- **Examples must run offline.** Every example supports a `--dry-run` flag (or has a fake client baked in). If you add an example that requires the API to run at all, you've written docs, not an example.

## Running

```bash
pip install -e ".[dev]"
pytest tests/                              # smoke tests
python examples/multi_skill_support/run.py --dry-run
python examples/research_assistant/run.py --dry-run
python examples/long_conversation/run.py  # already no-API
```

## Common operations

- Add a new skill: create a `.md` file in the relevant skills directory with YAML front-matter. The loader picks it up automatically.
- Add a new compression strategy: implement `__call__(window, tokens_to_free) -> int`, then `window.set_compaction_strategy(category, my_strategy)`.
- Add a new router: subclass `SkillRouter`, implement `route(...)`. The base class handles history flattening and top-K / threshold clamping.
- Change budget allocation: edit `TokenBudget.for_window(...)` call in the relevant example, or construct your own.

## Pitfalls

- The Anthropic SDK's `client.messages.count_tokens()` makes a network call. Don't use it in the hot path — use `HeuristicTokenCounter` or `TiktokenCounter` instead.
- `LLMRouter` and `llm_summarizer` default to Haiku — they're cheap but they're still API calls. Cache or batch if you're calling per turn.
- `HistoryManager`'s `rollover()` is idempotent only when `summarizer` is deterministic. With an LLM summarizer, repeated rollovers produce slightly different summaries — fine for production but surprising in tests.
- Skills with unique names; the registry raises on duplicate registration. This is intentional — silent last-wins hides authoring bugs.
- The `examples/*/run.py` scripts are not imported by the tests. If you restructure the package and an example breaks, the tests will still pass. Run the examples' `--dry-run` as part of any review.

## When to extend vs swap

This package's primitives are small by design. When you need behavior that doesn't fit:

- **Extend** when your needs are an additive change (a new router, a new summarizer, a new compaction strategy). The pluggable points (`set_compaction_strategy`, `summarizer=`, subclassing `SkillRouter`) are there for this.
- **Swap** when the change is architectural (a different budget shape, a different window layout). Fork the class — don't bend the existing one out of shape.

Ask whichever question clarifies the choice: "would three people use this?" (extend) vs "would this confuse the next reader of the module?" (swap).
