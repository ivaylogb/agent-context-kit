# agent-context-kit

Runtime context engineering for LLM agents.

> Agents fail at scale not because the model is bad, but because the context window fills with the wrong things. Stale tool results crowd out instructions. Conversation history grows until early context is forgotten. Relevant knowledge isn't loaded because the agent doesn't know it exists yet.

This package provides the runtime machinery to keep the context window lean and focused at every turn: token budgets across component types, dynamic skill loading with progressive disclosure, tool-result compaction, schema-based conversation summarization, and sub-agent isolation.

## Where this fits

Part of [agent-engineering](https://github.com/ivaylogb/agent-engineering) — a four-layer system for production-grade LLM agents.

This kit is **Layer 3: Context runtime**. It answers: how do we keep the model focused?

The other three layers:

- [agent-eval-loop](https://github.com/ivaylogb/agent-eval-loop) — evaluation. How do we know the agent got better?
- [agent-tool-kit](https://github.com/ivaylogb/agent-tool-kit) — tool contracts. How do we make agent actions reliable?
- [agent-skill-kit](https://github.com/ivaylogb/agent-skill-kit) — development workflows. How do we make agent-building repeatable?

---

## The framing

The LLM is the CPU. The context window is RAM. Context engineering is the operating system that manages what goes in.

This package is organised around four verbs — a write/select/compress/isolate taxonomy common in recent context engineering work:

| Verb | Problem | Primitives |
|------|---------|------------|
| **WRITE** | What enters context? When? | `ContextWindow`, `WorkingMemory`, `HistoryManager` |
| **SELECT** | Of everything we could load, what's actually relevant now? | `SkillRegistry`, `RuleBasedRouter`, `EmbeddingRouter`, `LLMRouter` |
| **COMPRESS** | The window is full. What gives? | `TokenBudget`, `ToolResultCompactor`, schema summarizers, `apply_cascade` |
| **ISOLATE** | This sub-task is too big to share one window. | `SubAgent`, `StaticPlanner`, `LLMPlanner` |

---

## Install

```bash
pip install -e .
pip install -e ".[tiktoken]"   # optional: closer-to-real token counts
pip install -e ".[dev]"        # pytest + ruff
```

Requires Python 3.11+, Pydantic v2, the Anthropic SDK.

---

## A 60-second tour

```python
from agent_context_kit import (
    ComponentCategory, ContextWindow, HistoryManager, SkillLoader,
    RuleBasedRouter, TokenBudget, ToolResultCompactor,
    ProgressiveCompactionStrategy, extractive_summarizer,
)

# 1. A budget carves the window into named components.
budget = TokenBudget.for_window(
    total_window=100_000,
    reply_reservation=4_000,
    hard_limits={
        ComponentCategory.INSTRUCTIONS: 3_000,
        ComponentCategory.ROUTINE: 2_000,
    },
    shares={
        ComponentCategory.HISTORY: 0.5,
        ComponentCategory.TOOL_RESULTS: 0.3,
        ComponentCategory.WORKING_MEMORY: 0.2,
    },
)

# 2. The window owns what goes into the LLM each turn.
window = ContextWindow(
    budget=budget,
    history=HistoryManager(
        keep_recent=8,
        summarizer=extractive_summarizer(max_lines=4),
    ),
)
window.set_instructions("You are a support agent. ...")

# 3. Skills are progressive-disclosure instruction bundles.
registry = SkillLoader().load_directory("examples/multi_skill_support/skills/")
router = RuleBasedRouter(registry)

# 4. Pluggable compaction strategies for tool results.
window.set_compaction_strategy(
    ComponentCategory.TOOL_RESULTS,
    ProgressiveCompactionStrategy(ToolResultCompactor()),
)

# 5. Per-turn: add the user turn, route skills, enforce budget, call the model.
window.add_user_turn("Where is my order ORD-123?")
matches = router.route(conversation_history=window.history.turns)
for name, body in registry.bodies_for([m.name for m in matches]).items():
    window.load_skill(name, body)
window.enforce_budget()

system_prompt = window.build_system_prompt()
api_messages = window.build_api_messages()
# -> feed into anthropic.Anthropic().messages.create(...)
```

---

## Core ideas

### Two-level progressive disclosure for skills

At scale you can't load every skill's full instructions upfront. The registry separates:

- **Level 1 metadata** (always loaded): `name + description + tags`, ~50-100 tokens per skill. The router sees the full catalogue cheaply.
- **Level 2 bodies** (loaded on demand): full instructions, routines, examples. Loaded only for the skills the router picks this turn.

This mirrors the `CapabilityRegistry` + `ToolClassifier` pair in `agent-tool-kit` — same principle, applied to instructions/routines instead of tool schemas. Use both together: the two catalogues produce a focused system prompt.

### Working memory is a superset of Scratchpad

`WorkingMemory` is drop-in compatible with `agent_eval_loop.agent.scratchpad.Scratchpad`. Every method carries over (`set`, `get`, `append_to`, `render`, `compact`, `entries`, `history`) but you also get:

- **Typed entries** — each entry has an `entry_type` tag for grouped rendering.
- **Priority entries** — `set_typed(..., priority=True)` pins facts that must survive compaction (customer-stated constraints, verified IDs).
- **Turn-aware rollup** — `rollup_old(older_than_turns=3)` folds stale non-pinned entries into a summary.
- **Auto-compaction** — pass `max_tokens=` or `max_entries=` and call `auto_compact_if_needed` to keep memory bounded.

Existing code using `Scratchpad` keeps working; new code benefits from the richer features.

### Token budget allocation

A fixed token budget split across component types. Instructions and the active routine are never compressed. Compressible components (history, tool results, working memory, skills) each get a share of the remainder. When one overruns, compression runs in a configurable priority order (default: history → tool results → working memory → skills).

Default ratios match the reference doc's table; see `docs/best-practices.md` for a deeper walkthrough.

### Sub-agent isolation

`SubAgent.run()` spins up a sub-agent with its own clean context — a focused task, only the tools it needs, a structured return shape. The main agent never sees the sub-agent's internal reasoning or tool calls. Use `StaticPlanner` when the breakdown is deterministic; `LLMPlanner` when you need the planner itself to generate the step list.

---

## Integrating with other agent engineering kit repos

### With `agent-eval-loop`'s `AgentRunner`

```python
from agent_eval_loop.agent.runner import AgentRunner
from agent_context_kit.managed_runner import ManagedAgentRunner

managed = ManagedAgentRunner(
    runner=agent_runner,
    window=context_window,
    skill_router=router,
    skill_registry=skill_registry,
)
reply = managed.send_message("What's my balance?")
```

`ManagedAgentRunner` replaces the runner's static system prompt + scratchpad with the window's runtime-managed versions. Tool-handler dispatch, audit logging, and everything else the runner already does keeps working.

### With `agent-tool-kit`'s `CapabilityRegistry`

A tool registry and a skill registry live side by side. Both use progressive disclosure; both feed the window:

```python
from agent_tool_kit import CapabilityRegistry, ToolClassifier

tools = CapabilityRegistry([...])
classifier = ToolClassifier(tools)
selected = classifier.select(user_message)
window.set_tool_descriptions(tools.menu())  # prose
runner.tool_handlers = tools.handlers(selected)  # dispatch
runner.config.tool_schemas = tools.schemas_for(selected)  # API
```

---

## The three worked examples

Each example is runnable with `python examples/<name>/run.py`. Where possible they support a `--dry-run` flag so you can see the mechanics without an API key.

- **`examples/multi_skill_support/`** — a support agent with three skills (billing, shipping, returns) dynamically loaded per turn by a keyword router. Shows the classifier → skill loader → response pipeline.
- **`examples/research_assistant/`** — a planner breaks a research task into two steps, each delegated to a sub-agent. Shows planner → executor with context isolation and inter-step dependencies.
- **`examples/long_conversation/`** — a 12-turn support scenario under a tight budget. Watch tool-result states age full → summary → reference as tokens climb and compression kicks in.

---

## Observability

Every non-trivial decision emits a `ContextEvent`:

- `skill_loaded` / `skill_unloaded`
- `compaction` / `budget_enforced` / `budget_exhausted`
- `sub_agent_started` / `sub_agent_completed` / `plan_step_failed`

Pass `event_log=ContextEventLog(path="context_events.jsonl")` at construction to flush events to disk. `ContextEventLog.replay(path)` iterates them back later. Same contract as `agent-tool-kit`'s `AuditLog`.

---

## License

MIT
