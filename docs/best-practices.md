# Best Practices for Runtime Context Engineering

Lessons from building agents that stay coherent across long conversations and complex tasks. These principles are domain-agnostic — they apply whether you're building customer support, research, code review, or anything else.

---

## The core thesis

The LLM is the CPU. The context window is RAM. Context engineering is the operating system that manages what goes in.

Every token in the window is a token the model has to attend to. The more noise, the worse the signal. Production agents fail at scale not because the model is bad but because the context window fills with the wrong things:

- Stale tool results crowd out instructions.
- Conversation history grows until early context is forgotten.
- Relevant knowledge isn't loaded because the agent doesn't know it exists yet.

The goal at each turn: the agent sees exactly what it needs for the next step — nothing more, nothing less.

---

## WRITE — what enters context

### Progressive disclosure for instructions

Don't load all instructions and capabilities into the system prompt upfront. Store them as files. The agent only loads the skill name and description initially. When a skill is relevant, load the full body.

| Level | When loaded | Token cost | Content |
|-------|-------------|------------|---------|
| Level 1: Metadata | Always (system prompt) | ~50-100 tokens per skill | Skill name + one-line description |
| Level 2: Instructions | On demand (when skill selected) | ~500-2000 tokens | Full instructions, routines, examples |
| Level 3: Reference data | On demand (as the skill needs it) | Variable | Knowledge-base articles, templates, policies |

Key insight: a cheap, fast classifier model solves a context-selection problem so the expensive, powerful model can focus on the actual work. Division of labor *between* models, not *within* one model.

In this package: `SkillRegistry.menu()` returns Level-1; `SkillRegistry.bodies_for(names)` returns Level-2. The router decides which names to pass.

### Working memory is structured, not a string

Production agents need cross-turn state — customer-stated constraints, verified facts, the current workflow step, partial results. A simple key-value scratchpad is a good start; a production system needs more:

- **Typed entries.** Group facts separately from scratch notes. The compactor should weight them differently.
- **Priority (pinned) entries.** A customer's stated order ID should never be compacted away. `set_typed(..., priority=True)`.
- **Turn-aware relevance.** Entries from 20 turns ago are less likely to be relevant than entries from 2 turns ago.
- **Automatic compaction.** When working memory grows beyond a threshold, fold older non-pinned entries into a summary.

### Turn-aware history management

Conversation history is a growing liability. An unmanaged 50-turn history is 30K tokens, mostly redundant with what the agent already knows via working memory.

Three policies, composable:

1. **Sliding window** — keep the last N turns in full. Simple; loses long-horizon context.
2. **Summarize-and-keep** — summarize turns older than N, keep recent turns verbatim. Preserves information at a summarization-latency cost.
3. **Critical-moment preservation** — regardless of age, keep turns where a tool was called, a decision was made, the customer stated a constraint, or an escalation happened. High-information-density turns.

Combine all three: recent turns in full + older turns summarized + critical moments preserved. `HistoryManager` with `critical=True` on relevant turns and a summarizer configured is this pattern out of the box.

---

## SELECT — choosing what's relevant

### Context routing

Before the main agent processes a message, a router determines what context to load. The message comes in, the router says "this is a billing question, load the billing skill," and by the time the main agent sees it, the right instructions are already in the window. The main agent never has to search for the skill.

Three router types:

- **Rule-based** — keyword matching. Fast, cheap, brittle.
- **Embedding-based** — compare message embedding to skill description embeddings. More robust to paraphrases.
- **LLM-based** — a cheap classifier model reads the skill menu and picks. Most flexible; adds ~200ms at Haiku.

Start with rule-based; upgrade when accuracy plateaus. Keep a fallback path: bad classifier should degrade into "load everything", never into "block the request".

### Multi-label classification

A single message can span multiple domains ("I was charged for my order but also need to return something"). The classifier should return a ranked list with confidence scores, not one label. All matches above a threshold get loaded.

The classifier receives the full conversation history, not just the current message. Without history, "compare fees" has no referent for which product, and "I don't understand" has zero signal.

### Route on intent, not on surface form

A message like "where's my stuff" should route to the shipping skill even though "shipping" isn't in the text. Rule-based routers fail on this. Embedding and LLM routers handle it. If your rule-based router keeps missing paraphrases, that's the signal to upgrade — not to add more keywords.

---

## COMPRESS — keeping context lean

### Tool result compaction

Tool results are usually the noisiest thing in the window. A verbose API response (an order lookup with 40 fields, a search result with 20 hits) is 80% noise by the time the model has extracted what it needs.

Progressive compaction:

- **Full** (verbatim): keep until older than `full_ttl_turns`.
- **Summary** (one-line): "Order ORD-78234: shipped, ETA Thursday." Keep until `summary_ttl_turns`.
- **Reference** (handle only): "see tool_result:lookup_order:turn7". Retrievable from audit logs if needed.

`ToolResultCompactor` does the transitions; `ProgressiveCompactionStrategy` wires it into the window's budget enforcer.

### Schema-based conversation summarization

When compaction hits diminishing returns, summarize the conversation trajectory using a consistent schema:

```json
{
  "customer_intent": "return defective headphones",
  "key_facts": ["order ORD-78234", "purchased 12 days ago", "item is defective"],
  "actions_taken": ["looked up order", "verified return eligibility"],
  "current_state": "awaiting customer confirmation to proceed with return",
  "unresolved": ["customer asked about refund timeline — not yet answered"]
}
```

The schema ensures summaries are uniform regardless of conversation shape. Uniform summaries are reliable inputs for the agent's reasoning. Free-form prose summaries vary; evals get flaky.

Use `llm_summarizer()` for the default schema; `build_schema_summarizer(MyModel)` for a custom one.

### Token budget allocation

A fixed budget split across component types. Example for a 100K-token window:

| Component | Budget | Priority | Compressible? |
|-----------|--------|----------|---------------|
| Instructions | 5K | Critical | No |
| Active routine | 3K | High | No |
| Tool descriptions | 5K | High | Partially |
| Working memory | 5K | Medium | Yes |
| Recent tool results | 10K | Medium | Yes |
| Conversation history | 15K | Low | Yes |
| Headroom | 57K | Reserved | — |

Compression priority: conversation history first, then tool results, then working memory. Instructions and active routine are never compressed.

Important: the **headroom reservation matters**. Reserve at least `max_tokens` (the model's reply budget) plus safety margin. A long model response is often the thing that tips you over the real window limit.

### Cascading automatic compression

Claude Code's approach: trigger progressively more aggressive compression as usage climbs.

- At 80%: compact aging tool results.
- At 90%: summarize older turns.
- At 95%: aggressive — compact working memory, drop inactive skill context.
- At 98%: emergency — summarize everything, keep only instructions + active routine + most recent turn.

`apply_cascade(window)` in `agent_context_kit.compress` implements this. Call it at the end of each turn.

---

## ISOLATE — containing complexity

### Sub-agent delegation

When a task requires more context than fits in one window, or requires a different skill set, spin up a sub-agent. The sub-agent gets:

- A focused task description.
- Only the tools it needs.
- Relevant context (not the full conversation history).
- A structured output schema for its result.

The sub-agent does its thing and returns a structured result. The main agent never sees the sub-agent's internal reasoning, tool calls, or intermediate state.

This is context isolation — the sub-agent's mess stays contained.

### Planner → executor pattern

For complex multi-step tasks:

1. **Planner** receives the full request and breaks it into discrete steps.
2. Each step is delegated to an **executor** sub-agent with exactly the context it needs.
3. Executors return structured results to the planner.
4. The planner synthesizes results and produces the final output.

The planner maintains high-level state. Executors are stateless — they receive a task, execute, and return. This prevents cross-step context contamination.

Important: the planner should budget how much total executor output it can accumulate. If 5 executors each return 2K tokens, the planner has 10K tokens of results in its window. Schema-based result summarization keeps this manageable.

Use `StaticPlanner` when the breakdown is deterministic (save the planning-model call). Use `LLMPlanner` when the breakdown needs domain-specific judgment.

---

## The filesystem as coordination layer

A key principle from both Claude Code and production agents: use the filesystem as extended memory.

- Tool results → write to files, keep references in context.
- Sub-agent outputs → write to files, pass paths to the planner.
- Knowledge bases → read from files on demand.
- Working memory → can be persisted to disk for crash recovery.

The filesystem is cheaper than the context window, unlimited in size, debuggable, and both humans and agents already have strong priors on how to use it.

This package is filesystem-friendly by design: skills are files, event logs are JSONL, tool-result references point to external storage the caller manages.

---

## Observability

Every non-trivial decision emits a `ContextEvent`. Pass an `event_log=ContextEventLog(path="events.jsonl")` at construction:

- `skill_loaded` / `skill_unloaded` — what the router chose this turn.
- `compaction` — which category got compressed and how many tokens freed.
- `budget_enforced` / `budget_exhausted` — whether the window fit or spilled.
- `sub_agent_started` / `sub_agent_completed` — delegation boundary events.

When an agent "loses" context mid-conversation, the event log is the first place to look. Replay with `ContextEventLog.replay(path)` the same way you'd replay an audit log in `agent-tool-kit`.

---

## Common anti-patterns

- **Hardcoding which skill to load.** Defeats the point of a router. If you know upfront, use a single-skill agent.
- **Summarizing the active routine.** Routines are the agent's step-by-step instructions for the current workflow. Summarizing them means the agent forgets the procedure mid-task.
- **Unbounded tool result history.** "I'll just let them accumulate — they're useful." They're not: old results are usually redundant with the current state, and they push the real instructions out of attention range.
- **Running the LLM router on every turn without caching.** Router decisions are often stable across turns ("the customer is still asking about billing"). Cache the last decision and only re-run when the message clearly shifts domains.
- **Sub-agents that call the main agent.** Breaks isolation. If you need a recursive agent, the planner should re-plan, not recurse into the main loop.
- **`keep_recent=1` with no summarizer.** You've thrown away every turn older than the current one. The agent can't reference its own earlier commitments.

---

## When to break these rules

A few legitimate exceptions:

- **Skip the router for single-domain agents.** If you only have one skill, the routing overhead isn't buying anything.
- **Skip compaction for short-conversation agents.** If the interaction is bounded to 3-5 turns, budget enforcement is overhead.
- **Skip sub-agent isolation for simple chains.** If a task is two sequential tool calls, the planner is overkill — just call the tools inline.

Every exception should be a deliberate, documented choice. If you can't articulate why you broke the pattern, you're cutting a corner that will cost you later — the same rule the sibling repos apply.
