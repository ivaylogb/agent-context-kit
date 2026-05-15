# Context Engineering Reference — Key Patterns

This document covers context-engineering patterns: principles for assembling, compressing, and routing context into agent calls.

---

## 1. The Core Principle

The LLM is the CPU. The context window is RAM. Context engineering is the operating system that manages what goes in.

The context window is finite and expensive. Every token matters. The goal: at any given moment, the agent sees exactly the information it needs for the next step — nothing more, nothing less.

This means actively managing what enters the context, compressing what's already there, and isolating complex tasks so they don't pollute the main window.

---

## 2. Write — What Enters Context

### 2.1 Progressive Disclosure (Skills Architecture)

Don't load all instructions and capabilities into the system prompt upfront. Store them as files in the filesystem. The agent only loads the skill name and description initially. When it determines a skill is relevant, it loads the full body.

Claude Agent Skills use a three-level progressive disclosure system:

| Level | When Loaded | Token Cost | Content |
|-------|-------------|------------|---------|
| Level 1: Metadata | Always (system prompt) | ~50-100 tokens per skill | Skill name and one-line description |
| Level 2: Instructions | On demand (when skill selected) | ~500-2000 tokens | Full instructions, routines, examples |
| Level 3: Reference data | On demand (when task requires it) | Variable | Knowledge base articles, templates, policies |

The key insight: a cheap, fast classifier model solves a context selection problem so the expensive, powerful model can focus on the actual work. Division of labor between models, not within one model.

### 2.2 Working Memory / Scratchpad

Persistent cross-turn state. The agent writes structured notes that accumulate across turns. On each turn, the full working memory is rendered into the context window.

Key design decisions:
- **Typed entries** — not just key-value strings. Support structured data (dicts, lists, typed objects).
- **Relevance decay** — entries from 20 turns ago are less likely to be relevant than entries from 2 turns ago. Weight accordingly.
- **Automatic compaction** — when working memory grows beyond a threshold, compact older entries into summaries.
- **Priority entries** — some entries (customer-stated constraints, verified facts) should never be compacted.

### 2.3 Conversation History Management

History is a growing liability. An unmanaged history of 50 turns might be 30K tokens — mostly redundant with what the agent already knows via working memory.

Patterns:
- **Sliding window** — keep the last N turns in full. Simple, fast, but loses context.
- **Summarize-and-keep** — summarize turns older than N, keep the summaries. Preserves key information but adds summarization latency.
- **Critical-moment preservation** — regardless of age, keep turns where: a tool was called, a decision was made, the customer stated a constraint, or an escalation happened. These are high-information-density turns.
- **Hybrid** — combine all three. Recent turns in full + older turns summarized + critical moments preserved regardless of age.

---

## 3. Select — Choosing What's Relevant

### 3.1 Context Routing

Before the main agent processes a message, a router determines what context to load. The message comes in, the router says "this is a billing question, load the billing skill," and by the time the main agent sees it, the right instructions are already in the context window. The main agent never has to search for the skill itself.

The router can be:
- **Rule-based** — keyword matching, regex patterns. Fast, cheap, brittle.
- **Embedding-based** — compare message embedding to skill description embeddings. More robust, requires an embedding index.
- **LLM-based** — a cheap model (Haiku-class, ~$1/M input tokens) with a structured prompt listing all skill descriptions. Most flexible, small latency cost.

### 3.2 Multi-Label Classification

A single message might span multiple domains. "Compare fees to card fees" touches fees AND cards. The classifier should return a ranked list of matching categories with confidence scores, not a single label.

All matched categories above a threshold get their context loaded. The primary category (highest confidence) selects the system prompt tone and persona. This is critical for cross-product queries.

The classifier receives the full conversation history, not just the current message. Without history, "compare fees" has no referent for which product's fees, and "I don't understand" has zero signal for any category.

---

## 4. Compress — Keeping Context Lean

### 4.1 Tool Result Compaction

Manus's pattern: use "full" vs "compact" representations of tool results.

- **Recent results** — stay in full. The agent might need to reference exact values.
- **Older results** — get compacted to summary references. "Order ORD-78234: shipped via FedEx, ETA Thursday April 17" instead of the full 500-token API response.
- **Stale results** — get reduced to file path references. "See order_details.json for full order data." The agent can re-retrieve if needed.

Threshold: tool results older than N turns get compacted. Results older than 2N turns get reduced to references.

### 4.2 Schema-Based Summarization

When compaction hits diminishing returns (the compacted context is still too large), summarize the entire conversation trajectory using a consistent schema:

```json
{
  "customer_intent": "return defective headphones",
  "key_facts": ["order ORD-78234", "purchased 12 days ago", "item is defective"],
  "actions_taken": ["looked up order", "verified return eligibility"],
  "current_state": "awaiting customer confirmation to proceed with return",
  "unresolved": ["customer asked about refund timeline — not yet answered"]
}
```

The schema ensures summaries are uniform regardless of conversation shape. This makes them reliable inputs for the agent's reasoning.

### 4.3 Token Budget Allocation

A fixed token budget split across component types. Example allocation for a 100K-token window:

| Component | Budget | Priority | Compressible? |
|-----------|--------|----------|---------------|
| Instructions | 5K | Critical — never compressed | No |
| Active routine | 3K | High — current procedure | No |
| Tool descriptions | 5K | High — needed for tool calling | Partially (inactive tools can be dropped) |
| Working memory | 5K | Medium — compactable | Yes |
| Recent tool results | 10K | Medium — compactable | Yes |
| Conversation history | 15K | Low — summarizable | Yes |
| **Headroom** | **57K** | Reserved for model output + safety margin | — |

When total context exceeds budget, compress in priority order: conversation history first, then tool results, then working memory. Instructions and active routine are never compressed.

---

## 5. Isolate — Containing Complexity

### 5.1 Sub-Agent Delegation

When a task requires more context than fits in one window, or requires a different skill set, spin up a sub-agent with its own clean context window.

The sub-agent gets:
- A focused task description
- Only the tools it needs
- Relevant context (not the full conversation history)
- A structured output schema for its result

The sub-agent does its thing and returns a structured result. The main agent never sees the sub-agent's internal reasoning, tool calls, or intermediate state. This is context isolation — the sub-agent's mess stays contained.

### 5.2 Planner → Executor Pattern

For complex multi-step tasks:

1. **Planner** receives the full request and breaks it into discrete steps
2. Each step is delegated to an **executor** sub-agent with exactly the context it needs
3. Executors return structured results to the planner
4. The planner synthesizes results and produces the final output

The planner maintains high-level state. Executors are stateless — they receive a task, execute it, and return. This prevents context contamination between steps.

Key design decision: the planner should have a token budget for how much total executor output it can accumulate. If 5 executors each return 2K tokens of results, the planner has 10K tokens of results in its window. Schema-based result summarization keeps this manageable.

---

## 6. The Filesystem as Coordination Layer

A key principle from both Claude Code and Manus: use the filesystem as extended memory.

- **Tool results** → write to files, keep references in context
- **Sub-agent outputs** → write to files, pass paths to the planner
- **Knowledge bases** → read from files on demand
- **Working memory** → can be persisted to disk for crash recovery

The filesystem is cheaper than the context window, unlimited in size, debuggable (you can inspect the files), and both humans and agents already have strong priors on how to use it.

---

## 7. Automatic Compression Trigger

Claude Code triggers automatic compression at 95% context usage. This is a good default. The system should:

1. Monitor context usage after each turn
2. At 80% — start compacting tool results
3. At 90% — summarize conversation history
4. At 95% — aggressive compression: compact working memory, drop inactive skill context
5. At 98% — emergency: summarize everything, keep only instructions + active routine + most recent turn

Each threshold triggers a progressively more aggressive compression strategy. The agent should be transparent about this: "I'm summarizing our earlier conversation to make room for more detailed help" (or equivalent in the working memory).
