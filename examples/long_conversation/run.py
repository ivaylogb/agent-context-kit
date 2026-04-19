"""Long conversation manager — compression under pressure.

Simulate a 20+ turn conversation and show how the context window stays
bounded. At each turn we:

1. Add the user's turn + a fake assistant reply with a verbose tool result.
2. Compact aging tool results (ProgressiveCompactionStrategy).
3. Roll older turns into a conversation summary (SummaryHistoryStrategy).
4. Roll old working-memory entries into ``memory_summary``.
5. Enforce the token budget.

The script prints per-turn usage so you can see the ceiling hold.

No API calls required: the assistant's replies are scripted, the
summarizer is the extractive (no-LLM) one, and tool outputs are static.
Run ``python examples/long_conversation/run.py``.
"""

from __future__ import annotations

import sys

from agent_context_kit import (
    ComponentCategory,
    ContextEventLog,
    ContextWindow,
    HistoryManager,
    ProgressiveCompactionStrategy,
    SummaryHistoryStrategy,
    TokenBudget,
    ToolResultCompactor,
    extractive_summarizer,
    truncate_summarizer,
)


# A verbose fake tool result — the kind of thing that eats context fast.
VERBOSE_TOOL_RESULT = (
    "{\n"
    '  "order_id": "ORD-%d",\n'
    '  "status": "shipped",\n'
    '  "carrier": "FastShip",\n'
    '  "tracking_id": "1Z999AA10123456784",\n'
    '  "items": [\n'
    '    {"sku": "SKU-A1", "name": "Noise-canceling headphones", '
    '"qty": 1, "price": 299.99, "metadata": {"color": "midnight blue", '
    '"warehouse": "WH-EAST", "lot": "LOT-4413A", "inspected_at": "2026-04-17T14:22:00Z"}},\n'
    '    {"sku": "SKU-B2", "name": "USB-C cable 2m", "qty": 2, '
    '"price": 14.99, "metadata": {"color": "black", "warehouse": "WH-EAST"}}\n'
    "  ],\n"
    '  "shipping_address": {"street": "1 Example St", "city": "Anytown", '
    '"state": "CA", "zip": "94000"},\n'
    '  "total": 329.97,\n'
    '  "tax": 27.05,\n'
    '  "grand_total": 356.02,\n'
    '  "internal_notes": "customer requested expedited handling; warehouse '
    'confirmed 2-day pickup; inventory reserved at 14:22 UTC"\n'
    "}"
)


def build_window(event_log: ContextEventLog) -> ContextWindow:
    # A deliberately tight budget so compression has to run frequently.
    budget = TokenBudget.for_window(
        total_window=6_000,
        reply_reservation=800,
        hard_limits={
            ComponentCategory.INSTRUCTIONS: 300,
            ComponentCategory.ROUTINE: 300,
            ComponentCategory.TOOL_DESCRIPTIONS: 200,
        },
        shares={
            ComponentCategory.HISTORY: 0.5,
            ComponentCategory.TOOL_RESULTS: 0.3,
            ComponentCategory.WORKING_MEMORY: 0.2,
        },
    )
    history = HistoryManager(
        keep_recent=4,
        summarizer=extractive_summarizer(max_lines=3),
    )
    window = ContextWindow(budget=budget, history=history, event_log=event_log)
    window.set_instructions(
        "You are a support agent. Be concise. Pull data before answering."
    )
    window.set_routine("Always confirm the order ID before acting.")

    # Plug in the progressive tool-result compactor.
    compactor = ToolResultCompactor(
        full_ttl_turns=1,
        summary_ttl_turns=3,
        summarizer=truncate_summarizer(80),
    )
    window.set_compaction_strategy(
        ComponentCategory.TOOL_RESULTS,
        ProgressiveCompactionStrategy(compactor),
    )

    # And the summary-backed history strategy.
    window.set_compaction_strategy(
        ComponentCategory.HISTORY,
        SummaryHistoryStrategy(extractive_summarizer(max_lines=3)),
    )
    return window


def run_scenario() -> int:
    event_log = ContextEventLog()
    window = build_window(event_log)

    scenario = [
        "My order ORD-100 was supposed to arrive yesterday — where is it?",
        "Any update on ORD-100?",
        "I'd also like to return an item in ORD-100 when it arrives.",
        "What's the refund timeline again?",
        "Actually, please also look at ORD-101.",
        "I was charged twice on invoice INV-55 — can you check?",
        "Are you sure INV-55's second charge is the subscription?",
        "Ok, refund the duplicate.",
        "What's the ETA on ORD-101 now?",
        "Can you email me the refund confirmation?",
        "One more — ORD-102 had the wrong size. Exchange?",
        "Thanks. Summarize everything we've done so far.",
    ]

    # Seed a pinned fact so we can confirm priority entries survive compaction.
    window.memory.set_typed(
        "customer_id", "CUST-42",
        entry_type="fact", priority=True,
    )

    print(f"{'turn':<4} {'user_msg':<40} {'tokens':<8} {'hist':<6} {'tools':<6} {'skills':<6}")
    for i, msg in enumerate(scenario, start=1):
        window.add_user_turn(msg)
        # Simulate an assistant action that produced a verbose tool result.
        window.add_tool_result(
            "lookup_order", VERBOSE_TOOL_RESULT % i, turn_index=i,
        )
        window.add_assistant_turn(f"Checked ORD-{i}. Let me help with that.")
        # Track a per-turn note so working memory grows too.
        window.memory.set(f"note_turn_{i}", f"processed message {i}")

        window.enforce_budget()

        usage = window.usage()
        hist = usage[ComponentCategory.HISTORY]
        tools = usage[ComponentCategory.TOOL_RESULTS]
        skills = usage[ComponentCategory.SKILLS]
        snippet = msg if len(msg) < 37 else msg[:34] + "..."
        print(f"{i:<4} {snippet:<40} {window.tokens():<8} {hist:<6} {tools:<6} {skills:<6}")

    print("\n--- final window state ---")
    print("loaded skills:", window.loaded_skill_names())
    print("pinned memory 'customer_id':", window.memory.get("customer_id"))
    print("memory_summary present:", "memory_summary" in window.memory.entries)
    print("tool result states:", [t.state for t in window.tool_results()])
    print("history summary present:", window.history.summary is not None)

    print("\n--- event log counts ---")
    for event_type, count in event_log.summary().items():
        print(f"  {event_type}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(run_scenario())
