"""Multi-skill customer support agent.

Shows the full classifier → skill loader → response pipeline:

1. Customer message arrives.
2. A router (keyword-based here; swap for ``LLMRouter`` in production)
   decides which skills are relevant for this turn.
3. The ``ContextWindow`` loads only those skills' bodies.
4. Stale skills from previous turns get unloaded so the window stays lean.
5. The budget is enforced.
6. The AgentRunner sees a focused system prompt with exactly the right
   instructions for the current domain.

Run ``python examples/multi_skill_support/run.py`` to see it in action.
Requires ``ANTHROPIC_API_KEY`` in the environment.

To run offline without hitting the API, pass ``--dry-run`` to see the
system-prompt assembly and skill-routing decisions without making calls.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_context_kit import (
    ComponentCategory,
    ContextEventLog,
    ContextWindow,
    HistoryManager,
    RuleBasedRouter,
    SkillLoader,
    TokenBudget,
    extractive_summarizer,
)

SKILLS_DIR = Path(__file__).parent / "skills"


def build_window(event_log: ContextEventLog) -> ContextWindow:
    """Assemble a window sized for a support-agent workload."""
    budget = TokenBudget.for_window(
        total_window=50_000,
        reply_reservation=2_000,
        hard_limits={
            ComponentCategory.INSTRUCTIONS: 500,
            ComponentCategory.ROUTINE: 500,
            ComponentCategory.TOOL_DESCRIPTIONS: 1_000,
        },
        shares={
            ComponentCategory.HISTORY: 0.5,
            ComponentCategory.TOOL_RESULTS: 0.25,
            ComponentCategory.WORKING_MEMORY: 0.15,
            ComponentCategory.SKILLS: 0.10,
        },
    )
    history = HistoryManager(
        keep_recent=6,
        summarizer=extractive_summarizer(max_lines=4),
    )
    window = ContextWindow(budget=budget, history=history, event_log=event_log)
    window.set_instructions(
        "You are a customer support agent. Be concise, accurate, and honest. "
        "Pull data before answering. Never fabricate IDs, prices, or policies. "
        "If you don't have a skill loaded for the domain, say so."
    )
    window.set_routine(
        "Investigate first, answer second. Use the loaded skills' routines — "
        "do not improvise steps they don't list."
    )
    return window


def turn(
    window: ContextWindow,
    router: RuleBasedRouter,
    registry,
    user_message: str,
    *,
    dry_run: bool,
    client=None,
):
    """Process one conversation turn with skill routing + budget enforcement."""
    window.add_user_turn(user_message)

    # Route skills for this turn, using the full conversation as context.
    matches = router.route(
        conversation_history=window.history.turns,
        threshold=0.0,
        top_k=3,
    )
    selected = [m.name for m in matches]

    # Drop skills that are no longer selected.
    for name in list(window.loaded_skill_names()):
        if name not in selected:
            window.unload_skill(name)

    # Load fresh skill content for the current selection.
    if selected:
        bodies = registry.bodies_for(selected)
        for name, body in bodies.items():
            window.load_skill(name, body)

    window.enforce_budget()

    system_prompt = window.build_system_prompt()
    api_messages = window.build_api_messages()

    print(f"\n--- turn: user asked {user_message!r} ---")
    print(f"matched skills: {[(m.name, round(m.confidence, 2)) for m in matches]}")
    print(f"loaded skills:  {window.loaded_skill_names()}")
    print(f"tokens in window: {window.tokens()}")

    if dry_run:
        # Show the last bit of the assembled system prompt as proof the skills
        # are actually in the window.
        print("--- system prompt tail (skills region) ---")
        tail = system_prompt.split("<skills>")[-1][:300]
        print("<skills>" + tail)
        reply = "(dry-run — no API call)"
    else:
        if client is None:
            raise RuntimeError("Non-dry-run mode requires a client.")
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            temperature=0.0,
            system=system_prompt,
            messages=api_messages,
        )
        reply = "".join(
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        )
        window.add_assistant_turn(reply)
        print(f"assistant: {reply}")
    return reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the API call; show routing and window state only.",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Re-run with --dry-run for offline demo.")
        return 1

    event_log = ContextEventLog()
    registry = SkillLoader().load_directory(SKILLS_DIR)
    router = RuleBasedRouter(registry)
    window = build_window(event_log)

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()

    scenario = [
        "Hey — where's my order ORD-4412? It was supposed to arrive yesterday.",
        "Actually, I also noticed I was charged twice on my last invoice INV-8891. Can you look?",
        "I'd also like to return one of the items from ORD-4412 when it shows up.",
    ]
    for msg in scenario:
        turn(window, router, registry, msg, dry_run=args.dry_run, client=client)

    print("\n--- event log summary ---")
    for event_type, count in event_log.summary().items():
        print(f"  {event_type}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
