"""Integration glue for ``agent_eval_loop.agent.runner.AgentRunner``.

The sibling ``agent-eval-loop`` package ships a simple ``AgentRunner`` with
a monolithic ``system_prompt`` string, a plain ``conversation_history``
list, and a ``Scratchpad``. ``ContextWindow`` replaces all three with
actively-managed runtime context. This module is the adapter that sits
between them.

``ManagedAgentRunner`` wraps an ``AgentRunner`` and, on every
``send_message`` call, (1) adds the user turn to the window, (2) routes
skills, (3) runs budget enforcement, (4) rebuilds the system prompt +
messages from the window, and (5) feeds them into the runner. After the
model replies, it records the assistant turn back into the window.

Importing this module requires ``agent-eval-loop`` to be installed. Callers
who don't have the eval-loop can still use the rest of this package; the
import only happens when ``ManagedAgentRunner`` is instantiated.
"""

from __future__ import annotations

import json
from typing import Any

from agent_context_kit.context.window import ContextWindow
from agent_context_kit.skills.loader import SkillRegistry
from agent_context_kit.skills.router import SkillRouter


class ManagedAgentRunner:
    """Drop-in wrapper that replaces ``AgentRunner``'s static context with a window.

    Usage::

        from agent_eval_loop.agent.runner import AgentRunner
        from agent_eval_loop.models import AgentConfig
        from agent_context_kit import ContextWindow, TokenBudget
        from agent_context_kit.managed_runner import ManagedAgentRunner

        runner = AgentRunner(config=cfg, tool_handlers=handlers)
        window = ContextWindow(
            budget=TokenBudget.for_window(total_window=100_000, ...),
        )
        window.set_instructions("You are ...")
        managed = ManagedAgentRunner(
            runner=runner,
            window=window,
            skill_router=router,
            skill_registry=skills,
        )
        reply = managed.send_message("Where's my order?")

    The managed runner preserves every feature of the underlying runner
    (tool handler execution, tool call bookkeeping) and layers on the
    window-managed context. Use ``runner.tool_calls`` for the audit trail;
    we don't try to duplicate it.
    """

    def __init__(
        self,
        *,
        runner: Any,  # agent_eval_loop.agent.runner.AgentRunner
        window: ContextWindow,
        skill_router: SkillRouter | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_threshold: float = 0.3,
        skill_top_k: int | None = 3,
        enforce_budget: bool = True,
    ) -> None:
        self.runner = runner
        self.window = window
        self.skill_router = skill_router
        self.skill_registry = skill_registry
        self.skill_threshold = skill_threshold
        self.skill_top_k = skill_top_k
        self.enforce_budget = enforce_budget
        # Replace the runner's scratchpad with a shim that proxies to
        # the window's working memory. The AgentRunner expects an object
        # with ``render()``; any superset works.
        self.runner.scratchpad = self.window.memory  # type: ignore[assignment]

    def send_message(self, user_message: str) -> str:
        """Send a user message; return the assistant's text reply.

        Mirrors ``AgentRunner.send_message`` but with context management.
        """
        # 1. Add the user turn to the window.
        self.window.add_user_turn(user_message)

        # 2. Route skills (if a router is configured).
        if self.skill_router is not None and self.skill_registry is not None:
            matches = self.skill_router.route(
                conversation_history=self.window.history.turns,
                threshold=self.skill_threshold,
                top_k=self.skill_top_k,
            )
            selected_names = [m.name for m in matches]
            # Unload skills no longer selected.
            for name in list(self.window.loaded_skill_names()):
                if name not in selected_names:
                    self.window.unload_skill(name)
            # Load selected skills.
            if selected_names:
                bodies = self.skill_registry.bodies_for(selected_names)
                for name, body in bodies.items():
                    self.window.load_skill(name, body)

        # 3. Enforce the budget.
        if self.enforce_budget:
            self.window.enforce_budget()

        # 4. Rebuild runner state from the window and issue the call.
        system_prompt = self.window.build_system_prompt()
        api_messages = self.window.build_api_messages()
        # Snapshot the runner's cumulative tool_calls before the call so we
        # can mirror exactly this turn's results into the window afterward.
        tool_calls_before = len(self.runner.tool_calls)
        reply = self._call_runner(system_prompt, api_messages)

        # 5. Mirror this turn's tool results into the window so progressive
        #    tool-result compaction has slots to age. Done before the
        #    assistant turn is added so the turn_index reflects the moment
        #    the tool ran (between user turn and assistant turn), matching
        #    the ordering used by ``examples/long_conversation/run.py``.
        self._mirror_tool_results(tool_calls_before)

        # 6. Record the assistant turn back in the window.
        self.window.add_assistant_turn(reply)
        return reply

    def _mirror_tool_results(self, snapshot_before: int) -> None:
        """Copy this turn's tool calls into ``window.tool_results``.

        ``AgentRunner._process_response`` runs tools inside one
        ``send_message`` round, feeding outputs back into the model via its
        own local ``messages`` list. Those results never surface in
        ``window.tool_results()`` unless we mirror them. We diff against
        ``snapshot_before`` rather than scanning the whole cumulative
        ``runner.tool_calls`` list so prior turns' results aren't double-
        recorded. Content is JSON-serialized to match how the runner
        already formats results for the API.
        """
        new_calls = self.runner.tool_calls[snapshot_before:]
        for call in new_calls:
            payload = call.result if call.result is not None else {"error": call.error}
            if isinstance(payload, str):
                content = payload
            else:
                content = json.dumps(payload, default=str)
            self.window.add_tool_result(call.tool_name, content)

    def _call_runner(
        self,
        system_prompt: str,
        api_messages: list[dict[str, Any]],
    ) -> str:
        """Issue a raw Anthropic call honoring runner config + tool handlers.

        We bypass ``AgentRunner.send_message`` because that method injects
        its own scratchpad rendering and builds its own messages list from
        ``runner.conversation_history``. We've already done both — we want
        the runner's tool-handler dispatch, not its context assembly.
        """
        runner = self.runner
        tools = runner._build_tool_definitions() or None  # noqa: SLF001 — documented integration point
        api_kwargs: dict[str, Any] = {
            "model": runner.config.model,
            "max_tokens": runner.config.max_tokens,
            "temperature": runner.config.temperature,
            "system": system_prompt,
            "messages": list(api_messages),
        }
        if tools:
            api_kwargs["tools"] = tools
        response = runner.client.messages.create(**api_kwargs)
        # Reuse the runner's tool-use loop for dispatch. Let the runner own
        # the mutable messages list during tool loops — that's what it was
        # designed to do.
        return runner._process_response(  # noqa: SLF001
            response, list(api_messages), system_prompt, tools,
        )
