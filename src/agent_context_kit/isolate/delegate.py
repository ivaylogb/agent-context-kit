"""Sub-agent delegation with context isolation.

When a task is bigger than one window can hold, or when it requires a
different skill set than the main agent, spin up a sub-agent. The sub-
agent gets its own clean context: a focused task description, only the
tools it needs, a subset of relevant context — but not the full
conversation history. It does its thing and returns a structured result.
The main agent never sees the sub-agent's internal reasoning, tool calls,
or intermediate state.

This is context isolation. The sub-agent's mess stays contained.

Two design choices worth calling out:

- **Results are structured, not prose.** The main agent receives an
  ``ExecutionResult`` with typed fields (status, output, errors, tool
  calls made). Prose summaries pollute the main window with tokens the
  main agent doesn't need to reason about.

- **The sub-agent runs as its own agent loop.** We don't try to
  interleave it with the main agent — that's how context contamination
  happens. The sub-agent has a defined task, it runs to completion (or
  a hard turn cap), and it returns.

Wiring::

    from anthropic import Anthropic
    from agent_context_kit.isolate import SubAgent

    sub = SubAgent(
        name="order_research",
        task="Look up order ORD-123 and report its current status.",
        tool_schemas=[lookup_order.tool_schema],
        tool_handlers={"lookup_order": lookup_order},
        client=Anthropic(),
    )
    result = sub.run()
    if result.ok:
        main_window.memory.set_typed(
            "order_research_result", result.output,
            entry_type="fact", priority=True,
        )

The ``SubAgent`` also accepts an optional ``ContextWindow`` if you want
the sub-agent to use the same runtime primitives (budget, memory,
summarization) as the main agent. Most sub-agent tasks are short enough
that the raw-message flow is sufficient — it's the default.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

import anthropic
from pydantic import BaseModel, Field

from agent_context_kit.context.window import ContextWindow
from agent_context_kit.observability import ContextEventLog


class ToolCallTrace(BaseModel):
    """One tool call made by the sub-agent. Useful for debugging / eval."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class ExecutionResult(BaseModel):
    """Structured result returned by a sub-agent.

    ``output`` is intentionally free-form: it may be a string, a dict,
    or a list depending on what the task asked for. Callers who need a
    typed contract should pass a ``result_model`` to ``SubAgent``, which
    validates the output against it.
    """

    ok: bool
    status: str = ""
    output: Any = None
    errors: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    turns_used: int = 0
    sub_agent: str = ""


class SubAgent:
    """A sub-agent with isolated context.

    The sub-agent maintains its own conversation history (starting empty),
    its own system prompt (from the ``task`` + optional ``context``
    sections), and its own tool handlers. It runs a standard Anthropic
    tool-use loop until the model returns text only or the turn cap is
    hit.

    Status values on the returned ``ExecutionResult``:
    - ``completed`` — the model returned a text response normally.
    - ``turn_limit`` — the loop hit ``max_turns`` without converging.
    - ``tool_error`` — a tool handler raised. The partial result is still
      returned with the error captured in ``errors``.
    - ``api_error`` — the Anthropic API call itself failed. Usually
      indicates a config problem; inspect ``errors``.
    """

    def __init__(
        self,
        *,
        name: str,
        task: str,
        context: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_turns: int = 8,
        result_model: type[BaseModel] | None = None,
        event_log: ContextEventLog | None = None,
        window: ContextWindow | None = None,
    ) -> None:
        if not name:
            raise ValueError("SubAgent name is required")
        if not task:
            raise ValueError("SubAgent task is required")
        self.name = name
        self.task = task
        self.context = context
        self.tool_schemas = list(tool_schemas or [])
        self.tool_handlers = dict(tool_handlers or {})
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_turns = max_turns
        self.result_model = result_model
        self.event_log = event_log
        self.window = window
        self.session_id = uuid.uuid4().hex[:12]

    # --------------------------------------------------------------- exec

    def run(self) -> ExecutionResult:
        """Execute the sub-agent loop. Always returns an ``ExecutionResult``.

        Structured so the main agent can always inspect a result — even
        on failure, the caller gets a typed envelope rather than an
        exception. The main agent should check ``ok`` before using the
        ``output``.
        """
        self._emit(
            "sub_agent_started",
            details={"task": self.task, "name": self.name, "model": self.model},
        )

        system = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self.task},
        ]
        tool_calls: list[ToolCallTrace] = []
        errors: list[str] = []

        for turn_idx in range(self.max_turns):
            try:
                api_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": system,
                    "messages": messages,
                }
                if self.tool_schemas:
                    api_kwargs["tools"] = self.tool_schemas
                response = self.client.messages.create(**api_kwargs)
            except Exception as e:  # noqa: BLE001
                errors.append(f"api_error: {type(e).__name__}: {e}")
                self._emit(
                    "sub_agent_api_error",
                    details={"error": str(e), "turn": turn_idx},
                )
                return ExecutionResult(
                    ok=False,
                    status="api_error",
                    errors=errors,
                    tool_calls=tool_calls,
                    turns_used=turn_idx,
                    sub_agent=self.name,
                )

            # Collect any tool_use blocks — these become the next turn's
            # tool-call round-trip.
            tool_uses = [
                b for b in response.content if getattr(b, "type", "") == "tool_use"
            ]
            text_blocks = [
                b for b in response.content if getattr(b, "type", "") == "text"
            ]

            if not tool_uses:
                # Model returned only text — we're done.
                text = " ".join(b.text for b in text_blocks).strip()
                validated_output = self._validate_output(text)
                self._emit(
                    "sub_agent_completed",
                    details={"turns": turn_idx + 1, "errors": errors},
                )
                return ExecutionResult(
                    ok=not errors,
                    status="completed",
                    output=validated_output,
                    errors=errors,
                    tool_calls=tool_calls,
                    turns_used=turn_idx + 1,
                    sub_agent=self.name,
                )

            # Execute tool calls, serialize the round-trip back into messages.
            assistant_content: list[dict[str, Any]] = []
            for block in response.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif btype == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                call_trace = self._execute_tool(tu.name, tu.input or {})
                tool_calls.append(call_trace)
                if call_trace.error:
                    errors.append(f"{tu.name}: {call_trace.error}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _serialize_tool_result(call_trace.result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # Hit the turn cap — return whatever we have.
        self._emit(
            "sub_agent_turn_limit",
            details={"max_turns": self.max_turns, "name": self.name},
        )
        return ExecutionResult(
            ok=False,
            status="turn_limit",
            output=None,
            errors=errors + [f"hit max_turns={self.max_turns}"],
            tool_calls=tool_calls,
            turns_used=self.max_turns,
            sub_agent=self.name,
        )

    # ------------------------------------------------------------ helpers

    def _build_system_prompt(self) -> str:
        sections: list[str] = []
        sections.append(
            f"You are a sub-agent named '{self.name}'. Execute the given "
            "task and return the result. Be concise — your output goes back "
            "to a parent agent, not to an end user."
        )
        if self.context:
            sections.append(f"<context>\n{self.context.strip()}\n</context>")
        if self.result_model is not None:
            schema = json.dumps(
                self.result_model.model_json_schema(), indent=2
            )
            sections.append(
                "Return your final answer as strict JSON matching this "
                f"schema (no prose, no markdown fencing):\n\n{schema}"
            )
        return "\n\n".join(sections)

    def _validate_output(self, text: str) -> Any:
        if self.result_model is None:
            return text
        # Extract JSON from the model output.
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return self.result_model.model_validate(data).model_dump()
        except (ValueError, json.JSONDecodeError, Exception):  # noqa: BLE001
            # Fall back to raw text — callers can check ``output`` type
            # against ``result_model`` if they care about strict parsing.
            return text

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallTrace:
        start = time.time()
        error: str | None = None
        result: Any = None
        handler = self.tool_handlers.get(tool_name)
        if handler is None:
            error = f"unknown tool: {tool_name}"
            result = {"error": error}
        else:
            try:
                # Prefer Tool.invoke(dict) if available (agent_tool_kit
                # Tool objects). Fall back to spread-kwargs calling.
                if hasattr(handler, "invoke"):
                    result = handler.invoke(arguments)
                else:
                    result = handler(**arguments)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                result = {"error": error}
        latency = (time.time() - start) * 1000.0
        return ToolCallTrace(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
            latency_ms=latency,
        )

    def _emit(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        if self.event_log is None:
            return
        self.event_log.record(
            event_type=event_type,
            component=f"sub_agent:{self.name}",
            details=details or {},
        )


def delegate(
    *,
    name: str,
    task: str,
    tool_schemas: list[dict[str, Any]] | None = None,
    tool_handlers: dict[str, Callable[..., Any]] | None = None,
    client: anthropic.Anthropic | None = None,
    context: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_turns: int = 8,
    result_model: type[BaseModel] | None = None,
    event_log: ContextEventLog | None = None,
) -> ExecutionResult:
    """Convenience function: construct and run a ``SubAgent`` in one call.

    For simple delegations where you don't need to hold a reference to the
    sub-agent object. For anything more complex (instrumentation, mid-run
    inspection), construct the ``SubAgent`` directly.
    """
    sub = SubAgent(
        name=name,
        task=task,
        context=context,
        tool_schemas=tool_schemas,
        tool_handlers=tool_handlers,
        client=client,
        model=model,
        max_turns=max_turns,
        result_model=result_model,
        event_log=event_log,
    )
    return sub.run()


def _serialize_tool_result(result: Any) -> str:
    """Anthropic's tool_result content is a string. Marshal anything we got."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
