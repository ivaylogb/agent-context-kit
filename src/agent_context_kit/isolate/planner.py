"""Planner → executor orchestration.

A planner reads a complex task, breaks it into discrete steps, and
delegates each step to an executor sub-agent. Results flow back to the
planner; it synthesizes them into the final answer.

Why the separation matters:

- **The planner is stateful.** It holds the overall task, the step list,
  and results as they come back. Its context grows, but with summaries
  of results rather than verbatim sub-agent transcripts.
- **Executors are stateless.** Each executor gets exactly one step, a
  clean context, the tools the step needs — and returns a structured
  result. No cross-step contamination.
- **Synthesis is a final pass.** Once all steps are done, the planner
  composes the answer from results. The synthesis step is where domain-
  specific formatting happens.

Two planners live here:

- ``StaticPlanner`` — takes a pre-computed plan (list of step dicts) and
  executes them in order. Use for workflows where the breakdown is
  deterministic (no need to pay for an LLM planning pass).
- ``LLMPlanner`` — uses an LLM to produce the plan from the task
  description. More flexible but costs one Haiku call upfront.

Both use ``SubAgent`` under the hood for execution.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import anthropic
from pydantic import BaseModel, Field

from agent_context_kit.isolate.delegate import (
    ExecutionResult,
    SubAgent,
)
from agent_context_kit.observability import ContextEventLog


class PlanStep(BaseModel):
    """One step in a plan.

    ``tools`` is a list of tool names the executor may call; the planner
    looks them up in the provided tool-handler registry. Steps can
    reference prior steps' outputs via the ``{step_NAME}`` template
    placeholder — the planner substitutes the placeholder with the
    prior step's output before dispatching.
    """

    name: str
    task: str
    tools: list[str] = Field(default_factory=list)
    # Keys in ``PlanResult.step_outputs`` whose outputs this step depends
    # on. Dependencies are substituted into ``task`` via ``{step_NAME}``
    # placeholders.
    depends_on: list[str] = Field(default_factory=list)
    model: str = "claude-haiku-4-5-20251001"
    max_turns: int = 6


class PlanResult(BaseModel):
    """Output of a full plan execution."""

    ok: bool
    task: str
    step_outputs: dict[str, Any] = Field(default_factory=dict)
    step_results: list[ExecutionResult] = Field(default_factory=list)
    synthesis: str = ""
    errors: list[str] = Field(default_factory=list)


PLANNER_SYSTEM = (
    "You are a task planner. Given a user task and a catalogue of tools, "
    "break the task into discrete steps. Each step has a name, a concrete "
    "sub-task description, the tools it may call, and any prior step names "
    "it depends on.\n\n"
    "Rules:\n"
    "- Each step should be something a focused sub-agent can accomplish.\n"
    "- Keep steps independent where possible — parallel execution is ideal.\n"
    "- When a step references a prior step's output, list that step in "
    "  ``depends_on`` and cite it in the step's task as ``{step_NAME}``.\n"
    "- Output strict JSON. No prose, no markdown fences.\n\n"
    "Output format:\n"
    '{{"steps": [{{"name": "...", "task": "...", "tools": [...], '
    '"depends_on": [...]}}, ...]}}\n\n'
    "Available tools:\n{tools}\n"
)


SYNTHESIZER_SYSTEM = (
    "You are a synthesizer. Given the original user task and the outputs "
    "of the steps that addressed it, compose a final answer. Cite facts "
    "from the step outputs; don't fabricate. If a step failed, note what "
    "didn't work and what's still known."
)


class StaticPlanner:
    """Executor for pre-computed plans.

    Use this when the breakdown is deterministic. It's cheap (no LLM
    planning pass) and reproducible — the same plan on the same inputs
    returns the same structure.
    """

    def __init__(
        self,
        *,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        client: anthropic.Anthropic | None = None,
        event_log: ContextEventLog | None = None,
        synthesizer: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        # Stored as dicts keyed by tool name so steps can pull just the
        # subset they need.
        self.tool_schemas = dict(tool_schemas or {})
        self.tool_handlers = dict(tool_handlers or {})
        self.client = client or anthropic.Anthropic()
        self.event_log = event_log
        self.synthesizer = synthesizer

    def execute(
        self,
        task: str,
        plan: Sequence[PlanStep],
    ) -> PlanResult:
        """Run every step in the plan, with dependency-aware substitution.

        Steps are executed in the order given; dependency references are
        resolved via placeholder substitution at dispatch time. If a step
        fails (executor returns ``ok=False``), downstream steps that
        depend on it are skipped and the plan's ``ok`` becomes False.
        """
        result = PlanResult(ok=True, task=task)
        completed_outputs: dict[str, Any] = {}
        for step in plan:
            self._emit(
                "plan_step_started",
                details={"step": step.name, "task": step.task[:120]},
            )
            # Skip if any upstream dependency failed.
            missing = [d for d in step.depends_on if d not in completed_outputs]
            if missing:
                msg = (
                    f"step '{step.name}' skipped — missing dependencies: "
                    f"{missing}"
                )
                result.errors.append(msg)
                result.ok = False
                self._emit("plan_step_skipped", details={"step": step.name})
                continue

            resolved_task = _substitute(step.task, completed_outputs)
            sub_schemas = [
                self.tool_schemas[t]
                for t in step.tools
                if t in self.tool_schemas
            ]
            sub_handlers = {
                t: self.tool_handlers[t]
                for t in step.tools
                if t in self.tool_handlers
            }
            sub = SubAgent(
                name=step.name,
                task=resolved_task,
                tool_schemas=sub_schemas,
                tool_handlers=sub_handlers,
                client=self.client,
                model=step.model,
                max_turns=step.max_turns,
                event_log=self.event_log,
            )
            exec_result = sub.run()
            result.step_results.append(exec_result)
            if exec_result.ok:
                completed_outputs[step.name] = exec_result.output
                result.step_outputs[step.name] = exec_result.output
            else:
                result.ok = False
                result.errors.extend(exec_result.errors)
                self._emit(
                    "plan_step_failed",
                    details={"step": step.name, "errors": exec_result.errors},
                )

        # Synthesis — either caller-supplied or an LLM pass.
        if completed_outputs:
            if self.synthesizer is not None:
                result.synthesis = self.synthesizer(task, completed_outputs)
            else:
                result.synthesis = self._llm_synthesize(task, completed_outputs)
        self._emit(
            "plan_complete",
            details={"ok": result.ok, "steps": len(plan)},
        )
        return result

    def _llm_synthesize(
        self,
        task: str,
        step_outputs: dict[str, Any],
    ) -> str:
        """Default synthesis pass: ask a cheap model to compose the final answer."""
        dumped = json.dumps(step_outputs, indent=2, default=str)
        user = (
            f"Original task:\n{task}\n\nStep outputs (JSON):\n{dumped}\n\n"
            "Compose the final answer."
        )
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                temperature=0.0,
                system=SYNTHESIZER_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            )
            return text.strip() or dumped
        except Exception:  # noqa: BLE001 — fall back to the raw dump
            return dumped

    def _emit(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        if self.event_log is None:
            return
        self.event_log.record(
            event_type=event_type,
            component="planner",
            details=details or {},
        )


class LLMPlanner(StaticPlanner):
    """Planner that generates the plan via an LLM before executing it.

    Uses a cheap model (Haiku by default) to produce the plan. The plan
    is then executed via ``StaticPlanner.execute``. Falls back to a
    single-step plan ("do the whole task in one shot") if the planning
    call fails — better to attempt the task than to block on a planner
    outage.
    """

    def __init__(
        self,
        *,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        client: anthropic.Anthropic | None = None,
        event_log: ContextEventLog | None = None,
        synthesizer: Callable[[str, dict[str, Any]], str] | None = None,
        planning_model: str = "claude-haiku-4-5-20251001",
        planning_max_tokens: int = 1500,
    ) -> None:
        super().__init__(
            tool_schemas=tool_schemas,
            tool_handlers=tool_handlers,
            client=client,
            event_log=event_log,
            synthesizer=synthesizer,
        )
        self.planning_model = planning_model
        self.planning_max_tokens = planning_max_tokens

    def plan(self, task: str) -> list[PlanStep]:
        """Ask the LLM to produce a plan. Returns a list of ``PlanStep``."""
        tool_lines = [
            f"- {name}: {schema.get('description', '').splitlines()[0]}"
            for name, schema in self.tool_schemas.items()
        ]
        menu = "\n".join(tool_lines) if tool_lines else "(no tools)"
        system = PLANNER_SYSTEM.format(tools=menu)
        try:
            response = self.client.messages.create(
                model=self.planning_model,
                max_tokens=self.planning_max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": task}],
            )
            raw = "".join(
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            )
        except Exception:  # noqa: BLE001 — fall back to single-step
            return [self._single_step_fallback(task)]

        parsed = self._parse_plan(raw)
        if not parsed:
            return [self._single_step_fallback(task)]
        return parsed

    def plan_and_execute(self, task: str) -> PlanResult:
        """Plan, then execute in one call."""
        plan = self.plan(task)
        self._emit("plan_generated", details={"steps": [s.name for s in plan]})
        return self.execute(task, plan)

    def _parse_plan(self, text: str) -> list[PlanStep]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data: Any = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return []
        steps: list[PlanStep] = []
        for rs in raw_steps:
            if not isinstance(rs, dict):
                continue
            try:
                # Filter out tool names that aren't registered so we don't
                # dispatch to missing handlers.
                tools = [
                    t for t in rs.get("tools", [])
                    if isinstance(t, str) and t in self.tool_schemas
                ]
                steps.append(
                    PlanStep(
                        name=rs.get("name", "step"),
                        task=rs.get("task", ""),
                        tools=tools,
                        depends_on=[
                            d for d in rs.get("depends_on", [])
                            if isinstance(d, str)
                        ],
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return steps

    def _single_step_fallback(self, task: str) -> PlanStep:
        return PlanStep(
            name="direct",
            task=task,
            tools=list(self.tool_schemas.keys()),
        )


def _substitute(template: str, values: dict[str, Any]) -> str:
    """Substitute ``{step_NAME}`` placeholders with prior-step outputs.

    Values are serialized via ``str()`` for scalars and ``json.dumps`` for
    lists/dicts. Missing placeholders are left verbatim so the sub-agent
    can see what was expected even if the upstream failed.
    """
    for key, value in values.items():
        token = f"{{step_{key}}}"
        if token not in template:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, default=str)
        else:
            rendered = str(value)
        template = template.replace(token, rendered)
    return template
