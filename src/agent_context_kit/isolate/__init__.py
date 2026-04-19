"""Sub-agent isolation patterns.

When a task needs more context than one window can hold, or a different
skill set than the main agent, spin up an isolated sub-agent.

- ``SubAgent`` / ``delegate`` — spawn a sub-agent with clean context and
  structured return.
- ``StaticPlanner`` — run a pre-computed plan of steps through sub-agents.
- ``LLMPlanner`` — generate the plan via an LLM, then execute it.
- ``PlanStep`` / ``PlanResult`` / ``ExecutionResult`` — the data shapes
  sub-agents and planners produce.
"""

from agent_context_kit.isolate.delegate import (
    ExecutionResult,
    SubAgent,
    ToolCallTrace,
    delegate,
)
from agent_context_kit.isolate.planner import (
    LLMPlanner,
    PlanResult,
    PlanStep,
    StaticPlanner,
)

__all__ = [
    "ExecutionResult",
    "LLMPlanner",
    "PlanResult",
    "PlanStep",
    "StaticPlanner",
    "SubAgent",
    "ToolCallTrace",
    "delegate",
]
