from __future__ import annotations

import json
from typing import Any

from .config import get_config
from .llm import llm_call
from .models import ContextSnapshot, Fact, PlanStep, Task

PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "tool": {"type": ["string", "null"]},
                    "args": {"type": "object"},
                    "precondition_fact_ids": {"type": "array", "items": {"type": "integer"}},
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "tool", "args", "precondition_fact_ids", "reasoning"],
            },
        }
    },
    "required": ["steps"],
}

PLANNER_SYSTEM_PROMPT = (
    "You are the planning engine for a persistent local AI assistant.\n"
    "Output only valid JSON.\n"
    "Allowed tools: respond, remember_fact, create_task, complete_task, noop.\n"
    "Prefer short, deterministic plans.\n"
    "If the user only needs a direct answer, emit one respond step."
)


def _planner_system_prompt() -> str:
    assistant_prompt = get_config().ASSISTANT_SYSTEM_PROMPT
    if not assistant_prompt:
        return PLANNER_SYSTEM_PROMPT
    return (
        f"{PLANNER_SYSTEM_PROMPT}\n"
        "When emitting a respond step, follow this assistant style instruction:\n"
        f"{assistant_prompt}"
    )


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "constraint": task.constraint,
        "active_from_version": task.active_from_version,
        "completed_version": task.completed_version,
    }


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "content": fact.content,
        "status": fact.status,
        "version_created": fact.version_created,
        "version_superseded": fact.version_superseded,
        "source_event_id": fact.source_event_id,
        "meta": fact.meta,
    }


def _planner_user_prompt(ctx: ContextSnapshot, goal: str) -> str:
    payload = {
        "goal": goal,
        "state_version": ctx.state_version,
        "vector_watermark": ctx.vector_watermark,
        "tasks": [_task_payload(task) for task in ctx.tasks],
        "constraints": ctx.constraints,
        "fts_results": [_fact_payload(fact) for fact in ctx.fts_results],
        "vector_results": [_fact_payload(fact) for fact in ctx.vector_results],
        "delta_facts": [_fact_payload(fact) for fact in ctx.delta_facts],
        "recent_messages": ctx.recent_messages,
    }
    return (
        "Return JSON using this exact structure:\n"
        "{\n"
        '  "steps": [{\n'
        '    "action": "string",\n'
        '    "tool": "string or null",\n'
        '    "args": {},\n'
        '    "precondition_fact_ids": [],\n'
        '    "reasoning": "string"\n'
        "  }]\n"
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_steps(raw: str) -> list[PlanStep]:
    payload = json.loads(raw)
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Planner output did not include a steps array.")
    steps: list[PlanStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        raw_preconditions = item.get("precondition_fact_ids") or []
        steps.append(
            PlanStep(
                action=str(item.get("action", "")).strip() or "respond",
                tool=item.get("tool") if item.get("tool") is None else str(item.get("tool")).strip(),
                args=item.get("args") if isinstance(item.get("args"), dict) else {},
                precondition_fact_ids=[
                    int(fid)
                    for fid in raw_preconditions
                    if isinstance(fid, int) or (isinstance(fid, str) and fid.strip().isdigit())
                ],
                reasoning=str(item.get("reasoning", "")).strip(),
            )
        )
    if not steps:
        raise ValueError("Planner returned no usable steps.")
    return steps


def _fallback_plan(goal: str) -> list[PlanStep]:
    message = "I need a bit more detail to continue." if not goal.strip() else f"I can help with: {goal}"
    return [
        PlanStep(
            action="respond",
            tool="respond",
            args={"message": message},
            precondition_fact_ids=[],
            reasoning="Fallback plan after planner output could not be parsed.",
        )
    ]


def plan(ctx: ContextSnapshot, goal: str) -> list[PlanStep]:
    prompt = _planner_user_prompt(ctx, goal)
    system_prompt = _planner_system_prompt()
    first_response = llm_call(system_prompt, prompt, schema=PLANNER_JSON_SCHEMA)
    try:
        return _parse_steps(first_response)
    except (json.JSONDecodeError, ValueError):
        repair_prompt = (
            "Repair the following output into valid JSON with the exact required structure.\n"
            f"{first_response}"
        )
        second_response = llm_call(system_prompt, repair_prompt, schema=PLANNER_JSON_SCHEMA)
        try:
            return _parse_steps(second_response)
        except (json.JSONDecodeError, ValueError):
            return _fallback_plan(goal)
