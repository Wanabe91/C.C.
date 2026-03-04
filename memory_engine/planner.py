from __future__ import annotations

import json
import re
from typing import Any

from .config import get_config
from .llm import llm_call
from .models import ContextSnapshot, Fact, PlannerRun, PlanStep, Task

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
    "Allowed tools: respond, remember_fact, create_task, complete_task, generate_weekly_review, noop.\n"
    "Prefer short, deterministic plans.\n"
    "Use remember_fact for durable knowledge. For decisions, encode structured metadata such as "
    "meta.kind='decision', meta.title, meta.context, meta.rationale, meta.alternatives, and meta.tags.\n"
    "When using remember_fact, args may also include importance (core|contextual|transient) to control memory tiering.\n"
    "Use generate_weekly_review to draft a weekly Markdown review from stored facts, tasks, and planner traces. "
    "Its args may include week_start (YYYY-MM-DD), week_offset (integer weeks from current week), title, and focus.\n"
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
        "importance": fact.importance,
        "tier": fact.tier,
        "version_created": fact.version_created,
        "version_superseded": fact.version_superseded,
        "source_event_id": fact.source_event_id,
        "created_at": fact.created_at,
        "last_accessed_at": fact.last_accessed_at,
        "access_count": fact.access_count,
        "meta": fact.meta,
    }


def snapshot_payload(ctx: ContextSnapshot, goal: str) -> dict[str, Any]:
    return {
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


def _planner_user_prompt(ctx: ContextSnapshot, goal: str) -> str:
    payload = snapshot_payload(ctx, goal)
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


def serialize_step(step: PlanStep) -> dict[str, Any]:
    return {
        "action": step.action,
        "tool": step.tool,
        "args": step.args,
        "precondition_fact_ids": step.precondition_fact_ids,
        "reasoning": step.reasoning,
    }


def _find_json_fragment(raw: str) -> str | None:
    start = -1
    stack: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(raw):
        if start < 0:
            if char not in "{[":
                continue
            start = index
            stack.append("}" if char == "{" else "]")
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue

        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return raw[start : index + 1]

    return None


def _normalize_planner_output(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in (cleaned, raw.strip()):
        if not candidate:
            continue
        if candidate[:1] in "{[":
            return candidate
        fragment = _find_json_fragment(candidate)
        if fragment is not None:
            return fragment
    raise ValueError("Planner output did not contain a JSON object.")


def _parse_steps(raw: str) -> list[PlanStep]:
    payload = json.loads(_normalize_planner_output(raw))
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
    message = (
        "I need a bit more detail to continue."
        if not goal.strip()
        else "I couldn't produce a structured response for that request. Please try again."
    )
    return [
        PlanStep(
            action="respond",
            tool="respond",
            args={"message": message},
            precondition_fact_ids=[],
            reasoning="Fallback plan after planner output could not be parsed.",
        )
    ]


def plan(ctx: ContextSnapshot, goal: str) -> PlannerRun:
    prompt = _planner_user_prompt(ctx, goal)
    system_prompt = _planner_system_prompt()
    snapshot = snapshot_payload(ctx, goal)
    first_response = llm_call(system_prompt, prompt, schema=PLANNER_JSON_SCHEMA)
    try:
        steps = _parse_steps(first_response)
        return PlannerRun(
            goal=goal,
            snapshot=snapshot,
            system_prompt=system_prompt,
            user_prompt=prompt,
            planner_status="ok",
            steps=steps,
            first_response=first_response,
        )
    except (json.JSONDecodeError, ValueError) as first_error:
        repair_prompt = (
            "Repair the following output into valid JSON with the exact required structure.\n"
            f"{first_response}"
        )
        second_response = llm_call(system_prompt, repair_prompt, schema=PLANNER_JSON_SCHEMA)
        try:
            steps = _parse_steps(second_response)
            return PlannerRun(
                goal=goal,
                snapshot=snapshot,
                system_prompt=system_prompt,
                user_prompt=prompt,
                planner_status="repaired",
                steps=steps,
                first_response=first_response,
                repair_prompt=repair_prompt,
                repair_response=second_response,
                error=str(first_error),
            )
        except (json.JSONDecodeError, ValueError) as second_error:
            return PlannerRun(
                goal=goal,
                snapshot=snapshot,
                system_prompt=system_prompt,
                user_prompt=prompt,
                planner_status="fallback",
                steps=_fallback_plan(goal),
                first_response=first_response,
                repair_prompt=repair_prompt,
                repair_response=second_response,
                error=f"{first_error}; {second_error}",
            )
