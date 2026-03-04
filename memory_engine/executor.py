from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel

from .config import get_config
from .db import get_fact_by_id
from .models import ValidatedPlanStep
from .tool_registry import get_tool


def _error_result(step: ValidatedPlanStep, error_code: str, details: str, assistant_message: str | None) -> dict[str, Any]:
    return {
        "status": "ok",
        "assistant_message": assistant_message,
        "facts": [],
        "created_tasks": [],
        "completed_task_ids": [],
        "generated_reviews": [],
        "meta": {
            "action": step.action,
            "tool": step.tool,
            "error": error_code,
            "details": details,
        },
    }


def _validate_step_args(step: ValidatedPlanStep, tool_definition) -> BaseModel:
    if isinstance(step.args, tool_definition.args_schema):
        return step.args
    raw_args = step.args.model_dump(exclude_none=True) if isinstance(step.args, BaseModel) else step.args
    return tool_definition.args_schema.model_validate(raw_args)


def _attach_step_meta(result: dict[str, Any], step: ValidatedPlanStep) -> dict[str, Any]:
    result["meta"] = {
        **dict(result.get("meta") or {}),
        "action": step.action,
        "tool": step.tool,
    }
    return result


async def execute_step(step: ValidatedPlanStep) -> dict[str, Any]:
    tool_definition = get_tool(step.tool)
    if tool_definition is None:
        return {
            **_error_result(
                step,
                "unsupported_tool",
                f"Unsupported tool: {step.tool}",
                f"Unsupported tool: {step.tool}",
            ),
            "status": "error",
        }

    try:
        validated_args = _validate_step_args(step, tool_definition)
    except Exception as exc:
        return {
            **_error_result(
                step,
                "invalid_args",
                str(exc),
                f"Invalid args for tool: {step.tool}",
            ),
            "status": "error",
        }

    result = tool_definition.handler(validated_args)
    if inspect.isawaitable(result):
        result = await result
    return _attach_step_meta(result, step)


def revalidate(step: ValidatedPlanStep, current_V: int, snap_V: int) -> tuple[bool, str | None]:
    config = get_config()
    drift = current_V - snap_V
    if drift > config.VERSION_DRIFT_THRESHOLD:
        return (
            False,
            f"state_version_drift:{drift}:threshold:{config.VERSION_DRIFT_THRESHOLD}",
        )
    for fid in step.precondition_fact_ids:
        try:
            fact_id = int(fid)
        except (TypeError, ValueError):
            return False, f"invalid_precondition_fact:{fid}"
        fact = get_fact_by_id(fact_id)
        if fact is None:
            return False, f"missing_precondition_fact:{fact_id}"
        if fact.status != "active":
            return False, f"stale_precondition_fact:{fact_id}:status:{fact.status}"
    return True, None
