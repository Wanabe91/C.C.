from __future__ import annotations

from typing import Any

from .config import get_config
from .db import get_fact_by_id
from .models import PlanStep


def _normalize_constraint(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


async def execute_step(step: PlanStep) -> dict[str, Any]:
    tool_name = step.tool or step.action
    args = step.args if isinstance(step.args, dict) else {}
    base_result = {
        "status": "ok",
        "assistant_message": None,
        "facts": [],
        "created_tasks": [],
        "completed_task_ids": [],
        "meta": {"action": step.action, "tool": tool_name},
    }

    if tool_name == "respond":
        message = str(
            args.get("message")
            or args.get("content")
            or args.get("text")
            or step.reasoning
            or "I have no additional response."
        ).strip()
        base_result["assistant_message"] = message
        return base_result

    if tool_name == "remember_fact":
        content = str(args.get("content") or args.get("fact") or "").strip()
        if not content:
            return {
                **base_result,
                "status": "error",
                "meta": {**base_result["meta"], "error": "empty_fact"},
            }
        base_result["facts"] = [{"content": content, "meta": _normalize_constraint(args.get("meta")) or {}}]
        return base_result

    if tool_name == "create_task":
        title = str(args.get("title") or args.get("task") or args.get("name") or "").strip()
        if not title:
            return {
                **base_result,
                "status": "error",
                "meta": {**base_result["meta"], "error": "empty_task"},
            }
        base_result["created_tasks"] = [
            {
                "title": title,
                "constraint_json": _normalize_constraint(
                    args.get("constraint_json") or args.get("constraint")
                ),
            }
        ]
        return base_result

    if tool_name == "complete_task":
        if "task_id" in args:
            task_ids = [int(args["task_id"])]
        else:
            task_ids = [int(item) for item in args.get("task_ids", [])]
        base_result["completed_task_ids"] = task_ids
        return base_result

    if tool_name == "noop":
        return base_result

    return {
        **base_result,
        "status": "error",
        "assistant_message": f"Unsupported tool: {tool_name}",
        "meta": {**base_result["meta"], "error": "unsupported_tool"},
    }


def revalidate(step: PlanStep, current_V: int, snap_V: int) -> bool:
    config = get_config()
    if current_V - snap_V > config.VERSION_DRIFT_THRESHOLD:
        return False
    for fid in step.precondition_fact_ids:
        fact = get_fact_by_id(fid)
        if fact is None or fact.status != "active":
            return False
    return True
