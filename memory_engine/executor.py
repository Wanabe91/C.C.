from __future__ import annotations

from datetime import date
import sqlite3
from typing import Any

from .config import get_config
from .db import get_active_tasks, get_fact_by_id, list_recent_active_facts
from .models import PlanStep


def _normalize_constraint(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _build_weekly_review_message(args: dict[str, Any]) -> str:
    raw_lookback = args.get("lookback_facts")
    if isinstance(raw_lookback, int):
        lookback = raw_lookback
    elif isinstance(raw_lookback, str) and raw_lookback.strip().isdigit():
        lookback = int(raw_lookback.strip())
    else:
        lookback = 25
    try:
        recent_facts = list_recent_active_facts(limit=max(5, min(lookback, 100)))
        active_tasks = get_active_tasks()
    except sqlite3.OperationalError:
        return "Weekly review draft is unavailable because the memory database is not initialized yet."
    today = date.today()

    overdue_reviews: list[tuple[str, str]] = []
    decision_notes: list[str] = []
    inbox_candidates: list[str] = []
    for fact in recent_facts:
        meta = fact.meta if isinstance(fact.meta, dict) else {}
        kind = str(meta.get("kind") or "").strip().lower()
        review_date = _parse_iso_date(str(meta.get("review_date") or "").strip())
        if kind == "decision_log":
            decision_notes.append(fact.content)
            if review_date and review_date < today:
                overdue_reviews.append((fact.content, review_date.isoformat()))
        elif kind in {"inbox", "thought"}:
            inbox_candidates.append(fact.content)

    lines = [
        "Weekly review draft",
        f"- Active tasks: {len(active_tasks)}",
        f"- Recent active facts scanned: {len(recent_facts)}",
        "",
        "1) Overdue decision reviews",
    ]
    if overdue_reviews:
        lines.extend([f"- [OVERDUE since {due}] {content}" for content, due in overdue_reviews[:10]])
    else:
        lines.append("- None")

    lines.extend(["", "2) Inbox escalation check (today/tomorrow impact)"])
    if inbox_candidates:
        lines.extend([f"- Consider escalating: {item}" for item in inbox_candidates[:10]])
    else:
        lines.append("- No explicit inbox-tagged facts found")

    lines.extend(["", "3) Recent decision log entries"])
    if decision_notes:
        lines.extend([f"- {item}" for item in decision_notes[:10]])
    else:
        lines.append("- No decision_log entries found")

    lines.extend(["", "4) Active tasks snapshot"])
    if active_tasks:
        lines.extend([f"- #{task.id}: {task.title}" for task in active_tasks[:10]])
    else:
        lines.append("- No active tasks")

    lines.extend([
        "",
        "Next step: convert any inbox item that affects today/tomorrow into a decision_log fact with a review_date.",
    ])
    return "\n".join(lines)


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

    if tool_name == "generate_weekly_review":
        base_result["assistant_message"] = _build_weekly_review_message(args)
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
