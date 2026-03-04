from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from .db import (
    count_events_between,
    get_active_tasks,
    get_planner_activity_between,
    list_facts_created_between,
    list_messages_between,
    list_tasks_completed_between,
    list_tasks_created_between,
)


def _local_today() -> date:
    return datetime.now().astimezone().date()


def _week_start_for(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _coerce_week_start(args: dict[str, Any]) -> date:
    raw_start = str(args.get("week_start") or "").strip()
    if raw_start:
        return date.fromisoformat(raw_start)

    raw_offset = args.get("week_offset", 0)
    if isinstance(raw_offset, str) and raw_offset.strip().lstrip("-").isdigit():
        raw_offset = int(raw_offset)
    if not isinstance(raw_offset, int):
        raw_offset = 0

    return _week_start_for(_local_today()) + timedelta(days=raw_offset * 7)


def _decision_kind(meta: dict[str, Any]) -> bool:
    kind = str(meta.get("kind") or meta.get("type") or "").strip().lower()
    return kind == "decision"


def _decision_title(fact: dict[str, Any]) -> str:
    meta = fact.get("meta") if isinstance(fact.get("meta"), dict) else {}
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    content = str(fact.get("content") or "").strip()
    return content[:120] + ("..." if len(content) > 120 else "")


def _task_title(task: dict[str, Any]) -> str:
    return str(task.get("title") or "").strip()


def _count_messages_by_role(messages: list[dict[str, Any]], role: str) -> int:
    return sum(message.get("role") == role for message in messages)


def generate_weekly_review(args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    week_start = _coerce_week_start(args)
    week_end = week_start + timedelta(days=7)
    tz = datetime.now().astimezone().tzinfo
    start_ts = datetime.combine(week_start, time.min, tzinfo=tz).timestamp()
    end_ts = datetime.combine(week_end, time.min, tzinfo=tz).timestamp()
    week_info = week_start.isocalendar()
    week_key = f"{week_info.year}-W{week_info.week:02d}"

    facts = list_facts_created_between(start_ts, end_ts)
    decisions = [fact for fact in facts if _decision_kind(fact.get("meta") or {})]
    general_facts = [fact for fact in facts if fact not in decisions]
    tasks_created = list_tasks_created_between(start_ts, end_ts)
    tasks_completed = list_tasks_completed_between(start_ts, end_ts)
    open_tasks = get_active_tasks()
    messages = list_messages_between(start_ts, end_ts)
    planner_activity = get_planner_activity_between(start_ts, end_ts)
    event_count = count_events_between(start_ts, end_ts)
    title = str(args.get("title") or f"Weekly Review {week_key}").strip()
    focus = str(args.get("focus") or "").strip()
    user_message_count = _count_messages_by_role(messages, "user")
    assistant_message_count = _count_messages_by_role(messages, "assistant")

    recent_user_messages = [
        str(message["content"]).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ][:5]

    summary = {
        "week_key": week_key,
        "period_start": week_start.isoformat(),
        "period_end": week_end.isoformat(),
        "event_count": event_count,
        "planner_runs": planner_activity["run_count"],
        "planner_repaired_runs": planner_activity["repaired_count"],
        "planner_fallback_runs": planner_activity["fallback_count"],
        "planner_rejections": planner_activity["rejections"],
        "facts_captured": len(facts),
        "decisions_logged": len(decisions),
        "tasks_created": len(tasks_created),
        "tasks_completed": len(tasks_completed),
        "open_tasks": len(open_tasks),
        "user_messages": user_message_count,
        "assistant_messages": assistant_message_count,
        "focus": focus,
    }

    lines = [
        f"# {title}",
        "",
        f"- Week: `{week_key}`",
        f"- Period: `{week_start.isoformat()}` -> `{week_end.isoformat()}`",
        f"- Events processed: `{event_count}`",
        f"- Planner runs: `{planner_activity['run_count']}`",
        f"- Planner repairs: `{planner_activity['repaired_count']}`",
        f"- Planner fallbacks: `{planner_activity['fallback_count']}`",
        f"- Facts captured: `{len(facts)}`",
        f"- Decisions logged: `{len(decisions)}`",
        f"- Tasks created: `{len(tasks_created)}`",
        f"- Tasks completed: `{len(tasks_completed)}`",
        f"- Open tasks now: `{len(open_tasks)}`",
    ]

    if focus:
        lines.extend(["", "## Focus", "", focus])

    lines.extend(["", "## Inbox Signals", ""])
    if recent_user_messages:
        lines.extend([f"- {message}" for message in recent_user_messages])
    else:
        lines.append("- No user messages captured in this period.")

    lines.extend(["", "## Decisions Logged", ""])
    if decisions:
        lines.extend([f"- {_decision_title(fact)}" for fact in decisions[:10]])
    else:
        lines.append("- No structured decisions were logged this week.")

    lines.extend(["", "## Facts Captured", ""])
    if general_facts:
        lines.extend(
            [
                f"- {str(fact.get('content') or '').strip()}"
                for fact in general_facts[:10]
                if str(fact.get("content") or "").strip()
            ]
        )
    else:
        lines.append("- No non-decision facts were captured this week.")

    lines.extend(["", "## Tasks", ""])
    if tasks_created:
        lines.append("### Created")
        lines.append("")
        lines.extend([f"- {_task_title(task)}" for task in tasks_created[:10] if _task_title(task)])
        lines.append("")
    if tasks_completed:
        lines.append("### Completed")
        lines.append("")
        lines.extend([f"- {_task_title(task)}" for task in tasks_completed[:10] if _task_title(task)])
        lines.append("")
    if open_tasks:
        lines.append("### Still Open")
        lines.append("")
        lines.extend([f"- {task.title}" for task in open_tasks[:10] if task.title])
        lines.append("")
    if not any((tasks_created, tasks_completed, open_tasks)):
        lines.append("- No task activity recorded.")

    lines.extend(["## Planner Diagnostics", ""])
    if planner_activity["rejections"]:
        lines.extend(
            [f"- {item['reason']}: {item['count']}" for item in planner_activity["rejections"][:10]]
        )
    else:
        lines.append("- No rejected steps recorded in this period.")

    lines.extend(
        [
            "",
            "## Draft Conclusions",
            "",
            "- Pattern:",
            "- Risk:",
            "- Next focus:",
        ]
    )

    return {
        "week_key": week_key,
        "title": title,
        "summary": summary,
        "markdown": "\n".join(lines).strip(),
    }
