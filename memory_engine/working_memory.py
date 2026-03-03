from __future__ import annotations

from threading import Lock
from typing import Any

from .models import Task


def _normalize_constraints(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _normalize_tasks(raw: Any) -> list[Task]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    tasks: list[Task] = []
    transient_id = -1
    for item in raw:
        if isinstance(item, str):
            title = item.strip()
            constraint = None
        elif isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            constraint = item.get("constraint_json") or item.get("constraint")
        else:
            continue
        if not title:
            continue
        tasks.append(
            Task(
                id=transient_id,
                title=title,
                status="active",
                constraint=constraint if isinstance(constraint, dict) else None,
                active_from_version=0,
                completed_version=None,
            )
        )
        transient_id -= 1
    return tasks


class WorkingMemory:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state_version = 0
        self._constraints: list[dict[str, Any]] = []
        self._tasks: list[Task] = []

    def update(self, raw: dict[str, Any], version: int) -> None:
        with self._lock:
            self._state_version = version
            self._constraints = _normalize_constraints(raw.get("constraints"))
            self._tasks = _normalize_tasks(raw.get("tasks"))
            for task in self._tasks:
                task.active_from_version = version

    def get_active_tasks(self) -> list[Task]:
        with self._lock:
            return [
                Task(
                    id=task.id,
                    title=task.title,
                    status=task.status,
                    constraint=dict(task.constraint) if task.constraint else None,
                    active_from_version=task.active_from_version,
                    completed_version=task.completed_version,
                )
                for task in self._tasks
            ]

    def get_constraints(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._constraints]


working_memory = WorkingMemory()
