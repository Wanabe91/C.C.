from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .config import get_config
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


def _copy_task(task: Task) -> Task:
    return Task(
        id=task.id,
        title=task.title,
        status=task.status,
        constraint=dict(task.constraint) if task.constraint else None,
        active_from_version=task.active_from_version,
        completed_version=task.completed_version,
    )


def _preview_text(text: str, limit: int = 140) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _stringify_payload(payload: Any) -> tuple[str, str]:
    if isinstance(payload, str):
        return payload, ".txt"
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), ".json"


@dataclass(frozen=True, slots=True)
class WorkingMemoryEntry:
    ref_id: str
    path: Path
    created_at: float
    source_tool: str | None
    event_id: int | None
    step_index: int | None
    field_name: str
    char_count: int
    preview: str

    def snapshot(self, *, include_path: bool = True) -> dict[str, Any]:
        payload = {
            "ref_id": self.ref_id,
            "created_at": self.created_at,
            "source_tool": self.source_tool,
            "event_id": self.event_id,
            "step_index": self.step_index,
            "field_name": self.field_name,
            "char_count": self.char_count,
            "preview": self.preview,
        }
        if include_path:
            payload["path"] = str(self.path)
        return payload


class WorkingMemory:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state_version = 0
        self._constraints: list[dict[str, Any]] = []
        self._tasks: list[Task] = []
        self._refs: dict[str, WorkingMemoryEntry] = {}
        self._session_id = uuid4().hex[:8]
        self._next_ref_number = 1

    @staticmethod
    def _storage_dir() -> Path:
        config = get_config()
        config.ensure_directories()
        return config.WORKING_MEMORY_PATH

    def _next_ref_id(self) -> str:
        ref_id = f"wm_{self._session_id}_{self._next_ref_number:04d}"
        self._next_ref_number += 1
        return ref_id

    def update(self, raw: dict[str, Any], version: int) -> None:
        with self._lock:
            self._state_version = version
            self._constraints = _normalize_constraints(raw.get("constraints"))
            self._tasks = _normalize_tasks(raw.get("tasks"))
            for task in self._tasks:
                task.active_from_version = version

    def get_active_tasks(self) -> list[Task]:
        with self._lock:
            return [_copy_task(task) for task in self._tasks]

    def get_constraints(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._constraints]

    def offload(
        self,
        payload: Any,
        *,
        source_tool: str | None,
        event_id: int | None,
        step_index: int | None,
        field_name: str,
    ) -> dict[str, Any]:
        text, suffix = _stringify_payload(payload)
        created_at = time.time()
        with self._lock:
            ref_id = self._next_ref_id()
            path = self._storage_dir() / f"{ref_id}{suffix}"
            path.write_text(text, encoding="utf-8")
            entry = WorkingMemoryEntry(
                ref_id=ref_id,
                path=path,
                created_at=created_at,
                source_tool=(str(source_tool).strip() or None) if source_tool is not None else None,
                event_id=event_id,
                step_index=step_index,
                field_name=str(field_name).strip() or "result",
                char_count=len(text),
                preview=_preview_text(text),
            )
            self._refs[ref_id] = entry
            return entry.snapshot()

    def placeholder_for(self, snapshot: dict[str, Any]) -> str:
        ref_id = str(snapshot.get("ref_id") or "").strip()
        source_tool = str(snapshot.get("source_tool") or "").strip()
        field_name = str(snapshot.get("field_name") or "result").strip() or "result"
        char_count = int(snapshot.get("char_count") or 0)
        tool_block = f" tool={source_tool}" if source_tool else ""
        return (
            f"[offloaded {field_name} to working memory ref={ref_id}{tool_block} "
            f"chars={char_count}; use grep_memory/read_memory to inspect it]"
        )

    def list_refs(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._refs.values())
        entries.sort(key=lambda item: (item.created_at, item.ref_id), reverse=True)
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return [entry.snapshot(include_path=False) for entry in entries]

    def grep(self, query: str, *, ref_id: str | None = None, limit: int | None = None) -> dict[str, Any]:
        needle = str(query).strip()
        if not needle:
            raise ValueError("query must not be empty.")
        config = get_config()
        resolved_limit = max(1, min(int(limit or config.WORKING_MEMORY_SEARCH_LIMIT), 20))
        lowered = needle.casefold()

        with self._lock:
            if ref_id is not None:
                entry = self._refs.get(str(ref_id).strip())
                entries = [entry] if entry is not None else []
            else:
                entries = list(self._refs.values())

        entries = [entry for entry in entries if entry is not None]
        entries.sort(key=lambda item: (item.created_at, item.ref_id), reverse=True)

        matches: list[dict[str, Any]] = []
        for entry in entries:
            text = entry.path.read_text(encoding="utf-8")
            position = text.casefold().find(lowered)
            if position < 0:
                continue
            start = max(0, position - 80)
            end = min(len(text), position + len(needle) + 80)
            snippet = _preview_text(text[start:end], limit=180)
            matches.append(
                {
                    "ref_id": entry.ref_id,
                    "source_tool": entry.source_tool,
                    "field_name": entry.field_name,
                    "char_count": entry.char_count,
                    "match_offset": position,
                    "snippet": snippet,
                }
            )
            if len(matches) >= resolved_limit:
                break

        return {
            "query": needle,
            "matches": matches,
            "searched_refs": len(entries),
        }

    def read(self, ref_id: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        normalized_ref_id = str(ref_id).strip()
        if not normalized_ref_id:
            raise ValueError("ref_id must not be empty.")
        config = get_config()
        resolved_limit = max(1, min(int(limit or config.WORKING_MEMORY_READ_CHAR_LIMIT), 20000))
        resolved_offset = max(0, int(offset))
        with self._lock:
            entry = self._refs.get(normalized_ref_id)
        if entry is None:
            raise ValueError(f"Unknown working memory ref '{normalized_ref_id}'.")

        text = entry.path.read_text(encoding="utf-8")
        chunk = text[resolved_offset : resolved_offset + resolved_limit]
        next_offset = resolved_offset + len(chunk)
        return {
            "ref_id": entry.ref_id,
            "source_tool": entry.source_tool,
            "field_name": entry.field_name,
            "offset": resolved_offset,
            "limit": resolved_limit,
            "next_offset": next_offset,
            "has_more": next_offset < len(text),
            "total_chars": len(text),
            "content": chunk,
        }


working_memory = WorkingMemory()
