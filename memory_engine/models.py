from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Event:
    id: int
    ts: float
    raw_json: dict[str, Any]
    state_version: int


@dataclass(slots=True)
class Fact:
    id: int
    content: str
    embedding_id: str | None
    version_created: int
    version_superseded: int | None
    status: str
    source_event_id: int | None = None
    importance: str = "contextual"
    tier: str = "active"
    created_at: float | None = None
    last_accessed_at: float | None = None
    access_count: int = 0
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Task:
    id: int
    title: str
    status: str
    constraint: dict[str, Any] | None
    active_from_version: int
    completed_version: int | None


@dataclass(slots=True)
class PlanStep:
    action: str
    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    precondition_fact_ids: list[int] = field(default_factory=list)
    reasoning: str = ""


@dataclass(slots=True)
class PlannerRun:
    goal: str
    snapshot: dict[str, Any]
    system_prompt: str
    user_prompt: str
    planner_status: str
    steps: list[PlanStep] = field(default_factory=list)
    first_response: str = ""
    repair_prompt: str | None = None
    repair_response: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ContextSnapshot:
    state_version: int
    vector_watermark: int
    tasks: list[Task]
    constraints: list[dict[str, Any]]
    fts_results: list[Fact]
    vector_results: list[Fact]
    delta_facts: list[Fact]
    recent_messages: list[dict[str, Any]]
