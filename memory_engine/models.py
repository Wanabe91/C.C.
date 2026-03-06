from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, ValidationError, model_validator


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
class ContextFingerprint:
    fact_versions: dict[str, tuple[int, str, str]]
    active_task_ids: set[str]
    last_summary_id: str | None
    last_message_id: str | None


@dataclass(slots=True)
class FingerprintDiff:
    changed_fact_ids: list[str]
    removed_fact_ids: list[str]
    task_changes: bool
    message_changes: bool
    is_empty: bool


def _coerce_precondition_fact_ids(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError("precondition_fact_ids must be an array of fact id strings.")

    normalized: list[str] = []
    for index, item in enumerate(raw_value):
        if isinstance(item, bool):
            raise ValueError(f"precondition_fact_ids[{index}] must be a string or integer fact id.")
        if isinstance(item, int):
            if item <= 0:
                raise ValueError(f"precondition_fact_ids[{index}] must be a positive fact id.")
            normalized.append(str(item))
            continue
        if isinstance(item, str):
            candidate = item.strip()
            if not candidate or not candidate.isdigit():
                raise ValueError(f"precondition_fact_ids[{index}] must be a numeric fact id string.")
            normalized.append(str(int(candidate)))
            continue
        raise ValueError(f"precondition_fact_ids[{index}] must be a string or integer fact id.")
    return normalized


class ValidatedPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    tool: str
    args: SerializeAsAny[BaseModel]
    precondition_fact_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="before")
    @classmethod
    def _validate_against_registry(cls, data: Any) -> dict[str, Any]:
        from .tool_registry import get_tool

        if not isinstance(data, dict):
            raise ValueError("Each plan step must be an object.")

        raw_action = str(data.get("action") or "").strip()
        raw_tool = data.get("tool")
        tool_name = raw_action
        if raw_tool is not None:
            tool_name = str(raw_tool).strip()
        if not tool_name:
            raise ValueError("tool is required.")

        tool_definition = get_tool(tool_name)
        if tool_definition is None:
            raise ValueError(f"Unknown tool '{tool_name}'.")

        raw_args = data.get("args")
        if not isinstance(raw_args, dict):
            raise ValueError(f"args for tool '{tool_definition.name}' must be an object.")
        try:
            validated_args = tool_definition.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            message = "; ".join(error.get("msg", str(exc)) for error in exc.errors(include_url=False)) or str(exc)
            raise ValueError(f"args for tool '{tool_definition.name}' are invalid: {message}") from exc

        return {
            "action": raw_action or tool_definition.name,
            "tool": tool_definition.name,
            "args": validated_args,
            # Planner validation owns the raw-to-string coercion so execution never sees ambiguous ids.
            "precondition_fact_ids": _coerce_precondition_fact_ids(data.get("precondition_fact_ids")),
            "reasoning": str(data.get("reasoning") or "").strip(),
        }

    def args_dict(self) -> dict[str, Any]:
        return self.args.model_dump(exclude_none=True, exclude_defaults=True)


@dataclass(slots=True)
class PlannerRun:
    goal: str
    snapshot: dict[str, Any]
    system_prompt: str
    user_prompt: str
    planner_status: str
    steps: list[ValidatedPlanStep] = field(default_factory=list)
    first_response: str = ""
    repair_prompt: str | None = None
    repair_response: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ContextSnapshot:
    state_version: int
    event_version: int
    vector_watermark: int
    fingerprint: ContextFingerprint
    tasks: list[Task]
    constraints: list[dict[str, Any]]
    fts_results: list[Fact]
    vector_results: list[Fact]
    delta_facts: list[Fact]
    recent_messages: list[dict[str, Any]]
    working_memory_refs: list[dict[str, Any]] = field(default_factory=list)
    pinned_facts: list[Fact] = field(default_factory=list)

    @property
    def facts(self) -> list[Fact]:
        merged: list[Fact] = []
        seen_fact_ids: set[int] = set()
        for fact in [*self.pinned_facts, *self.fts_results, *self.vector_results, *self.delta_facts]:
            if fact.id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact.id)
            merged.append(fact)
        return merged
