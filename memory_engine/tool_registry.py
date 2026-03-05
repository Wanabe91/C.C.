from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .identity import CORE
from .weekly_review import generate_weekly_review


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    args_schema: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]
    planner_hint: str


class ToolArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RespondArgs(ToolArgsModel):
    message: str = Field(validation_alias=AliasChoices("message", "content", "text"))

    @model_validator(mode="after")
    def _ensure_message(self) -> "RespondArgs":
        if not self.message:
            raise ValueError("message must not be empty.")
        return self


class RememberFactArgs(ToolArgsModel):
    content: str = Field(validation_alias=AliasChoices("content", "fact"))
    importance: str | None = None
    tier: str | None = None
    meta: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _ensure_content(self) -> "RememberFactArgs":
        if not self.content:
            raise ValueError("content must not be empty.")
        return self


class CreateTaskArgs(ToolArgsModel):
    title: str = Field(validation_alias=AliasChoices("title", "task", "name"))
    description: str | None = None
    constraint_json: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("constraint_json", "constraint"),
    )

    @model_validator(mode="after")
    def _ensure_title(self) -> "CreateTaskArgs":
        if not self.title:
            raise ValueError("title must not be empty.")
        return self


class CompleteTaskArgs(ToolArgsModel):
    task_id: str | int | None = None
    task_ids: list[str | int] | None = None

    @model_validator(mode="after")
    def _ensure_task_ids(self) -> "CompleteTaskArgs":
        if self.task_id is not None:
            normalized = str(self.task_id).strip()
            if not normalized or not normalized.isdigit():
                raise ValueError("task_id must be a numeric string.")
            self.task_id = normalized
            return self
        if not self.task_ids:
            raise ValueError("task_id is required.")
        normalized_ids = [str(item).strip() for item in self.task_ids]
        if any(not item or not item.isdigit() for item in normalized_ids):
            raise ValueError("task_ids must contain numeric strings only.")
        self.task_ids = normalized_ids
        return self


class GenerateWeeklyReviewArgs(ToolArgsModel):
    week_start: str | None = None
    week_offset: int | None = None
    title: str | None = None
    focus: str | None = None

    @model_validator(mode="after")
    def _validate_week_start(self) -> "GenerateWeeklyReviewArgs":
        if self.week_start:
            date.fromisoformat(self.week_start)
        return self


class NoopArgs(ToolArgsModel):
    pass


def _base_result(tool_name: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "assistant_message": None,
        "facts": [],
        "created_tasks": [],
        "completed_task_ids": [],
        "generated_reviews": [],
        "meta": {"action": tool_name, "tool": tool_name},
    }


# These handlers are the executor branches lifted into registry entries.
def _handle_respond(args: RespondArgs) -> dict[str, Any]:
    result = _base_result("respond")
    result["assistant_message"] = args.message
    return result


def _handle_remember_fact(args: RememberFactArgs) -> dict[str, Any]:
    result = _base_result("remember_fact")
    kind = CORE.classify_memory_request(args.content)
    if kind == "core_attack":
        result["assistant_message"] = (
            "I cannot store that memory request because it attempts to override core identity rules."
        )
        return result
    meta = dict(args.meta or {})
    meta["kind"] = kind
    result["facts"] = [
        {
            "content": args.content,
            "importance": args.importance,
            "tier": args.tier,
            "meta": meta,
        }
    ]
    return result


def _handle_create_task(args: CreateTaskArgs) -> dict[str, Any]:
    result = _base_result("create_task")
    result["created_tasks"] = [
        {
            "title": args.title,
            "constraint_json": dict(args.constraint_json) if args.constraint_json is not None else None,
        }
    ]
    return result


def _handle_complete_task(args: CompleteTaskArgs) -> dict[str, Any]:
    result = _base_result("complete_task")
    result["completed_task_ids"] = (
        [int(args.task_id)]
        if args.task_id is not None
        else [int(item) for item in args.task_ids or []]
    )
    return result


def _handle_generate_weekly_review(args: GenerateWeeklyReviewArgs) -> dict[str, Any]:
    result = _base_result("generate_weekly_review")
    review = generate_weekly_review(args.model_dump(exclude_none=True))
    result["generated_reviews"] = [review]
    result["assistant_message"] = review["markdown"]
    result["meta"] = {**result["meta"], "week_key": review["week_key"]}
    return result


def _handle_noop(_: NoopArgs) -> dict[str, Any]:
    return _base_result("noop")


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "respond": ToolDefinition(
        name="respond",
        args_schema=RespondArgs,
        handler=_handle_respond,
        planner_hint='Use when the user only needs a direct answer. Args: {"message": "assistant reply"}',
    ),
    "remember_fact": ToolDefinition(
        name="remember_fact",
        args_schema=RememberFactArgs,
        handler=_handle_remember_fact,
        planner_hint=(
            'Store durable knowledge. Args: {"content": "fact to store", '
            '"importance": "core|contextual|transient"?, "tier": "active|cold|archived"?, "meta": {...}?}'
        ),
    ),
    "create_task": ToolDefinition(
        name="create_task",
        args_schema=CreateTaskArgs,
        handler=_handle_create_task,
        planner_hint='Create a follow-up task. Args: {"title": "task title", "description": "optional detail"?}',
    ),
    "complete_task": ToolDefinition(
        name="complete_task",
        args_schema=CompleteTaskArgs,
        handler=_handle_complete_task,
        planner_hint='Mark an existing task complete. Args: {"task_id": "task id"}',
    ),
    "generate_weekly_review": ToolDefinition(
        name="generate_weekly_review",
        args_schema=GenerateWeeklyReviewArgs,
        handler=_handle_generate_weekly_review,
        planner_hint=(
            'Draft a weekly Markdown review from stored activity. Args: {"week_start": "YYYY-MM-DD"?, '
            '"week_offset": integer?, "title": "optional title"?, "focus": "optional focus"?}'
        ),
    ),
    "noop": ToolDefinition(
        name="noop",
        args_schema=NoopArgs,
        handler=_handle_noop,
        planner_hint="Do nothing when no action is required. Args: {}",
    ),
}


def get_tool(name: str) -> ToolDefinition | None:
    return TOOL_REGISTRY.get(str(name or "").strip())


@functools.lru_cache(maxsize=None)
def registry_prompt_block() -> str:
    lines = ["Available tools:"]
    for tool_definition in TOOL_REGISTRY.values():
        lines.append(f"- {tool_definition.name}: {tool_definition.planner_hint}")
    return "\n".join(lines)


def assert_registry_integrity() -> None:
    seen_names: set[str] = set()
    for key, tool_definition in TOOL_REGISTRY.items():
        assert tool_definition.name == key, f"Registry key mismatch for tool '{key}'."
        assert tool_definition.name not in seen_names, f"Duplicate tool name '{tool_definition.name}'."
        seen_names.add(tool_definition.name)
        assert tool_definition.planner_hint.strip(), f"Tool '{tool_definition.name}' is missing planner_hint."
        assert callable(tool_definition.handler), f"Tool '{tool_definition.name}' handler is not callable."
        assert isinstance(tool_definition.args_schema, type) and issubclass(
            tool_definition.args_schema, BaseModel
        ), f"Tool '{tool_definition.name}' args_schema must be a BaseModel subclass."
