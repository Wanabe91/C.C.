from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .config import get_config
from .db import get_fact_by_id, record_fact_verification
from .epistemics import remembered_fact_epistemics
from .identity import CORE
from .weekly_review import generate_weekly_review
from .working_memory import working_memory


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


class VerifyFactArgs(ToolArgsModel):
    fact_id: str | int
    method: str
    source_ref: str | None = None
    note: str | None = None
    contradiction_group_id: str | None = None

    @model_validator(mode="after")
    def _validate_args(self) -> "VerifyFactArgs":
        normalized_fact_id = str(self.fact_id).strip()
        if not normalized_fact_id or not normalized_fact_id.isdigit():
            raise ValueError("fact_id must be a numeric string.")
        self.fact_id = normalized_fact_id

        normalized_method = str(self.method or "").strip().lower()
        if normalized_method not in {"user_confirmed", "external_match", "logical_consistency", "contradicted"}:
            raise ValueError(
                "method must be one of: user_confirmed, external_match, logical_consistency, contradicted."
            )
        self.method = normalized_method
        self.source_ref = self.source_ref.strip() if self.source_ref is not None else None
        self.note = self.note.strip() if self.note is not None else None
        self.contradiction_group_id = (
            self.contradiction_group_id.strip()
            if self.contradiction_group_id is not None
            else None
        )
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


class GrepMemoryArgs(ToolArgsModel):
    query: str
    ref_id: str | None = None
    limit: int | None = None

    @model_validator(mode="after")
    def _validate_args(self) -> "GrepMemoryArgs":
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query must not be empty.")
        if self.ref_id is not None:
            self.ref_id = self.ref_id.strip() or None
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be a positive integer.")
        return self


class ReadMemoryArgs(ToolArgsModel):
    ref_id: str
    offset: int = 0
    limit: int | None = None

    @model_validator(mode="after")
    def _validate_args(self) -> "ReadMemoryArgs":
        self.ref_id = self.ref_id.strip()
        if not self.ref_id:
            raise ValueError("ref_id must not be empty.")
        if self.offset < 0:
            raise ValueError("offset must be greater than or equal to zero.")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be a positive integer.")
        return self


def _base_result(tool_name: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "assistant_message": None,
        "tool_output": None,
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
    verification_status, confidence_score, evidence = remembered_fact_epistemics(meta)
    result["facts"] = [
        {
            "content": args.content,
            "importance": args.importance,
            "tier": args.tier,
            "confidence_score": confidence_score,
            "verification_status": verification_status,
            "verification_count": 1,
            "evidence": evidence,
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


def _handle_verify_fact(args: VerifyFactArgs) -> dict[str, Any]:
    result = _base_result("verify_fact")
    fact_id = int(args.fact_id)
    updated = record_fact_verification(
        fact_id,
        method=args.method,
        source_ref=args.source_ref,
        note=args.note,
        contradiction_group_id=args.contradiction_group_id,
    )
    if not updated:
        result["status"] = "error"
        result["assistant_message"] = f"Fact {fact_id} was not found for verification."
        return result

    fact = get_fact_by_id(fact_id)
    if fact is not None:
        result["tool_output"] = {
            "fact_id": fact.id,
            "verification_status": fact.verification_status,
            "confidence_score": fact.confidence_score,
            "verification_count": fact.verification_count,
            "contradiction_group_id": fact.contradiction_group_id,
        }
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


def _handle_grep_memory(args: GrepMemoryArgs) -> dict[str, Any]:
    result = _base_result("grep_memory")
    grep_result = working_memory.grep(args.query, ref_id=args.ref_id, limit=args.limit)
    matches = grep_result["matches"]
    if not matches:
        result["assistant_message"] = f"No working memory matches found for query: {args.query}"
        result["meta"] = {
            **result["meta"],
            "query": args.query,
            "searched_refs": grep_result["searched_refs"],
            "matches": [],
        }
        return result

    lines = [f"Working memory matches for query: {args.query}"]
    for match in matches:
        source_tool = f" tool={match['source_tool']}" if match.get("source_tool") else ""
        lines.append(
            f"- ref={match['ref_id']}{source_tool} field={match['field_name']} "
            f"offset={match['match_offset']} chars={match['char_count']} snippet={json.dumps(match['snippet'], ensure_ascii=False)}"
        )
    result["assistant_message"] = "\n".join(lines)
    result["meta"] = {
        **result["meta"],
        "query": args.query,
        "searched_refs": grep_result["searched_refs"],
        "matches": matches,
    }
    return result


def _handle_read_memory(args: ReadMemoryArgs) -> dict[str, Any]:
    result = _base_result("read_memory")
    read_result = working_memory.read(args.ref_id, offset=args.offset, limit=args.limit)
    header = (
        f"Working memory read ref={read_result['ref_id']} field={read_result['field_name']} "
        f"offset={read_result['offset']} next_offset={read_result['next_offset']} "
        f"total_chars={read_result['total_chars']} has_more={str(read_result['has_more']).lower()}"
    )
    content = read_result["content"]
    result["assistant_message"] = f"{header}\n\n{content}" if content else header
    result["meta"] = {
        **result["meta"],
        **{key: value for key, value in read_result.items() if key != "content"},
    }
    return result


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "respond": ToolDefinition(
        name="respond",
        args_schema=RespondArgs,
        handler=_handle_respond,
        planner_hint=(
            'Use when the user only needs a direct answer and is not asking to save memory. '
            'Do not use this alone for explicit remember/save/store requests. '
            'Args: {"message": "assistant reply"}'
        ),
    ),
    "remember_fact": ToolDefinition(
        name="remember_fact",
        args_schema=RememberFactArgs,
        handler=_handle_remember_fact,
        planner_hint=(
            'Highest priority for explicit memory-save requests. Use when the user asks you '
            'to remember, save, store, or not forget durable information. '
            'If the user also wants a reply, use remember_fact before respond. '
            'Args: {"content": "fact to store", '
            '"importance": "core|contextual|transient"?, "tier": "active|cold|archived"?, "meta": {...}?}. '
            "Facts captured through this tool start as self-reported memory, not externally verified truth."
        ),
    ),
    "create_task": ToolDefinition(
        name="create_task",
        args_schema=CreateTaskArgs,
        handler=_handle_create_task,
        planner_hint='Create a follow-up task. Args: {"title": "task title", "description": "optional detail"?}',
    ),
    "verify_fact": ToolDefinition(
        name="verify_fact",
        args_schema=VerifyFactArgs,
        handler=_handle_verify_fact,
        planner_hint=(
            'Update the epistemic status of an existing fact when the user confirms it, an external source '
            'matches it, a consistency check passes, or a contradiction is detected. '
            'Args: {"fact_id": "numeric id", "method": '
            '"user_confirmed|external_match|logical_consistency|contradicted", '
            '"source_ref": "optional citation", "note": "optional note", '
            '"contradiction_group_id": "optional shared conflict id"}'
        ),
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
    "grep_memory": ToolDefinition(
        name="grep_memory",
        args_schema=GrepMemoryArgs,
        handler=_handle_grep_memory,
        planner_hint=(
            'Search offloaded working-memory outputs from earlier tool runs. Use when the context snapshot '
            'mentions working_memory_refs or a prior step stored a large result. '
            'Args: {"query": "needle", "ref_id": "optional specific ref", "limit": 1-20?}'
        ),
    ),
    "read_memory": ToolDefinition(
        name="read_memory",
        args_schema=ReadMemoryArgs,
        handler=_handle_read_memory,
        planner_hint=(
            "Read a specific working-memory chunk after grep_memory identified a ref. "
            'Args: {"ref_id": "wm_0001", "offset": 0?, "limit": positive integer?}'
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
    config = get_config()
    lines = [
        "Tool selection rules:",
        "- Highest priority: if the user explicitly asks you to remember, save, store, or not forget durable information, use remember_fact before respond.",
        "- Never answer such explicit memory-save requests with only respond.",
        "- If the context snapshot contains working_memory_refs, those are offloaded large tool outputs. Use grep_memory before read_memory when you need to inspect them.",
        f"- Keep read_memory chunks at or below about {config.WORKING_MEMORY_READ_CHAR_LIMIT} characters unless you have a clear reason to page further.",
        "Available tools:",
    ]
    for tool_definition in TOOL_REGISTRY.values():
        lines.append(f"- {tool_definition.name}: {tool_definition.planner_hint}")
    return "\n".join(lines)


def clear_registry_prompt_cache() -> None:
    registry_prompt_block.cache_clear()


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
