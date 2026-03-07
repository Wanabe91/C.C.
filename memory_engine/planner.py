from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .config import get_config
from .llm import LLMRequest, llm_call
from .models import ContextSnapshot, Fact, PlannerRun, Task, ValidatedPlanStep
from .tool_registry import registry_prompt_block

PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "tool": {"type": ["string", "null"]},
                    "args": {"type": "object"},
                    "precondition_fact_ids": {
                        "type": "array",
                        "items": {"type": ["string", "integer"]},
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "tool", "args", "precondition_fact_ids", "reasoning"],
            },
        }
    },
    "required": ["steps"],
}

PLANNER_SYSTEM_PROMPT_PREFIX = (
    "You are the planning engine for a persistent local AI assistant.\n"
    "Output only valid JSON.\n"
    "Prefer short, deterministic plans.\n"
    "You receive a list of already-executed steps under '## Steps already executed'.\n"
    "Use their results to decide the NEXT single step only.\n"
    "If the goal is achieved, emit one respond step.\n"
    "If the user only needs a direct answer, emit one respond step.\n"
    "Treat salience and truth separately: access_count and last_accessed_at show reuse, not certainty.\n"
    "Use verification_status and confidence_score when deciding whether a memory is tentative or confirmed.\n"
    "CRITICAL: The message in any respond step MUST be written in the same "
    "language as the user's goal. If the user wrote in Russian — respond in Russian."
)


PLANNER_MEMORY_RULES = (
    "Memory-save rule:\n"
    "When the user explicitly asks to save, remember, store, or not forget information, "
    "you MUST include a remember_fact step before any respond step.\n"
    "This applies to requests such as 'remember that', 'remember...', 'store this', "
    "'save this', 'don't forget', 'запомни', 'запомни, что', 'сохрани', 'не забудь', "
    "'я хочу, чтобы ты знал', and also clear durable user facts phrased like 'my X is Y' "
    "or 'у меня X = Y' when the user is clearly presenting a persistent fact to retain.\n"
    "Never emit only respond when the user is explicitly asking to save information.\n"
    "If the user asks to save information and also wants an acknowledgement, the plan "
    "must be remember_fact first and respond second."
)


class PlannerValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _clean_validation_message(message: str) -> str:
    prefix = "Value error, "
    return message[len(prefix) :] if message.startswith(prefix) else message


def _validation_error_message(exc: ValidationError) -> str:
    return "; ".join(error.get("msg", str(exc)) for error in exc.errors(include_url=False)) or str(exc)


def _planner_validation_code(message: str) -> str:
    if "Unknown tool" in message:
        return "unknown_tool"
    if "precondition_fact_ids" in message:
        return "invalid_precondition_fact_ids"
    if "args for tool" in message:
        return "invalid_args"
    return "invalid_step"


def _planner_system_prompt() -> str:
    assistant_prompt = get_config().ASSISTANT_SYSTEM_PROMPT
    if not assistant_prompt:
        return f"{PLANNER_SYSTEM_PROMPT_PREFIX}\n{PLANNER_MEMORY_RULES}"
    return (
        f"{PLANNER_SYSTEM_PROMPT_PREFIX}\n"
        f"{PLANNER_MEMORY_RULES}\n"
        "When emitting a respond step, follow this assistant style instruction:\n"
        f"{assistant_prompt}\n"
        "Regardless of the language facts are stored in, always reply in the "
        "language the user used in their goal."
    )


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "constraint": task.constraint,
        "active_from_version": task.active_from_version,
        "completed_version": task.completed_version,
    }


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "content": fact.content,
        "status": fact.status,
        "importance": fact.importance,
        "tier": fact.tier,
        "version_created": fact.version_created,
        "version_superseded": fact.version_superseded,
        "source_event_id": fact.source_event_id,
        "created_at": fact.created_at,
        "last_accessed_at": fact.last_accessed_at,
        "access_count": fact.access_count,
        "confidence_score": fact.confidence_score,
        "verification_status": fact.verification_status,
        "verification_count": fact.verification_count,
        "last_verified_at": fact.last_verified_at,
        "evidence": fact.evidence,
        "contradiction_group_id": fact.contradiction_group_id,
        "meta": fact.meta,
    }


def snapshot_payload(ctx: ContextSnapshot, goal: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "state_version": ctx.state_version,
        "vector_watermark": ctx.vector_watermark,
        "tasks": [_task_payload(task) for task in ctx.tasks],
        "constraints": ctx.constraints,
        "fts_results": [_fact_payload(fact) for fact in ctx.fts_results],
        "vector_results": [_fact_payload(fact) for fact in ctx.vector_results],
        "delta_facts": [_fact_payload(fact) for fact in ctx.delta_facts],
        "pinned_facts": [_fact_payload(fact) for fact in ctx.pinned_facts],
        "recent_messages": ctx.recent_messages,
        "working_memory_refs": ctx.working_memory_refs,
    }


def _planner_user_prompt(goal: str) -> str:
    return (
        "Return JSON using this exact structure:\n"
        "{\n"
        '  "steps": [{\n'
        '    "action": "string",\n'
        '    "tool": "string or null",\n'
        '    "args": {},\n'
        '    "precondition_fact_ids": [],\n'
        '    "reasoning": "string"\n'
        "  }]\n"
        "}\n\n"
        f"Goal:\n{goal.strip()}"
    )


_EXPLICIT_MEMORY_PREFIXES = (
    r"remember(?:\s+that)?",
    r"store(?:\s+this|\s+that)?",
    r"save(?:\s+this|\s+that)?",
    r"don't forget(?:\s+that)?",
    r"запомни(?:,\s*что)?",
    r"сохрани(?:,\s*что)?",
    r"не забудь(?:,\s*что)?",
    r"я хочу,\s*чтобы ты знал(?:,\s*что)?",
)

_MEMORY_TAIL_MARKERS = (
    r"(?:[.!?]\s*|\s+)(?:then\s+)?(?:reply|respond|answer|confirm)\b.*$",
    r"(?:[.!?]\s*|\s+)(?:please\s+)?(?:reply|respond|answer|confirm)\b.*$",
    r"(?:[.!?]\s*|\s+)(?:ответь|подтверди|скажи)\b.*$",
)


def _extract_explicit_memory_fact(goal: str) -> str | None:
    text = goal.strip()
    if not text:
        return None

    for prefix in _EXPLICIT_MEMORY_PREFIXES:
        match = re.match(
            rf"^\s*(?:please\s+)?{prefix}\s*[:,-]?\s*(?P<fact>.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        fact = match.group("fact").strip()
        for tail_pattern in _MEMORY_TAIL_MARKERS:
            fact = re.sub(tail_pattern, "", fact, flags=re.IGNORECASE).strip()
        fact = fact.strip().strip("\"'").rstrip(".!?;,")
        return fact or None
    return None


def _acknowledgement_message(goal: str) -> str:
    return "Запомнил." if re.search(r"[А-Яа-яЁё]", goal) else "I'll remember that."


def _remember_step(fact: str) -> ValidatedPlanStep:
    return ValidatedPlanStep.model_validate(
        {
            "action": "remember explicit user fact",
            "tool": "remember_fact",
            "args": {"content": fact},
            "precondition_fact_ids": [],
            "reasoning": "The user explicitly asked the assistant to save this information.",
        }
    )


def _respond_step(message: str) -> ValidatedPlanStep:
    return ValidatedPlanStep.model_validate(
        {
            "action": "acknowledge saved memory",
            "tool": "respond",
            "args": {"message": message},
            "precondition_fact_ids": [],
            "reasoning": "Acknowledge that the requested memory was saved.",
        }
    )


def _observation_has_tool(observations: tuple[dict[str, Any], ...], tool_name: str) -> bool:
    normalized_tool_name = str(tool_name).strip()
    return any(str(item.get("tool") or "").strip() == normalized_tool_name for item in observations)


def _enforce_explicit_memory_rules(
    goal: str,
    steps: list[ValidatedPlanStep],
    observations: tuple[dict[str, Any], ...] = (),
) -> list[ValidatedPlanStep]:
    fact = _extract_explicit_memory_fact(goal)
    if fact is None:
        return steps

    if _observation_has_tool(observations, "remember_fact"):
        normalized_steps = [step for step in steps if step.tool != "remember_fact"]
        if any(step.tool == "respond" for step in normalized_steps):
            return normalized_steps
        return [*normalized_steps, _respond_step(_acknowledgement_message(goal))]

    remember_steps = [step for step in steps if step.tool == "remember_fact"]
    respond_indices = [index for index, step in enumerate(steps) if step.tool == "respond"]
    first_respond_index = respond_indices[0] if respond_indices else None

    if not remember_steps:
        normalized_steps = list(steps)
        insert_at = first_respond_index if first_respond_index is not None else 0
        normalized_steps.insert(insert_at, _remember_step(fact))
        if first_respond_index is None:
            normalized_steps.append(_respond_step(_acknowledgement_message(goal)))
        return normalized_steps

    if first_respond_index is None:
        return [*steps, _respond_step(_acknowledgement_message(goal))]

    first_remember_index = next(index for index, step in enumerate(steps) if step.tool == "remember_fact")
    if first_remember_index <= first_respond_index:
        return steps

    normalized_steps = [step for index, step in enumerate(steps) if index != first_remember_index]
    normalized_steps.insert(first_respond_index, steps[first_remember_index])
    return normalized_steps


def _fact_kind(fact: Fact) -> str:
    meta = fact.meta if isinstance(fact.meta, dict) else {}
    return str(meta.get("kind") or "").strip().lower()


def _snapshot_facts(ctx: ContextSnapshot) -> list[Fact]:
    explicit_facts = getattr(ctx, "facts", None)
    if isinstance(explicit_facts, list):
        facts = [fact for fact in explicit_facts if isinstance(fact, Fact)]
    else:
        facts = [*ctx.fts_results, *ctx.vector_results, *ctx.delta_facts]

    deduplicated: list[Fact] = []
    seen_fact_ids: set[int] = set()
    for fact in facts:
        if fact.id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact.id)
        deduplicated.append(fact)
    return deduplicated


def _split_pinned_facts(ctx: ContextSnapshot) -> tuple[list[Fact], list[Fact]]:
    user_model_facts: list[Fact] = []
    preference_facts: list[Fact] = []
    for fact in ctx.pinned_facts:
        kind = _fact_kind(fact)
        if kind == "user_model":
            user_model_facts.append(fact)
        elif kind == "preference":
            preference_facts.append(fact)
    return user_model_facts, preference_facts


def _facts_to_block(facts: list[Fact]) -> str:
    if not facts:
        return ""
    return json.dumps([_fact_payload(fact) for fact in facts], ensure_ascii=False, indent=2)


def _planner_context_snapshot(ctx: ContextSnapshot, goal: str, context_facts: list[Fact]) -> str:
    payload = {
        "goal": goal,
        "state_version": ctx.state_version,
        "vector_watermark": ctx.vector_watermark,
        "tasks": [_task_payload(task) for task in ctx.tasks],
        "constraints": ctx.constraints,
        "facts": [_fact_payload(fact) for fact in context_facts],
        "recent_messages": ctx.recent_messages,
        "working_memory_refs": ctx.working_memory_refs,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def serialize_step(step: ValidatedPlanStep) -> dict[str, Any]:
    return {
        "action": step.action,
        "tool": step.tool,
        "args": step.args_dict(),
        "precondition_fact_ids": step.precondition_fact_ids,
        "reasoning": step.reasoning,
    }


def _find_json_fragment(raw: str) -> str | None:
    start = -1
    stack: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(raw):
        if start < 0:
            if char not in "{[":
                continue
            start = index
            stack.append("}" if char == "{" else "]")
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue

        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return raw[start : index + 1]

    return None


def _normalize_planner_output(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in (cleaned, raw.strip()):
        if not candidate:
            continue
        if candidate[:1] in "{[":
            return candidate
        fragment = _find_json_fragment(candidate)
        if fragment is not None:
            return fragment
    raise ValueError("Planner output did not contain a JSON object.")


def _parse_steps(raw: str) -> list[ValidatedPlanStep]:
    payload = json.loads(_normalize_planner_output(raw))
    if not isinstance(payload, dict):
        raise PlannerValidationError("invalid_shape", "Planner output must be a JSON object.")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise PlannerValidationError("invalid_shape", "Planner output did not include a steps array.")
    steps: list[ValidatedPlanStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise PlannerValidationError("invalid_step", f"steps[{index}] must be an object.")
        try:
            steps.append(ValidatedPlanStep.model_validate(item))
        except ValidationError as exc:
            message = _clean_validation_message(_validation_error_message(exc))
            raise PlannerValidationError(
                _planner_validation_code(message),
                f"steps[{index}] {message}",
            ) from exc
        except ValueError as exc:
            message = _clean_validation_message(str(exc))
            raise PlannerValidationError(
                _planner_validation_code(message),
                f"steps[{index}] {message}",
            ) from exc
    if not steps:
        raise PlannerValidationError("empty_plan", "Planner returned no usable steps.")
    return steps


def _fallback_plan(goal: str) -> list[ValidatedPlanStep]:
    message = (
        "Уточни запрос, пожалуйста."
        if not goal.strip()
        else "Не удалось сформировать ответ. Я туплю."
    )
    return [
        ValidatedPlanStep.model_validate(
            {
                "action": "respond",
                "tool": "respond",
                "args": {"message": message},
                "precondition_fact_ids": [],
                "reasoning": "Fallback plan after planner output could not be parsed.",
            }
        )
    ]


def _repair_prompt(raw_response: str, error_message: str) -> str:
    return (
        "Repair the following planner output into valid JSON with the exact required structure.\n"
        f"Validation error: {error_message}\n\n"
        f"{raw_response}"
    )


def plan(
    ctx: ContextSnapshot,
    goal: str,
    observations: tuple[dict[str, Any], ...] = (),
) -> PlannerRun:
    prompt = _planner_user_prompt(goal)
    system_prompt = _planner_system_prompt()
    snapshot = snapshot_payload(ctx, goal)
    user_model_facts, preference_facts = _split_pinned_facts(ctx)
    context_facts = _snapshot_facts(ctx)
    request = LLMRequest(
        goal=prompt,
        context_snapshot=_planner_context_snapshot(ctx, goal, context_facts),
        tool_registry_block=f"{system_prompt}\n{registry_prompt_block()}".strip(),
        user_model=_facts_to_block(user_model_facts),
        preferences=_facts_to_block(preference_facts),
        observation_history=observations,
    )
    first_response = llm_call(request, schema=PLANNER_JSON_SCHEMA)
    try:
        steps = _enforce_explicit_memory_rules(goal, _parse_steps(first_response), observations)
        return PlannerRun(
            goal=goal,
            snapshot=snapshot,
            system_prompt=system_prompt,
            user_prompt=prompt,
            planner_status="ok",
            steps=steps,
            first_response=first_response,
        )
    except (json.JSONDecodeError, PlannerValidationError, ValueError) as first_error:
        repair_prompt = _repair_prompt(first_response, str(first_error))
        repair_request = LLMRequest(
            goal=repair_prompt,
            context_snapshot=request.context_snapshot,
            tool_registry_block=request.tool_registry_block,
            user_model=request.user_model,
            preferences=request.preferences,
            observation_history=observations,
            identity=request.identity,
        )
        second_response = llm_call(
            repair_request,
            schema=PLANNER_JSON_SCHEMA,
        )
        try:
            steps = _enforce_explicit_memory_rules(goal, _parse_steps(second_response), observations)
            return PlannerRun(
                goal=goal,
                snapshot=snapshot,
                system_prompt=system_prompt,
                user_prompt=prompt,
                planner_status="repaired",
                steps=steps,
                first_response=first_response,
                repair_prompt=repair_prompt,
                repair_response=second_response,
                error=str(first_error),
            )
        except (json.JSONDecodeError, PlannerValidationError, ValueError) as second_error:
            return PlannerRun(
                goal=goal,
                snapshot=snapshot,
                system_prompt=system_prompt,
                user_prompt=prompt,
                planner_status="fallback",
                steps=_fallback_plan(goal),
                first_response=first_response,
                repair_prompt=repair_prompt,
                repair_response=second_response,
                error=f"{first_error}; {second_error}",
            )
