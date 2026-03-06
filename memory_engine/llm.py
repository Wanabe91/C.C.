from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import litellm
import yaml
from dotenv import load_dotenv

from .identity import CORE, IdentityCore
from .router import TaskType, route

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

TASK_MODEL_MAP: dict[TaskType, str] = {
    TaskType.FAST: "groq/llama-3.3-70b-versatile",
    TaskType.REASON: "claude-sonnet-4-6",
    TaskType.MATH: "o3-mini",
    TaskType.VISION: "gpt-4o",
    TaskType.CONSOLIDATE: "groq/llama-3.3-70b-versatile",
    TaskType.SUMMARIZE: "groq/llama-3.3-70b-versatile",
    TaskType.PLAN: "groq/llama-3.3-70b-versatile",
}

FALLBACK_CHAIN = [
    "groq/llama-3.3-70b-versatile",
    "claude-sonnet-4-6",
    "gpt-4o",
]

TASK_PROVIDER_MAP: dict[TaskType, str] = {
    TaskType.FAST: "groq",
    TaskType.REASON: "claude",
    TaskType.MATH: "openai_o3",
    TaskType.VISION: "openai_gpt4o",
    TaskType.CONSOLIDATE: "groq",
    TaskType.SUMMARIZE: "groq",
    TaskType.PLAN: "groq",
}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "groq": TASK_MODEL_MAP[TaskType.FAST],
    "claude": TASK_MODEL_MAP[TaskType.REASON],
    "openai_gpt4o": TASK_MODEL_MAP[TaskType.VISION],
    "openai_o3": TASK_MODEL_MAP[TaskType.MATH],
}


@dataclass(slots=True)
class LLMRequest:
    goal: str
    context_snapshot: str
    tool_registry_block: str
    user_model: str = ""
    preferences: str = ""
    image_data_urls: tuple[str, ...] = ()
    identity: IdentityCore = field(default_factory=lambda: CORE)

    def build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        identity_block = self.identity.as_system_block().strip()
        if identity_block:
            messages.append({"role": "system", "content": identity_block})
        tool_registry_block = self.tool_registry_block.strip()
        if tool_registry_block:
            messages.append({"role": "system", "content": tool_registry_block})
        user_model = self.user_model.strip()
        if user_model:
            messages.append({"role": "system", "content": f"User model:\n{user_model}"})
        preferences = self.preferences.strip()
        if preferences:
            messages.append({"role": "system", "content": f"User preferences:\n{preferences}"})
        context_snapshot = self.context_snapshot.strip()
        if context_snapshot:
            messages.append({"role": "system", "content": f"Context snapshot:\n{context_snapshot}"})
        goal = self.goal.strip()
        image_data_urls = tuple(url.strip() for url in self.image_data_urls if isinstance(url, str) and url.strip())
        if image_data_urls:
            content: list[dict[str, Any]] = [{"type": "text", "text": goal or " "}]
            for image_url in image_data_urls:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": goal or " "})
        return messages


def llm_call(
    request: LLMRequest | str,
    schema: dict[str, Any] | str | None = None,
    *,
    task: str | TaskType = TaskType.PLAN,
    context_len: int = 0,
    has_image: bool = False,
    **completion_overrides: Any,
) -> str:
    resolved_request, resolved_schema = _resolve_request(request, schema)
    requested_task = _normalize_task(task, resolved_request, resolved_schema)
    has_image = has_image or bool(resolved_request.image_data_urls)
    effective_task = route(
        requested_task,
        context_len=context_len,
        has_image=has_image,
        user_message=resolved_request.goal,
    )
    messages = resolved_request.build_messages()
    if resolved_schema is not None:
        messages = _messages_with_schema(messages, resolved_schema)

    completion_kwargs = _completion_kwargs(**completion_overrides)
    model = _model_for_task(effective_task)
    models_to_try = _candidate_models(model)
    errors: list[Exception] = []

    for index, candidate in enumerate(models_to_try):
        try:
            logger.info(
                "llm_call task=%s routed=%s model=%s context_len=%d messages=%d attempt=%d",
                requested_task.value,
                effective_task.value,
                candidate,
                context_len,
                len(messages),
                index + 1,
            )
            response = litellm.completion(
                model=candidate,
                messages=messages,
                **completion_kwargs,
            )
            content = _extract_text(response)
            if resolved_schema is not None and not _looks_like_json(content):
                repair_messages = _build_repair_messages(messages, content, resolved_schema)
                logger.info(
                    "llm_call task=%s routed=%s model=%s context_len=%d messages=%d attempt=%d repair=1",
                    requested_task.value,
                    effective_task.value,
                    candidate,
                    context_len,
                    len(repair_messages),
                    index + 1,
                )
                repair_response = litellm.completion(
                    model=candidate,
                    messages=repair_messages,
                    **completion_kwargs,
                )
                return _extract_text(repair_response)
            return content
        except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
            errors.append(exc)
            logger.warning("Provider retryable failure on %s: %s", candidate, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            logger.warning("Provider failure on %s: %s", candidate, exc)
            continue

    raise RuntimeError("All providers failed") from (errors[-1] if errors else None)


def _resolve_request(
    request: LLMRequest | str,
    schema: dict[str, Any] | str | None,
) -> tuple[LLMRequest, dict[str, Any] | str | None]:
    if isinstance(request, LLMRequest):
        return request, schema
    if not isinstance(schema, str):
        raise TypeError(
            "Legacy llm_call usage requires llm_call(system_prompt: str, user_prompt: str)."
        )
    return (
        LLMRequest(
            goal=schema,
            context_snapshot="",
            tool_registry_block=request,
        ),
        None,
    )


def _normalize_task(
    task: str | TaskType,
    request: LLMRequest,
    schema: dict[str, Any] | str | None,
) -> TaskType:
    if isinstance(task, TaskType):
        resolved = task
    else:
        normalized = str(task or "").strip().lower()
        try:
            resolved = TaskType(normalized)
        except ValueError:
            resolved = _default_task()

    if resolved != TaskType.PLAN:
        return resolved

    inferred = _infer_task_from_request(request, schema)
    return inferred or resolved


def _default_task() -> TaskType:
    raw_value = str(_llm_config().get("default_task") or TaskType.FAST.value).strip().lower()
    try:
        return TaskType(raw_value)
    except ValueError:
        return TaskType.FAST


def _infer_task_from_request(
    request: LLMRequest,
    schema: dict[str, Any] | str | None,
) -> TaskType | None:
    goal = request.goal.lower()
    block = request.tool_registry_block.lower()
    schema_text = _schema_text(schema).lower() if schema is not None else ""
    summarize_markers = ("summarize", "summarise", "суммар")

    if "memory optimization engine" in block or "\"proposals\"" in schema_text:
        return TaskType.CONSOLIDATE
    if any(marker in goal or marker in block for marker in summarize_markers):
        return TaskType.SUMMARIZE
    return None


def _messages_with_schema(
    messages: list[dict[str, Any]],
    schema: dict[str, Any] | str,
) -> list[dict[str, Any]]:
    schema_prompt = (
        "Return valid JSON only.\n"
        f"Required shape: {_schema_text(schema)}"
    )
    updated = [dict(message) for message in messages]
    if updated and updated[-1].get("role") == "user":
        existing = updated[-1].get("content", "")
        if isinstance(existing, str):
            existing = existing.strip()
            updated[-1]["content"] = f"{schema_prompt}\n\n{existing}" if existing else schema_prompt
            return updated
        if isinstance(existing, list):
            merged_content: list[dict[str, Any]] = []
            inserted_schema = False
            for item in existing:
                if (
                    not inserted_schema
                    and isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    current_text = item["text"].strip()
                    merged_content.append(
                        {
                            **item,
                            "text": f"{schema_prompt}\n\n{current_text}" if current_text else schema_prompt,
                        }
                    )
                    inserted_schema = True
                    continue
                merged_content.append(item)
            if not inserted_schema:
                merged_content.insert(0, {"type": "text", "text": schema_prompt})
            updated[-1]["content"] = merged_content
            return updated
        updated[-1]["content"] = schema_prompt
        return updated
    updated.append({"role": "user", "content": schema_prompt})
    return updated


def _build_repair_messages(
    messages: list[dict[str, Any]],
    response_text: str,
    schema: dict[str, Any] | str,
) -> list[dict[str, Any]]:
    repair_messages = [dict(message) for message in messages]
    repair_messages.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )
    repair_messages.append(
        {
            "role": "user",
            "content": (
                "Repair your last answer. Return valid JSON only.\n"
                f"Required shape: {_schema_text(schema)}"
            ),
        }
    )
    return repair_messages


def _schema_text(schema: dict[str, Any] | str) -> str:
    if isinstance(schema, str):
        return schema.strip()
    return json.dumps(schema, ensure_ascii=False, sort_keys=True)


def _looks_like_json(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return True


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise RuntimeError("LiteLLM response did not include any choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
    if message is None:
        raise RuntimeError("LiteLLM response did not include a message payload.")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RuntimeError("LiteLLM response did not include string message content.")


def _completion_kwargs(**overrides: Any) -> dict[str, Any]:
    llm_cfg = _llm_config()
    temperature = _coerce_float(llm_cfg.get("temperature"), 0.3)
    max_tokens = _coerce_int(llm_cfg.get("max_tokens"), 2048)
    params: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
    }
    params.update(overrides)
    return params


def _model_for_task(task: TaskType) -> str:
    provider_key = TASK_PROVIDER_MAP[task]
    provider_cfg = _provider_config(provider_key)
    model = str(provider_cfg.get("model") or TASK_MODEL_MAP[task]).strip() or TASK_MODEL_MAP[task]
    if _coerce_bool(provider_cfg.get("enabled"), True):
        return model

    logger.info("Provider %s is disabled for task=%s; falling back to FAST model.", provider_key, task.value)
    fast_provider_cfg = _provider_config(TASK_PROVIDER_MAP[TaskType.FAST])
    fast_model = str(
        fast_provider_cfg.get("model") or TASK_MODEL_MAP[TaskType.FAST]
    ).strip()
    return fast_model or TASK_MODEL_MAP[TaskType.FAST]


def _candidate_models(primary_model: str) -> list[str]:
    candidates = [primary_model, *_configured_fallback_chain()]
    deduplicated: list[str] = []
    for model in candidates:
        cleaned = str(model or "").strip()
        if not cleaned or cleaned in deduplicated:
            continue
        deduplicated.append(cleaned)
    return deduplicated


def _configured_fallback_chain() -> list[str]:
    chain: list[str] = []
    for provider_key, fallback_model in (
        ("groq", FALLBACK_CHAIN[0]),
        ("claude", FALLBACK_CHAIN[1]),
        ("openai_gpt4o", FALLBACK_CHAIN[2]),
    ):
        provider_cfg = _provider_config(provider_key)
        if not _coerce_bool(provider_cfg.get("enabled"), True):
            continue
        configured_model = str(provider_cfg.get("model") or fallback_model).strip()
        chain.append(configured_model or fallback_model)

    return chain or list(FALLBACK_CHAIN)


def _provider_config(provider_key: str) -> dict[str, Any]:
    providers = _llm_config().get("providers", {})
    if not isinstance(providers, dict):
        return {"model": PROVIDER_DEFAULT_MODELS.get(provider_key, ""), "enabled": True}
    raw_config = providers.get(provider_key, {})
    if not isinstance(raw_config, dict):
        raw_config = {}
    if "model" not in raw_config and provider_key in PROVIDER_DEFAULT_MODELS:
        raw_config = {
            **raw_config,
            "model": PROVIDER_DEFAULT_MODELS[provider_key],
        }
    if "enabled" not in raw_config:
        raw_config = {**raw_config, "enabled": True}
    return raw_config


def _llm_config() -> dict[str, Any]:
    cfg = _load_yaml_config()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return llm_cfg if isinstance(llm_cfg, dict) else {}


@lru_cache(maxsize=1)
def _load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
