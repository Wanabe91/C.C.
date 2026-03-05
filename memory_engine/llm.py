from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import get_config
from .http_client import post
from .identity import CORE, IdentityCore


class LMStudioError(Exception):
    def __init__(self, status_code: int | None, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code
        self.body = body


_JSON_RESPONSE_FORMAT_SUPPORTED: bool | None = None


@dataclass(slots=True)
class LLMRequest:
    goal: str
    context_snapshot: str
    tool_registry_block: str
    user_model: str = ""
    preferences: str = ""
    identity: IdentityCore = field(default_factory=lambda: CORE)

    def build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
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
        messages.append({"role": "user", "content": goal or " "})
        return messages


def _post_chat(payload: dict[str, Any]) -> str:
    config = get_config()
    try:
        response = post(
            f"{config.LLM_BASE_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LMStudioError(exc.response.status_code, exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("LM Studio response did not include any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LM Studio response did not include a string message content.")
    return content.strip()


def llm_call(request: LLMRequest | str, schema: dict[str, Any] | str | None = None) -> str:
    global _JSON_RESPONSE_FORMAT_SUPPORTED

    resolved_request: LLMRequest
    resolved_schema: dict[str, Any] | None
    if isinstance(request, LLMRequest):
        if schema is not None and not isinstance(schema, dict):
            raise TypeError("schema must be a dict or None when request is LLMRequest.")
        resolved_request = request
        resolved_schema = schema
    else:
        if not isinstance(schema, str):
            raise TypeError(
                "Legacy llm_call usage requires llm_call(system_prompt: str, user_prompt: str)."
            )
        resolved_request = LLMRequest(
            goal=schema,
            context_snapshot="",
            tool_registry_block=request,
        )
        resolved_schema = None

    config = get_config()
    messages: list[dict[str, str]] = resolved_request.build_messages()
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if resolved_schema is None:
        try:
            return _post_chat(payload)
        except LMStudioError as exc:
            raise RuntimeError(
                f"LM Studio request failed with HTTP {exc.status_code}: {exc.body}"
            ) from exc

    if _JSON_RESPONSE_FORMAT_SUPPORTED is not False:
        first_payload = dict(payload)
        first_payload["response_format"] = {"type": "json_object"}
        try:
            response = _post_chat(first_payload)
            _JSON_RESPONSE_FORMAT_SUPPORTED = True
            return response
        except LMStudioError as exc:
            if exc.status_code != 400:
                raise RuntimeError(
                    f"LM Studio request failed with HTTP {exc.status_code}: {exc.body}"
                ) from exc
            _JSON_RESPONSE_FORMAT_SUPPORTED = False

    repair_messages = [dict(message) for message in messages]
    last_user_index = next(
        (
            index
            for index in range(len(repair_messages) - 1, -1, -1)
            if repair_messages[index].get("role") == "user"
        ),
        -1,
    )
    original_user = (
        repair_messages[last_user_index].get("content", "")
        if last_user_index >= 0
        else resolved_request.goal.strip()
    )
    repaired_user_message = {
        "role": "user",
        "content": (
            "Return valid JSON only.\n"
            f"Required shape: {resolved_schema}\n\n"
            f"{original_user}"
        ),
    }
    if last_user_index >= 0:
        repair_messages[last_user_index] = repaired_user_message
    else:
        repair_messages.append(repaired_user_message)
    repair_payload = {
        "model": config.LLM_MODEL,
        "messages": repair_messages,
        "temperature": 0,
    }
    try:
        return _post_chat(repair_payload)
    except LMStudioError as retry_exc:
        raise RuntimeError(
            f"LM Studio request failed with HTTP {retry_exc.status_code}: {retry_exc.body}"
        ) from retry_exc
