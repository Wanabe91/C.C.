from __future__ import annotations

from typing import Any

import httpx

from .config import get_config
from .http_client import post


class LMStudioError(Exception):
    def __init__(self, status_code: int | None, body: str) -> None:
        super().__init__(body)
        self.status_code = status_code
        self.body = body


_JSON_RESPONSE_FORMAT_SUPPORTED: bool | None = None


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


def llm_call(system: str, user: str, schema: dict[str, Any] | None = None) -> str:
    global _JSON_RESPONSE_FORMAT_SUPPORTED

    config = get_config()
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if schema is None:
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

    repair_messages = list(messages)
    repair_messages[-1] = {
        "role": "user",
        "content": (
            "Return valid JSON only.\n"
            f"Required shape: {schema}\n\n"
            f"{user}"
        ),
    }
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
