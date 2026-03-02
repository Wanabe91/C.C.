from urllib.parse import urlparse

import httpx

class LLMClient:
    def __init__(self, cfg):
        base_url = cfg["base_url"].rstrip("/")
        if base_url.endswith("/chat/completions"):
            self.url = base_url
        else:
            self.url = base_url + "/chat/completions"
        self.model = cfg["model"]
        self.system = cfg.get("system_prompt", "")
        self.timeout = cfg.get("timeout", 30)
        self.params = {
            "max_tokens": cfg.get("max_tokens", 1024),
            "temperature": cfg.get("temperature", 0.7),
        }
        self.headers = {"Content-Type": "application/json"}
        host = (urlparse(self.url).hostname or "").lower()
        self.is_local = host in {"localhost", "127.0.0.1", "::1"}
        self.trust_env = cfg.get("trust_env", not self.is_local)
        if cfg.get("api_key"):
            self.headers["Authorization"] = f"Bearer {cfg['api_key']}"

    def _msgs(self, history=None, user=None):
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if history:
            messages.extend(history)
        if user is not None:
            messages.append({"role": "user", "content": user})
        return messages

    async def _post(self, payload):
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=self.trust_env) as c:
                r = await c.post(self.url, headers=self.headers, json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text.strip() or "empty response body"
            hint = ""
            if self.is_local and status == 502:
                hint = " Local LLM returned 502; verify the LM server is running and a model is loaded."
            raise RuntimeError(f"LLM request failed with HTTP {status}: {detail}.{hint}") from exc
        except httpx.HTTPError as exc:
            hint = ""
            if self.is_local:
                hint = " Check that the local LLM server is reachable on the configured host and port."
            raise RuntimeError(f"LLM request failed: {exc}.{hint}") from exc

    async def chat(self, user, history=None):
        messages = user if isinstance(user, list) else self._msgs(history or [], user)
        data = await self._post({
            "model": self.model,
            "messages": messages,
            **self.params,
        })
        return data["choices"][0]["message"]["content"].strip()

    async def chat_with_tools(self, user, tools, history=None):
        messages = user if isinstance(user, list) else self._msgs(history or [], user)
        data = await self._post({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            **self.params,
        })
        return data["choices"][0]["message"]
