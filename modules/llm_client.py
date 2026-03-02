import json, httpx

class LLMClient:
    def __init__(self, cfg):
        self.url  = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.model = cfg["model"]
        self.system = cfg.get("system_prompt", "")
        self.params = {"max_tokens": cfg.get("max_tokens",1024),
                       "temperature": cfg.get("temperature",0.7)}
        self.headers = {"Content-Type":"application/json"}
        if cfg.get("api_key"):
            self.headers["Authorization"] = f"Bearer {cfg['api_key']}"

    def _msgs(self, history, user):
        return [{"role":"system","content":self.system},
                *history,
                {"role":"user","content":user}]

    async def chat(self, user, history=None):
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(self.url, headers=self.headers, json={
                "model": self.model,
                "messages": self._msgs(history or [], user),
                **self.params
            })
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def chat_with_tools(self, user, tools, history=None):
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(self.url, headers=self.headers, json={
                "model": self.model,
                "messages": self._msgs(history or [], user),
                "tools": tools, "tool_choice": "auto",
                **self.params
            })
            r.raise_for_status()
            return r.json()["choices"][0]["message"]