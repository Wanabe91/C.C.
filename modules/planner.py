import json
from modules.llm_client import LLMClient
from modules.memory import Memory
from datetime import datetime

TOOLS = [
    {"type":"function","function":{
        "name":"get_current_time",
        "description":"Текущие дата и время",
        "parameters":{"type":"object","properties":{},"required":[]}
    }},
    {"type":"function","function":{
        "name":"remember",
        "description":"Запомнить важный факт",
        "parameters":{"type":"object","properties":{
            "content":{"type":"string"}
        },"required":["content"]}
    }},
    {"type":"function","function":{
        "name":"recall",
        "description":"Вспомнить информацию из памяти",
        "parameters":{"type":"object","properties":{
            "query":{"type":"string"}
        },"required":["query"]}
    }},
]

class Planner:
    def __init__(self, llm: LLMClient, memory: Memory, cfg=None):
        self.llm = llm
        self.memory = memory
        self.max_steps = (cfg or {}).get("max_steps", 10)

    async def run(self, user_input: str) -> str:
        history = self.memory.get_history()
        self.memory.add("user", user_input)

        for _ in range(self.max_steps):
            msg = await self.llm.chat_with_tools(user_input, TOOLS, history)
            if not msg.get("tool_calls"):
                answer = msg.get("content","").strip()
                self.memory.add("assistant", answer)
                return answer

            tool_results = []
            for call in msg["tool_calls"]:
                result = await self._execute(
                    call["function"]["name"],
                    json.loads(call["function"].get("arguments","{}"))
                )
                tool_results.append({"tool_call_id":call["id"],
                                     "role":"tool","content":str(result)})
            history = history + [
                {"role":"assistant","content":None,
                 "tool_calls":msg["tool_calls"]},
                *tool_results
            ]
        return "Не смог выполнить задачу."

    async def _execute(self, name, args):
        if name == "get_current_time":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if name == "remember":
            self.memory.save("fact", args["content"]); return "Запомнено."
        if name == "recall":
            return self.memory.recall(args["query"])
        return f"Неизвестный инструмент: {name}"
```

---

## 🗺 Дорожная карта

| Фаза | Что делать |
|---|---|
| **1. Скелет** | Запустить text mode + подключить свой LLM сервер |
| **2. Голос** | Добавить Whisper STT + pyttsx3 TTS, запустить voice mode |
| **3. Память** | Заменить LIKE-поиск в memory.py на embedding-поиск (SQLite+FAISS) |
| **4. Зрение** | Подключить YOLOv8, протестировать vision mode |
| **5. Инструменты** | Реализовать web_tool (DuckDuckGo), calendar_tool (Google API) |
| **6. Продакшн** | Wake word, демон-сервис, GUI (tkinter/web), многопоточность |

## Зависимости для старта
```
pip install httpx pyyaml openai-whisper pyttsx3 ultralytics opencv-python