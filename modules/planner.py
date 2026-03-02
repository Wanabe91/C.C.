import json
from datetime import datetime
from modules.llm_client import LLMClient
from modules.memory import Memory

TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Текущие дата и время",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Запомнить важный факт",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            },
            "required": ["content"]
        }
    }},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Вспомнить информацию из памяти",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }},
    {"type": "function",
     "function": {
         "name": "update_profile",
         "description": "Обновить профиль пользователя когда он сообщает о себе что-то важное",
         "parameters": {
             "type": "object",
             "properties": {
                 "updates": {
                     "type": "object",
                     "description": 'Например: {"name":"Алексей"} или {"preferences":{"кофе":"американо"}} или {"goals":["научиться плавать"]}'
                 }
             },
             "required": ["updates"]
         }
     }}
]

class Planner:
    def __init__(self, llm: LLMClient, memory: Memory, cfg=None):
        self.llm = llm
        self.memory = memory
        self.max_steps = (cfg or {}).get("max_steps", 10)

    async def run(self, user_input: str) -> str:
        history = self.memory.get_history()
        history.append({"role": "user", "content": user_input})
        self.memory.add("user", user_input)

        for _ in range(self.max_steps):
            msg = await self.llm.chat_with_tools(history, TOOLS)
            if not msg.get("tool_calls"):
                answer = (msg.get("content") or "").strip()
                if not answer:
                    answer = "Пустой ответ от модели."
                self.memory.add("assistant", answer)
                self.memory.after_turn(user_input, answer)
                return answer

            history.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": msg["tool_calls"],
            })

            for call in msg["tool_calls"]:
                try:
                    arguments = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                result = await self._execute(call["function"]["name"], arguments)
                if isinstance(result, (dict, list)):
                    content = json.dumps(result, ensure_ascii=False)
                else:
                    content = str(result)

                history.append({
                    "tool_call_id": call["id"],
                    "role": "tool",
                    "content": content,
                })

        answer = "Не смог выполнить задачу."
        self.memory.add("assistant", answer)
        self.memory.after_turn(user_input, answer)
        return answer

    async def _execute(self, name, args):
        if name == "get_current_time":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if name == "remember":
            content = args.get("content", "").strip()
            if not content:
                return "Пустое содержимое для памяти."
            self.memory.save(content, role="fact")
            return "Запомнено."
        if name == "recall":
            query = args.get("query", "").strip()
            if not query:
                return []
            return self.memory.recall(query)
        if name == "update_profile":
            updates = args.get("updates", {})
            if not isinstance(updates, dict):
                return "Некорректные данные профиля."
            self.memory.update_profile(updates)
            return "Профиль обновлён."
        return f"Неизвестный инструмент: {name}"
