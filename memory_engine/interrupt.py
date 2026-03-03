from __future__ import annotations

import asyncio
from enum import IntEnum


class Priority(IntEnum):
    HIGH = 0
    NORMAL = 10
    LOW = 20


class InterruptChannel:
    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue[tuple[int, dict]] = asyncio.PriorityQueue()

    async def send(self, p: Priority, payload: dict) -> None:
        await self._q.put((int(p), payload))

    async def check(self) -> dict | None:
        try:
            _, payload = self._q.get_nowait()
            return payload
        except asyncio.QueueEmpty:
            return None
