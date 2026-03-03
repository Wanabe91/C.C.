from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from .config import get_config
from .consolidator import run_consolidator
from .db import init_db
from .indexer import run_indexer
from .interrupt import InterruptChannel
from .loop import ingest_event


def _parse_event_stream(payload: str) -> list[dict[str, Any]]:
    payload = payload.strip()
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    events: list[dict[str, Any]] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed_line = json.loads(line)
        if isinstance(parsed_line, dict):
            events.append(parsed_line)
    return events


async def _collect_events() -> list[dict[str, Any]]:
    if not sys.stdin.isatty():
        payload = await asyncio.to_thread(sys.stdin.read)
        return _parse_event_stream(payload)

    events: list[dict[str, Any]] = []
    while True:
        try:
            line = await asyncio.to_thread(input, "json> ")
        except EOFError:
            break
        line = line.strip()
        if not line:
            continue
        events.extend(_parse_event_stream(line))
    return events


async def start() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    get_config()
    init_db()
    stop_event = asyncio.Event()
    interrupt = InterruptChannel()
    workers = [
        asyncio.create_task(run_indexer(stop_event), name="memory-indexer"),
        asyncio.create_task(run_consolidator(stop_event), name="memory-consolidator"),
    ]

    try:
        for raw in await _collect_events():
            await ingest_event(raw, interrupt)
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=5)
        except asyncio.TimeoutError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(start())
