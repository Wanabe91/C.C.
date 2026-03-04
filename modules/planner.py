from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

from memory_engine.config import get_config
from memory_engine.consolidator import run_consolidator
from memory_engine.db import init_db
from memory_engine.indexer import run_indexer
from memory_engine.interrupt import InterruptChannel
from memory_engine.loop import ingest_event


def _normalize_llm_base_url(raw_url: str) -> str:
    value = (raw_url or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value[: -len("/chat/completions")]
    return value


def _default_sqlite_path(memory_cfg: dict) -> str:
    explicit = str(memory_cfg.get("sqlite_path") or "").strip()
    if explicit:
        return explicit

    episodic_db = str(memory_cfg.get("episodic_db") or "").strip()
    if episodic_db.endswith(".db"):
        return episodic_db
    if episodic_db:
        return str(Path(episodic_db).parent / "agent_memory.db")
    return "./agent_memory.db"


def _default_chroma_path(memory_cfg: dict) -> str:
    explicit = str(memory_cfg.get("chroma_path") or "").strip()
    if explicit:
        return explicit

    episodic_db = str(memory_cfg.get("episodic_db") or "").strip()
    if episodic_db:
        if episodic_db.endswith(".db"):
            return str(Path(episodic_db).with_suffix("").parent / "chroma_store")
        return episodic_db
    return "./chroma_store"


def _default_obsidian_vault(memory_cfg: dict) -> str:
    explicit = str(memory_cfg.get("obsidian_vault") or "").strip()
    if explicit:
        return explicit

    sqlite_path = _default_sqlite_path(memory_cfg)
    return str(Path(sqlite_path).resolve().parent / "obsidian")


def _configure_memory_engine(app_cfg: dict) -> None:
    llm_cfg = app_cfg.get("llm", {}) if isinstance(app_cfg, dict) else {}
    memory_cfg = app_cfg.get("memory", {}) if isinstance(app_cfg, dict) else {}
    planner_cfg = app_cfg.get("planner", {}) if isinstance(app_cfg, dict) else {}

    base_url = _normalize_llm_base_url(str(llm_cfg.get("base_url") or "http://localhost:1234/v1"))
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid LLM base URL: {base_url}")

    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_MODEL"] = str(llm_cfg.get("model") or "local-model")
    os.environ["ASSISTANT_SYSTEM_PROMPT"] = str(llm_cfg.get("system_prompt") or "")
    os.environ["SQLITE_PATH"] = _default_sqlite_path(memory_cfg)
    os.environ["CHROMA_PATH"] = _default_chroma_path(memory_cfg)
    os.environ["OBSIDIAN_VAULT_PATH"] = _default_obsidian_vault(memory_cfg)

    embed_backend = str(memory_cfg.get("embed_backend") or os.environ.get("EMBED_BACKEND") or "").strip()
    if embed_backend:
        os.environ["EMBED_BACKEND"] = embed_backend

    embed_model = str(memory_cfg.get("embed_model") or os.environ.get("EMBED_MODEL") or "").strip()
    if embed_model:
        os.environ["EMBED_MODEL"] = embed_model

    max_recent_messages = planner_cfg.get("max_recent_messages")
    if isinstance(max_recent_messages, int) and max_recent_messages > 0:
        os.environ["MAX_RECENT_MESSAGES"] = str(max_recent_messages)

    max_context_facts = memory_cfg.get("max_context_facts")
    if isinstance(max_context_facts, int) and max_context_facts > 0:
        os.environ["MAX_CONTEXT_FACTS"] = str(max_context_facts)

    get_config.cache_clear()


class Planner:
    def __init__(self, app_cfg: dict | None = None):
        self.app_cfg = app_cfg or {}
        _configure_memory_engine(self.app_cfg)
        init_db()

        self._interrupt = InterruptChannel()
        self._stop_event: asyncio.Event | None = None
        self._workers: list[asyncio.Task] = []

    async def _ensure_workers(self) -> None:
        if self._workers and all(not worker.done() for worker in self._workers):
            return
        await self.close()
        self._stop_event = asyncio.Event()
        self._workers = [
            asyncio.create_task(run_indexer(self._stop_event), name="memory-indexer"),
            asyncio.create_task(run_consolidator(self._stop_event), name="memory-consolidator"),
        ]

    async def run(self, user_input: str) -> str:
        await self._ensure_workers()
        responses = await ingest_event({"text": user_input}, self._interrupt)
        if not responses:
            return "No assistant response was produced."
        return "\n".join(item for item in responses if item)

    async def close(self) -> None:
        workers = self._workers
        stop_event = self._stop_event
        if not workers:
            self._stop_event = None
            return

        self._workers = []
        self._stop_event = None
        if stop_event is not None:
            stop_event.set()
        try:
            await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=5)
        except asyncio.TimeoutError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
