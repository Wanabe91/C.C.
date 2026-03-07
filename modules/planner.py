from __future__ import annotations

import asyncio
from pathlib import Path

from memory_engine.config import Config, set_active_config
from memory_engine.consolidator import run_consolidator
from memory_engine.db import init_db
from memory_engine.indexer import run_indexer
from memory_engine.interrupt import InterruptChannel
from memory_engine.llm import set_runtime_llm_config
from memory_engine.loop import ingest_event
from memory_engine.tool_registry import assert_registry_integrity


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


def _default_working_memory_path(memory_cfg: dict) -> str:
    explicit = str(memory_cfg.get("working_memory_path") or "").strip()
    if explicit:
        return explicit

    sqlite_path = _default_sqlite_path(memory_cfg)
    return str(Path(sqlite_path).resolve().parent / "working_memory")


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


def _build_memory_engine_config(app_cfg: dict) -> Config:
    llm_cfg = app_cfg.get("llm", {}) if isinstance(app_cfg, dict) else {}
    memory_cfg = app_cfg.get("memory", {}) if isinstance(app_cfg, dict) else {}
    planner_cfg = app_cfg.get("planner", {}) if isinstance(app_cfg, dict) else {}

    sqlite_path = Path(_default_sqlite_path(memory_cfg)).expanduser().resolve()
    # Keep these defaults aligned with load_config_from_env() so both startup modes behave the same.
    return Config(
        SQLITE_PATH=sqlite_path,
        CHROMA_PATH=Path(_default_chroma_path(memory_cfg)).expanduser().resolve(),
        OBSIDIAN_VAULT_PATH=Path(_default_obsidian_vault(memory_cfg)).expanduser().resolve(),
        WORKING_MEMORY_PATH=Path(_default_working_memory_path(memory_cfg)).expanduser().resolve(),
        ASSISTANT_SYSTEM_PROMPT=str(llm_cfg.get("system_prompt") or ""),
        EMBED_MODEL=str(
            memory_cfg.get("embed_model")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        VERSION_DRIFT_THRESHOLD=_positive_int(planner_cfg.get("version_drift_threshold"), 5),
        CONSOLIDATION_INTERVAL_SEC=_positive_int(memory_cfg.get("consolidation_interval_sec"), 60),
        INDEXER_POLL_INTERVAL_SEC=_positive_int(memory_cfg.get("indexer_poll_interval_sec"), 2),
        MAX_CONTEXT_FACTS=_positive_int(memory_cfg.get("max_context_facts"), 20),
        MAX_RECENT_MESSAGES=_positive_int(planner_cfg.get("max_recent_messages"), 10),
        WORKING_MEMORY_OFFLOAD_CHAR_THRESHOLD=_positive_int(
            memory_cfg.get("working_memory_offload_char_threshold"),
            8000,
        ),
        WORKING_MEMORY_READ_CHAR_LIMIT=_positive_int(
            memory_cfg.get("working_memory_read_char_limit"),
            4000,
        ),
        WORKING_MEMORY_SEARCH_LIMIT=_positive_int(
            memory_cfg.get("working_memory_search_limit"),
            5,
        ),
        WORKING_MEMORY_SNAPSHOT_REF_LIMIT=_positive_int(
            memory_cfg.get("working_memory_snapshot_ref_limit"),
            10,
        ),
        max_steps_per_event=_positive_int(planner_cfg.get("max_steps_per_event"), 10),
    )


class Planner:
    def __init__(self, app_cfg: dict | None = None):
        self.app_cfg = app_cfg or {}
        set_runtime_llm_config(self.app_cfg)
        memory_cfg = _build_memory_engine_config(self.app_cfg)
        set_active_config(memory_cfg)
        assert_registry_integrity()
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
            asyncio.create_task(
                run_consolidator(self._stop_event),
                name="memory-consolidator",
            ),
        ]

    async def run(self, user_input: str) -> str:
        await self._ensure_workers()
        responses = await ingest_event(
            {"text": user_input},
            self._interrupt,
        )
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
