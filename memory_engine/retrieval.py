from __future__ import annotations

import logging
import re
from threading import Lock
from typing import Any
import warnings

from .config import get_config
from .db import (
    _connect,
    _fact_from_row,
    get_active_tasks as get_persisted_active_tasks,
    get_delta_facts as get_db_delta_facts,
    get_fact_by_id,
    get_recent_messages,
)
from .embeddings import embed
from .models import ContextSnapshot, Fact, Task
from .working_memory import working_memory

_collection = None
_collection_lock = Lock()
_vector_store_disabled = False
_vector_store_disable_reason = ""
logger = logging.getLogger(__name__)


class MemoryEmbeddingFunction:
    def __call__(self, input: Any) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        return [embed(text) for text in texts]

    def embed_query(self, input: Any) -> list[list[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "memory_engine"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "MemoryEmbeddingFunction":
        MemoryEmbeddingFunction.validate_config(config)
        return MemoryEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        return {}

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def validate_config_update(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> None:
        self.validate_config(old_config)
        self.validate_config(new_config)

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError("Embedding function config must be a dict.")


def _disable_vector_store(exc: Exception) -> None:
    global _vector_store_disabled, _vector_store_disable_reason

    if _vector_store_disabled:
        return
    _vector_store_disabled = True
    _vector_store_disable_reason = str(exc).strip() or exc.__class__.__name__
    logger.info(
        "Vector store disabled; continuing without ChromaDB-backed retrieval. Reason: %s",
        _vector_store_disable_reason,
    )


def vector_store_is_available() -> bool:
    return get_collection() is not None


def get_collection():
    global _collection
    if _vector_store_disabled:
        return None
    if _collection is not None:
        return _collection
    with _collection_lock:
        if _vector_store_disabled:
            return None
        if _collection is None:
            config = get_config()
            config.ensure_directories()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
                        category=UserWarning,
                    )
                    import chromadb
                    from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=str(config.CHROMA_PATH),
                    settings=Settings(anonymized_telemetry=False),
                )
                _collection = client.get_or_create_collection(
                    name="facts",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=MemoryEmbeddingFunction(),
                )
            except Exception as exc:
                _disable_vector_store(exc)
                return None
    return _collection


def _fts_query(query: str) -> str:
    tokens = list(dict.fromkeys(re.findall(r"\w+", query.lower())))
    return " OR ".join(f"{token}*" for token in tokens[:8])


def fts_search(query: str, n: int | None = None) -> list[Fact]:
    resolved_n = n or get_config().MAX_CONTEXT_FACTS
    fts_query_text = _fts_query(query)
    if not fts_query_text:
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT f.*
            FROM facts_fts
            JOIN facts f ON f.id = facts_fts.rowid
            WHERE facts_fts MATCH ? AND f.status = 'active'
            ORDER BY bm25(facts_fts), f.id DESC
            LIMIT ?
            """,
            (fts_query_text, resolved_n),
        ).fetchall()
        return [_fact_from_row(row) for row in rows if row is not None]
    finally:
        conn.close()


def chroma_search(query: str, n: int | None = None, vector_watermark: int = 0) -> list[Fact]:
    resolved_n = n or get_config().MAX_CONTEXT_FACTS
    if not query.strip():
        return []
    collection = get_collection()
    if collection is None:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=resolved_n,
        where={"version_created": {"$lte": vector_watermark}},
        include=["documents", "metadatas", "distances"],
    )
    metadatas = (results.get("metadatas") or [[]])[0]
    seen: set[int] = set()
    facts: list[Fact] = []
    for metadata in metadatas:
        if not metadata:
            continue
        fact_id = int(metadata.get("fact_id", 0))
        if fact_id in seen:
            continue
        fact = get_fact_by_id(fact_id)
        if fact is None or fact.status != "active":
            continue
        seen.add(fact_id)
        facts.append(fact)
    return facts


def get_delta_facts(current_version: int, vector_watermark: int) -> list[Fact]:
    return get_db_delta_facts(current_version, vector_watermark)


def _merge_tasks(persisted_tasks: list[Task], transient_tasks: list[Task]) -> list[Task]:
    merged: list[Task] = []
    seen: set[tuple[int, str]] = set()
    for task in [*persisted_tasks, *transient_tasks]:
        key = (task.id, task.title)
        if key in seen:
            continue
        seen.add(key)
        merged.append(task)
    return merged


def build_context_snapshot(current_version: int, vector_watermark: int, query: str) -> ContextSnapshot:
    config = get_config()
    persisted_tasks = get_persisted_active_tasks()
    transient_tasks = working_memory.get_active_tasks()
    constraints = working_memory.get_constraints()
    fts_results = fts_search(query, config.MAX_CONTEXT_FACTS)
    vector_results = chroma_search(query, config.MAX_CONTEXT_FACTS, vector_watermark)
    delta_facts = get_db_delta_facts(current_version, vector_watermark)
    recent_messages = get_recent_messages(config.MAX_RECENT_MESSAGES)
    return ContextSnapshot(
        state_version=current_version,
        vector_watermark=vector_watermark,
        tasks=_merge_tasks(persisted_tasks, transient_tasks),
        constraints=constraints,
        fts_results=fts_results,
        vector_results=vector_results,
        delta_facts=delta_facts,
        recent_messages=recent_messages,
    )
