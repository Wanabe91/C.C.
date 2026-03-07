from __future__ import annotations

import logging
import re
import time
import warnings
from threading import Lock
from typing import Any

from .config import get_config
from .db import (
    _connect,
    _fact_from_row,
    get_active_tasks as get_persisted_active_tasks,
    get_delta_facts as get_db_delta_facts,
    get_fact_rows_for_ids,
    get_recent_messages,
    touch_fact_accesses,
)
from .embeddings import embed_many
from .models import ContextFingerprint, ContextSnapshot, Fact, FingerprintDiff, Task
from .working_memory import working_memory

_collection = None
_collection_lock = Lock()
_VECTOR_STORE_RETRY_COOLDOWN_SEC = 5.0
_vector_store_retry_after = 0.0
_vector_store_disable_reason = ""
logger = logging.getLogger(__name__)

_FTS_STOPWORDS = {
    "и",
    "в",
    "на",
    "с",
    "по",
    "к",
    "а",
    "но",
    "что",
    "это",
    "я",
    "не",
    "он",
    "она",
    "они",
    "мы",
    "вы",
    "то",
    "как",
    "the",
    "a",
    "an",
    "is",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "and",
    "or",
    "not",
    "it",
    "be",
    "do",
}


class MemoryEmbeddingFunction:
    def __call__(self, input: Any) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else list(input)
        return embed_many([str(text) for text in texts])

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


def invalidate_vector_store(exc: Exception) -> None:
    global _collection, _vector_store_retry_after, _vector_store_disable_reason

    reason = str(exc).strip() or exc.__class__.__name__
    now = time.time()
    should_log = reason != _vector_store_disable_reason or now >= _vector_store_retry_after
    _collection = None
    _vector_store_disable_reason = reason
    _vector_store_retry_after = now + _VECTOR_STORE_RETRY_COOLDOWN_SEC
    if should_log:
        logger.warning(
            "Vector store unavailable; retrying in %.1fs. Reason: %s",
            _VECTOR_STORE_RETRY_COOLDOWN_SEC,
            _vector_store_disable_reason,
        )


def reset_vector_store_state() -> None:
    global _collection, _vector_store_retry_after, _vector_store_disable_reason

    _collection = None
    _vector_store_retry_after = 0.0
    _vector_store_disable_reason = ""


def vector_store_is_available() -> bool:
    return get_collection() is not None


def get_collection():
    global _collection, _vector_store_retry_after, _vector_store_disable_reason
    if _collection is not None:
        return _collection
    if time.time() < _vector_store_retry_after:
        return None
    with _collection_lock:
        if time.time() < _vector_store_retry_after:
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
                _vector_store_retry_after = 0.0
                _vector_store_disable_reason = ""
            except Exception as exc:
                invalidate_vector_store(exc)
                return None
    return _collection


def _fts_query(query: str) -> str:
    tokens = list(
        dict.fromkeys(
            t
            for t in re.findall(r"\w+", query.lower())
            if t not in _FTS_STOPWORDS
        )
    )
    return " OR ".join(f"{token}*" for token in tokens[:8])


def fts_search(query: str, n: int | None = None, conn=None) -> list[Fact]:
    resolved_n = n or get_config().MAX_CONTEXT_FACTS
    fts_query_text = _fts_query(query)
    if not fts_query_text:
        return []
    local_conn = conn if conn is not None else _connect()
    try:
        rows = local_conn.execute(
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
        if conn is None:
            local_conn.close()


def chroma_search(query: str, n: int | None = None, vector_watermark: int = 0) -> list[Fact]:
    resolved_n = n or get_config().MAX_CONTEXT_FACTS
    if not query.strip():
        return []
    collection = get_collection()
    if collection is None:
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=resolved_n,
            where={"version_created": {"$lte": vector_watermark}},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        invalidate_vector_store(exc)
        return []
    metadatas = (results.get("metadatas") or [[]])[0]
    ordered_fact_ids: list[int] = []
    seen: set[int] = set()
    for metadata in metadatas:
        if not metadata:
            continue
        try:
            fact_id = int(metadata.get("fact_id", 0))
        except (TypeError, ValueError):
            continue
        if fact_id <= 0:
            continue
        if fact_id in seen:
            continue
        seen.add(fact_id)
        ordered_fact_ids.append(fact_id)
    if not ordered_fact_ids:
        return []

    conn = _connect()
    try:
        db_facts = get_fact_rows_for_ids(conn, ordered_fact_ids)
    finally:
        conn.close()

    facts_by_id = {fact.id: fact for fact in db_facts}
    facts: list[Fact] = []
    for fact_id in ordered_fact_ids:
        fact = facts_by_id.get(fact_id)
        if fact is None or fact.status != "active":
            continue
        facts.append(fact)
    return facts


def get_delta_facts(
    current_version: int,
    vector_watermark: int,
    conn=None,
) -> list[Fact]:
    return get_db_delta_facts(current_version, vector_watermark, conn=conn)


def get_pinned_facts(conn) -> list[Fact]:
    rows = conn.execute(
        """
        SELECT *
        FROM facts
        WHERE status = 'active'
          AND json_extract(meta_json, '$.kind') IN ('user_model', 'preference')
        ORDER BY importance DESC, COALESCE(confidence_score, 0) DESC, last_accessed_at DESC
        """
    ).fetchall()
    return [_fact_from_row(row) for row in rows if row is not None]


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


def _fact_fingerprint_tuple(fact: Fact) -> tuple[int, str, str]:
    return (fact.version_created, fact.status, fact.tier)


def _load_last_entity_id(conn, table_name: str) -> str | None:
    row = conn.execute(f"SELECT id FROM {table_name} ORDER BY id DESC LIMIT 1").fetchone()
    return None if row is None else str(int(row["id"]))


def _load_active_task_ids(conn) -> set[str]:
    rows = conn.execute(
        """
        SELECT id
        FROM tasks
        WHERE status = 'active'
        ORDER BY id ASC
        """
    ).fetchall()
    return {str(int(row["id"])) for row in rows}


def _build_fact_versions(facts: list[Fact]) -> dict[str, tuple[int, str, str]]:
    return {str(fact.id): _fact_fingerprint_tuple(fact) for fact in facts}


def build_context_fingerprint(
    facts: list[Fact],
    active_task_ids: set[str],
    conn=None,
) -> ContextFingerprint:
    local_conn = conn if conn is not None else _connect()
    try:
        return ContextFingerprint(
            fact_versions=_build_fact_versions(facts),
            active_task_ids=set(active_task_ids),
            last_summary_id=_load_last_entity_id(local_conn, "summaries"),
            last_message_id=_load_last_entity_id(local_conn, "messages"),
            # vector_watermark is excluded because index lag is async bookkeeping, not semantic drift.
        )
    finally:
        if conn is None:
            local_conn.close()


def refresh_context_fingerprint(snapshot_fingerprint: ContextFingerprint) -> ContextFingerprint:
    tracked_fact_ids: list[int] = []
    for fact_id in snapshot_fingerprint.fact_versions:
        try:
            tracked_fact_ids.append(int(fact_id))
        except (TypeError, ValueError):
            continue

    conn = _connect()
    try:
        current_fact_versions: dict[str, tuple[int, str, str]] = {}
        if tracked_fact_ids:
            placeholders = ",".join("?" for _ in tracked_fact_ids)
            rows = conn.execute(
                f"""
                SELECT id, version_created, status, tier
                FROM facts
                WHERE id IN ({placeholders})
                """,
                tuple(tracked_fact_ids),
            ).fetchall()
            for row in rows:
                current_fact_versions[str(int(row["id"]))] = (
                    int(row["version_created"]),
                    row["status"],
                    row["tier"],
                )
        return ContextFingerprint(
            fact_versions=current_fact_versions,
            active_task_ids=_load_active_task_ids(conn),
            last_summary_id=_load_last_entity_id(conn, "summaries"),
            last_message_id=_load_last_entity_id(conn, "messages"),
        )
    finally:
        conn.close()


def fingerprint_diff(a: ContextFingerprint, b: ContextFingerprint) -> FingerprintDiff:
    changed_fact_ids = sorted(
        fact_id
        for fact_id, snapshot_state in a.fact_versions.items()
        if fact_id in b.fact_versions and b.fact_versions[fact_id] != snapshot_state
    )
    removed_fact_ids = sorted(
        fact_id for fact_id in a.fact_versions if fact_id not in b.fact_versions
    )
    task_changes = a.active_task_ids != b.active_task_ids
    message_changes = (
        a.last_summary_id != b.last_summary_id
        or a.last_message_id != b.last_message_id
    )
    return FingerprintDiff(
        changed_fact_ids=changed_fact_ids,
        removed_fact_ids=removed_fact_ids,
        task_changes=task_changes,
        message_changes=message_changes,
        is_empty=not (changed_fact_ids or removed_fact_ids or task_changes or message_changes),
    )


def build_context_snapshot(
    current_version: int,
    event_version: int,
    vector_watermark: int,
    query: str,
) -> ContextSnapshot:
    config = get_config()
    conn = _connect()
    try:
        pinned_facts = get_pinned_facts(conn)
        pinned_fact_ids = {fact.id for fact in pinned_facts}
        persisted_tasks = get_persisted_active_tasks(conn=conn)
        transient_tasks = working_memory.get_active_tasks()
        constraints = working_memory.get_constraints()
        fts_results = [
            fact
            for fact in fts_search(query, config.MAX_CONTEXT_FACTS, conn=conn)
            if fact.id not in pinned_fact_ids
        ]
        vector_results = [
            fact
            for fact in chroma_search(query, config.MAX_CONTEXT_FACTS, vector_watermark)
            if fact.id not in pinned_fact_ids
        ]
        delta_facts = get_db_delta_facts(current_version, vector_watermark, conn=conn)
        all_context_facts = [*pinned_facts, *fts_results, *vector_results, *delta_facts]
        seen_ids: set[int] = set()
        deduped: list[Fact] = []
        for fact in all_context_facts:
            if fact.id not in seen_ids:
                seen_ids.add(fact.id)
                deduped.append(fact)
        all_context_facts = deduped
        touched_fact_ids = sorted({fact.id for fact in all_context_facts})
        if touched_fact_ids:
            touch_fact_accesses(touched_fact_ids, conn=conn)
        fingerprint = build_context_fingerprint(
            all_context_facts,
            {str(task.id) for task in persisted_tasks if task.status == "active"},
            conn=conn,
        )
    finally:
        conn.close()
    recent_messages = get_recent_messages(config.MAX_RECENT_MESSAGES)
    working_memory_refs = working_memory.list_refs(limit=config.WORKING_MEMORY_SNAPSHOT_REF_LIMIT)
    return ContextSnapshot(
        state_version=current_version,
        event_version=event_version,
        vector_watermark=vector_watermark,
        fingerprint=fingerprint,
        tasks=_merge_tasks(persisted_tasks, transient_tasks),
        constraints=constraints,
        fts_results=fts_results,
        vector_results=vector_results,
        delta_facts=delta_facts,
        recent_messages=recent_messages,
        working_memory_refs=working_memory_refs,
        pinned_facts=pinned_facts,
    )
