from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .config import get_config
from .models import Fact, Task

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS state_versions (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL
);
INSERT OR IGNORE INTO state_versions(id, version) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, raw_json TEXT NOT NULL, state_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL, embedding_id TEXT,
  version_created INTEGER NOT NULL, version_superseded INTEGER,
  status TEXT NOT NULL DEFAULT 'active',
  source_event_id INTEGER REFERENCES events(id), meta_json TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
  constraint_json TEXT, active_from_version INTEGER NOT NULL,
  completed_version INTEGER
);
CREATE TABLE IF NOT EXISTS embedding_outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fact_id INTEGER NOT NULL REFERENCES facts(id),
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS consolidation_proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_fact_ids TEXT NOT NULL, proposed_content TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL, content TEXT NOT NULL,
  state_version INTEGER NOT NULL, ts REAL NOT NULL, summary_id INTEGER
);
CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
  covers_versions_from INTEGER NOT NULL, covers_versions_to INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
  USING fts5(content, content='facts', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TABLE IF NOT EXISTS vector_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  watermark INTEGER NOT NULL,
  updated_at REAL NOT NULL
);
INSERT OR IGNORE INTO vector_state(id, watermark, updated_at) VALUES (1, 0, 0);
"""


def _migrate_state_versions(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'state_versions'
        """
    ).fetchone()
    if row is None:
        return

    columns = conn.execute("PRAGMA table_info(state_versions)").fetchall()
    column_names = [str(column["name"]) for column in columns]
    if column_names == ["id", "version"]:
        current = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM state_versions"
        ).fetchone()
        current_version = int(current[0]) if current else 0
        conn.execute("DELETE FROM state_versions WHERE id != 1")
        conn.execute(
            """
            INSERT INTO state_versions(id, version)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET version = excluded.version
            """,
            (current_version,),
        )
        return

    current = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM state_versions"
    ).fetchone()
    current_version = int(current[0]) if current else 0
    conn.execute("ALTER TABLE state_versions RENAME TO state_versions_legacy")
    conn.execute(
        """
        CREATE TABLE state_versions (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          version INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO state_versions(id, version) VALUES (1, ?)",
        (current_version,),
    )
    conn.execute("DROP TABLE state_versions_legacy")


def _get_state_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT version FROM state_versions WHERE id = 1"
    ).fetchone()
    return int(row[0]) if row else 0


def _connect() -> sqlite3.Connection:
    config = get_config()
    conn = sqlite3.connect(config.SQLITE_PATH, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _loads_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    return json.loads(value)


def _fact_from_row(row: sqlite3.Row | None) -> Fact | None:
    if row is None:
        return None
    return Fact(
        id=int(row["id"]),
        content=row["content"],
        embedding_id=row["embedding_id"],
        version_created=int(row["version_created"]),
        version_superseded=row["version_superseded"],
        status=row["status"],
        source_event_id=row["source_event_id"],
        meta=_loads_json(row["meta_json"]),
    )


def _task_from_row(row: sqlite3.Row) -> Task:
    return Task(
        id=int(row["id"]),
        title=row["title"],
        status=row["status"],
        constraint=_loads_json(row["constraint_json"]),
        active_from_version=int(row["active_from_version"]),
        completed_version=row["completed_version"],
    )


def init_db() -> None:
    config = get_config()
    config.ensure_directories()
    conn = _connect()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _migrate_state_versions(conn)
        conn.executescript(SCHEMA_SQL)
    finally:
        conn.close()


@contextmanager
def db_transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def bump_version(conn: sqlite3.Connection) -> int:
    conn.execute("UPDATE state_versions SET version = version + 1 WHERE id = 1")
    return _get_state_version(conn)


def get_version() -> int:
    conn = _connect()
    try:
        return _get_state_version(conn)
    finally:
        conn.close()


def get_vector_watermark() -> int:
    conn = _connect()
    try:
        row = conn.execute("SELECT watermark FROM vector_state WHERE id = 1").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def insert_event(conn: sqlite3.Connection, raw: dict[str, Any]) -> int:
    ts = time.time()
    state_version = _get_state_version(conn)
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    cur = conn.execute(
        "INSERT INTO events(ts, raw_json, state_version) VALUES (?, ?, ?)",
        (ts, payload, state_version),
    )
    return int(cur.lastrowid)


def insert_message(
    conn: sqlite3.Connection,
    role: str,
    content: str,
    state_version: int | None = None,
    summary_id: int | None = None,
) -> int:
    resolved_version = (
        state_version
        if state_version is not None
        else _get_state_version(conn)
    )
    cur = conn.execute(
        """
        INSERT INTO messages(role, content, state_version, ts, summary_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (role, content, resolved_version, time.time(), summary_id),
    )
    return int(cur.lastrowid)


def insert_fact(conn: sqlite3.Connection, fact: Fact | dict[str, Any] | str, version_created: int) -> int:
    if isinstance(fact, Fact):
        content = fact.content
        source_event_id = fact.source_event_id
        meta_json = json.dumps(fact.meta or {}, ensure_ascii=False)
    elif isinstance(fact, str):
        content = fact.strip()
        source_event_id = None
        meta_json = json.dumps({}, ensure_ascii=False)
    else:
        content = str(fact.get("content", "")).strip()
        source_event_id = fact.get("source_event_id")
        meta_json = json.dumps(fact.get("meta", {}), ensure_ascii=False)
    if not content:
        raise ValueError("Fact content must not be empty.")
    cur = conn.execute(
        """
        INSERT INTO facts(
            content, embedding_id, version_created, version_superseded,
            status, source_event_id, meta_json
        ) VALUES (?, NULL, ?, NULL, 'active', ?, ?)
        """,
        (content, version_created, source_event_id, meta_json),
    )
    fact_id = int(cur.lastrowid)
    embedding_id = f"fact:{fact_id}"
    conn.execute("UPDATE facts SET embedding_id = ? WHERE id = ?", (embedding_id, fact_id))
    return fact_id


def insert_outbox(conn: sqlite3.Connection, fact_id: int) -> int:
    cur = conn.execute(
        """
        INSERT INTO embedding_outbox(fact_id, status, attempts, created_at)
        VALUES (?, 'pending', 0, ?)
        """,
        (fact_id, time.time()),
    )
    return int(cur.lastrowid)


def persist_result(conn: sqlite3.Connection, result: dict[str, Any], event_id: int) -> None:
    del event_id
    state_version = _get_state_version(conn)
    assistant_message = str(result.get("assistant_message") or "").strip()
    if assistant_message:
        insert_message(conn, "assistant", assistant_message, state_version=state_version)

    for task_data in result.get("created_tasks", []):
        title = str(task_data.get("title", "")).strip()
        if not title:
            continue
        constraint_json = json.dumps(task_data.get("constraint_json"), ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO tasks(title, status, constraint_json, active_from_version, completed_version)
            VALUES (?, 'active', ?, ?, NULL)
            """,
            (title, constraint_json, state_version),
        )

    for task_id in result.get("completed_task_ids", []):
        conn.execute(
            """
            UPDATE tasks
            SET status = 'completed', completed_version = ?
            WHERE id = ? AND status != 'completed'
            """,
            (state_version, int(task_id)),
        )


def get_fact_by_id(fact_id: int) -> Fact | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _fact_from_row(row)
    finally:
        conn.close()


def get_recent_messages(limit: int | None = None) -> list[dict[str, Any]]:
    resolved_limit = limit or get_config().MAX_RECENT_MESSAGES
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, role, content, state_version, ts, summary_id
            FROM messages
            WHERE summary_id IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (resolved_limit,),
        ).fetchall()
        rows = list(reversed(rows))
        messages = [
            {
                "id": int(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "state_version": int(row["state_version"]),
                "ts": float(row["ts"]),
                "summary_id": row["summary_id"],
            }
            for row in rows
        ]
        summary_row = conn.execute(
            """
            SELECT id, content, covers_versions_from, covers_versions_to, created_at
            FROM summaries
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        if summary_row is not None:
            summary_message = {
                "id": int(summary_row["id"]),
                "role": "system",
                "content": f"Summary of earlier conversation: {summary_row['content']}",
                "state_version": int(summary_row["covers_versions_to"]),
                "ts": float(summary_row["created_at"]),
                "summary_id": int(summary_row["id"]),
            }
            return [summary_message, *messages]
        return messages
    finally:
        conn.close()


def get_delta_facts(current_version: int, vector_watermark: int) -> list[Fact]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM facts
            WHERE version_created > ? AND version_created <= ?
            ORDER BY version_created ASC, id ASC
            """,
            (vector_watermark, current_version),
        ).fetchall()
        return [_fact_from_row(row) for row in rows if row is not None]
    finally:
        conn.close()


def get_active_tasks() -> list[Task]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM tasks
            WHERE status = 'active'
            ORDER BY active_from_version ASC, id ASC
            """
        ).fetchall()
        return [_task_from_row(row) for row in rows]
    finally:
        conn.close()


def claim_pending_outbox_item() -> dict[str, Any] | None:
    with db_transaction() as conn:
        row = conn.execute(
            """
            SELECT eo.id, eo.fact_id, eo.attempts, f.embedding_id, f.version_created
            FROM embedding_outbox eo
            JOIN facts f ON f.id = eo.fact_id
            WHERE eo.status = 'pending'
            ORDER BY eo.id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        updated = conn.execute(
            "UPDATE embedding_outbox SET status = 'processing' WHERE id = ? AND status = 'pending'",
            (row["id"],),
        ).rowcount
        if updated != 1:
            return None
        return {
            "id": int(row["id"]),
            "fact_id": int(row["fact_id"]),
            "attempts": int(row["attempts"]),
            "embedding_id": row["embedding_id"],
            "version_created": int(row["version_created"]),
        }


def mark_outbox_done(outbox_id: int) -> None:
    with db_transaction() as conn:
        conn.execute(
            "UPDATE embedding_outbox SET status = 'done' WHERE id = ?",
            (outbox_id,),
        )


def mark_outbox_failed(outbox_id: int) -> None:
    with db_transaction() as conn:
        conn.execute(
            """
            UPDATE embedding_outbox
            SET status = 'pending', attempts = attempts + 1
            WHERE id = ?
            """,
            (outbox_id,),
        )


def recompute_vector_watermark() -> int:
    with db_transaction() as conn:
        pending_row = conn.execute(
            """
            SELECT MIN(f.version_created) AS min_pending_version
            FROM embedding_outbox eo
            JOIN facts f ON f.id = eo.fact_id
            WHERE eo.status IN ('pending', 'processing')
            """
        ).fetchone()
        max_row = conn.execute(
            "SELECT COALESCE(MAX(version_created), 0) AS max_version FROM facts"
        ).fetchone()
        min_pending_version = pending_row["min_pending_version"] if pending_row else None
        max_version = int(max_row["max_version"]) if max_row else 0
        watermark = int(min_pending_version) - 1 if min_pending_version is not None else max_version
        conn.execute(
            "UPDATE vector_state SET watermark = ?, updated_at = ? WHERE id = 1",
            (watermark, time.time()),
        )
        return watermark


def list_recent_active_facts(limit: int = 50) -> list[Fact]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM facts
            WHERE status = 'active'
            ORDER BY version_created DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_fact_from_row(row) for row in rows if row is not None]
    finally:
        conn.close()


def insert_consolidation_proposal(source_fact_ids: list[int], proposed_content: str) -> int:
    with db_transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO consolidation_proposals(source_fact_ids, proposed_content, status, created_at)
            VALUES (?, ?, 'pending', ?)
            """,
            (json.dumps(source_fact_ids), proposed_content, time.time()),
        )
        return int(cur.lastrowid)


def list_pending_consolidation_proposals(limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, source_fact_ids, proposed_content, status, created_at
            FROM consolidation_proposals
            WHERE status = 'pending'
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "source_fact_ids": row["source_fact_ids"],
                "proposed_content": row["proposed_content"],
                "status": row["status"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_pending_proposal_pairs() -> set[tuple[int, ...]]:
    pairs: set[tuple[int, ...]] = set()
    for proposal in list_pending_consolidation_proposals(limit=500):
        try:
            raw_ids = json.loads(proposal["source_fact_ids"])
            pair = tuple(sorted(int(item) for item in raw_ids))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        pairs.add(pair)
    return pairs


def count_uncompacted_messages() -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE summary_id IS NULL"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def get_messages_for_compaction(limit: int = 30) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, role, content, state_version, ts
            FROM messages
            WHERE summary_id IS NULL
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "state_version": int(row["state_version"]),
                "ts": float(row["ts"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def create_summary_and_mark_messages(message_ids: list[int], summary_text: str) -> int | None:
    if not message_ids:
        return None
    with db_transaction() as conn:
        placeholders = ",".join("?" for _ in message_ids)
        rows = conn.execute(
            f"""
            SELECT MIN(state_version) AS from_version, MAX(state_version) AS to_version
            FROM messages
            WHERE id IN ({placeholders})
            """,
            tuple(message_ids),
        ).fetchone()
        cur = conn.execute(
            """
            INSERT INTO summaries(content, covers_versions_from, covers_versions_to, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                summary_text,
                int(rows["from_version"]),
                int(rows["to_version"]),
                time.time(),
            ),
        )
        summary_id = int(cur.lastrowid)
        conn.execute(
            f"UPDATE messages SET summary_id = ? WHERE id IN ({placeholders})",
            (summary_id, *message_ids),
        )
        return summary_id


def get_fact_rows_for_ids(conn: sqlite3.Connection, fact_ids: list[int]) -> list[Fact]:
    if not fact_ids:
        return []
    placeholders = ",".join("?" for _ in fact_ids)
    rows = conn.execute(
        f"SELECT * FROM facts WHERE id IN ({placeholders})",
        tuple(fact_ids),
    ).fetchall()
    return [_fact_from_row(row) for row in rows if row is not None]
