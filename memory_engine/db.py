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
  version INTEGER NOT NULL,
  event_version INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO state_versions(id, version, event_version) VALUES (1, 0, 0);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, raw_json TEXT NOT NULL, state_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL, embedding_id TEXT,
  version_created INTEGER NOT NULL, version_superseded INTEGER,
  status TEXT NOT NULL DEFAULT 'active',
  importance TEXT NOT NULL DEFAULT 'contextual',
  tier TEXT NOT NULL DEFAULT 'active',
  source_event_id INTEGER REFERENCES events(id), meta_json TEXT,
  created_at REAL,
  last_accessed_at REAL,
  access_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
  constraint_json TEXT, active_from_version INTEGER NOT NULL,
  completed_version INTEGER,
  created_at REAL,
  completed_at REAL
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
  proposal_type TEXT,
  proposal_json TEXT,
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
CREATE TABLE IF NOT EXISTS planner_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL REFERENCES events(id),
  goal TEXT NOT NULL,
  snapshot_state_version INTEGER NOT NULL,
  vector_watermark INTEGER NOT NULL,
  planner_status TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  user_prompt TEXT NOT NULL,
  first_response TEXT NOT NULL,
  repair_prompt TEXT,
  repair_response TEXT,
  final_steps_json TEXT NOT NULL,
  error_text TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS step_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  planner_run_id INTEGER NOT NULL REFERENCES planner_runs(id),
  step_index INTEGER NOT NULL,
  action TEXT NOT NULL,
  tool TEXT,
  args_json TEXT NOT NULL,
  precondition_fact_ids TEXT NOT NULL,
  reasoning TEXT NOT NULL,
  snapshot_state_version INTEGER NOT NULL,
  current_state_version INTEGER,
  revalidation_status TEXT NOT NULL,
  rejection_reason TEXT,
  replan_reason TEXT,
  replan_count INTEGER,
  replan_diff_summary TEXT,
  execution_status TEXT,
  result_json TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS weekly_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER REFERENCES events(id),
  week_key TEXT NOT NULL,
  title TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  markdown TEXT NOT NULL,
  note_path TEXT,
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
    if set(column_names) == {"id", "version", "event_version"}:
        current = conn.execute(
            "SELECT COALESCE(MAX(version), 0), COALESCE(MAX(event_version), 0) FROM state_versions"
        ).fetchone()
        current_version = int(current[0]) if current else 0
        current_event_version = int(current[1]) if current else 0
        conn.execute("DELETE FROM state_versions WHERE id != 1")
        conn.execute(
            """
            INSERT INTO state_versions(id, version, event_version)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              version = excluded.version,
              event_version = excluded.event_version
            """,
            (current_version, max(current_event_version, current_version)),
        )
        return

    if set(column_names) == {"id", "version"}:
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
        conn.execute(
            "ALTER TABLE state_versions ADD COLUMN event_version INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "UPDATE state_versions SET event_version = ? WHERE id = 1",
            (current_version,),
        )
        return

    current = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM state_versions"
    ).fetchone()
    current_version = int(current[0]) if current else 0
    if "event_version" in column_names:
        current_event = conn.execute(
            "SELECT COALESCE(MAX(event_version), 0) FROM state_versions"
        ).fetchone()
        current_event_version = int(current_event[0]) if current_event else 0
    else:
        current_event_version = current_version
    conn.execute("ALTER TABLE state_versions RENAME TO state_versions_legacy")
    conn.execute(
        """
        CREATE TABLE state_versions (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          version INTEGER NOT NULL,
          event_version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO state_versions(id, version, event_version) VALUES (1, ?, ?)",
        (current_version, max(current_event_version, current_version)),
    )
    conn.execute("DROP TABLE state_versions_legacy")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    if column_name in _table_columns(conn, table_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _migrate_runtime_tables(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "facts", "created_at", "REAL")
    _ensure_column(conn, "facts", "importance", "TEXT NOT NULL DEFAULT 'contextual'")
    _ensure_column(conn, "facts", "tier", "TEXT NOT NULL DEFAULT 'active'")
    _ensure_column(conn, "facts", "last_accessed_at", "REAL")
    _ensure_column(conn, "facts", "access_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "tasks", "created_at", "REAL")
    _ensure_column(conn, "tasks", "completed_at", "REAL")
    _ensure_column(conn, "weekly_reviews", "note_path", "TEXT")
    _ensure_column(conn, "consolidation_proposals", "proposal_type", "TEXT")
    _ensure_column(conn, "consolidation_proposals", "proposal_json", "TEXT")
    _ensure_column(conn, "step_traces", "replan_reason", "TEXT")
    _ensure_column(conn, "step_traces", "replan_count", "INTEGER")
    _ensure_column(conn, "step_traces", "replan_diff_summary", "TEXT")
    conn.execute(
        """
        UPDATE facts
        SET importance = 'contextual'
        WHERE importance IS NULL OR TRIM(importance) NOT IN ('core', 'contextual', 'transient')
        """
    )
    conn.execute(
        """
        UPDATE facts
        SET tier = 'active'
        WHERE tier IS NULL OR TRIM(tier) NOT IN ('active', 'cold', 'archived')
        """
    )
    conn.execute(
        """
        UPDATE facts
        SET access_count = 0
        WHERE access_count IS NULL OR access_count < 0
        """
    )


def _get_state_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT version FROM state_versions WHERE id = 1"
    ).fetchone()
    return int(row[0]) if row else 0


def _get_event_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT event_version FROM state_versions WHERE id = 1"
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


def _normalize_fact_importance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"core", "contextual", "transient"}:
        return normalized
    return "contextual"


def _normalize_fact_tier(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"active", "cold", "archived"}:
        return normalized
    return "active"


def _normalize_access_count(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 0
    return normalized if normalized >= 0 else 0


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
        importance=_normalize_fact_importance(row["importance"] if "importance" in row.keys() else None),
        tier=_normalize_fact_tier(row["tier"] if "tier" in row.keys() else None),
        source_event_id=row["source_event_id"],
        created_at=float(row["created_at"]) if "created_at" in row.keys() and row["created_at"] is not None else None,
        last_accessed_at=(
            float(row["last_accessed_at"])
            if "last_accessed_at" in row.keys() and row["last_accessed_at"] is not None
            else None
        ),
        access_count=_normalize_access_count(row["access_count"] if "access_count" in row.keys() else None),
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
        _migrate_runtime_tables(conn)
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


def bump_event_version(conn: sqlite3.Connection) -> int:
    # event_version advances once per ingested event; version still tracks every write.
    conn.execute("UPDATE state_versions SET event_version = event_version + 1 WHERE id = 1")
    return _get_event_version(conn)


def get_version() -> int:
    conn = _connect()
    try:
        return _get_state_version(conn)
    finally:
        conn.close()


def get_event_version(conn: sqlite3.Connection | None = None) -> int:
    if conn is not None:
        return _get_event_version(conn)
    local_conn = _connect()
    try:
        return _get_event_version(local_conn)
    finally:
        local_conn.close()


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
        importance = _normalize_fact_importance(fact.importance)
        tier = _normalize_fact_tier(fact.tier)
        created_at = fact.created_at if fact.created_at is not None else time.time()
        last_accessed_at = fact.last_accessed_at
        access_count = _normalize_access_count(fact.access_count)
    elif isinstance(fact, str):
        content = fact.strip()
        source_event_id = None
        meta_json = json.dumps({}, ensure_ascii=False)
        importance = "contextual"
        tier = "active"
        created_at = time.time()
        last_accessed_at = None
        access_count = 0
    else:
        content = str(fact.get("content", "")).strip()
        source_event_id = fact.get("source_event_id")
        meta_json = json.dumps(fact.get("meta", {}), ensure_ascii=False)
        importance = _normalize_fact_importance(fact.get("importance"))
        tier = _normalize_fact_tier(fact.get("tier"))
        created_at = float(fact["created_at"]) if fact.get("created_at") is not None else time.time()
        last_accessed_at = (
            float(fact["last_accessed_at"])
            if fact.get("last_accessed_at") is not None
            else None
        )
        access_count = _normalize_access_count(fact.get("access_count"))
    if not content:
        raise ValueError("Fact content must not be empty.")
    cur = conn.execute(
        """
        INSERT INTO facts(
            content, embedding_id, version_created, version_superseded,
            status, importance, tier, source_event_id, meta_json, created_at, last_accessed_at, access_count
        ) VALUES (?, NULL, ?, NULL, 'active', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            version_created,
            importance,
            tier,
            source_event_id,
            meta_json,
            created_at,
            last_accessed_at,
            access_count,
        ),
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
            INSERT INTO tasks(
                title, status, constraint_json, active_from_version, completed_version, created_at, completed_at
            )
            VALUES (?, 'active', ?, ?, NULL, ?, NULL)
            """,
            (title, constraint_json, state_version, time.time()),
        )

    for task_id in result.get("completed_task_ids", []):
        conn.execute(
            """
            UPDATE tasks
            SET status = 'completed', completed_version = ?, completed_at = ?
            WHERE id = ? AND status != 'completed'
            """,
            (state_version, time.time(), int(task_id)),
        )

    for review_data in result.get("generated_reviews", []):
        insert_weekly_review(conn, review_data, event_id=event_id)


def get_fact_by_id(fact_id: int) -> Fact | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return _fact_from_row(row)
    finally:
        conn.close()


def get_fact_record_by_id(fact_id: int) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT
              id, content, status, importance, tier, source_event_id, meta_json,
              created_at, last_accessed_at, access_count, version_created, version_superseded
            FROM facts
            WHERE id = ?
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "content": row["content"],
            "status": row["status"],
            "importance": _normalize_fact_importance(row["importance"]),
            "tier": _normalize_fact_tier(row["tier"]),
            "source_event_id": row["source_event_id"],
            "meta": _loads_json(row["meta_json"]) or {},
            "created_at": float(row["created_at"]) if row["created_at"] is not None else None,
            "last_accessed_at": (
                float(row["last_accessed_at"]) if row["last_accessed_at"] is not None else None
            ),
            "access_count": _normalize_access_count(row["access_count"]),
            "version_created": int(row["version_created"]),
            "version_superseded": row["version_superseded"],
        }
    finally:
        conn.close()


def get_event_by_id(event_id: int) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, ts, raw_json, state_version FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "ts": float(row["ts"]),
            "raw_json": json.loads(row["raw_json"]),
            "state_version": int(row["state_version"]),
        }
    finally:
        conn.close()


def insert_planner_run(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    goal: str,
    snapshot: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    planner_status: str,
    first_response: str,
    repair_prompt: str | None,
    repair_response: str | None,
    final_steps: list[dict[str, Any]],
    error_text: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO planner_runs(
            event_id, goal, snapshot_state_version, vector_watermark, planner_status,
            snapshot_json, system_prompt, user_prompt, first_response, repair_prompt,
            repair_response, final_steps_json, error_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            goal,
            int(snapshot.get("state_version", 0)),
            int(snapshot.get("vector_watermark", 0)),
            planner_status,
            json.dumps(snapshot, ensure_ascii=False),
            system_prompt,
            user_prompt,
            first_response,
            repair_prompt,
            repair_response,
            json.dumps(final_steps, ensure_ascii=False),
            error_text,
            time.time(),
        ),
    )
    return int(cur.lastrowid)


def insert_step_trace(
    conn: sqlite3.Connection,
    *,
    planner_run_id: int,
    step_index: int,
    action: str,
    tool: str | None,
    args: dict[str, Any],
    precondition_fact_ids: list[int],
    reasoning: str,
    snapshot_state_version: int,
    current_state_version: int | None,
    revalidation_status: str,
    rejection_reason: str | None,
    replan_reason: str | None = None,
    replan_count: int | None = None,
    replan_diff_summary: dict[str, Any] | None = None,
    execution_status: str | None = None,
    result: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO step_traces(
            planner_run_id, step_index, action, tool, args_json, precondition_fact_ids,
            reasoning, snapshot_state_version, current_state_version, revalidation_status,
            rejection_reason, replan_reason, replan_count, replan_diff_summary,
            execution_status, result_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            planner_run_id,
            step_index,
            action,
            tool,
            json.dumps(args, ensure_ascii=False),
            json.dumps(precondition_fact_ids, ensure_ascii=False),
            reasoning,
            snapshot_state_version,
            current_state_version,
            revalidation_status,
            rejection_reason,
            replan_reason,
            replan_count,
            json.dumps(replan_diff_summary, ensure_ascii=False) if replan_diff_summary is not None else None,
            execution_status,
            json.dumps(result, ensure_ascii=False) if result is not None else None,
            time.time(),
        ),
    )
    return int(cur.lastrowid)


def list_planner_runs_for_event(event_id: int) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM planner_runs
            WHERE event_id = ?
            ORDER BY id ASC
            """,
            (event_id,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "event_id": int(row["event_id"]),
                "goal": row["goal"],
                "snapshot_state_version": int(row["snapshot_state_version"]),
                "vector_watermark": int(row["vector_watermark"]),
                "planner_status": row["planner_status"],
                "snapshot": _loads_json(row["snapshot_json"]) or {},
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
                "first_response": row["first_response"],
                "repair_prompt": row["repair_prompt"],
                "repair_response": row["repair_response"],
                "final_steps": json.loads(row["final_steps_json"]),
                "error_text": row["error_text"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_step_traces_for_planner_run(planner_run_id: int) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM step_traces
            WHERE planner_run_id = ?
            ORDER BY step_index ASC, id ASC
            """,
            (planner_run_id,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "planner_run_id": int(row["planner_run_id"]),
                "step_index": int(row["step_index"]),
                "action": row["action"],
                "tool": row["tool"],
                "args": _loads_json(row["args_json"]) or {},
                "precondition_fact_ids": json.loads(row["precondition_fact_ids"]),
                "reasoning": row["reasoning"],
                "snapshot_state_version": int(row["snapshot_state_version"]),
                "current_state_version": (
                    int(row["current_state_version"])
                    if row["current_state_version"] is not None
                    else None
                ),
                "revalidation_status": row["revalidation_status"],
                "rejection_reason": row["rejection_reason"],
                "replan_reason": row["replan_reason"] if "replan_reason" in row.keys() else None,
                "replan_count": (
                    int(row["replan_count"])
                    if "replan_count" in row.keys() and row["replan_count"] is not None
                    else None
                ),
                "replan_diff_summary": _loads_json(row["replan_diff_summary"] if "replan_diff_summary" in row.keys() else None),
                "execution_status": row["execution_status"],
                "result": _loads_json(row["result_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def insert_weekly_review(conn: sqlite3.Connection, review: dict[str, Any], event_id: int | None = None) -> int:
    cur = conn.execute(
        """
        INSERT INTO weekly_reviews(event_id, week_key, title, summary_json, markdown, note_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            str(review.get("week_key") or ""),
            str(review.get("title") or "Weekly Review").strip(),
            json.dumps(review.get("summary", {}), ensure_ascii=False),
            str(review.get("markdown") or "").strip(),
            review.get("note_path"),
            time.time(),
        ),
    )
    return int(cur.lastrowid)


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


def list_stale_active_facts(limit: int = 50) -> list[Fact]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM facts
            WHERE status = 'active'
            ORDER BY COALESCE(last_accessed_at, created_at, 0) ASC, created_at ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_fact_from_row(row) for row in rows if row is not None]
    finally:
        conn.close()


def touch_fact_accesses(fact_ids: list[int], accessed_at: float | None = None) -> None:
    normalized_ids = sorted({int(fact_id) for fact_id in fact_ids if int(fact_id) > 0})
    if not normalized_ids:
        return
    with db_transaction() as conn:
        conn.execute(
            f"""
            UPDATE facts
            SET last_accessed_at = ?, access_count = COALESCE(access_count, 0) + 1
            WHERE id IN ({",".join("?" for _ in normalized_ids)})
            """,
            (accessed_at if accessed_at is not None else time.time(), *normalized_ids),
        )


def _legacy_proposal_payload(row: sqlite3.Row) -> dict[str, Any] | None:
    try:
        source_fact_ids = json.loads(row["source_fact_ids"])
        normalized_ids = [int(item) for item in source_fact_ids]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    proposed_content = str(row["proposed_content"] or "").strip()
    if not normalized_ids or not proposed_content:
        return None
    return {
        "type": "merge",
        "source_fact_ids": normalized_ids,
        "merged_content": proposed_content,
        "merged_importance": "contextual",
        "source_tier_after": "cold",
        "reasoning": "Legacy merge proposal imported from the previous consolidator format.",
    }


def _proposal_payload_from_row(row: sqlite3.Row) -> dict[str, Any] | None:
    payload = _loads_json(row["proposal_json"] if "proposal_json" in row.keys() else None)
    if isinstance(payload, dict) and str(payload.get("type") or "").strip():
        return payload
    return _legacy_proposal_payload(row)


def _proposal_signature_from_payload(payload: dict[str, Any]) -> tuple[str, ...] | None:
    proposal_type = str(payload.get("type") or "").strip()
    if proposal_type == "merge":
        raw_ids = payload.get("source_fact_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            return None
        try:
            normalized_ids = sorted({int(item) for item in raw_ids})
        except (TypeError, ValueError):
            return None
        return ("merge", *[str(item) for item in normalized_ids])
    if proposal_type == "tier_change":
        try:
            fact_id = int(payload.get("fact_id"))
        except (TypeError, ValueError):
            return None
        new_tier = str(payload.get("new_tier") or "").strip().lower()
        if fact_id <= 0 or new_tier not in {"cold", "archived"}:
            return None
        return ("tier_change", str(fact_id), new_tier)
    return None


def insert_consolidation_proposal(proposal: dict[str, Any]) -> int:
    proposal_type = str(proposal.get("type") or "").strip()
    if proposal_type not in {"merge", "tier_change"}:
        raise ValueError(f"Unsupported consolidation proposal type: {proposal_type or 'unknown'}")
    if proposal_type == "merge":
        raw_ids = proposal.get("source_fact_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("Merge proposals require source_fact_ids.")
        source_fact_ids = [int(item) for item in raw_ids]
        proposed_content = str(proposal.get("merged_content") or "").strip()
    else:
        source_fact_ids = [int(proposal.get("fact_id"))]
        proposed_content = ""
    with db_transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO consolidation_proposals(
                source_fact_ids, proposed_content, proposal_type, proposal_json, status, created_at
            )
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (
                json.dumps(source_fact_ids, ensure_ascii=False),
                proposed_content,
                proposal_type,
                json.dumps(proposal, ensure_ascii=False),
                time.time(),
            ),
        )
        return int(cur.lastrowid)


def list_pending_consolidation_proposals(limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, source_fact_ids, proposed_content, proposal_type, proposal_json, status, created_at
            FROM consolidation_proposals
            WHERE status = 'pending'
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        proposals: list[dict[str, Any]] = []
        for row in rows:
            proposals.append(
                {
                    "id": int(row["id"]),
                    "source_fact_ids": row["source_fact_ids"],
                    "proposed_content": row["proposed_content"],
                    "proposal_type": row["proposal_type"],
                    "proposal_json": row["proposal_json"],
                    "proposal": _proposal_payload_from_row(row),
                    "status": row["status"],
                    "created_at": float(row["created_at"]),
                }
            )
        return proposals
    finally:
        conn.close()


def list_pending_proposal_pairs() -> set[tuple[int, ...]]:
    pairs: set[tuple[int, ...]] = set()
    for proposal in list_pending_consolidation_proposals(limit=500):
        payload = proposal.get("proposal")
        if not isinstance(payload, dict) or payload.get("type") != "merge":
            continue
        try:
            raw_ids = payload["source_fact_ids"]
            pair = tuple(sorted(int(item) for item in raw_ids))
        except (KeyError, TypeError, ValueError):
            continue
        pairs.add(pair)
    return pairs


def list_pending_proposal_signatures() -> set[tuple[str, ...]]:
    signatures: set[tuple[str, ...]] = set()
    for proposal in list_pending_consolidation_proposals(limit=500):
        payload = proposal.get("proposal")
        if not isinstance(payload, dict):
            continue
        signature = _proposal_signature_from_payload(payload)
        if signature is not None:
            signatures.add(signature)
    return signatures


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


def list_facts_created_between(start_ts: float, end_ts: float) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, content, status, source_event_id, meta_json, created_at
            FROM facts
            WHERE created_at IS NOT NULL AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC, id ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "content": row["content"],
                "status": row["status"],
                "source_event_id": row["source_event_id"],
                "meta": _loads_json(row["meta_json"]) or {},
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_tasks_created_between(start_ts: float, end_ts: float) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, title, status, constraint_json, active_from_version, completed_version, created_at, completed_at
            FROM tasks
            WHERE created_at IS NOT NULL AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC, id ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "status": row["status"],
                "constraint": _loads_json(row["constraint_json"]) or {},
                "active_from_version": int(row["active_from_version"]),
                "completed_version": row["completed_version"],
                "created_at": float(row["created_at"]),
                "completed_at": row["completed_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_tasks_completed_between(start_ts: float, end_ts: float) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, title, status, constraint_json, active_from_version, completed_version, created_at, completed_at
            FROM tasks
            WHERE completed_at IS NOT NULL AND completed_at >= ? AND completed_at < ?
            ORDER BY completed_at ASC, id ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "status": row["status"],
                "constraint": _loads_json(row["constraint_json"]) or {},
                "active_from_version": int(row["active_from_version"]),
                "completed_version": row["completed_version"],
                "created_at": row["created_at"],
                "completed_at": float(row["completed_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_messages_between(start_ts: float, end_ts: float, limit: int | None = None) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        sql = """
            SELECT id, role, content, state_version, ts, summary_id
            FROM messages
            WHERE ts >= ? AND ts < ?
            ORDER BY ts ASC, id ASC
        """
        params: tuple[Any, ...]
        if limit is None:
            params = (start_ts, end_ts)
        else:
            sql += " LIMIT ?"
            params = (start_ts, end_ts, int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [
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
    finally:
        conn.close()


def get_planner_activity_between(start_ts: float, end_ts: float) -> dict[str, Any]:
    conn = _connect()
    try:
        run_row = conn.execute(
            """
            SELECT
              COUNT(*) AS run_count,
              SUM(CASE WHEN planner_status = 'fallback' THEN 1 ELSE 0 END) AS fallback_count,
              SUM(CASE WHEN planner_status = 'repaired' THEN 1 ELSE 0 END) AS repaired_count
            FROM planner_runs
            WHERE created_at >= ? AND created_at < ?
            """,
            (start_ts, end_ts),
        ).fetchone()
        rejection_rows = conn.execute(
            """
            SELECT rejection_reason, COUNT(*) AS count
            FROM step_traces
            WHERE created_at >= ? AND created_at < ? AND revalidation_status = 'rejected'
            GROUP BY rejection_reason
            ORDER BY count DESC, rejection_reason ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
        return {
            "run_count": int(run_row["run_count"] or 0) if run_row else 0,
            "fallback_count": int(run_row["fallback_count"] or 0) if run_row else 0,
            "repaired_count": int(run_row["repaired_count"] or 0) if run_row else 0,
            "rejections": [
                {
                    "reason": row["rejection_reason"] or "unknown",
                    "count": int(row["count"]),
                }
                for row in rejection_rows
            ],
        }
    finally:
        conn.close()


def count_events_between(start_ts: float, end_ts: float) -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE ts >= ? AND ts < ?",
            (start_ts, end_ts),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()
