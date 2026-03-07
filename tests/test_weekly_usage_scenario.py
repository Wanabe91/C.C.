from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path

from memory_engine.config import Config, set_active_config
from memory_engine.db import (
    _connect,
    bump_version,
    create_summary_and_mark_messages,
    db_transaction,
    get_fact_by_id,
    init_db,
    insert_event,
    insert_fact,
    insert_message,
    insert_planner_run,
    insert_step_trace,
    touch_fact_accesses,
)
from memory_engine.weekly_review import generate_weekly_review


def _test_config(root: Path) -> Config:
    return Config(
        SQLITE_PATH=root / "memory.sqlite",
        CHROMA_PATH=root / "chroma",
        OBSIDIAN_VAULT_PATH=root / "obsidian",
        WORKING_MEMORY_PATH=root / "working_memory",
        ASSISTANT_SYSTEM_PROMPT="",
        EMBED_MODEL="test-model",
        VERSION_DRIFT_THRESHOLD=100,
        CONSOLIDATION_INTERVAL_SEC=60,
        INDEXER_POLL_INTERVAL_SEC=1,
        MAX_CONTEXT_FACTS=20,
        MAX_RECENT_MESSAGES=10,
    )


class WeeklyUsageScenarioTests(unittest.TestCase):
    def test_week_of_usage_converges_into_weekly_memory_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            tz = datetime.now().astimezone().tzinfo
            week_start = date(2026, 3, 2)

            def ts(day_offset: int, hour: int, minute: int = 0) -> float:
                return datetime.combine(
                    week_start + timedelta(days=day_offset),
                    time(hour=hour, minute=minute),
                    tzinfo=tz,
                ).timestamp()

            with db_transaction() as conn:
                monday_version = bump_version(conn)
                monday_event_id = insert_event(conn, {"text": "Запомни, что я люблю короткие ответы"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(0, 9), monday_event_id))
                monday_user_message_id = insert_message(
                    conn,
                    "user",
                    "Запомни, что я люблю короткие ответы.",
                    state_version=monday_version,
                )
                monday_assistant_message_id = insert_message(
                    conn,
                    "assistant",
                    "Запомнил предпочтение к коротким ответам.",
                    state_version=monday_version,
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(0, 9, 1), monday_user_message_id),
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(0, 9, 2), monday_assistant_message_id),
                )
                preference_fact_id = insert_fact(
                    conn,
                    {
                        "content": "Пользователь предпочитает короткие прямые ответы.",
                        "importance": "core",
                        "meta": {"kind": "preference"},
                        "created_at": ts(0, 9, 2),
                    },
                    monday_version,
                )

                tuesday_version = bump_version(conn)
                tuesday_event_id = insert_event(conn, {"text": "Мы храним память в SQLite и Chroma"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(1, 11), tuesday_event_id))
                tuesday_user_message_id = insert_message(
                    conn,
                    "user",
                    "Запомни, что Veronia хранит память в SQLite и Chroma.",
                    state_version=tuesday_version,
                )
                tuesday_assistant_message_id = insert_message(
                    conn,
                    "assistant",
                    "Сохранил архитектурный факт о памяти.",
                    state_version=tuesday_version,
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(1, 11, 1), tuesday_user_message_id),
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(1, 11, 2), tuesday_assistant_message_id),
                )
                architecture_fact_id = insert_fact(
                    conn,
                    {
                        "content": "Veronia хранит долговременную память в SQLite и Chroma.",
                        "created_at": ts(1, 11, 2),
                    },
                    tuesday_version,
                )

                wednesday_version = bump_version(conn)
                wednesday_event_id = insert_event(conn, {"text": "Напомни, где хранится память"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(2, 14), wednesday_event_id))
                wednesday_user_message_id = insert_message(
                    conn,
                    "user",
                    "Напомни, где хранится память проекта.",
                    state_version=wednesday_version,
                )
                wednesday_assistant_message_id = insert_message(
                    conn,
                    "assistant",
                    "Память проекта хранится в SQLite и Chroma.",
                    state_version=wednesday_version,
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(2, 14, 1), wednesday_user_message_id),
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(2, 14, 2), wednesday_assistant_message_id),
                )
                touch_fact_accesses([architecture_fact_id], accessed_at=ts(2, 14, 2), conn=conn)

                thursday_version = bump_version(conn)
                thursday_event_id = insert_event(conn, {"text": "Нужно перейти к недельным обзорам памяти"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(3, 16), thursday_event_id))
                thursday_user_message_id = insert_message(
                    conn,
                    "user",
                    "Давай перейдём к недельным semantic review памяти.",
                    state_version=thursday_version,
                )
                thursday_assistant_message_id = insert_message(
                    conn,
                    "assistant",
                    "Зафиксировал решение перейти к недельным обзорам памяти.",
                    state_version=thursday_version,
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(3, 16, 1), thursday_user_message_id),
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(3, 16, 2), thursday_assistant_message_id),
                )
                decision_fact_id = insert_fact(
                    conn,
                    {
                        "content": "Перейти к weekly semantic review для долговременной памяти.",
                        "meta": {"kind": "decision", "title": "Включить weekly semantic review"},
                        "created_at": ts(3, 16, 2),
                    },
                    thursday_version,
                )
                created_task = conn.execute(
                    """
                    INSERT INTO tasks(
                        title, status, constraint_json, active_from_version, completed_version, created_at, completed_at
                    )
                    VALUES (?, 'active', ?, ?, NULL, ?, NULL)
                    """,
                    ("Добавить e2e тест недели памяти", "{}", thursday_version, ts(3, 16, 5)),
                )
                completed_task_id = int(created_task.lastrowid)

                friday_version = bump_version(conn)
                friday_event_id = insert_event(conn, {"text": "Тест недели памяти готов"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(4, 18), friday_event_id))
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'completed', completed_version = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (friday_version, ts(4, 18, 10), completed_task_id),
                )
                open_task = conn.execute(
                    """
                    INSERT INTO tasks(
                        title, status, constraint_json, active_from_version, completed_version, created_at, completed_at
                    )
                    VALUES (?, 'active', ?, ?, NULL, ?, NULL)
                    """,
                    ("Проверить качество consolidation merge", "{}", friday_version, ts(4, 18, 20)),
                )
                open_task_id = int(open_task.lastrowid)

                saturday_version = bump_version(conn)
                saturday_event_id = insert_event(conn, {"text": "У нас был replan из-за drift"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(5, 13), saturday_event_id))
                planner_run_id = insert_planner_run(
                    conn,
                    event_id=saturday_event_id,
                    goal="Handle planner drift and continue memory work",
                    snapshot={"state_version": saturday_version, "vector_watermark": saturday_version},
                    system_prompt="system",
                    user_prompt="user",
                    planner_status="repaired",
                    first_response='{"steps":[]}',
                    repair_prompt="repair",
                    repair_response='{"steps":[]}',
                    final_steps=[],
                    error_text=None,
                )
                insert_step_trace(
                    conn,
                    planner_run_id=planner_run_id,
                    step_index=0,
                    action="respond",
                    tool="respond",
                    args={"message": "continue"},
                    precondition_fact_ids=[],
                    reasoning="Existing memory context changed.",
                    snapshot_state_version=saturday_version - 1,
                    current_state_version=saturday_version,
                    revalidation_status="rejected",
                    rejection_reason="state_version_drift:1:threshold:0",
                    execution_status="skipped",
                    result={"status": "skipped"},
                )
                conn.execute(
                    "UPDATE planner_runs SET created_at = ? WHERE id = ?",
                    (ts(5, 13, 5), planner_run_id),
                )
                conn.execute(
                    "UPDATE step_traces SET created_at = ? WHERE planner_run_id = ?",
                    (ts(5, 13, 6), planner_run_id),
                )
                fallback_run_id = insert_planner_run(
                    conn,
                    event_id=saturday_event_id,
                    goal="Fallback after repeated drift",
                    snapshot={"state_version": saturday_version, "vector_watermark": saturday_version},
                    system_prompt="system",
                    user_prompt="user",
                    planner_status="fallback",
                    first_response='{"steps":[]}',
                    repair_prompt=None,
                    repair_response=None,
                    final_steps=[],
                    error_text="too much drift",
                )
                conn.execute(
                    "UPDATE planner_runs SET created_at = ? WHERE id = ?",
                    (ts(5, 13, 15), fallback_run_id),
                )

                sunday_version = bump_version(conn)
                sunday_event_id = insert_event(conn, {"text": "Подведи итоги недели по памяти"})
                conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts(6, 19), sunday_event_id))
                sunday_user_message_id = insert_message(
                    conn,
                    "user",
                    "Подведи итоги недели по развитию памяти.",
                    state_version=sunday_version,
                )
                sunday_assistant_message_id = insert_message(
                    conn,
                    "assistant",
                    "Готов собрать недельное обобщение.",
                    state_version=sunday_version,
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(6, 19, 1), sunday_user_message_id),
                )
                conn.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts(6, 19, 2), sunday_assistant_message_id),
                )

            touch_fact_accesses([architecture_fact_id], accessed_at=ts(6, 12))

            summary_id = create_summary_and_mark_messages(
                [
                    monday_user_message_id,
                    monday_assistant_message_id,
                    tuesday_user_message_id,
                    tuesday_assistant_message_id,
                ],
                "Early week summary: the user stored communication preference and core memory architecture.",
            )
            self.assertIsNotNone(summary_id)

            conn = _connect()
            try:
                conn.execute(
                    "UPDATE summaries SET created_at = ? WHERE id = ?",
                    (ts(6, 18), summary_id),
                )
                conn.commit()
            finally:
                conn.close()

            review = generate_weekly_review(
                {
                    "week_start": week_start.isoformat(),
                    "title": "Weekly Review Memory Scenario",
                    "focus": "How memory should evolve across one week of active use.",
                }
            )

            summary = review["summary"]
            markdown = review["markdown"]

            architecture_fact = get_fact_by_id(architecture_fact_id)
            preference_fact = get_fact_by_id(preference_fact_id)
            decision_fact = get_fact_by_id(decision_fact_id)
            self.assertIsNotNone(architecture_fact)
            self.assertIsNotNone(preference_fact)
            self.assertIsNotNone(decision_fact)

            self.assertEqual(int(summary["event_count"]), 7)
            self.assertEqual(int(summary["planner_runs"]), 2)
            self.assertEqual(int(summary["planner_repaired_runs"]), 1)
            self.assertEqual(int(summary["planner_fallback_runs"]), 1)
            self.assertEqual(int(summary["tasks_created"]), 2)
            self.assertEqual(int(summary["tasks_completed"]), 1)
            self.assertEqual(int(summary["open_tasks"]), 1)
            self.assertGreaterEqual(int(summary["facts_captured"]), 3)
            self.assertEqual(int(summary["decisions_logged"]), 1)
            self.assertGreaterEqual(int(summary["user_messages"]), 4)
            self.assertGreaterEqual(int(summary["assistant_messages"]), 4)

            self.assertEqual(architecture_fact.access_count, 2)
            self.assertEqual(preference_fact.meta or {}, {"kind": "preference"})
            self.assertEqual((decision_fact.meta or {}).get("kind"), "decision")

            self.assertIn("Weekly Review Memory Scenario", markdown)
            self.assertIn("How memory should evolve across one week of active use.", markdown)
            self.assertIn("Включить weekly semantic review", markdown)
            self.assertIn("Veronia хранит долговременную память в SQLite и Chroma.", markdown)
            self.assertIn("Проверить качество consolidation merge", markdown)
            self.assertIn("state_version_drift:1:threshold:0", markdown)
            self.assertIn("Запомни, что я люблю короткие ответы.", markdown)
            self.assertNotIn("No task activity recorded.", markdown)
            self.assertGreater(open_task_id, 0)


if __name__ == "__main__":
    unittest.main()
