from __future__ import annotations

import importlib
import tempfile
import time
import unittest
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
    touch_fact_accesses,
)
from memory_engine.memory_model import DEFAULT_MEMORY_MODEL
from memory_engine.models import Fact
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


class MemoryModelSpecificationTests(unittest.TestCase):
    def test_default_memory_model_defines_full_lifecycle(self) -> None:
        self.assertEqual(
            DEFAULT_MEMORY_MODEL.stage_keys(),
            ("capture", "stabilize", "rehearse", "consolidate", "reflect"),
        )
        self.assertEqual(
            DEFAULT_MEMORY_MODEL.invariant_keys(),
            (
                "explicit_memories_are_durable",
                "pinned_identity_is_not_auto_compressed",
                "compression_preserves_traceability",
                "reuse_strengthens_salience",
                "confidence_requires_verification",
                "reflection_is_additive",
            ),
        )


class MemoryModelRuntimeContractTests(unittest.TestCase):
    def test_rehearsal_updates_salience_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(conn, "I like jasmine tea", version)

            before = get_fact_by_id(fact_id)
            self.assertIsNotNone(before)
            self.assertEqual(before.access_count, 0)
            self.assertIsNone(before.last_accessed_at)

            touched_at = time.time()
            touch_fact_accesses([fact_id], accessed_at=touched_at)

            after = get_fact_by_id(fact_id)
            self.assertIsNotNone(after)
            self.assertEqual(after.access_count, 1)
            self.assertAlmostEqual(after.last_accessed_at or 0.0, touched_at, places=3)

    def test_pinned_identity_memory_is_excluded_from_consolidation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            consolidator = importlib.import_module("memory_engine.consolidator")

            with db_transaction() as conn:
                version = bump_version(conn)
                pinned_fact_id = insert_fact(
                    conn,
                    {
                        "content": "My preferred editor is Neovim",
                        "meta": {"kind": "preference"},
                    },
                    version,
                )
                normal_fact_id = insert_fact(conn, "The memory refactor started today", version)

            candidate_ids = {fact.id for fact in consolidator._candidate_facts(limit=10)}
            self.assertIn(normal_fact_id, candidate_ids)
            self.assertNotIn(pinned_fact_id, candidate_ids)

    def test_merge_contract_requires_preserving_active_source_facts(self) -> None:
        consolidator = importlib.import_module("memory_engine.consolidator")
        facts_by_id = {
            1: Fact(
                id=1,
                content="We discussed memory design on Monday",
                embedding_id="fact:1",
                version_created=1,
                version_superseded=None,
                status="active",
                tier="active",
            ),
            2: Fact(
                id=2,
                content="We refined the memory model on Tuesday",
                embedding_id="fact:2",
                version_created=1,
                version_superseded=None,
                status="active",
                tier="active",
            ),
        }

        proposal = consolidator._normalize_merge_proposal(
            {
                "type": "merge",
                "source_fact_ids": ["fact_000001", "fact_000002"],
                "merged_content": "This week we iterated on the memory model design.",
                "merged_importance": "contextual",
                "source_tier_after": "cold",
                "reasoning": "Both facts describe the same ongoing design thread.",
            },
            facts_by_id,
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["source_fact_ids"], [1, 2])
        self.assertEqual(proposal["source_tier_after"], "cold")

    def test_message_compaction_adds_summary_without_erasing_message_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            with db_transaction() as conn:
                state_version = bump_version(conn)
                first_message_id = insert_message(conn, "user", "Day one memory note", state_version=state_version)
                second_message_id = insert_message(
                    conn,
                    "assistant",
                    "Acknowledged and stored.",
                    state_version=state_version,
                )

            summary_id = create_summary_and_mark_messages(
                [first_message_id, second_message_id],
                "Summary: the user recorded an explicit memory note.",
            )
            self.assertIsNotNone(summary_id)

            conn = _connect()
            try:
                message_rows = conn.execute(
                    "SELECT id, summary_id FROM messages WHERE id IN (?, ?)",
                    (first_message_id, second_message_id),
                ).fetchall()
                summary_row = conn.execute(
                    "SELECT id, content FROM summaries WHERE id = ?",
                    (summary_id,),
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual({int(row["summary_id"]) for row in message_rows}, {int(summary_id)})
            self.assertEqual(int(summary_row["id"]), int(summary_id))
            self.assertIn("explicit memory note", summary_row["content"])

    def test_weekly_review_builds_macro_summary_from_recent_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            now_ts = time.time()

            with db_transaction() as conn:
                state_version = bump_version(conn)
                insert_event(conn, {"text": "review memory work"})
                insert_message(conn, "user", "We discussed long-term memory behavior", state_version=state_version)
                insert_fact(
                    conn,
                    {
                        "content": "Adopt semantic weekly review for memory reflection",
                        "meta": {"kind": "decision", "title": "Adopt weekly reflection"},
                        "created_at": now_ts,
                    },
                    state_version,
                )
                insert_fact(
                    conn,
                    {
                        "content": "Need a compression model for older conversations",
                        "created_at": now_ts,
                    },
                    state_version,
                )

            review = generate_weekly_review({})
            summary = review["summary"]
            markdown = review["markdown"]

            self.assertGreaterEqual(int(summary["facts_captured"]), 2)
            self.assertGreaterEqual(int(summary["decisions_logged"]), 1)
            self.assertIn("Weekly Review", review["title"])
            self.assertIn("Adopt weekly reflection", markdown)
            self.assertIn("Need a compression model for older conversations", markdown)


if __name__ == "__main__":
    unittest.main()
