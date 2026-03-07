from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from memory_engine.config import Config, set_active_config
from memory_engine.db import (
    bump_version,
    db_transaction,
    get_fact_by_id,
    init_db,
    insert_fact,
    insert_outbox,
    list_pending_consolidation_proposals,
    list_recent_active_facts,
    record_fact_verification,
    touch_fact_accesses,
)
from memory_engine.loop import apply_pending_proposals
from memory_engine.tool_registry import RememberFactArgs, VerifyFactArgs, get_tool


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


class MemoryEpistemicsTests(unittest.TestCase):
    def test_remember_fact_starts_as_self_reported_not_globally_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            remember_tool = get_tool("remember_fact")
            self.assertIsNotNone(remember_tool)

            result = remember_tool.handler(RememberFactArgs(content="I prefer short replies."))

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(conn, result["facts"][0], version)

            fact = get_fact_by_id(fact_id)
            self.assertIsNotNone(fact)
            self.assertEqual(fact.verification_status, "self_reported")
            self.assertAlmostEqual(fact.confidence_score, 0.55, places=2)
            self.assertEqual(fact.verification_count, 1)
            self.assertIsNotNone(fact.last_verified_at)
            self.assertEqual((fact.evidence or [])[0]["method"], "user_statement")

    def test_retrieval_rehearsal_changes_salience_but_not_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(conn, "The project has a local SQLite store.", version)

            before = get_fact_by_id(fact_id)
            self.assertIsNotNone(before)
            self.assertEqual(before.access_count, 0)
            self.assertEqual(before.verification_status, "unverified")
            self.assertAlmostEqual(before.confidence_score, 0.20, places=2)

            touch_fact_accesses([fact_id])
            touch_fact_accesses([fact_id])

            after = get_fact_by_id(fact_id)
            self.assertIsNotNone(after)
            self.assertEqual(after.access_count, 2)
            self.assertEqual(after.verification_status, "unverified")
            self.assertAlmostEqual(after.confidence_score, 0.20, places=2)

    def test_verification_events_raise_confidence_without_retrieval_hacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(conn, "The preferred timezone is Europe/Warsaw.", version)

            self.assertTrue(
                record_fact_verification(
                    fact_id,
                    method="user_confirmed",
                    source_ref="chat:event:1",
                    note="The user explicitly confirmed this fact.",
                )
            )
            after_user_confirmation = get_fact_by_id(fact_id)
            self.assertIsNotNone(after_user_confirmation)
            self.assertEqual(after_user_confirmation.verification_status, "user_confirmed")
            self.assertAlmostEqual(after_user_confirmation.confidence_score, 0.80, places=2)
            self.assertEqual(after_user_confirmation.verification_count, 1)

            self.assertTrue(
                record_fact_verification(
                    fact_id,
                    method="external_match",
                    source_ref="calendar://profile",
                    note="External profile data matched the stored timezone.",
                )
            )
            after_external_match = get_fact_by_id(fact_id)
            self.assertIsNotNone(after_external_match)
            self.assertEqual(after_external_match.verification_status, "externally_confirmed")
            self.assertAlmostEqual(after_external_match.confidence_score, 0.95, places=2)
            self.assertEqual(after_external_match.verification_count, 2)
            self.assertEqual((after_external_match.evidence or [])[-1]["method"], "external_match")

    def test_verify_fact_tool_exposes_runtime_verification_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            verify_tool = get_tool("verify_fact")
            self.assertIsNotNone(verify_tool)

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(conn, "The assistant should prefer concise answers.", version)

            result = verify_tool.handler(
                VerifyFactArgs(
                    fact_id=str(fact_id),
                    method="user_confirmed",
                    source_ref="chat:event:3",
                    note="The user confirmed this memory explicitly.",
                )
            )
            updated = get_fact_by_id(fact_id)
            self.assertEqual(result["status"], "ok")
            self.assertIsNotNone(updated)
            self.assertEqual(updated.verification_status, "user_confirmed")
            self.assertAlmostEqual(updated.confidence_score, 0.80, places=2)
            self.assertEqual(result["tool_output"]["verification_status"], "user_confirmed")

    def test_contradiction_marks_fact_without_silent_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_id = insert_fact(
                    conn,
                    {
                        "content": "The dog is named Beam.",
                        "verification_status": "user_confirmed",
                        "confidence_score": 0.80,
                        "verification_count": 1,
                    },
                    version,
                )

            self.assertTrue(
                record_fact_verification(
                    fact_id,
                    method="contradicted",
                    source_ref="chat:event:2",
                    note="The user corrected the dog's name.",
                    contradiction_group_id="dog-name",
                )
            )
            contradicted = get_fact_by_id(fact_id)
            self.assertIsNotNone(contradicted)
            self.assertEqual(contradicted.verification_status, "contradicted")
            self.assertAlmostEqual(contradicted.confidence_score, 0.05, places=2)
            self.assertEqual(contradicted.contradiction_group_id, "dog-name")
            self.assertEqual((contradicted.evidence or [])[-1]["method"], "contradicted")

    def test_merge_inherits_conservative_epistemics_from_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            now_ts = time.time()

            with db_transaction() as conn:
                version = bump_version(conn)
                fact_a = insert_fact(
                    conn,
                    {
                        "content": "Veronia stores memory in SQLite.",
                        "verification_status": "externally_confirmed",
                        "confidence_score": 0.95,
                        "verification_count": 2,
                        "last_verified_at": now_ts,
                    },
                    version,
                )
                insert_outbox(conn, fact_a)
                fact_b = insert_fact(
                    conn,
                    {
                        "content": "Veronia stores embeddings in Chroma.",
                        "verification_status": "self_reported",
                        "confidence_score": 0.55,
                        "verification_count": 1,
                        "last_verified_at": now_ts - 60,
                    },
                    version,
                )
                insert_outbox(conn, fact_b)

            from memory_engine.db import insert_consolidation_proposal

            proposal_id = insert_consolidation_proposal(
                {
                    "type": "merge",
                    "source_fact_ids": [fact_a, fact_b],
                    "merged_content": "Veronia stores long-term memory in SQLite and Chroma.",
                    "merged_importance": "contextual",
                    "source_tier_after": "cold",
                    "reasoning": "Both facts describe the same memory storage architecture.",
                }
            )
            self.assertGreater(proposal_id, 0)
            self.assertEqual(len(list_pending_consolidation_proposals()), 1)

            applied = apply_pending_proposals(version=2)
            self.assertEqual(applied, 1)

            active_facts = list_recent_active_facts(limit=10)
            merged = next(
                fact for fact in active_facts if fact.content == "Veronia stores long-term memory in SQLite and Chroma."
            )
            self.assertEqual(merged.verification_status, "self_reported")
            self.assertAlmostEqual(merged.confidence_score, 0.55, places=2)
            self.assertEqual(merged.verification_count, 1)
            self.assertEqual((merged.evidence or [])[0]["method"], "consolidation_merge")

            source_a = get_fact_by_id(fact_a)
            source_b = get_fact_by_id(fact_b)
            self.assertIsNotNone(source_a)
            self.assertIsNotNone(source_b)
            self.assertEqual(source_a.tier, "cold")
            self.assertEqual(source_b.tier, "cold")


if __name__ == "__main__":
    unittest.main()
