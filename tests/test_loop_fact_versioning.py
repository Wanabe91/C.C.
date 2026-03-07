from __future__ import annotations

import importlib
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, patch

from memory_engine.config import Config, set_active_config
from memory_engine.db import (
    bump_event_version,
    bump_version,
    claim_pending_outbox_item,
    db_transaction,
    get_delta_facts,
    get_fact_by_id,
    get_vector_watermark,
    get_version,
    init_db,
    insert_event,
    mark_outbox_done,
    recompute_vector_watermark,
)
from memory_engine.models import ContextFingerprint, ContextSnapshot


@dataclass
class FakeStep:
    action: str = "respond"
    tool: str = "respond"
    precondition_fact_ids: list[str] = field(default_factory=list)
    reasoning: str = ""

    def args_dict(self) -> dict[str, str]:
        return {"message": "ok"}


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


def _snapshot(state_version: int) -> ContextSnapshot:
    return ContextSnapshot(
        state_version=state_version,
        event_version=1,
        vector_watermark=get_vector_watermark(),
        fingerprint=ContextFingerprint(
            fact_versions={},
            active_task_ids=set(),
            last_summary_id=None,
            last_message_id=None,
        ),
        tasks=[],
        constraints=[],
        fts_results=[],
        vector_results=[],
        delta_facts=[],
        recent_messages=[],
    )


class ExecuteAndPersistFactVersionTests(unittest.IsolatedAsyncioTestCase):
    async def test_later_step_fact_stays_in_delta_after_watermark_reaches_previous_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            loop = importlib.import_module("memory_engine.loop")

            with db_transaction() as conn:
                event_id = insert_event(conn, {"text": "remember this"})
                bump_event_version(conn)
                first_step_version = bump_version(conn)

            decision_fact_ids: list[int] = []
            generated_reviews: list[dict[str, object]] = []
            step = FakeStep()

            with (
                patch.object(loop, "insert_step_trace", return_value=None),
                patch.object(
                    loop,
                    "execute_step",
                    new=AsyncMock(
                        return_value={
                            "status": "ok",
                            "assistant_message": "saved",
                            "facts": ["first fact"],
                            "created_tasks": [],
                            "completed_task_ids": [],
                            "generated_reviews": [],
                        }
                    ),
                ),
            ):
                await loop._execute_and_persist(
                    step=step,
                    event_id=event_id,
                    planner_run_id=1,
                    snapshot=_snapshot(first_step_version),
                    step_index=0,
                    decision_fact_ids=decision_fact_ids,
                    generated_reviews=generated_reviews,
                )

            first_outbox_item = claim_pending_outbox_item()
            self.assertIsNotNone(first_outbox_item)
            mark_outbox_done(first_outbox_item["id"])
            self.assertEqual(recompute_vector_watermark(), first_step_version)

            with (
                patch.object(loop, "insert_step_trace", return_value=None),
                patch.object(
                    loop,
                    "execute_step",
                    new=AsyncMock(
                        return_value={
                            "status": "ok",
                            "assistant_message": "saved again",
                            "facts": ["second fact"],
                            "created_tasks": [],
                            "completed_task_ids": [],
                            "generated_reviews": [],
                        }
                    ),
                ),
            ):
                await loop._execute_and_persist(
                    step=step,
                    event_id=event_id,
                    planner_run_id=1,
                    snapshot=_snapshot(get_version()),
                    step_index=1,
                    decision_fact_ids=decision_fact_ids,
                    generated_reviews=generated_reviews,
                )

            first_fact = get_fact_by_id(decision_fact_ids[0])
            second_fact = get_fact_by_id(decision_fact_ids[1])
            self.assertIsNotNone(first_fact)
            self.assertIsNotNone(second_fact)
            self.assertEqual(first_fact.version_created, first_step_version)
            self.assertEqual(second_fact.version_created, first_step_version + 1)

            delta_facts = get_delta_facts(get_version(), get_vector_watermark())
            self.assertEqual([fact.content for fact in delta_facts], ["second fact"])


if __name__ == "__main__":
    unittest.main()
