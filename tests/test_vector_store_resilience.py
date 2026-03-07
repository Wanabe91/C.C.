from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from memory_engine.config import Config, set_active_config
from memory_engine.db import (
    _connect,
    bump_version,
    db_transaction,
    get_vector_watermark,
    init_db,
    insert_event,
    insert_fact,
    insert_outbox,
)


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
        INDEXER_POLL_INTERVAL_SEC=0,
        MAX_CONTEXT_FACTS=20,
        MAX_RECENT_MESSAGES=10,
    )


def _fake_chromadb_modules(persistent_client):
    chromadb_module = types.ModuleType("chromadb")
    chromadb_module.PersistentClient = persistent_client

    config_module = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    config_module.Settings = Settings
    return chromadb_module, config_module


class VectorStoreRetryTests(unittest.TestCase):
    def test_get_collection_recovers_after_transient_init_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            retrieval = importlib.import_module("memory_engine.retrieval")
            retrieval = importlib.reload(retrieval)

            failing_chromadb, failing_config = _fake_chromadb_modules(
                Mock(side_effect=RuntimeError("boom"))
            )
            with patch.dict(
                sys.modules,
                {"chromadb": failing_chromadb, "chromadb.config": failing_config},
            ):
                self.assertIsNone(retrieval.get_collection())

            self.assertIsNone(retrieval._collection)
            self.assertGreater(retrieval._vector_store_retry_after, 0.0)

            fake_collection = object()
            fake_client = Mock()
            fake_client.get_or_create_collection.return_value = fake_collection
            retrieval._vector_store_retry_after = 0.0

            working_chromadb, working_config = _fake_chromadb_modules(
                Mock(return_value=fake_client)
            )
            with patch.dict(
                sys.modules,
                {"chromadb": working_chromadb, "chromadb.config": working_config},
            ):
                self.assertIs(retrieval.get_collection(), fake_collection)
                self.assertIs(retrieval.get_collection(), fake_collection)


class IndexerRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_indexer_requeues_outbox_when_collection_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_active_config(_test_config(Path(tmpdir)))
            init_db()
            indexer = importlib.import_module("memory_engine.indexer")

            with db_transaction() as conn:
                event_id = insert_event(conn, {"text": "remember this"})
                version = bump_version(conn)
                fact_id = insert_fact(conn, "fact awaiting vectorization", version)
                outbox_id = insert_outbox(conn, fact_id)

            stop_event = indexer.asyncio.Event()

            async def stop_after_retry(stop_event_obj, _seconds: float) -> None:
                stop_event_obj.set()

            with (
                patch.object(indexer, "get_collection", return_value=None),
                patch.object(indexer, "_wait_or_stop", new=stop_after_retry),
            ):
                await indexer.run_indexer(stop_event)

            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT status, attempts FROM embedding_outbox WHERE id = ?",
                    (outbox_id,),
                ).fetchone()
            finally:
                conn.close()

            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "pending")
            self.assertEqual(int(row["attempts"]), 1)
            self.assertEqual(get_vector_watermark(), 0)


if __name__ == "__main__":
    unittest.main()
