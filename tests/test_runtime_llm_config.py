from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_engine.config import Config
from memory_engine.router import TaskType


def _test_config(root: Path, *, embed_model: str = "model-a", read_limit: int = 111) -> Config:
    return Config(
        SQLITE_PATH=root / "memory.sqlite",
        CHROMA_PATH=root / "chroma",
        OBSIDIAN_VAULT_PATH=root / "obsidian",
        WORKING_MEMORY_PATH=root / "working_memory",
        ASSISTANT_SYSTEM_PROMPT="",
        EMBED_MODEL=embed_model,
        VERSION_DRIFT_THRESHOLD=100,
        CONSOLIDATION_INTERVAL_SEC=60,
        INDEXER_POLL_INTERVAL_SEC=1,
        MAX_CONTEXT_FACTS=20,
        MAX_RECENT_MESSAGES=10,
        WORKING_MEMORY_READ_CHAR_LIMIT=read_limit,
    )


class RuntimeLLMConfigTests(unittest.TestCase):
    def test_runtime_llm_config_overrides_file_backed_settings(self) -> None:
        llm = importlib.import_module("memory_engine.llm")
        llm = importlib.reload(llm)
        llm.set_runtime_llm_config(
            {
                "llm": {
                    "default_task": "reason",
                    "temperature": 0.9,
                    "max_tokens": 321,
                    "providers": {
                        "groq": {
                            "model": "groq/custom-fast",
                            "enabled": True,
                        }
                    },
                }
            }
        )

        try:
            with patch.object(
                llm,
                "_load_yaml_config",
                side_effect=AssertionError("runtime llm config should bypass file config"),
            ):
                self.assertEqual(llm._default_task(), TaskType.REASON)
                self.assertEqual(llm._completion_kwargs()["temperature"], 0.9)
                self.assertEqual(llm._completion_kwargs()["max_tokens"], 321)
                self.assertEqual(llm._provider_config("groq")["model"], "groq/custom-fast")
                self.assertEqual(llm._model_for_task(TaskType.FAST), "groq/custom-fast")
        finally:
            llm.set_runtime_llm_config(None)

    def test_planner_registers_runtime_llm_config_from_app_cfg(self) -> None:
        planner_module = importlib.import_module("modules.planner")
        app_cfg = {
            "llm": {
                "default_task": "math",
            }
        }

        with (
            patch.object(planner_module, "set_runtime_llm_config") as set_runtime_llm_config,
            patch.object(planner_module, "set_active_config"),
            patch.object(planner_module, "assert_registry_integrity"),
            patch.object(planner_module, "init_db"),
        ):
            planner_module.Planner(app_cfg)

        set_runtime_llm_config.assert_called_once_with(app_cfg)

    def test_set_active_config_invalidates_runtime_caches(self) -> None:
        config_module = importlib.import_module("memory_engine.config")
        embeddings_module = importlib.import_module("memory_engine.embeddings")
        retrieval_module = importlib.import_module("memory_engine.retrieval")
        tool_registry_module = importlib.import_module("memory_engine.tool_registry")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _test_config(Path(tmpdir))
            with (
                patch.object(embeddings_module, "clear_embedding_model_cache") as clear_embedding_model_cache,
                patch.object(retrieval_module, "reset_vector_store_state") as reset_vector_store_state,
                patch.object(tool_registry_module, "clear_registry_prompt_cache") as clear_registry_prompt_cache,
            ):
                config_module.set_active_config(cfg)

        clear_embedding_model_cache.assert_called_once_with()
        reset_vector_store_state.assert_called_once_with()
        clear_registry_prompt_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
