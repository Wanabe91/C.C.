from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Config:
    SQLITE_PATH: Path
    CHROMA_PATH: Path
    OBSIDIAN_VAULT_PATH: Path
    LLM_BASE_URL: str
    LLM_MODEL: str
    ASSISTANT_SYSTEM_PROMPT: str
    EMBED_BACKEND: str
    EMBED_MODEL: str
    VERSION_DRIFT_THRESHOLD: int
    CONSOLIDATION_INTERVAL_SEC: int
    INDEXER_POLL_INTERVAL_SEC: int
    MAX_CONTEXT_FACTS: int
    MAX_RECENT_MESSAGES: int

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(override=False)
        sqlite_path = Path(os.getenv("SQLITE_PATH", "./agent_memory.db")).expanduser().resolve()
        config = cls(
            SQLITE_PATH=sqlite_path,
            CHROMA_PATH=Path(os.getenv("CHROMA_PATH", "./chroma_store")).expanduser().resolve(),
            OBSIDIAN_VAULT_PATH=Path(
                os.getenv("OBSIDIAN_VAULT_PATH", str(sqlite_path.parent / "obsidian"))
            ).expanduser().resolve(),
            LLM_BASE_URL=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/"),
            LLM_MODEL=os.getenv("LLM_MODEL", "local-model"),
            ASSISTANT_SYSTEM_PROMPT=os.getenv("ASSISTANT_SYSTEM_PROMPT", "").strip(),
            EMBED_BACKEND=os.getenv("EMBED_BACKEND", "sentence_transformers").strip().lower(),
            EMBED_MODEL=os.getenv(
                "EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            VERSION_DRIFT_THRESHOLD=int(os.getenv("VERSION_DRIFT_THRESHOLD", "5")),
            CONSOLIDATION_INTERVAL_SEC=int(os.getenv("CONSOLIDATION_INTERVAL_SEC", "60")),
            INDEXER_POLL_INTERVAL_SEC=int(os.getenv("INDEXER_POLL_INTERVAL_SEC", "2")),
            MAX_CONTEXT_FACTS=int(os.getenv("MAX_CONTEXT_FACTS", "20")),
            MAX_RECENT_MESSAGES=int(os.getenv("MAX_RECENT_MESSAGES", "10")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        parsed = urlparse(self.LLM_BASE_URL)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid LLM_BASE_URL: {self.LLM_BASE_URL}")
        if self.CHROMA_PATH == self.SQLITE_PATH:
            raise ValueError("CHROMA_PATH must not point to the SQLite file.")
        if self.CHROMA_PATH == self.SQLITE_PATH.parent:
            raise ValueError("CHROMA_PATH must be a separate directory from SQLite.")
        if self.EMBED_BACKEND not in {"sentence_transformers", "lmstudio"}:
            raise ValueError(
                "EMBED_BACKEND must be 'sentence_transformers' or 'lmstudio'."
            )

    def ensure_directories(self) -> None:
        self.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.OBSIDIAN_VAULT_PATH.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config.from_env()
