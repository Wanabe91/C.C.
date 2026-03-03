from __future__ import annotations

from functools import lru_cache

import httpx
from sentence_transformers import SentenceTransformer

from .config import get_config


@lru_cache(maxsize=1)
def _get_st_model() -> SentenceTransformer:
    config = get_config()
    return SentenceTransformer(config.EMBED_MODEL)


def embed(text: str) -> list[float]:
    config = get_config()
    if config.EMBED_BACKEND == "lmstudio":
        response = httpx.post(
            f"{config.LLM_BASE_URL}/embeddings",
            json={"model": config.EMBED_MODEL, "input": text},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            raise RuntimeError("LM Studio embeddings response did not contain any vectors.")
        return list(data[0]["embedding"])
    return _get_st_model().encode(text).tolist()
