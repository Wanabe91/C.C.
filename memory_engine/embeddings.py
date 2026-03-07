from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .config import get_config
from .hf_auth import configure_huggingface_auth

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any


@lru_cache(maxsize=1)
def _get_st_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    config = get_config()
    configure_huggingface_auth()
    return _SentenceTransformer(config.EMBED_MODEL)


def clear_embedding_model_cache() -> None:
    _get_st_model.cache_clear()


def embed_many(texts: list[str] | tuple[str, ...]) -> list[list[float]]:
    if not texts:
        return []
    encoded = _get_st_model().encode(list(texts))
    return encoded.tolist()


def embed(text: str) -> list[float]:
    return embed_many([text])[0]
