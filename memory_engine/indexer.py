from __future__ import annotations

import asyncio
import logging

from .config import get_config
from .db import (
    claim_pending_outbox_item,
    get_fact_by_id,
    mark_outbox_done,
    mark_outbox_failed,
    recompute_vector_watermark,
)
from .embeddings import embed
from .retrieval import get_collection

logger = logging.getLogger(__name__)


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


async def run_indexer(stop_event: asyncio.Event) -> None:
    config = get_config()
    while not stop_event.is_set():
        item = claim_pending_outbox_item()
        if item is None:
            await _wait_or_stop(stop_event, config.INDEXER_POLL_INTERVAL_SEC)
            continue

        fact = get_fact_by_id(item["fact_id"])
        if fact is None or fact.status != "active":
            mark_outbox_done(item["id"])
            recompute_vector_watermark()
            continue

        try:
            collection = get_collection()
            vector = embed(fact.content)
            metadata = {
                "fact_id": fact.id,
                "version_created": fact.version_created,
                "status": fact.status,
            }
            if fact.source_event_id is not None:
                metadata["source_event_id"] = fact.source_event_id
            collection.upsert(
                ids=[fact.embedding_id or f"fact:{fact.id}"],
                embeddings=[vector],
                documents=[fact.content],
                metadatas=[metadata],
            )
            mark_outbox_done(item["id"])
            recompute_vector_watermark()
        except Exception:
            logger.exception("Indexer failed to embed or upsert fact %s", item["fact_id"])
            mark_outbox_failed(item["id"])
