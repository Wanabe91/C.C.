from __future__ import annotations

import asyncio
import json
import logging
import re

from .config import get_config
from .db import (
    insert_consolidation_proposal,
    list_pending_proposal_pairs,
    list_recent_active_facts,
)
from .llm import llm_call
from .models import Fact

logger = logging.getLogger(__name__)

MERGE_PROMPT = (
    "You are merging two potentially contradictory facts.\n"
    "Output ONLY a JSON object: {{\"merged_fact\": \"<resolved text>\"}}.\n"
    "No explanation. No extra keys. Preserve the most recent information.\n\n"
    "Fact A (version {va}): {content_a}\n"
    "Fact B (version {vb}): {content_b}"
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _looks_mergeable(a: Fact, b: Fact) -> bool:
    tokens_a = _tokens(a.content)
    tokens_b = _tokens(b.content)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    normalized_a = _normalized(a.content)
    normalized_b = _normalized(b.content)
    containment = normalized_a in normalized_b or normalized_b in normalized_a
    return overlap >= 0.6 or containment


def _merge_content(a: Fact, b: Fact) -> str | None:
    raw = llm_call(
        "",
        MERGE_PROMPT.format(
            va=a.version_created,
            vb=b.version_created,
            content_a=a.content,
            content_b=b.content,
        ),
        schema={"type": "object", "properties": {"merged_fact": {"type": "string"}}, "required": ["merged_fact"]},
    )
    payload = json.loads(raw)
    merged_fact = str(payload.get("merged_fact", "")).strip()
    return merged_fact or None


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


async def run_consolidator(stop_event: asyncio.Event) -> None:
    config = get_config()
    while not stop_event.is_set():
        try:
            facts = list_recent_active_facts(limit=max(config.MAX_CONTEXT_FACTS * 4, 40))
            pending_pairs = list_pending_proposal_pairs()
            created_pairs: set[tuple[int, int]] = set()
            for index, fact_a in enumerate(facts):
                for fact_b in facts[index + 1 :]:
                    pair = tuple(sorted((fact_a.id, fact_b.id)))
                    if pair in pending_pairs or pair in created_pairs:
                        continue
                    if not _looks_mergeable(fact_a, fact_b):
                        continue
                    merged = _merge_content(fact_a, fact_b)
                    if not merged:
                        continue
                    insert_consolidation_proposal(list(pair), merged)
                    created_pairs.add(pair)
                    if len(created_pairs) >= 5:
                        break
                if len(created_pairs) >= 5:
                    break
        except Exception:
            logger.exception("Consolidator loop failed")
        await _wait_or_stop(stop_event, config.CONSOLIDATION_INTERVAL_SEC)
