from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .config import get_config
from .db import (
    insert_consolidation_proposal,
    list_pending_proposal_signatures,
    list_recent_active_facts,
    list_stale_active_facts,
)
from .llm import llm_call
from .models import Fact

logger = logging.getLogger(__name__)

TIERING_SYSTEM_PROMPT = """
You are a memory optimization engine for a personal AI assistant.
Your job is to analyze a set of facts and produce structured proposals
for memory tiering. You must never destroy information - only reorganize it.

---

## Tiers

- active   : fact is recent, specific, or frequently relevant
- cold     : fact is older or redundant but must be preserved
- archived : fact is superseded by a merge - keep for audit only

## Importance levels

- core        : identity, values, long-term goals, explicit user preferences
                -> never auto-tiered, never merged without explicit flag
- contextual  : situational facts, project state, relationships
                -> can be merged if semantically equivalent
- transient   : temporary states, short-term tasks, incidental observations
                -> candidate for cold after 14 days without access

---

## Your output

Respond ONLY with a valid JSON object. No explanation, no markdown, no preamble.

Schema:
{
  "proposals": [
    {
      "type": "merge",
      "source_fact_ids": ["id1", "id2"],
      "merged_content": "single merged fact as plain string",
      "merged_importance": "core | contextual | transient",
      "source_tier_after": "cold",
      "reasoning": "one sentence"
    },
    {
      "type": "tier_change",
      "fact_id": "id",
      "new_tier": "cold | archived",
      "reasoning": "one sentence"
    }
  ]
}

If no changes are needed, return: { "proposals": [] }

---

## Hard rules

1. NEVER propose tier_change to "archived" unless fact_id appears in a merge's source_fact_ids.
2. NEVER merge facts with importance "core" unless the field merged_importance is also "core"
   and the semantic content is identical - not just similar.
3. NEVER produce a merge where merged_content loses specific detail present in any source fact.
   If in doubt, do not merge.
4. source_tier_after for merge proposals must always be "cold", never "archived".
5. Maximum 5 proposals per call.
6. reasoning must be one sentence, no longer.
""".strip()

TIERING_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {"type": "object"},
        }
    },
    "required": ["proposals"],
}


def _normalize_json_object(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in (cleaned, raw.strip()):
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            return candidate[start : end + 1]
    raise ValueError("Consolidator output did not contain a JSON object.")


def _fact_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {
        "id": f"fact_{fact.id:06d}",
        "content": fact.content,
        "importance": fact.importance,
        "tier": fact.tier,
        "created_at": _fact_timestamp(fact.created_at),
        "last_accessed_at": _fact_timestamp(fact.last_accessed_at),
        "access_count": fact.access_count,
    }


def _candidate_facts(limit: int) -> list[Fact]:
    recent_limit = max(limit // 2, 1)
    stale_limit = max(limit - recent_limit, 1)
    recent = list_recent_active_facts(limit=recent_limit)
    stale = list_stale_active_facts(limit=stale_limit)
    merged: list[Fact] = []
    seen: set[int] = set()
    for fact in [*recent, *stale, *list_recent_active_facts(limit=max(limit, 1)), *list_stale_active_facts(limit=max(limit, 1))]:
        if fact.id in seen:
            continue
        seen.add(fact.id)
        merged.append(fact)
        if len(merged) >= limit:
            break
    return merged


def _tiering_user_prompt(facts: list[Fact]) -> str:
    payload = [_fact_payload(fact) for fact in facts]
    return (
        "Analyze the following facts and propose memory optimizations.\n\n"
        f"Current timestamp: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n\n"
        "Facts:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Each fact includes:\n"
        "- id\n"
        "- content\n"
        "- importance (core | contextual | transient)\n"
        "- tier (active | cold)\n"
        "- created_at\n"
        "- last_accessed_at\n"
        "- access_count\n\n"
        "Apply the rules strictly. Return only JSON."
    )


def _parse_proposals(raw: str) -> list[dict[str, Any]]:
    payload = json.loads(_normalize_json_object(raw))
    proposals = payload.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("Consolidator output did not include a proposals array.")
    return [item for item in proposals if isinstance(item, dict)]


def _normalize_fact_ref(raw_id: Any) -> int | None:
    text = str(raw_id or "").strip()
    if not text:
        return None
    if text.startswith("fact_"):
        text = text[5:]
    if text.isdigit():
        normalized = int(text)
        return normalized if normalized > 0 else None
    return None


def _proposal_signature(proposal: dict[str, Any]) -> tuple[str, ...] | None:
    proposal_type = str(proposal.get("type") or "").strip()
    if proposal_type == "merge":
        raw_ids = proposal.get("source_fact_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) < 2:
            return None
        normalized_ids: list[int] = []
        for raw_id in raw_ids:
            normalized_id = _normalize_fact_ref(raw_id)
            if normalized_id is None:
                return None
            normalized_ids.append(normalized_id)
        return ("merge", *[str(item) for item in sorted(set(normalized_ids))])
    if proposal_type == "tier_change":
        fact_id = _normalize_fact_ref(proposal.get("fact_id"))
        new_tier = str(proposal.get("new_tier") or "").strip().lower()
        if fact_id is None or new_tier not in {"cold", "archived"}:
            return None
        return ("tier_change", str(fact_id), new_tier)
    return None


def _normalize_merge_proposal(proposal: dict[str, Any], facts_by_id: dict[int, Fact]) -> dict[str, Any] | None:
    raw_ids = proposal.get("source_fact_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) < 2:
        return None
    normalized_ids: list[int] = []
    for raw_id in raw_ids:
        normalized_id = _normalize_fact_ref(raw_id)
        if normalized_id is None or normalized_id not in facts_by_id:
            return None
        normalized_ids.append(normalized_id)
    normalized_ids = sorted(set(normalized_ids))
    if len(normalized_ids) < 2:
        return None
    if any(facts_by_id[fact_id].tier != "active" for fact_id in normalized_ids):
        return None
    merged_content = str(proposal.get("merged_content") or "").strip()
    merged_importance = str(proposal.get("merged_importance") or "").strip().lower()
    reasoning = str(proposal.get("reasoning") or "").strip()
    if not merged_content or merged_importance not in {"core", "contextual", "transient"} or not reasoning:
        return None
    if str(proposal.get("source_tier_after") or "").strip().lower() != "cold":
        return None
    return {
        "type": "merge",
        "source_fact_ids": normalized_ids,
        "merged_content": merged_content,
        "merged_importance": merged_importance,
        "source_tier_after": "cold",
        "reasoning": reasoning,
    }


def _normalize_tier_change_proposal(proposal: dict[str, Any], facts_by_id: dict[int, Fact]) -> dict[str, Any] | None:
    fact_id = _normalize_fact_ref(proposal.get("fact_id"))
    new_tier = str(proposal.get("new_tier") or "").strip().lower()
    reasoning = str(proposal.get("reasoning") or "").strip()
    if fact_id is None or fact_id not in facts_by_id or new_tier not in {"cold", "archived"} or not reasoning:
        return None
    fact = facts_by_id[fact_id]
    if fact.tier == new_tier or fact.status != "active":
        return None
    return {
        "type": "tier_change",
        "fact_id": fact_id,
        "new_tier": new_tier,
        "reasoning": reasoning,
    }


def _normalize_proposal(proposal: dict[str, Any], facts_by_id: dict[int, Fact]) -> dict[str, Any] | None:
    proposal_type = str(proposal.get("type") or "").strip()
    if proposal_type == "merge":
        return _normalize_merge_proposal(proposal, facts_by_id)
    if proposal_type == "tier_change":
        return _normalize_tier_change_proposal(proposal, facts_by_id)
    return None


def _proposal_sort_key(proposal: dict[str, Any]) -> tuple[int, int]:
    proposal_type = str(proposal.get("type") or "").strip()
    if proposal_type == "merge":
        return (0, 0)
    if proposal_type == "tier_change" and str(proposal.get("new_tier") or "").strip().lower() == "cold":
        return (1, 0)
    return (2, 0)


def _propose_memory_optimizations(facts: list[Fact]) -> list[dict[str, Any]]:
    raw = llm_call(
        TIERING_SYSTEM_PROMPT,
        _tiering_user_prompt(facts),
        schema=TIERING_RESPONSE_SCHEMA,
    )
    return _parse_proposals(raw)


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


async def run_consolidator(stop_event: asyncio.Event) -> None:
    config = get_config()
    while not stop_event.is_set():
        try:
            candidate_limit = max(config.MAX_CONTEXT_FACTS * 4, 40)
            facts = _candidate_facts(candidate_limit)
            if facts:
                facts_by_id = {fact.id: fact for fact in facts}
                pending_signatures = list_pending_proposal_signatures()
                proposals = _propose_memory_optimizations(facts)
                created_signatures: set[tuple[str, ...]] = set()
                for raw_proposal in sorted(proposals, key=_proposal_sort_key):
                    proposal = _normalize_proposal(raw_proposal, facts_by_id)
                    if proposal is None:
                        continue
                    signature = _proposal_signature(proposal)
                    if signature is None or signature in pending_signatures or signature in created_signatures:
                        continue
                    insert_consolidation_proposal(proposal)
                    created_signatures.add(signature)
                    if len(created_signatures) >= 5:
                        break
        except Exception:
            logger.exception("Consolidator loop failed")
        await _wait_or_stop(stop_event, config.CONSOLIDATION_INTERVAL_SEC)
