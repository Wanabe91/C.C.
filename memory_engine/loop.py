from __future__ import annotations

import json
from typing import Any

from . import planner
from .db import (
    _get_state_version,
    bump_version,
    count_uncompacted_messages,
    create_summary_and_mark_messages,
    db_transaction,
    get_fact_rows_for_ids,
    get_messages_for_compaction,
    get_vector_watermark,
    get_version,
    insert_event,
    insert_fact,
    insert_message,
    insert_outbox,
    list_pending_consolidation_proposals,
    persist_result,
)
from .executor import execute_step, revalidate
from .interrupt import InterruptChannel
from .llm import llm_call
from .retrieval import build_context_snapshot
from .working_memory import working_memory


def extract_goal(raw: dict[str, Any]) -> str:
    goal = raw.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    text = raw.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return json.dumps(raw, ensure_ascii=False, sort_keys=True)


def extract_facts_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for item in result.get("facts", []):
        if isinstance(item, str) and item.strip():
            facts.append({"content": item.strip(), "meta": {}})
        elif isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            if content:
                facts.append({"content": content, "meta": item.get("meta", {})})
    return facts


def _maybe_compact_messages() -> None:
    if count_uncompacted_messages() <= 50:
        return
    rows = get_messages_for_compaction(limit=30)
    if len(rows) < 30:
        return
    transcript = "\n".join(f"{row['role']}: {row['content']}" for row in rows)
    summary = llm_call(
        "Summarize these older assistant messages for future planning. Preserve durable facts and open commitments.",
        transcript,
    )
    create_summary_and_mark_messages([row["id"] for row in rows], summary)


def apply_pending_proposals(version: int) -> int:
    applied = 0
    for proposal in list_pending_consolidation_proposals():
        try:
            source_fact_ids = json.loads(proposal["source_fact_ids"])
            if not isinstance(source_fact_ids, list) or not source_fact_ids:
                raise ValueError("Proposal source ids must be a non-empty list.")
            normalized_ids = [int(item) for item in source_fact_ids]
        except (TypeError, ValueError, json.JSONDecodeError):
            with db_transaction() as conn:
                conn.execute(
                    "UPDATE consolidation_proposals SET status = 'rejected' WHERE id = ?",
                    (proposal["id"],),
                )
            continue

        with db_transaction() as conn:
            facts = get_fact_rows_for_ids(conn, normalized_ids)
            by_id = {fact.id: fact for fact in facts}
            if any(fid not in by_id or by_id[fid].status != "active" for fid in normalized_ids):
                conn.execute(
                    "UPDATE consolidation_proposals SET status = 'rejected' WHERE id = ?",
                    (proposal["id"],),
                )
                continue
            conn.execute(
                f"""
                UPDATE facts
                SET status = 'superseded', version_superseded = ?
                WHERE id IN ({",".join("?" for _ in normalized_ids)})
                """,
                (version, *normalized_ids),
            )
            merged_fact_id = insert_fact(
                conn,
                {"content": proposal["proposed_content"], "meta": {"merged_from": normalized_ids}},
                version,
            )
            insert_outbox(conn, merged_fact_id)
            conn.execute(
                "UPDATE consolidation_proposals SET status = 'applied' WHERE id = ?",
                (proposal["id"],),
            )
            applied += 1
    return applied


async def ingest_event(raw: dict[str, Any], interrupt: InterruptChannel) -> list[str]:
    raw_text = str(raw.get("text") or "").strip()
    assistant_messages: list[str] = []
    with db_transaction() as conn:
        event_id = insert_event(conn, raw)
        if raw_text:
            current_version = _get_state_version(conn)
            insert_message(conn, "user", raw_text, state_version=current_version)
        V = bump_version(conn)

    working_memory.update(raw, V)
    W = get_vector_watermark()
    snapshot = build_context_snapshot(V, W, raw_text)
    if apply_pending_proposals(V):
        W = get_vector_watermark()
        snapshot = build_context_snapshot(V, W, raw_text)
    goal = extract_goal(raw)
    steps = planner.plan(snapshot, goal)

    step_index = 0
    while step_index < len(steps):
        if await interrupt.check():
            return assistant_messages
        step = steps[step_index]
        cur_V = get_version()
        if not revalidate(step, cur_V, snapshot.state_version):
            W2 = get_vector_watermark()
            snapshot = build_context_snapshot(cur_V, W2, raw_text)
            steps = planner.plan(snapshot, goal)
            step_index = 0
            continue
        result = await execute_step(step)
        assistant_message = str(result.get("assistant_message") or "").strip()
        if assistant_message:
            assistant_messages.append(assistant_message)
        with db_transaction() as conn:
            persist_result(conn, result, event_id)
            for fact in extract_facts_from_result(result):
                fact["source_event_id"] = event_id
                fact_id = insert_fact(conn, fact, V)
                insert_outbox(conn, fact_id)
            bump_version(conn)
        _maybe_compact_messages()
        if await interrupt.check():
            return assistant_messages
        step_index += 1
    return assistant_messages
