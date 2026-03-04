from __future__ import annotations

import json
import logging
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
    insert_planner_run,
    insert_step_trace,
    list_pending_consolidation_proposals,
    persist_result,
)
from .executor import execute_step, revalidate
from .interrupt import InterruptChannel
from .llm import llm_call
from .obsidian import write_decision_note, write_event_note, write_weekly_review
from .retrieval import build_context_snapshot
from .working_memory import working_memory

logger = logging.getLogger(__name__)


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
            facts.append({"content": item.strip(), "importance": "contextual", "tier": "active", "meta": {}})
        elif isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            if content:
                facts.append(
                    {
                        "content": content,
                        "importance": item.get("importance"),
                        "tier": item.get("tier"),
                        "meta": item.get("meta", {}),
                    }
                )
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


def _normalize_importance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"core", "contextual", "transient"}:
        return normalized
    return "contextual"


def _normalize_tier_change(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"cold", "archived"}:
        return normalized
    return None


def _normalize_source_fact_ids(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("Proposal source ids must be a non-empty list.")
    normalized_ids = [int(item) for item in value]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Proposal source ids must be unique.")
    return normalized_ids


def _reject_proposal(conn: Any, proposal_id: int) -> None:
    conn.execute(
        "UPDATE consolidation_proposals SET status = 'rejected' WHERE id = ?",
        (proposal_id,),
    )


def _merge_source_meta(meta: dict[str, Any] | None, *, merged_fact_id: int, proposal_id: int, version: int) -> dict[str, Any]:
    updated = dict(meta or {})
    updated["merged_into_fact_id"] = merged_fact_id
    updated["merged_by_proposal_id"] = proposal_id
    updated["merged_at_version"] = version
    return updated


def _archive_source_meta(meta: dict[str, Any] | None, *, proposal_id: int, version: int) -> dict[str, Any]:
    updated = dict(meta or {})
    updated["archived_by_proposal_id"] = proposal_id
    updated["archived_at_version"] = version
    return updated


def _apply_merge_proposal(conn: Any, proposal_record: dict[str, Any], proposal: dict[str, Any], version: int) -> bool:
    try:
        normalized_ids = _normalize_source_fact_ids(proposal.get("source_fact_ids"))
    except (TypeError, ValueError):
        _reject_proposal(conn, proposal_record["id"])
        return False

    merged_content = str(proposal.get("merged_content") or "").strip()
    if not merged_content or str(proposal.get("source_tier_after") or "").strip().lower() != "cold":
        _reject_proposal(conn, proposal_record["id"])
        return False

    facts = get_fact_rows_for_ids(conn, normalized_ids)
    by_id = {fact.id: fact for fact in facts}
    if any(fid not in by_id for fid in normalized_ids):
        _reject_proposal(conn, proposal_record["id"])
        return False
    if any(by_id[fid].status != "active" or by_id[fid].tier != "active" for fid in normalized_ids):
        _reject_proposal(conn, proposal_record["id"])
        return False

    merged_fact_id = insert_fact(
        conn,
        {
            "content": merged_content,
            "importance": _normalize_importance(proposal.get("merged_importance")),
            "tier": "active",
            "meta": {
                "merged_from": normalized_ids,
                "merge_source_proposal_id": proposal_record["id"],
            },
        },
        version,
    )
    insert_outbox(conn, merged_fact_id)
    for fact_id in normalized_ids:
        source_fact = by_id[fact_id]
        conn.execute(
            """
            UPDATE facts
            SET tier = 'cold', meta_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    _merge_source_meta(
                        source_fact.meta,
                        merged_fact_id=merged_fact_id,
                        proposal_id=proposal_record["id"],
                        version=version,
                    ),
                    ensure_ascii=False,
                ),
                fact_id,
            ),
        )
    conn.execute(
        "UPDATE consolidation_proposals SET status = 'applied' WHERE id = ?",
        (proposal_record["id"],),
    )
    return True


def _apply_tier_change_proposal(conn: Any, proposal_record: dict[str, Any], proposal: dict[str, Any], version: int) -> bool:
    try:
        fact_id = int(proposal.get("fact_id"))
    except (TypeError, ValueError):
        _reject_proposal(conn, proposal_record["id"])
        return False
    new_tier = _normalize_tier_change(proposal.get("new_tier"))
    if fact_id <= 0 or new_tier is None:
        _reject_proposal(conn, proposal_record["id"])
        return False

    facts = get_fact_rows_for_ids(conn, [fact_id])
    if not facts:
        _reject_proposal(conn, proposal_record["id"])
        return False
    fact = facts[0]
    if fact.status != "active":
        _reject_proposal(conn, proposal_record["id"])
        return False
    if fact.tier == new_tier:
        _reject_proposal(conn, proposal_record["id"])
        return False

    if new_tier == "cold":
        conn.execute(
            "UPDATE facts SET tier = 'cold' WHERE id = ?",
            (fact_id,),
        )
    else:
        merged_into_fact_id = (fact.meta or {}).get("merged_into_fact_id")
        if fact.tier != "cold" or not merged_into_fact_id:
            _reject_proposal(conn, proposal_record["id"])
            return False
        conn.execute(
            """
            UPDATE facts
            SET tier = 'archived', status = 'superseded', version_superseded = ?, meta_json = ?
            WHERE id = ?
            """,
            (
                version,
                json.dumps(
                    _archive_source_meta(
                        fact.meta,
                        proposal_id=proposal_record["id"],
                        version=version,
                    ),
                    ensure_ascii=False,
                ),
                fact_id,
            ),
        )

    conn.execute(
        "UPDATE consolidation_proposals SET status = 'applied' WHERE id = ?",
        (proposal_record["id"],),
    )
    return True


def apply_pending_proposals(version: int) -> int:
    applied = 0
    for proposal_record in list_pending_consolidation_proposals():
        proposal = proposal_record.get("proposal")
        if not isinstance(proposal, dict):
            with db_transaction() as conn:
                _reject_proposal(conn, proposal_record["id"])
            continue
        with db_transaction() as conn:
            proposal_type = str(proposal.get("type") or "").strip()
            if proposal_type == "merge":
                applied += int(_apply_merge_proposal(conn, proposal_record, proposal, version))
                continue
            if proposal_type == "tier_change":
                applied += int(_apply_tier_change_proposal(conn, proposal_record, proposal, version))
                continue
            _reject_proposal(conn, proposal_record["id"])
    return applied


async def ingest_event(raw: dict[str, Any], interrupt: InterruptChannel) -> list[str]:
    raw_text = str(raw.get("text") or "").strip()
    assistant_messages: list[str] = []
    decision_fact_ids: list[int] = []
    generated_reviews: list[dict[str, Any]] = []
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
    plan_run = planner.plan(snapshot, goal)
    with db_transaction() as conn:
        planner_run_id = insert_planner_run(
            conn,
            event_id=event_id,
            goal=plan_run.goal,
            snapshot=plan_run.snapshot,
            system_prompt=plan_run.system_prompt,
            user_prompt=plan_run.user_prompt,
            planner_status=plan_run.planner_status,
            first_response=plan_run.first_response,
            repair_prompt=plan_run.repair_prompt,
            repair_response=plan_run.repair_response,
            final_steps=[planner.serialize_step(step) for step in plan_run.steps],
            error_text=plan_run.error,
        )
    steps = plan_run.steps

    try:
        step_index = 0
        while step_index < len(steps):
            if await interrupt.check():
                return assistant_messages
            step = steps[step_index]
            cur_V = get_version()
            is_valid, rejection_reason = revalidate(step, cur_V, snapshot.state_version)
            if not is_valid:
                with db_transaction() as conn:
                    insert_step_trace(
                        conn,
                        planner_run_id=planner_run_id,
                        step_index=step_index,
                        action=step.action,
                        tool=step.tool,
                        args=step.args,
                        precondition_fact_ids=step.precondition_fact_ids,
                        reasoning=step.reasoning,
                        snapshot_state_version=snapshot.state_version,
                        current_state_version=cur_V,
                        revalidation_status="rejected",
                        rejection_reason=rejection_reason,
                        execution_status=None,
                        result=None,
                    )
                W2 = get_vector_watermark()
                snapshot = build_context_snapshot(cur_V, W2, raw_text)
                plan_run = planner.plan(snapshot, goal)
                with db_transaction() as conn:
                    planner_run_id = insert_planner_run(
                        conn,
                        event_id=event_id,
                        goal=plan_run.goal,
                        snapshot=plan_run.snapshot,
                        system_prompt=plan_run.system_prompt,
                        user_prompt=plan_run.user_prompt,
                        planner_status=plan_run.planner_status,
                        first_response=plan_run.first_response,
                        repair_prompt=plan_run.repair_prompt,
                        repair_response=plan_run.repair_response,
                        final_steps=[planner.serialize_step(item) for item in plan_run.steps],
                        error_text=plan_run.error,
                    )
                steps = plan_run.steps
                step_index = 0
                continue
            result = await execute_step(step)
            assistant_message = str(result.get("assistant_message") or "").strip()
            if assistant_message:
                assistant_messages.append(assistant_message)
            with db_transaction() as conn:
                persist_result(conn, result, event_id)
                created_fact_ids: list[int] = []
                for fact in extract_facts_from_result(result):
                    fact["source_event_id"] = event_id
                    fact_id = insert_fact(conn, fact, V)
                    insert_outbox(conn, fact_id)
                    created_fact_ids.append(fact_id)
                logged_result = dict(result)
                if created_fact_ids:
                    logged_result["persisted_fact_ids"] = created_fact_ids
                insert_step_trace(
                    conn,
                    planner_run_id=planner_run_id,
                    step_index=step_index,
                    action=step.action,
                    tool=step.tool,
                    args=step.args,
                    precondition_fact_ids=step.precondition_fact_ids,
                    reasoning=step.reasoning,
                    snapshot_state_version=snapshot.state_version,
                    current_state_version=cur_V,
                    revalidation_status="executed",
                    rejection_reason=None,
                    execution_status=str(result.get("status") or "").strip() or "ok",
                    result=logged_result,
                )
                bump_version(conn)
            decision_fact_ids.extend(created_fact_ids)
            generated_reviews.extend(result.get("generated_reviews", []))
            _maybe_compact_messages()
            if await interrupt.check():
                return assistant_messages
            step_index += 1
        return assistant_messages
    finally:
        for fact_id in decision_fact_ids:
            try:
                write_decision_note(fact_id)
            except Exception:
                logger.exception("Failed to write decision note for fact %s", fact_id)
        for review in generated_reviews:
            try:
                write_weekly_review(review)
            except Exception:
                logger.exception("Failed to write weekly review note for week %s", review.get("week_key"))
        try:
            write_event_note(event_id, assistant_messages)
        except Exception:
            logger.exception("Failed to write inbox note for event %s", event_id)
